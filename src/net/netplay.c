/*
 * netplay.c -- lockstep multiplayer: every peer runs the whole game, only
 * controller input travels.
 *
 *   RECOMP_NET=host:<udp port>          RECOMP_NET_PLAYERS=<2-4> (default 2)
 *   RECOMP_NET=join:<host ip>:<port>    RECOMP_NET_DELAY=<frames> (default 3)
 *
 * The guest runs in virtual mode (sched.c, RECOMP_SCHED_VIRTUAL): the same
 * inputs at the same frames give the same game on every machine. A frame is
 * a 60th of a second of virtual time, so it is the same frame everywhere.
 *
 * - Session: the host waits for its players; each joiner says hello with a
 *   hash of its exe (it must be the same build) and gets its player number
 *   (= Xbox port, the host is port 1). Then all start from power-on.
 * - Input: each peer samples its own pad (its port-1 bindings) once per
 *   frame and sends it for frame f + delay, the last 16 frames in every
 *   packet so a lost one costs nothing. Joiners send to the host, which
 *   relays (a star: LAN, direct IP or a VPN). The emulated USB pad of port p
 *   reports exactly player p's input for the current frame; if it has not
 *   arrived the guest waits for it -- inside a turn, so everyone simply
 *   pauses. Frames before the delay are neutral.
 * - Sync check: every 60 frames each peer hashes guest RAM at the same point
 *   and sends the hash; a mismatch stops the session with a report.
 */
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "netplay.h"
#include "../kernel/kernel.h"
#include "../kernel/xbox_memory_layout.h"
#include "../usb/usb_gamepad.h"

#pragma comment(lib, "ws2_32.lib")

#define NP_MAXP     4
#define NP_RING     2048
#define NP_REDUND   16
#define NP_FRAME_NS 16666667ull
#define NP_CHECK    60

enum { PK_HELLO = 1, PK_WELCOME, PK_START, PK_INPUT, PK_HASH, PK_BYE };

typedef struct { uint32_t f; uint8_t have; uint8_t s[18]; } NpIn;

static int s_on, s_is_host, s_me, s_np = 2, s_delay = 3;
static SOCKET s_sock = INVALID_SOCKET;
static struct sockaddr_in s_peer[NP_MAXP];       /* host: joiners; join: [0] = host */
static int s_npeer;
static NpIn s_in[NP_MAXP][NP_RING];
static SRWLOCK s_lock = SRWLOCK_INIT;
static uint32_t s_sampled;                       /* local frames sampled so far */
static uint64_t s_exe_hash;
static uint32_t s_hash_f[NP_MAXP];
static uint64_t s_hash_v[NP_MAXP][2];            /* last two checks per player */
static uint32_t s_hash_fr[NP_MAXP][2];
static volatile LONG s_failed;

int xbox_NetplayActive(void) { return s_on; }
int xbox_NetplayPlayer(void) { return s_me; }
int xbox_NetplayPlayers(void) { return s_np; }

/* ---- packets --------------------------------------------------------------- */

#pragma pack(push, 1)
typedef struct { uint8_t type, player, players, delay; uint64_t exe; } PkHello;
typedef struct { uint8_t type, player; uint16_t count; uint32_t first; uint8_t s[NP_REDUND][18]; } PkInput;
typedef struct { uint8_t type, player; uint16_t pad; uint32_t frame; uint64_t hash; } PkHash;
#pragma pack(pop)

static void send_to(const struct sockaddr_in *a, const void *p, int n)
{
    sendto(s_sock, (const char *)p, n, 0, (const struct sockaddr *)a, sizeof *a);
}

/* Host: to every joiner except `except` (a player number, or -1). */
static void broadcast(const void *p, int n, int except)
{
    int i;
    for (i = 0; i < s_npeer; i++)
        if (s_is_host ? i + 1 != except : 1)
            send_to(&s_peer[i], p, n);
}

static void store_input(int player, uint32_t f, const uint8_t *s)
{
    NpIn *e;
    if (player < 0 || player >= NP_MAXP)
        return;
    e = &s_in[player][f % NP_RING];
    AcquireSRWLockExclusive(&s_lock);
    if (!(e->have && e->f == f)) {
        e->f = f;
        memcpy(e->s, s, 18);
        e->have = 1;
    }
    ReleaseSRWLockExclusive(&s_lock);
}

static void send_local_inputs(void)
{
    PkInput pk;
    uint32_t last = s_sampled ? s_sampled - 1 : 0, first, f;
    int n = 0;
    if (!s_sampled)
        return;
    first = last + 1 > NP_REDUND ? last + 1 - NP_REDUND : 0;
    memset(&pk, 0, sizeof pk);
    pk.type = PK_INPUT;
    pk.player = (uint8_t)s_me;
    pk.first = first;
    AcquireSRWLockShared(&s_lock);
    for (f = first; f <= last; f++, n++)
        memcpy(pk.s[n], s_in[s_me][f % NP_RING].s, 18);
    ReleaseSRWLockShared(&s_lock);
    pk.count = (uint16_t)n;
    broadcast(&pk, (int)sizeof pk, -1);
}

static void fail(const char *why)
{
    char msg[512];
    if (InterlockedExchange(&s_failed, 1))
        return;
    snprintf(msg, sizeof msg, "Network game stopped: %s", why);
    fprintf(stderr, "[NET] %s\n", msg);
    fflush(stderr);
    {
        FILE *f = fopen("netplay_report.txt", "a");
        if (f) {
            fprintf(f, "%s\n", msg);
            fclose(f);
        }
    }
    if (!getenv("RECOMP_HEADLESS"))
        MessageBoxA(NULL, msg, "Blinx 2 network game", MB_OK | MB_ICONWARNING | MB_TOPMOST);
    TerminateProcess(GetCurrentProcess(), 5);
}

static void check_hashes(int p)
{
    int k, j;
    /* Compare player p's latest hash against everyone's at the same frame. */
    for (k = 0; k < 2; k++) {
        uint32_t f = s_hash_fr[p][k];
        uint64_t v = s_hash_v[p][k];
        if (!f)
            continue;
        for (j = 0; j < s_np; j++) {
            int m;
            if (j == p)
                continue;
            for (m = 0; m < 2; m++)
                if (s_hash_fr[j][m] == f && s_hash_v[j][m] != v) {
                    char why[160];
                    snprintf(why, sizeof why, "desync at frame %u (player %d "
                             "%016llX, player %d %016llX)", f, p + 1,
                             (unsigned long long)v, j + 1,
                             (unsigned long long)s_hash_v[j][m]);
                    fail(why);
                }
        }
    }
}

static void note_hash(int p, uint32_t f, uint64_t v)
{
    if (p < 0 || p >= NP_MAXP)
        return;
    AcquireSRWLockExclusive(&s_lock);
    if (s_hash_fr[p][1] != f && s_hash_fr[p][0] != f) {
        s_hash_fr[p][0] = s_hash_fr[p][1];
        s_hash_v[p][0] = s_hash_v[p][1];
        s_hash_fr[p][1] = f;
        s_hash_v[p][1] = v;
    }
    ReleaseSRWLockExclusive(&s_lock);
    check_hashes(p);
}

/* Receives inputs and hashes (the host relays them), and resends this
 * peer's recent inputs every 10 ms so a lost packet is repaired at once. */
static DWORD WINAPI net_thread(LPVOID unused)
{
    char buf[1024];
    (void)unused;
    for (;;) {
        struct sockaddr_in from;
        int fl = sizeof from;
        int n = recvfrom(s_sock, buf, sizeof buf, 0, (struct sockaddr *)&from, &fl);
        if (n >= (int)sizeof(PkInput) && buf[0] == PK_INPUT) {
            PkInput *pk = (PkInput *)buf;
            int i;
            for (i = 0; i < pk->count && i < NP_REDUND; i++)
                store_input(pk->player, pk->first + (uint32_t)i, pk->s[i]);
            if (s_is_host)
                broadcast(pk, n, pk->player);
        } else if (n >= (int)sizeof(PkHash) && buf[0] == PK_HASH) {
            PkHash *pk = (PkHash *)buf;
            note_hash(pk->player, pk->frame, pk->hash);
            if (s_is_host)
                broadcast(pk, n, pk->player);
        } else if (n >= 1 && buf[0] == PK_BYE) {
            fail("another player left the game");
        }
        send_local_inputs();
    }
    return 0;
}

/* ---- the session ------------------------------------------------------------ */

static uint64_t exe_hash(void)
{
    char path[MAX_PATH];
    uint64_t h = 1469598103934665603ull;
    FILE *f;
    unsigned char b[65536];
    size_t n, i;
    GetModuleFileNameA(NULL, path, MAX_PATH);
    f = fopen(path, "rb");
    if (!f)
        return 0;
    while ((n = fread(b, 1, sizeof b, f)) > 0)
        for (i = 0; i < n; i++)
            h = (h ^ b[i]) * 1099511628211ull;
    fclose(f);
    return h;
}

static int handshake_host(int port)
{
    struct sockaddr_in a;
    int joined = 0;
    memset(&a, 0, sizeof a);
    a.sin_family = AF_INET;
    a.sin_port = htons((u_short)port);
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    if (bind(s_sock, (struct sockaddr *)&a, sizeof a)) {
        fprintf(stderr, "[NET] cannot listen on UDP port %d (%d)\n", port, WSAGetLastError());
        return 0;
    }
    fprintf(stderr, "[NET] hosting on UDP port %d: waiting for %d player(s)\n",
            port, s_np - 1);
    fflush(stderr);
    while (joined < s_np - 1) {
        char buf[256];
        struct sockaddr_in from;
        int fl = sizeof from, i, known = -1;
        int n = recvfrom(s_sock, buf, sizeof buf, 0, (struct sockaddr *)&from, &fl);
        PkHello w;
        if (n < (int)sizeof(PkHello) || buf[0] != PK_HELLO)
            continue;
        for (i = 0; i < s_npeer; i++)
            if (s_peer[i].sin_addr.s_addr == from.sin_addr.s_addr
                && s_peer[i].sin_port == from.sin_port)
                known = i;
        if (((PkHello *)buf)->exe != s_exe_hash) {
            fprintf(stderr, "[NET] a player with a different build tried to join; refused\n");
            continue;
        }
        if (known < 0) {
            known = s_npeer;
            s_peer[s_npeer++] = from;
            joined++;
            fprintf(stderr, "[NET] player %d joined\n", known + 2);
            fflush(stderr);
        }
        memset(&w, 0, sizeof w);
        w.type = PK_WELCOME;
        w.player = (uint8_t)(known + 1);
        w.players = (uint8_t)s_np;
        w.delay = (uint8_t)s_delay;
        w.exe = s_exe_hash;
        send_to(&from, &w, sizeof w);
    }
    {
        PkHello st;
        int k;
        memset(&st, 0, sizeof st);
        st.type = PK_START;
        st.players = (uint8_t)s_np;
        st.delay = (uint8_t)s_delay;
        for (k = 0; k < 5; k++)                     /* UDP: say it a few times */
            broadcast(&st, sizeof st, -1);
    }
    s_me = 0;
    return 1;
}

static int handshake_join(const char *ip, int port)
{
    PkHello h;
    int got_welcome = 0;
    DWORD t0 = GetTickCount();
    memset(&s_peer[0], 0, sizeof s_peer[0]);
    s_peer[0].sin_family = AF_INET;
    s_peer[0].sin_port = htons((u_short)port);
    if (inet_pton(AF_INET, ip, &s_peer[0].sin_addr) != 1) {
        fprintf(stderr, "[NET] not an IPv4 address: %s\n", ip);
        return 0;
    }
    s_npeer = 1;
    memset(&h, 0, sizeof h);
    h.type = PK_HELLO;
    h.exe = s_exe_hash;
    fprintf(stderr, "[NET] joining %s:%d ...\n", ip, port);
    fflush(stderr);
    for (;;) {
        char buf[256];
        struct sockaddr_in from;
        int fl = sizeof from, n;
        if (!got_welcome)
            send_to(&s_peer[0], &h, sizeof h);
        n = recvfrom(s_sock, buf, sizeof buf, 0, (struct sockaddr *)&from, &fl);
        if (n >= (int)sizeof(PkHello) && buf[0] == PK_WELCOME) {
            PkHello *w = (PkHello *)buf;
            s_me = w->player;
            s_np = w->players;
            s_delay = w->delay;
            got_welcome = 1;
        } else if (n >= (int)sizeof(PkHello) && buf[0] == PK_START && got_welcome) {
            return 1;
        }
        if (!got_welcome && GetTickCount() - t0 > 120000) {
            fprintf(stderr, "[NET] no answer from the host in 2 minutes\n");
            return 0;
        }
    }
}

/* Called by the game before anything reads the schedule or USB settings. */
int xbox_NetplayStart(void)
{
    const char *e = getenv("RECOMP_NET");
    WSADATA wd;
    int ok = 0;
    if (!e || !*e)
        return 0;
    if (getenv("RECOMP_NET_PLAYERS"))
        s_np = atoi(getenv("RECOMP_NET_PLAYERS"));
    if (s_np < 2 || s_np > NP_MAXP)
        s_np = 2;
    if (getenv("RECOMP_NET_DELAY"))
        s_delay = atoi(getenv("RECOMP_NET_DELAY"));
    if (s_delay < 1 || s_delay > 30)
        s_delay = 3;
    if (WSAStartup(MAKEWORD(2, 2), &wd))
        return 0;
    s_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (s_sock == INVALID_SOCKET)
        return 0;
    {
        DWORD to = 200;                             /* handshake polls */
        setsockopt(s_sock, SOL_SOCKET, SO_RCVTIMEO, (const char *)&to, sizeof to);
    }
    s_exe_hash = exe_hash();
    if (!strncmp(e, "host:", 5)) {
        s_is_host = 1;
        ok = handshake_host(atoi(e + 5));
    } else if (!strncmp(e, "join:", 5)) {
        char ip[64];
        const char *c = strrchr(e + 5, ':');
        size_t n = c ? (size_t)(c - (e + 5)) : 0;
        if (c && n < sizeof ip) {
            memcpy(ip, e + 5, n);
            ip[n] = 0;
            ok = handshake_join(ip, atoi(c + 1));
        }
    }
    if (!ok) {
        fprintf(stderr, "[NET] RECOMP_NET=%s: no network game\n", e);
        fflush(stderr);
        return 0;
    }
    {
        DWORD to = 10;                              /* the resend cadence */
        setsockopt(s_sock, SOL_SOCKET, SO_RCVTIMEO, (const char *)&to, sizeof to);
    }
    {   /* frames before the delay: nothing pressed, for everyone */
        static const uint8_t zero[18];
        uint32_t f;
        int p;
        for (p = 0; p < s_np; p++)
            for (f = 0; f < (uint32_t)s_delay; f++)
                store_input(p, f, zero);
        s_sampled = (uint32_t)s_delay;
    }
    /* What every peer must run alike: virtual mode, one pad per player,
     * software rendering for anything the guest reads back. */
    _putenv_s("RECOMP_SCHED_VIRTUAL", "1");
    {
        char n[8];
        snprintf(n, sizeof n, "%d", s_np);
        _putenv_s("RECOMP_USB_PADS", n);
    }
    _putenv_s("RECOMP_GPU", "software");
    /* One implementation of the x87 maths (fsin, fcos, ...) on every peer:
     * each OS's C library differs in the last bit (P12, RECOMP_PORTABLE_MATH). */
    _putenv_s("RECOMP_PORTABLE_MATH", "1");
    s_on = 1;
    usb_gamepad_set_override(xbox_NetplayPadState);
    CloseHandle(CreateThread(NULL, 0, net_thread, NULL, 0, NULL));
    fprintf(stderr, "[NET] network game: player %d of %d, input delay %d frames\n",
            s_me + 1, s_np, s_delay);
    fflush(stderr);
    return 1;
}

/* ---- the guest side ------------------------------------------------------------- */

static uint64_t hash_guest(void)
{
    const uint64_t *p = (const uint64_t *)((uintptr_t)xbox_GetMemoryOffset() + 0x10000);
    const uint64_t *e = (const uint64_t *)((uintptr_t)xbox_GetMemoryOffset() + 0x4000000);
    uint64_t h = 1469598103934665603ull;
    for (; p < e; p++)
        h = (h ^ *p) * 1099511628211ull;
    return h;
}

/* The emulated pad of `port` for the current frame (usb_gamepad.c): exactly
 * that player's input, waited for if it has not arrived. Runs in the OHCI
 * agent's turn, so a wait pauses the whole guest, the same on every peer. */
int xbox_NetplayPadState(int port, uint8_t *state18)
{
    uint32_t f = (uint32_t)(xbox_SchedVirtualNs() / NP_FRAME_NS);
    static uint32_t checked;
    DWORD t0;

    /* This peer's own input, sampled once per frame ahead by the delay. */
    while (s_sampled <= f + (uint32_t)s_delay) {
        uint8_t local[18];
        usb_gamepad_sample_local(local);
        store_input(s_me, s_sampled, local);
        s_sampled++;
    }
    send_local_inputs();

    /* The sync check, at the first pad report of every 60th frame. */
    if (f && f % NP_CHECK == 0 && f != checked) {
        PkHash pk;
        checked = f;
        memset(&pk, 0, sizeof pk);
        pk.type = PK_HASH;
        pk.player = (uint8_t)s_me;
        pk.frame = f;
        pk.hash = hash_guest();
        note_hash(s_me, f, pk.hash);
        if (f % (NP_CHECK * 10) == 0) {     /* a line every 10 s of play */
            fprintf(stderr, "[NET] frame %u: guest hash %016llX\n", f,
                    (unsigned long long)pk.hash);
            fflush(stderr);
        }
        broadcast(&pk, sizeof pk, -1);
    }

    if (port >= s_np) {
        memset(state18, 0, 18);
        return 0;
    }
    t0 = GetTickCount();
    for (;;) {
        NpIn *e = &s_in[port][f % NP_RING];
        int ok;
        AcquireSRWLockShared(&s_lock);
        ok = e->have && e->f == f;
        if (ok)
            memcpy(state18, e->s, 18);
        ReleaseSRWLockShared(&s_lock);
        if (ok)
            return 1;
        if (GetTickCount() - t0 > 15000) {
            char why[96];
            snprintf(why, sizeof why, "no input from player %d for 15 s (frame %u)",
                     port + 1, f);
            fail(why);
        }
        Sleep(1);
    }
}
