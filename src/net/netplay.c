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
 *   hash of its build and game files (build_identity: they must match) and
 *   gets its player number
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
static char s_why[256];                          /* why the session did not start */
static uint32_t s_hash_f[NP_MAXP];
static uint64_t s_hash_v[NP_MAXP][2];            /* last two checks per player */
static uint32_t s_hash_fr[NP_MAXP][2];
static volatile LONG s_failed;
static void broadcast(const void *p, int n, int except);
static void dump_pages(uint32_t f);         /* below, with the hashing */

int xbox_NetplayActive(void) { return s_on; }

/* Leaving (the window closed): tell the others, who stop with "player n
 * left" at once instead of after 15 s of silence. */
void xbox_NetplayLeave(void)
{
    struct { uint8_t type, player; uint16_t pad; uint32_t desync_frame; } b;
    int k;
    if (!s_on || s_sock == INVALID_SOCKET)
        return;
    memset(&b, 0, sizeof b);
    b.type = 6;                                  /* PK_BYE */
    b.player = (uint8_t)s_me;
    for (k = 0; k < 3; k++)
        broadcast(&b, sizeof b, -1);
}
int xbox_NetplayPlayer(void) { return s_me; }
int xbox_NetplayPlayers(void) { return s_np; }

/* ---- packets --------------------------------------------------------------- */

#pragma pack(push, 1)
typedef struct { uint8_t type, player, players, delay; uint64_t exe; } PkHello;
typedef struct { uint8_t type, player; uint16_t count; uint32_t first; uint8_t s[NP_REDUND][18]; } PkInput;
typedef struct { uint8_t type, player; uint16_t pad; uint32_t frame; uint64_t hash; } PkHash;
typedef struct { uint8_t type, player; uint16_t pad; uint32_t desync_frame; } PkBye;
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

static uint32_t s_fail_frame;                    /* a desync's frame, for the others */

/* The game stops: say why (log, netplay_report.txt, a message box unless
 * headless), and tell the other players first -- a desync's frame with it,
 * so each of them dumps its own page hashes for that frame too. */
static void fail(const char *why)
{
    char msg[512];
    if (InterlockedExchange(&s_failed, 1))
        return;
    if (s_sock != INVALID_SOCKET) {
        PkBye b;
        int k;
        memset(&b, 0, sizeof b);
        b.type = PK_BYE;
        b.player = (uint8_t)s_me;
        b.desync_frame = s_fail_frame;
        for (k = 0; k < 3; k++)
            broadcast(&b, sizeof b, -1);
    }
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
                    char why[200];
                    dump_pages(f);
                    s_fail_frame = f;
                    snprintf(why, sizeof why, "desync at frame %u (player %d "
                             "%016llX, player %d %016llX); compare every player's "
                             "netplay_desync_f%u_p*.bin", f, p + 1,
                             (unsigned long long)v, j + 1,
                             (unsigned long long)s_hash_v[j][m], f);
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
        } else if (n >= (int)sizeof(PkBye) && buf[0] == PK_BYE) {
            PkBye *b = (PkBye *)buf;
            char why[160];
            if (s_is_host)
                broadcast(b, n, b->player);
            if (b->desync_frame) {
                dump_pages(b->desync_frame);
                s_fail_frame = b->desync_frame;
                snprintf(why, sizeof why, "player %d found a desync at frame %u; "
                         "compare every player's netplay_desync_f%u_p*.bin",
                         b->player + 1, b->desync_frame, b->desync_frame);
            } else {
                snprintf(why, sizeof why, "player %d left the game", b->player + 1);
            }
            fail(why);
        }
        send_local_inputs();
    }
    return 0;
}


/* ---- the session's saves ----------------------------------------------------
 *
 * Every peer must start from the same saves. The host's UDATA/TDATA (its own
 * saves: RECOMP_SAVE_DIR, else the game folder) go to every joiner over TCP
 * on the session's port, and everyone -- the host too -- plays on a copy in
 * %LOCALAPPDATA%\xboxrecomp\netplay\p<n> with an empty Z: cache, which the
 * game rebuilds from the disc the same way everywhere. A network game never
 * writes to anyone's real saves. */
static char s_session[MAX_PATH];

static void rm_tree(const char *dir)
{
    char pat[MAX_PATH], p[MAX_PATH];
    WIN32_FIND_DATAA fd;
    HANDLE h;
    snprintf(pat, sizeof pat, "%s\\*", dir);
    h = FindFirstFileA(pat, &fd);
    if (h != INVALID_HANDLE_VALUE) {
        do {
            if (!strcmp(fd.cFileName, ".") || !strcmp(fd.cFileName, ".."))
                continue;
            snprintf(p, sizeof p, "%s\\%s", dir, fd.cFileName);
            if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
                rm_tree(p);
            else {
                SetFileAttributesA(p, FILE_ATTRIBUTE_NORMAL);
                DeleteFileA(p);
            }
        } while (FindNextFileA(h, &fd));
        FindClose(h);
    }
    RemoveDirectoryA(dir);
}

static void mk_dirs(const char *path)            /* every level of path */
{
    char p[MAX_PATH];
    size_t i;
    snprintf(p, sizeof p, "%s", path);
    for (i = 3; p[i]; i++)
        if (p[i] == '\\') {
            p[i] = 0;
            CreateDirectoryA(p, NULL);
            p[i] = '\\';
        }
    CreateDirectoryA(p, NULL);
}

/* %LOCALAPPDATA%\xboxrecomp\netplay\<port>\p<n>: by port too, so two games
 * on one PC (a loopback test) do not share a folder. */
static void session_dir(int player, int port)
{
    char base[MAX_PATH];
    if (!GetEnvironmentVariableA("LOCALAPPDATA", base, MAX_PATH))
        snprintf(base, sizeof base, ".");
    snprintf(s_session, sizeof s_session, "%s\\xboxrecomp\\netplay\\%d\\p%d", base, port,
             player + 1);
    rm_tree(s_session);
    mk_dirs(s_session);
}

/* Each file under dir as: u16 path length, the path (relative), u64 size, bytes. */
static int send_all(SOCKET s, const void *p, int n)
{
    const char *c = (const char *)p;
    while (n > 0) {
        int k = send(s, c, n, 0);
        if (k <= 0)
            return 0;
        c += k;
        n -= k;
    }
    return 1;
}

static int recv_all(SOCKET s, void *p, int n)
{
    char *c = (char *)p;
    while (n > 0) {
        int k = recv(s, c, n, 0);
        if (k <= 0)
            return 0;
        c += k;
        n -= k;
    }
    return 1;
}

static int send_tree(SOCKET s, const char *root, const char *rel)
{
    char pat[MAX_PATH], full[MAX_PATH], r[MAX_PATH];
    WIN32_FIND_DATAA fd;
    HANDLE h;
    snprintf(pat, sizeof pat, "%s\\%s\\*", root, rel);
    h = FindFirstFileA(pat, &fd);
    if (h == INVALID_HANDLE_VALUE)
        return 1;
    do {
        if (!strcmp(fd.cFileName, ".") || !strcmp(fd.cFileName, ".."))
            continue;
        snprintf(r, sizeof r, "%s\\%s", rel, fd.cFileName);
        snprintf(full, sizeof full, "%s\\%s", root, r);
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            if (!send_tree(s, root, r))
                break;
        } else {
            FILE *f = fopen(full, "rb");
            uint16_t len = (uint16_t)strlen(r);
            uint64_t size = ((uint64_t)fd.nFileSizeHigh << 32) | fd.nFileSizeLow;
            char buf[65536];
            size_t n;
            if (!f)
                continue;
            if (!send_all(s, &len, 2) || !send_all(s, r, len) || !send_all(s, &size, 8)) {
                fclose(f);
                break;
            }
            while ((n = fread(buf, 1, sizeof buf, f)) > 0)
                if (!send_all(s, buf, (int)n))
                    break;
            fclose(f);
        }
    } while (FindNextFileA(h, &fd));
    FindClose(h);
    return 1;
}

static int recv_tree(SOCKET s, const char *root)
{
    int files = 0;
    for (;;) {
        uint16_t len;
        uint64_t size;
        char r[MAX_PATH], full[MAX_PATH], buf[65536], *slash;
        FILE *f;
        if (!recv_all(s, &len, 2))
            return -1;
        if (!len)
            return files;
        if (len >= sizeof r || !recv_all(s, r, len) || !recv_all(s, &size, 8))
            return -1;
        r[len] = 0;
        if (strstr(r, ".."))
            return -1;                          /* nothing outside the session */
        snprintf(full, sizeof full, "%s\\%s", root, r);
        slash = strrchr(full, '\\');
        if (slash) {
            *slash = 0;
            mk_dirs(full);
            *slash = '\\';
        }
        f = fopen(full, "wb");
        while (size > 0) {
            int k = (int)(size < sizeof buf ? size : sizeof buf);
            if (!recv_all(s, buf, k)) {
                if (f)
                    fclose(f);
                return -1;
            }
            if (f)
                fwrite(buf, 1, (size_t)k, f);
            size -= (uint64_t)k;
        }
        if (f)
            fclose(f);
        files++;
    }
}

/* Host: the saves into its own session, then to each joiner. */
static void copy_tree(const char *src, const char *dst)
{
    char pat[MAX_PATH], a[MAX_PATH], b[MAX_PATH];
    WIN32_FIND_DATAA fd;
    HANDLE h;
    mk_dirs(dst);
    snprintf(pat, sizeof pat, "%s\\*", src);
    h = FindFirstFileA(pat, &fd);
    if (h == INVALID_HANDLE_VALUE)
        return;
    do {
        if (!strcmp(fd.cFileName, ".") || !strcmp(fd.cFileName, ".."))
            continue;
        snprintf(a, sizeof a, "%s\\%s", src, fd.cFileName);
        snprintf(b, sizeof b, "%s\\%s", dst, fd.cFileName);
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
            copy_tree(a, b);
        else
            CopyFileA(a, b, FALSE);
    } while (FindNextFileA(h, &fd));
    FindClose(h);
}

/* ---- the session ------------------------------------------------------------ */

static const char *s_build_id;    /* xbox_NetplaySetBuild */

void xbox_NetplaySetBuild(const char *id)
{
    s_build_id = id;
}

static uint64_t fnv(uint64_t h, const void *p, size_t n)
{
    const unsigned char *b = (const unsigned char *)p;
    size_t i;
    for (i = 0; i < n; i++)
        h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

static uint64_t fnv_file(uint64_t h, const char *path)
{
    static unsigned char b[65536];
    FILE *f = fopen(path, "rb");
    size_t n;
    if (!f)
        return fnv(h, "missing", 7);
    while ((n = fread(b, 1, sizeof b, f)) > 0)
        h = fnv(h, b, n);
    fclose(f);
    return h;
}

/* What every player must have alike, as one number in the hello:
 * - the code: the game's build identity (both repos' commits and a hash of
 *   the generated C, from xbox_NetplaySetBuild) -- builds of the same
 *   commits match whichever compiler or OS made them. A build with
 *   uncommitted changes, or none given, uses the exe's own bytes instead.
 * - the game: default.xbe (another disc or region is another game).
 * - the maths: portable x87 transcendentals, on for every network game. */
static uint64_t build_identity(const char *game_dir)
{
    uint64_t h = 1469598103934665603ull;
    char path[MAX_PATH];
    if (s_build_id && *s_build_id && !strstr(s_build_id, "dirty")
        && !strstr(s_build_id, "unknown")) {
        h = fnv(h, s_build_id, strlen(s_build_id));
        fprintf(stderr, "[NET] build: %s\n", s_build_id);
    } else {
        GetModuleFileNameA(NULL, path, MAX_PATH);
        h = fnv_file(h, path);
        fprintf(stderr, "[NET] build: %s; matching the exe's bytes\n",
                s_build_id && *s_build_id ? s_build_id : "(no identity)");
    }
    snprintf(path, sizeof path, "%s\\default.xbe", game_dir ? game_dir : ".");
    h = fnv_file(h, path);
    return fnv(h, "portable-math", 13);
}

static int handshake_host(int port, const char *save_src)
{
    struct sockaddr_in a;
    int joined = 0;
    SOCKET ls;
    memset(&a, 0, sizeof a);
    a.sin_family = AF_INET;
    a.sin_port = htons((u_short)port);
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    if (bind(s_sock, (struct sockaddr *)&a, sizeof a)) {
        fprintf(stderr, "[NET] cannot listen on UDP port %d (%d)\n", port, WSAGetLastError());
        return 0;
    }
    /* The saves go out on TCP, same port: listen now so a joiner that
     * connects right after its welcome is queued. */
    ls = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (ls == INVALID_SOCKET || bind(ls, (struct sockaddr *)&a, sizeof a)
        || listen(ls, NP_MAXP)) {
        fprintf(stderr, "[NET] cannot listen on TCP port %d (%d)\n", port, WSAGetLastError());
        return 0;
    }
    session_dir(0, port);
    {
        char src[MAX_PATH], dst[MAX_PATH];
        snprintf(src, sizeof src, "%s\\UDATA", save_src);
        snprintf(dst, sizeof dst, "%s\\UDATA", s_session);
        copy_tree(src, dst);
        snprintf(src, sizeof src, "%s\\TDATA", save_src);
        snprintf(dst, sizeof dst, "%s\\TDATA", s_session);
        copy_tree(src, dst);
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
            PkBye no;
            memset(&no, 0, sizeof no);
            no.type = PK_BYE;
            send_to(&from, &no, sizeof no);      /* so it can say why */
            fprintf(stderr, "[NET] a player with a different build or game files tried to join; refused\n");
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
    {   /* the saves, to every joiner */
        int k;
        for (k = 0; k < s_np - 1; k++) {
            SOCKET c = accept(ls, NULL, NULL);
            uint16_t end = 0;
            if (c == INVALID_SOCKET) {
                fprintf(stderr, "[NET] a joiner did not fetch the saves\n");
                closesocket(ls);
                return 0;
            }
            send_tree(c, s_session, "UDATA");
            send_tree(c, s_session, "TDATA");
            send_all(c, &end, 2);
            shutdown(c, SD_SEND);
            closesocket(c);
        }
        closesocket(ls);
        fprintf(stderr, "[NET] saves sent to %d player(s)\n", s_np - 1);
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
        if (n >= (int)sizeof(PkHello) && buf[0] == PK_WELCOME && !got_welcome) {
            PkHello *w = (PkHello *)buf;
            SOCKET c;
            int files;
            s_me = w->player;
            s_np = w->players;
            s_delay = w->delay;
            got_welcome = 1;
            session_dir(s_me, port);
            c = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
            if (c == INVALID_SOCKET
                || connect(c, (struct sockaddr *)&s_peer[0], sizeof s_peer[0])) {
                snprintf(s_why, sizeof s_why, "Could not fetch the host's saves (TCP port "
                         "%d). The host's firewall must allow the game on TCP as well as "
                         "UDP.", port);
                return 0;
            }
            files = recv_tree(c, s_session);
            closesocket(c);
            if (files < 0) {
                snprintf(s_why, sizeof s_why, "The host's saves did not arrive whole.");
                return 0;
            }
            fprintf(stderr, "[NET] received the host's saves (%d files)\n", files);
        } else if (n >= (int)sizeof(PkHello) && buf[0] == PK_START && got_welcome) {
            return 1;
        } else if (n >= 1 && buf[0] == PK_BYE && !got_welcome) {
            snprintf(s_why, sizeof s_why, "The host refused: it runs a different build or "
                     "different game files. Every player needs the same build of the "
                     "game and the same disc's files.");
            return 0;
        }
        if (!got_welcome && GetTickCount() - t0 > 120000) {
            snprintf(s_why, sizeof s_why, "No answer from the host at %s:%d in 2 minutes. "
                     "Check the address, that the host is waiting, and its firewall.",
                     ip, port);
            return 0;
        }
    }
}

/* Called by the game before anything reads the schedule or USB settings. */
int xbox_NetplayStart(const char *game_dir)
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
    s_exe_hash = build_identity(game_dir);
    if (!strncmp(e, "host:", 5)) {
        const char *sd = getenv("RECOMP_SAVE_DIR");
        s_is_host = 1;
        ok = handshake_host(atoi(e + 5), sd && *sd ? sd : (game_dir ? game_dir : "."));
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
        const char *hl = getenv("RECOMP_HEADLESS");
        if (s_why[0])
            fprintf(stderr, "[NET] %s\n", s_why);
        fprintf(stderr, "[NET] RECOMP_NET=%s: no network game\n", e);
        fflush(stderr);
        if (s_why[0] && !(hl && *hl == '1')) {
            char msg[400];
            snprintf(msg, sizeof msg, "%s\n\nThe game continues on its own.", s_why);
            MessageBoxA(NULL, msg, "Blinx 2 network game", MB_OK | MB_ICONWARNING | MB_TOPMOST);
        }
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
     * software rendering for anything the guest reads back. RECOMP_NET_VK=1
     * (the Network tab's "Vulkan display") changes only what this player
     * sees: software stays authoritative and Vulkan presents (P10's shadow
     * mode). Headless there is nothing to present. */
    _putenv_s("RECOMP_SCHED_VIRTUAL", "1");
    _putenv_s("RECOMP_SAVE_DIR", s_session);
    {
        char n[8];
        snprintf(n, sizeof n, "%d", s_np);
        _putenv_s("RECOMP_USB_PADS", n);
    }
    {
        const char *vk = getenv("RECOMP_NET_VK"), *hl = getenv("RECOMP_HEADLESS");
        if (vk && *vk == '1' && !(hl && *hl == '1')) {
            _putenv_s("RECOMP_GPU", "vulkan");
            _putenv_s("RECOMP_VK_SHADOW", "1");
            _putenv_s("RECOMP_VK_PRESENT", "1");
        } else {
            _putenv_s("RECOMP_GPU", "software");
        }
    }
    /* One implementation of the x87 maths (fsin, fcos, ...) on every peer:
     * each OS's C library differs in the last bit (src/kernel/recomp_math.c). */
    _putenv_s("RECOMP_PORTABLE_MATH", "1");
    recomp_set_portable_math(1);
    s_on = 1;
    usb_gamepad_set_override(xbox_NetplayPadState);
    CloseHandle(CreateThread(NULL, 0, net_thread, NULL, 0, NULL));
    fprintf(stderr, "[NET] network game: player %d of %d, input delay %d frames; "
            "saves: %s\n", s_me + 1, s_np, s_delay, s_session);
    fflush(stderr);
    return 1;
}

/* ---- the guest side ------------------------------------------------------------- */

/* Guest RAM 0x10000-0x4000000 by 4 KB page; the sync hash is the hash of
 * the page hashes. The page hashes of the last two checks are kept, so a
 * desync can say which pages differ (dump_pages). */
#define NP_PAGE_LO  0x10000u
#define NP_PAGES    ((0x4000000u - NP_PAGE_LO) / 4096u)
static uint64_t s_pg[2][NP_PAGES];
static volatile uint32_t s_pg_frame[2];
static int s_pg_i;

static uint64_t hash_guest(uint32_t f)
{
    const uint64_t *p = (const uint64_t *)((uintptr_t)xbox_GetMemoryOffset() + NP_PAGE_LO);
    uint64_t total = 1469598103934665603ull;
    uint32_t i, k;
    s_pg_i ^= 1;
    for (i = 0; i < NP_PAGES; i++) {
        uint64_t h = 1469598103934665603ull;
        for (k = 0; k < 512; k++, p++)
            h = (h ^ *p) * 1099511628211ull;
        s_pg[s_pg_i][i] = h;
        total = (total ^ h) * 1099511628211ull;
    }
    s_pg_frame[s_pg_i] = f;
    return total;
}

/* netplay_desync_f<frame>_p<player>.bin next to the exe: u32 frame, u32 page
 * count, u32 first page's address, then a u64 hash per 4 KB page. The files
 * of two players, compared, name the pages that differ. */
static void dump_pages(uint32_t f)
{
    int k;
    for (k = 0; k < 2; k++)
        if (s_pg_frame[k] == f) {
            char name[64];
            FILE *o;
            uint32_t hdr[3] = { f, NP_PAGES, NP_PAGE_LO };
            snprintf(name, sizeof name, "netplay_desync_f%u_p%d.bin", f, s_me + 1);
            o = fopen(name, "wb");
            if (o) {
                fwrite(hdr, 4, 3, o);
                fwrite(s_pg[k], 8, NP_PAGES, o);
                fclose(o);
                fprintf(stderr, "[NET] page hashes of frame %u in %s\n", f, name);
            }
            return;
        }
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

    /* RECOMP_NET_TEST_DESYNC=<frame> (testing the desync report): this peer
     * changes one byte of guest RAM at that frame, so the next check fails
     * and the page dumps name exactly that page (0x02FF0000). */
    {
        static int test = -1;
        static uint32_t at;
        if (test < 0) {
            const char *e = getenv("RECOMP_NET_TEST_DESYNC");
            test = e && *e;
            at = test ? (uint32_t)atoi(e) : 0;
        }
        if (test && f >= at) {
            test = 0;
            (*(volatile uint8_t *)((uintptr_t)xbox_GetMemoryOffset() + 0x02FF0000u))++;
            fprintf(stderr, "[NET] test: changed guest 0x02FF0000 at frame %u\n", f);
        }
    }

    /* The sync check, at the first pad report of every 60th frame. */
    if (f && f % NP_CHECK == 0 && f != checked) {
        PkHash pk;
        checked = f;
        memset(&pk, 0, sizeof pk);
        pk.type = PK_HASH;
        pk.player = (uint8_t)s_me;
        pk.frame = f;
        pk.hash = hash_guest(f);
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
