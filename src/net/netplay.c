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
#include "net_os.h"         /* sockets etc.: Winsock or POSIX; first, for winsock2.h */
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "netplay.h"
#include "../kernel/kernel.h"
#include "../kernel/xbox_memory_layout.h"
#include "../usb/usb_gamepad.h"

#define NP_MAXP     4
#define NP_RING     2048
#define NP_REDUND   16
#define NP_FRAME_NS 16666667ull
#define NP_CHECK    60

enum { PK_HELLO = 1, PK_WELCOME, PK_START, PK_INPUT, PK_HASH, PK_BYE };

/* One player's frame: the 18 bytes of pad state, then an optional cheat
 * event (a flag and 22 bytes, cheats.c's request), so a cheat applies on every
 * peer at the same frame. NP_PROTO is in the join identity: peers whose
 * records differ cannot join each other. */
#define NP_PROTO    3
#define NP_REC      (18 + 1 + 22)
#define NP_EV_FLAG  18
#define NP_EV       19
typedef struct { uint32_t f; uint8_t have; uint8_t s[NP_REC]; } NpIn;

static int s_on, s_is_host, s_me, s_np = 2, s_delay = 3;
static SOCKET s_sock = INVALID_SOCKET;
static struct sockaddr_in s_peer[NP_MAXP];       /* host: joiners; join: [0] = host */
static int s_npeer;
static NpIn s_in[NP_MAXP][NP_RING];
static SRWLOCK s_lock = SRWLOCK_INIT;
static uint32_t s_sampled;                       /* local frames sampled so far */
static uint64_t s_exe_hash;
/* How every peer renders. The guest's render targets are part of the sync
 * hash, so this is the session's choice (the host's), not each player's:
 * - NP_MODE_SOFTWARE: the software renderer draws everything (as before);
 *   RECOMP_NET_VK=1 lets a player present through Vulkan on top.
 * - NP_MODE_DEPTH: Vulkan draws what is seen; the software executor does
 *   only what the guest can see of it -- depth, stencil, coverage and the
 *   visibility reports -- and no colour reaches guest memory (P10's
 *   renderer side: RECOMP_SW_DEPTH_ONLY). Much cheaper on a slow CPU.
 * The host takes RECOMP_NET_RENDER=software|depth (default software). */
enum { NP_MODE_SOFTWARE = 0, NP_MODE_DEPTH = 1 };
enum { NP_CAN_VULKAN = 1 };

#if defined(XBOXRECOMP_HAVE_VULKAN)
extern int xbox_vk_usable(void);                 /* src/gpu_vk */
#endif

/* Whether this PC can render a depth-only game (it needs Vulkan). */
static int vulkan_usable(void)
{
    static int ok = -1;
#if defined(XBOXRECOMP_HAVE_VULKAN)
    if (ok < 0)
        ok = xbox_vk_usable() ? 1 : 0;
#else
    ok = 0;
#endif
    return ok;
}
static int s_mode = NP_MODE_SOFTWARE;
static const char *const k_mode_name[] = { "software", "depth-only + Vulkan display" };
static char s_why[256];                          /* why the session did not start */
static char s_host_info[400];                    /* the host's status lines */
static uint32_t s_hash_f[NP_MAXP];
static uint64_t s_hash_v[NP_MAXP][2];            /* last two checks per player */
static uint32_t s_hash_fr[NP_MAXP][2];
static volatile LONG s_failed;
static uint64_t s_wait_ticks;                    /* QPC ticks spent waiting for input */
static volatile DWORD s_heard[NP_MAXP];          /* when each player was last heard from */
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
/* mode: in WELCOME, how every peer renders, chosen by the host (NP_MODE_*);
 * in HELLO, what the joiner can do (NP_CAN_*). */
typedef struct { uint8_t type, player, players, delay; uint64_t exe; uint8_t mode, pad[3]; } PkHello;
typedef struct { uint8_t type, player; uint16_t count; uint32_t first; uint8_t s[NP_REDUND][NP_REC]; } PkInput;
typedef struct { uint8_t type, player; uint16_t pad; uint32_t frame; uint64_t hash; } PkHash;
/* A refused hello is answered with a BYE whose reason says why. */
enum { BYE_LEFT = 0, BYE_BUILD = 1, BYE_FULL = 2, BYE_NOVULKAN = 3 };
typedef struct { uint8_t type, player; uint16_t reason; uint32_t desync_frame; } PkBye;

static void send_to(const struct sockaddr_in *a, const void *p, int n);

static void refuse(const struct sockaddr_in *to, int reason)
{
    PkBye no;
    memset(&no, 0, sizeof no);
    no.type = PK_BYE;
    no.reason = (uint16_t)reason;
    send_to(to, &no, sizeof no);
}
#pragma pack(pop)

static void send_to(const struct sockaddr_in *a, const void *p, int n)
{
    sendto(s_sock, (const char *)p, n, NP_SENDFLAGS, (const struct sockaddr *)a, sizeof *a);
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
        memcpy(e->s, s, NP_REC);
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
        memcpy(pk.s[n], s_in[s_me][f % NP_RING].s, NP_REC);
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
        if (n >= 2 && (buf[0] == PK_INPUT || buf[0] == PK_HASH)
            && (uint8_t)buf[1] < NP_MAXP)
            s_heard[(uint8_t)buf[1]] = GetTickCount();
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
        } else if (n >= (int)sizeof(PkHello) && buf[0] == PK_HELLO && s_is_host) {
            /* After the start: a joiner that missed every START says hello
             * again and gets it again; anyone else is too late. */
            int i, known = 0;
            for (i = 0; i < s_npeer; i++)
                if (s_peer[i].sin_addr.s_addr == from.sin_addr.s_addr
                    && s_peer[i].sin_port == from.sin_port)
                    known = 1;
            if (known) {
                PkHello st;
                memset(&st, 0, sizeof st);
                st.type = PK_START;
                st.players = (uint8_t)s_np;
                st.delay = (uint8_t)s_delay;
                send_to(&from, &st, sizeof st);
            } else {
                refuse(&from, BYE_FULL);
            }
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
 * <data>/xboxrecomp/netplay/<port>/p<n> with an empty Z: cache, which the
 * game rebuilds from the disc the same way everywhere. A network game never
 * writes to anyone's real saves. */
static char s_session[MAX_PATH];

static void rm_tree(const char *dir)
{
    char p[MAX_PATH];
    NpDir d;
    if (np_dir_open(&d, dir)) {
        while (np_dir_next(&d)) {
            snprintf(p, sizeof p, "%s%c%s", dir, NP_SEP, d.name);
            if (d.is_dir)
                rm_tree(p);
            else
                np_unlink(p);
        }
        np_dir_close(&d);
    }
    np_rmdir(dir);
}

static void mk_dirs(const char *path)            /* every level of path */
{
    char p[MAX_PATH];
    size_t i;
    snprintf(p, sizeof p, "%s", path);
    for (i = 1; p[i]; i++)
        if (p[i] == NP_SEP && p[i - 1] != ':') {
            p[i] = 0;
            np_mkdir(p);
            p[i] = NP_SEP;
        }
    np_mkdir(p);
}

/* <data>/netplay/<port>/p<n> (data: %LOCALAPPDATA%\xboxrecomp, or
 * $XDG_DATA_HOME/xboxrecomp): by port too, so two games on one PC (a
 * loopback test) do not share a folder. */
static void session_dir(int player, int port)
{
    char base[MAX_PATH];
    np_data_dir(base, sizeof base);
    snprintf(s_session, sizeof s_session, "%s%cnetplay%c%d%cp%d", base, NP_SEP, NP_SEP,
             port, NP_SEP, player + 1);
    rm_tree(s_session);
    mk_dirs(s_session);
}

/* Each file under dir as: u16 path length, the path (relative), u64 size, bytes. */
static int send_all(SOCKET s, const void *p, int n)
{
    const char *c = (const char *)p;
    while (n > 0) {
        int k = (int)send(s, c, n, NP_SENDFLAGS);
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
        int k = (int)recv(s, c, n, 0);
        if (k <= 0)
            return 0;
        c += k;
        n -= k;
    }
    return 1;
}

/* rel is sent with '/' whatever this host uses: the receiver may be another OS. */
static int send_tree(SOCKET s, const char *root, const char *rel)
{
    char dir[MAX_PATH], full[MAX_PATH], r[MAX_PATH];
    NpDir d;
    snprintf(dir, sizeof dir, "%s%c%s", root, NP_SEP, rel);
    np_native_path(dir + strlen(root));
    if (!np_dir_open(&d, dir))
        return 1;
    while (np_dir_next(&d)) {
        snprintf(r, sizeof r, "%s/%s", rel, d.name);
        snprintf(full, sizeof full, "%s%c%s", dir, NP_SEP, d.name);
        if (d.is_dir) {
            if (!send_tree(s, root, r))
                break;
        } else {
            FILE *f = fopen(full, "rb");
            uint16_t len = (uint16_t)strlen(r);
            uint64_t size = d.size;
            char buf[65536];
            size_t n;
            if (!f)
                continue;
            np_wire_path(r);
            if (!send_all(s, &len, 2) || !send_all(s, r, len) || !send_all(s, &size, 8)) {
                fclose(f);
                break;
            }
            while ((n = fread(buf, 1, sizeof buf, f)) > 0)
                if (!send_all(s, buf, (int)n))
                    break;
            fclose(f);
        }
    }
    np_dir_close(&d);
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
        if (strstr(r, "..") || r[0] == '/' || r[0] == '\\' || strchr(r, ':'))
            return -1;                          /* nothing outside the session */
        np_native_path(r);
        snprintf(full, sizeof full, "%s%c%s", root, NP_SEP, r);
        slash = strrchr(full, NP_SEP);
        if (slash) {
            *slash = 0;
            mk_dirs(full);
            *slash = NP_SEP;
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
    char a[MAX_PATH], b[MAX_PATH];
    NpDir d;
    mk_dirs(dst);
    if (!np_dir_open(&d, src))
        return;
    while (np_dir_next(&d)) {
        snprintf(a, sizeof a, "%s%c%s", src, NP_SEP, d.name);
        snprintf(b, sizeof b, "%s%c%s", dst, NP_SEP, d.name);
        if (d.is_dir)
            copy_tree(a, b);
        else
            np_copy_file(a, b);
    }
    np_dir_close(&d);
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
        np_exe_path(path, sizeof path);
        h = fnv_file(h, path);
        fprintf(stderr, "[NET] build: %s; matching the exe's bytes\n",
                s_build_id && *s_build_id ? s_build_id : "(no identity)");
    }
    snprintf(path, sizeof path, "%s%cdefault.xbe", game_dir ? game_dir : ".", NP_SEP);
    h = fnv_file(h, path);
    h = fnv(h, "portable-math", 13);
    {
        const uint32_t proto = NP_PROTO;
        h = fnv(h, &proto, sizeof proto);
    }
    return h;
}

/* ---- the status window -------------------------------------------------------
 *
 * Starting a network game restarts the exe, and the session is set up before
 * the game's own window exists: without this the player saw nothing at all
 * for as long as it took (up to 2 minutes, or for ever while a host waited
 * for players). It shows what is happening, and Cancel, Escape or closing it
 * gives up: the game then starts on its own. Windows only (and not headless);
 * the same lines go to the log everywhere. */
static char s_stat[640];
static volatile LONG s_cancel;                   /* the player gave up */
static volatile LONG s_stat_done;
#if defined(_WIN32)
static SRWLOCK s_stat_lock = SRWLOCK_INIT;
static HANDLE s_stat_thread;

static LRESULT CALLBACK stat_proc(HWND h, UINT m, WPARAM w, LPARAM l)
{
    switch (m) {
    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC dc = BeginPaint(h, &ps);
        RECT r;
        char t[sizeof s_stat];
        AcquireSRWLockShared(&s_stat_lock);
        memcpy(t, s_stat, sizeof t);
        ReleaseSRWLockShared(&s_stat_lock);
        GetClientRect(h, &r);
        r.left += 18; r.top += 16; r.right -= 18; r.bottom -= 56;
        SetBkMode(dc, TRANSPARENT);
        SelectObject(dc, GetStockObject(DEFAULT_GUI_FONT));
        DrawTextA(dc, t, -1, &r, DT_WORDBREAK | DT_NOPREFIX);
        EndPaint(h, &ps);
        return 0;
    }
    case WM_COMMAND:
        if (LOWORD(w) == 1)
            InterlockedExchange(&s_cancel, 1);
        return 0;
    case WM_CLOSE:
        InterlockedExchange(&s_cancel, 1);
        return 0;
    case WM_TIMER:
        if (s_stat_done) {
            DestroyWindow(h);
            return 0;
        }
        InvalidateRect(h, NULL, TRUE);
        return 0;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcA(h, m, w, l);
}

static DWORD WINAPI stat_thread(LPVOID unused)
{
    WNDCLASSA wc;
    HWND h;
    MSG msg;
    int W = 520, H = 260;
    (void)unused;
    memset(&wc, 0, sizeof wc);
    wc.lpfnWndProc = stat_proc;
    wc.hInstance = GetModuleHandleA(NULL);
    wc.hCursor = LoadCursorA(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_BTNFACE + 1);
    wc.lpszClassName = "Blinx2NetStatus";
    RegisterClassA(&wc);
    h = CreateWindowExA(WS_EX_TOPMOST, wc.lpszClassName, "Blinx 2 network game",
                        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU,
                        (GetSystemMetrics(SM_CXSCREEN) - W) / 2,
                        (GetSystemMetrics(SM_CYSCREEN) - H) / 2, W, H,
                        NULL, NULL, wc.hInstance, NULL);
    if (!h)
        return 0;
    {
        HWND b = CreateWindowA("BUTTON", "Cancel", WS_CHILD | WS_VISIBLE | BS_DEFPUSHBUTTON,
                               W - 130, H - 80, 100, 28, h, (HMENU)(INT_PTR)1,
                               wc.hInstance, NULL);
        SendMessageA(b, WM_SETFONT, (WPARAM)GetStockObject(DEFAULT_GUI_FONT), TRUE);
    }
    ShowWindow(h, SW_SHOWNORMAL);
    SetForegroundWindow(h);
    SetTimer(h, 1, 250, NULL);
    while (GetMessageA(&msg, NULL, 0, 0) > 0) {
        if (msg.message == WM_KEYDOWN && msg.wParam == VK_ESCAPE)
            InterlockedExchange(&s_cancel, 1);
        TranslateMessage(&msg);
        DispatchMessageA(&msg);
    }
    return 0;
}
#endif

/* The status text (also logged when it changes). */
static void np_status(const char *fmt, ...)
{
    char t[sizeof s_stat];
    va_list ap;
    int changed;
    va_start(ap, fmt);
    vsnprintf(t, sizeof t, fmt, ap);
    va_end(ap);
#if defined(_WIN32)
    AcquireSRWLockExclusive(&s_stat_lock);
#endif
    changed = strcmp(t, s_stat) != 0;
    if (changed)
        memcpy(s_stat, t, sizeof t);
#if defined(_WIN32)
    ReleaseSRWLockExclusive(&s_stat_lock);
#endif
    if (changed) {
        fprintf(stderr, "[NET] %s\n", t);
        fflush(stderr);
    }
}

static void stat_open(void)
{
#if defined(_WIN32)
    const char *hl = getenv("RECOMP_HEADLESS");
    if (!(hl && *hl == '1'))
        s_stat_thread = CreateThread(NULL, 0, stat_thread, NULL, 0, NULL);
#endif
}

static void stat_close(void)
{
    InterlockedExchange(&s_stat_done, 1);
#if defined(_WIN32)
    if (s_stat_thread) {
        WaitForSingleObject(s_stat_thread, 2000);
        CloseHandle(s_stat_thread);
        s_stat_thread = NULL;
    }
#endif
}

/* This PC's IPv4 addresses, for the host to tell the others. */
static void local_addresses(char *out, size_t n)
{
    char name[256];
    struct addrinfo hints, *res = NULL, *a;
    size_t used = 0;
    out[0] = 0;
    if (gethostname(name, sizeof name))
        return;
    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_INET;
    if (getaddrinfo(name, NULL, &hints, &res))
        return;
    for (a = res; a && used + 20 < n; a = a->ai_next) {
        char ip[64];
        const struct sockaddr_in *sa = (const struct sockaddr_in *)a->ai_addr;
        if (!inet_ntop(AF_INET, &sa->sin_addr, ip, sizeof ip) || !strncmp(ip, "127.", 4))
            continue;
        if (strstr(out, ip))
            continue;
        used += (size_t)snprintf(out + used, n - used, "%s%s", used ? ", " : "", ip);
    }
    freeaddrinfo(res);
}

/* Wait up to ms for a socket to become readable (a connection to accept). */
static int wait_readable(SOCKET sk, unsigned ms)
{
    fd_set r;
    struct timeval tv;
    FD_ZERO(&r);
    FD_SET(sk, &r);
    tv.tv_sec = (long)(ms / 1000u);
    tv.tv_usec = (long)((ms % 1000u) * 1000u);
    return select((int)sk + 1, &r, NULL, NULL, &tv) > 0;
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
        snprintf(s_why, sizeof s_why, "Cannot host on UDP port %d (error %d): is another "
                 "game already using it? Choose another port.", port, np_sockerr());
        return 0;
    }
    /* The saves go out on TCP, same port: listen now so a joiner that
     * connects right after its welcome is queued. */
    ls = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (ls == INVALID_SOCKET || bind(ls, (struct sockaddr *)&a, sizeof a)
        || listen(ls, NP_MAXP)) {
        snprintf(s_why, sizeof s_why, "Cannot host on TCP port %d (error %d): is another "
                 "program already using it? Choose another port.", port, np_sockerr());
        return 0;
    }
    session_dir(0, port);
    {
        char src[MAX_PATH], dst[MAX_PATH];
        snprintf(src, sizeof src, "%s%cUDATA", save_src, NP_SEP);
        snprintf(dst, sizeof dst, "%s%cUDATA", s_session, NP_SEP);
        copy_tree(src, dst);
        snprintf(src, sizeof src, "%s%cTDATA", save_src, NP_SEP);
        snprintf(dst, sizeof dst, "%s%cTDATA", s_session, NP_SEP);
        copy_tree(src, dst);
    }
    {
        char ips[200];
        local_addresses(ips, sizeof ips);
        snprintf(s_host_info, sizeof s_host_info,
                 "Hosting a network game for %d players on port %d (rendering: %s).\n\n"
                 "The others choose Join and enter this PC's address (%s) and port "
                 "%d. This PC's firewall must let the game in (UDP and TCP).",
                 s_np, port, k_mode_name[s_mode], ips[0] ? ips : "see ipconfig", port);
    }
    while (joined < s_np - 1) {
        char buf[256];
        np_status("%s\n\nWaiting for players: %d of %d joined.", s_host_info,
                  joined, s_np - 1);
        if (s_cancel) {
            closesocket(ls);
            return 0;
        }
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
        if (s_mode == NP_MODE_DEPTH && !(((PkHello *)buf)->mode & NP_CAN_VULKAN)) {
            refuse(&from, BYE_NOVULKAN);
            fprintf(stderr, "[NET] a player without usable Vulkan tried to join a "
                    "depth-only game; refused\n");
            continue;
        }
        if (((PkHello *)buf)->exe != s_exe_hash) {
            refuse(&from, BYE_BUILD);              /* so it can say why */
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
        w.mode = (uint8_t)s_mode;
        send_to(&from, &w, sizeof w);
    }
    {   /* the saves, to every joiner */
        int k;
        for (k = 0; k < s_np - 1; k++) {
            SOCKET c;
            uint16_t end = 0;
            DWORD t0 = GetTickCount();
            np_status("%s\n\nAll players joined. Sending the saves (%d of %d)...",
                      s_host_info, k + 1, s_np - 1);
            /* A joiner fetches them right after its welcome; one that cannot
             * reach TCP (a firewall) would block this for ever. */
            while (!wait_readable(ls, 250)) {
                if (s_cancel) {
                    closesocket(ls);
                    return 0;
                }
                if (GetTickCount() - t0 > 60000) {
                    snprintf(s_why, sizeof s_why, "A player joined but could not fetch the "
                             "saves in 60 s (TCP port %d). This PC's firewall must let the "
                             "game in on TCP as well as UDP.", port);
                    closesocket(ls);
                    return 0;
                }
            }
            c = accept(ls, NULL, NULL);
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
    DWORD t0 = GetTickCount(), t_hello = 0;
    memset(&s_peer[0], 0, sizeof s_peer[0]);
    s_peer[0].sin_family = AF_INET;
    s_peer[0].sin_port = htons((u_short)port);
    if (inet_pton(AF_INET, ip, &s_peer[0].sin_addr) != 1) {
        /* A name (a VPN's, e.g. Tailscale's): its IPv4 address. */
        struct addrinfo hints, *res = NULL;
        memset(&hints, 0, sizeof hints);
        hints.ai_family = AF_INET;
        np_status("Looking up %s ...", ip);
        if (getaddrinfo(ip, NULL, &hints, &res) || !res) {
            snprintf(s_why, sizeof s_why, "\"%s\" is not an address this PC can find. "
                     "Enter the host's IP address (e.g. 192.168.1.23).", ip);
            return 0;
        }
        s_peer[0].sin_addr = ((const struct sockaddr_in *)res->ai_addr)->sin_addr;
        freeaddrinfo(res);
    }
    s_npeer = 1;
    memset(&h, 0, sizeof h);
    h.type = PK_HELLO;
    h.exe = s_exe_hash;
    h.mode = (uint8_t)(vulkan_usable() ? NP_CAN_VULKAN : 0);
    for (;;) {
        char buf[256];
        struct sockaddr_in from;
        int fl = sizeof from, n;
        if (s_cancel)
            return 0;
        if (!got_welcome) {
            unsigned secs = (unsigned)((GetTickCount() - t0) / 1000u);
            if (secs < 5)
                np_status("Connecting to %s, port %d ...", ip, port);
            else
                np_status("No answer yet from %s, port %d (%u s).\n\nCheck that this is "
                          "the host's address, that the host has started \"Host a game\", "
                          "and that the host's firewall lets the game in (UDP and TCP port "
                          "%d). Trying for 2 minutes.", ip, port, secs, port);
        } else {
            np_status("Connected: you are player %d of %d.\n\nWaiting for the host to "
                      "start: it starts when all %d players have joined.",
                      s_me + 1, s_np, s_np);
        }
        /* Hello until welcomed, then once a second until the START: the
         * host answers a known player's hello with the START again. */
        if (!got_welcome || GetTickCount() - t_hello >= 1000) {
            send_to(&s_peer[0], &h, sizeof h);
            t_hello = GetTickCount();
        }
        n = recvfrom(s_sock, buf, sizeof buf, 0, (struct sockaddr *)&from, &fl);
        if (n >= (int)sizeof(PkHello) && buf[0] == PK_WELCOME && !got_welcome) {
            PkHello *w = (PkHello *)buf;
            SOCKET c;
            int files;
            s_me = w->player;
            s_np = w->players;
            s_delay = w->delay;
            s_mode = w->mode <= NP_MODE_DEPTH ? w->mode : NP_MODE_SOFTWARE;
            got_welcome = 1;
            np_status("Connected: you are player %d of %d.\n\nReceiving the host's "
                      "saves (TCP port %d) ...", s_me + 1, s_np, port);
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
            np_status("Connected: you are player %d of %d. Saves received.\n\nWaiting for "
                      "the host to start: it starts when all %d players have joined.",
                      s_me + 1, s_np, s_np);
        } else if (n >= (int)sizeof(PkHello) && buf[0] == PK_START && got_welcome) {
            return 1;
        } else if (n >= (int)sizeof(PkBye) && buf[0] == PK_BYE) {
            if (((PkBye *)buf)->reason == BYE_FULL)
                snprintf(s_why, sizeof s_why, "The host's game is full or has already "
                         "started. Ask the host to start a new network game.");
            else if (((PkBye *)buf)->reason == BYE_NOVULKAN)
                snprintf(s_why, sizeof s_why, "This network game needs Vulkan on every "
                         "PC, and this PC has no usable Vulkan. Ask the host to choose "
                         "Compatibility rendering.");
            else if (((PkBye *)buf)->reason == BYE_BUILD)
                snprintf(s_why, sizeof s_why, "The host refused: it runs a different build "
                         "or different game files. Every player needs the same build of "
                         "the game and the same disc's files.");
            else
                snprintf(s_why, sizeof s_why, "The host left before the game started.");
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
    if (!np_net_init())
        return 0;
    s_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (s_sock == INVALID_SOCKET)
        return 0;
    np_rcvtimeo(s_sock, 200);                       /* handshake polls */
    s_exe_hash = build_identity(game_dir);
    np_status("Starting the network game ...");
    stat_open();
    if (!strncmp(e, "host:", 5)) {
        const char *sd = getenv("RECOMP_SAVE_DIR"), *rm = getenv("RECOMP_NET_RENDER");
        if (rm && !strcmp(rm, "depth"))
            s_mode = NP_MODE_DEPTH;
        if (s_mode == NP_MODE_DEPTH && !vulkan_usable()) {
            snprintf(s_why, sizeof s_why, "Fast rendering needs Vulkan, and this PC has "
                     "no usable Vulkan. Choose Compatibility rendering to host.");
        } else {
            s_is_host = 1;
            ok = handshake_host(atoi(e + 5), sd && *sd ? sd : (game_dir ? game_dir : "."));
        }
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
    stat_close();
    if (!ok && s_cancel) {
        fprintf(stderr, "[NET] cancelled by the player: no network game\n");
        fflush(stderr);
        closesocket(s_sock);
        s_sock = INVALID_SOCKET;
        return 0;
    }
    if (!ok) {
        const char *hl = getenv("RECOMP_HEADLESS");
        if (!s_why[0])
            snprintf(s_why, sizeof s_why, "The network game could not start (%s).", e);
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
    np_rcvtimeo(s_sock, 10);                        /* the resend cadence */
    {   /* frames before the delay: nothing pressed, for everyone */
        static const uint8_t zero[NP_REC];
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
    np_setenv("RECOMP_SCHED_VIRTUAL", "1");
    np_setenv("RECOMP_SAVE_DIR", s_session);
    {
        char n[8];
        snprintf(n, sizeof n, "%d", s_np);
        np_setenv("RECOMP_USB_PADS", n);
    }
    {
        const char *vk = getenv("RECOMP_NET_VK"), *hl = getenv("RECOMP_HEADLESS");
        int headless = hl && *hl == '1';
        if (s_mode == NP_MODE_DEPTH) {
            np_setenv("RECOMP_GPU", "vulkan");
            np_setenv("RECOMP_SW_DEPTH_ONLY", "1");
            np_setenv("RECOMP_VK_READBACK", "0");
            if (!headless)
                np_setenv("RECOMP_VK_PRESENT", "1");
        } else if (vk && *vk == '1' && !headless) {
            np_setenv("RECOMP_GPU", "vulkan");
            np_setenv("RECOMP_VK_SHADOW", "1");
            np_setenv("RECOMP_VK_PRESENT", "1");
        } else {
            np_setenv("RECOMP_GPU", "software");
        }
    }
    /* One implementation of the x87 maths (fsin, fcos, ...) on every peer:
     * each OS's C library differs in the last bit (src/kernel/recomp_math.c). */
    np_setenv("RECOMP_PORTABLE_MATH", "1");
    recomp_set_portable_math(1);
    {
        int p;
        for (p = 0; p < NP_MAXP; p++)
            s_heard[p] = GetTickCount();
    }
    s_on = 1;
    usb_gamepad_set_override(xbox_NetplayPadState);
    CloseHandle(CreateThread(NULL, 0, net_thread, NULL, 0, NULL));
    fprintf(stderr, "[NET] network game: player %d of %d, input delay %d frames, "
            "rendering: %s; saves: %s\n", s_me + 1, s_np, s_delay, k_mode_name[s_mode],
            s_session);
    fflush(stderr);
    return 1;
}

/* ---- cheats ----------------------------------------------------------------------
 *
 * xbox_NetplayCheatSend queues a local cheat request; one per frame leaves
 * with this player's next sampled frame (cheat_take). When the emulated pad of
 * a port hands a frame's input to the guest, that frame's event becomes due
 * (cheat_due) -- in the guest's own order, so the same on every peer -- and
 * cheats.c applies it from xbox_NetplayCheatNext at its poll. */
#define NP_CQ 64
static uint8_t s_cq[NP_CQ][22];                  /* local requests not yet sent */
static unsigned s_cq_head, s_cq_tail;
static uint8_t s_dq[NP_CQ][22];                  /* due, not yet applied */
static unsigned s_dq_head, s_dq_tail;
static uint32_t s_ev_done[NP_MAXP];              /* last frame + 1 taken per port */

int xbox_NetplayCheatSend(const uint8_t ev[22])
{
    int ok = 0;
    if (!s_on)
        return 0;
    AcquireSRWLockExclusive(&s_lock);
    if (s_cq_tail - s_cq_head < NP_CQ) {
        memcpy(s_cq[s_cq_tail++ % NP_CQ], ev, 22);
        ok = 1;
    }
    ReleaseSRWLockExclusive(&s_lock);
    if (!ok)
        fprintf(stderr, "[NET] cheat request dropped: %d already waiting\n", NP_CQ);
    return 1;                                   /* taken (or dropped): not local */
}

/* flag_ev: the record's flag byte, then 22 bytes. */
static void cheat_take(uint8_t *flag_ev)
{
    AcquireSRWLockExclusive(&s_lock);
    if (s_cq_head != s_cq_tail) {
        flag_ev[0] = 1;
        memcpy(flag_ev + 1, s_cq[s_cq_head++ % NP_CQ], 22);
    }
    ReleaseSRWLockExclusive(&s_lock);
}

static void cheat_due(const uint8_t *ev)
{
    int full;
    AcquireSRWLockExclusive(&s_lock);
    full = s_dq_tail - s_dq_head >= NP_CQ;
    if (!full)
        memcpy(s_dq[s_dq_tail++ % NP_CQ], ev, 22);
    ReleaseSRWLockExclusive(&s_lock);
    if (full)          /* the same on every peer, so no desync; but say so */
        fprintf(stderr, "[NET] cheat request %u dropped: %d due and not yet applied\n",
                ev[0], NP_CQ);
}

int xbox_NetplayCheatNext(uint8_t ev[22])
{
    int got = 0;
    if (!s_on)
        return 0;
    AcquireSRWLockExclusive(&s_lock);
    if (s_dq_head != s_dq_tail) {
        memcpy(ev, s_dq[s_dq_head++ % NP_CQ], 22);
        got = 1;
    }
    ReleaseSRWLockExclusive(&s_lock);
    return got;
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
    LARGE_INTEGER q0;
    int waited;

    /* RECOMP_NET_TEST_CHEAT=<frame> (testing the cheat path): this peer sends
     * three requests (op 0xEE, ignored by cheats.c) at that frame; every
     * peer logs the frame each comes due at, which must be the same. */
    {
        static int test = -1;
        static uint32_t at;
        if (test < 0) {
            const char *e = getenv("RECOMP_NET_TEST_CHEAT");
            test = e && *e;
            at = test ? (uint32_t)atoi(e) : 0;
        }
        if (test && f >= at) {
            uint8_t ev[22];
            int k;
            test = 0;
            memset(ev, 0, sizeof ev);
            ev[0] = 0xEE;
            for (k = 0; k < 3; k++)
                xbox_NetplayCheatSend(ev);
        }
    }

    /* RECOMP_NET_TEST_STALL=<frame>,<ms> (testing): this peer stops for
     * <ms> at that frame, so the others wait for it that long. */
    {
        static int stall = -1;
        static uint32_t at, ms;
        if (stall < 0) {
            const char *e = getenv("RECOMP_NET_TEST_STALL"), *c;
            stall = e && (c = strchr(e, ',')) != NULL;
            if (stall) {
                at = (uint32_t)atoi(e);
                ms = (uint32_t)atoi(c + 1);
            }
        }
        if (stall && f >= at) {
            stall = 0;
            fprintf(stderr, "[NET] test: stalling %u ms at frame %u\n", ms, f);
            Sleep(ms);
        }
    }

    /* This peer's own input, sampled once per frame ahead by the delay. */
    while (s_sampled <= f + (uint32_t)s_delay) {
        uint8_t local[NP_REC];
        memset(local, 0, sizeof local);
        usb_gamepad_sample_local(local);
        {   /* RECOMP_NET_TEST_INPUT=1 (testing): this player's input is a
             * function of the frame -- Start for 15 of every 120 frames, as
             * scratch/fakehost.py plays -- so runs can be compared. */
            static int ti = -1;
            if (ti < 0)
                ti = getenv("RECOMP_NET_TEST_INPUT") != NULL;
            if (ti) {
                memset(local, 0, 18);
                if (s_sampled >= (uint32_t)s_delay && s_sampled % 120u < 15u)
                    local[0] = 0x10;
            }
        }
        cheat_take(local + NP_EV_FLAG);         /* one queued cheat rides along */
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
            /* ... and how long this peer stood waiting for the others' input
             * since the last line: the cost of the lockstep itself (a slower
             * peer, or input arriving later than the delay covers). */
            static LARGE_INTEGER qf, last;
            LARGE_INTEGER now;
            if (!qf.QuadPart)
                QueryPerformanceFrequency(&qf);
            QueryPerformanceCounter(&now);
            fprintf(stderr, "[NET] frame %u: guest hash %016llX; waited for the "
                    "others %.0f ms of the last %.0f ms\n", f,
                    (unsigned long long)pk.hash,
                    s_wait_ticks * 1000.0 / (double)qf.QuadPart,
                    last.QuadPart ? (now.QuadPart - last.QuadPart) * 1000.0
                                    / (double)qf.QuadPart : 0.0);
            fflush(stderr);
            s_wait_ticks = 0;
            last = now;
        }
        broadcast(&pk, sizeof pk, -1);
    }

    if (port >= s_np) {
        memset(state18, 0, 18);
        return 0;
    }
    t0 = GetTickCount();
    waited = 0;
    for (;;) {
        NpIn *e = &s_in[port][f % NP_RING];
        int ok, due = 0;
        uint8_t ev[22];
        AcquireSRWLockExclusive(&s_lock);       /* s_ev_done is written here */
        ok = e->have && e->f == f;
        if (ok) {
            memcpy(state18, e->s, 18);
            if (e->s[NP_EV_FLAG] && s_ev_done[port] != f + 1u) {
                memcpy(ev, e->s + NP_EV, 22);
                due = 1;
            }
            s_ev_done[port] = f + 1u;
        }
        ReleaseSRWLockExclusive(&s_lock);
        if (due) {                              /* now, the same on every peer */
            cheat_due(ev);
            fprintf(stderr, "[NET] frame %u: cheat request %u from player %d due\n",
                    f, ev[0], port + 1);
        }
        if (ok) {
            if (waited) {
                LARGE_INTEGER q1;
                QueryPerformanceCounter(&q1);
                s_wait_ticks += (uint64_t)(q1.QuadPart - q0.QuadPart);
            }
            return 1;
        }
        if (!waited) {
            waited = 1;
            QueryPerformanceCounter(&q0);
        }
        /* A slower player is waited for as long as it takes -- a slow disk,
         * or one OS logging slower than another, can put a peer many
         * seconds behind -- but one not heard from at all for 15 s is gone:
         * every peer resends its recent inputs every 10 ms. */
        if (GetTickCount() - s_heard[port] > 15000) {
            char why[96];
            snprintf(why, sizeof why, "nothing from player %d for 15 s (frame %u)",
                     port + 1, f);
            fail(why);
        }
        if (GetTickCount() - t0 >= 10000) {
            fprintf(stderr, "[NET] frame %u: waiting for player %d, who is behind\n",
                    f, port + 1);
            fflush(stderr);
            t0 = GetTickCount();
        }
        Sleep(1);
    }
}
