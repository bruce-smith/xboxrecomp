/*
 * sched.c -- one guest CPU, and a record of who had it when.
 *
 * The Xbox runs guest code on one CPU. Here every guest thread runs on a
 * host thread of its own, and so do the device models that call guest code
 * (the timer thread's vblank, DPCs and timers; the OHCI controller's
 * interrupt). Run concurrently, the order in which they touch shared memory
 * is whatever the host scheduler made of it, and two runs of the same input
 * end up in different places: a voice line finishes a frame earlier, a
 * mashed button is taken on another frame.
 *
 * With RECOMP_SCHED_RECORD=<file> or RECOMP_SCHED_REPLAY=<file>, every
 * "agent" -- a guest thread, or a device model -- holds the guest CPU while
 * it runs guest code or touches guest-visible state, and gives it up only
 * where it would block (a wait that is not yet satisfied, a sleep, a
 * contended lock), at a spin loop's yield point, or when preempted at a
 * loop head (below). That makes each agent's own sequence of acquisitions a
 * deterministic function of what it has seen. Recording writes down the
 * order the agents took the CPU in, as runs: (agent, how many acquisitions
 * in a row). Replaying hands the CPU out in exactly that order. What an
 * agent takes from the host -- a time read (QPC, system and interrupt time,
 * rdtsc, KeTickCount), whether a timed wait timed out, whether a vblank or a
 * kernel timer was due, the pad's report, file times, free disk space --
 * goes into the same file as a per-agent value stream, and replay gives the
 * recorded value back. Sleeps do not sleep in a replay.
 *
 * Agents: 1 timer thread (vblank, DPCs, kernel timers, KeTickCount, and
 * main.c's device shims as its "ticks"), 2 OHCI
 * controller, 16 the title's main thread, 17+ the guest threads in the order
 * they were created (creation happens on a thread that holds the CPU, so the
 * numbering is the same in every run). Whatever an agent allocates from the
 * guest heap it does holding the CPU, so the heap's layout repeats too.
 *
 * A guest thread spinning in guest code on another agent's flag never
 * blocks. tools.recomp --loop-heads counts its loop iterations per thread;
 * every 65536 of them, if another agent has waited 2 ms, it lets go, and the
 * count is recorded so a replay lets go at exactly the same iteration.
 * --yield-points names known spins that also run "pumps" (the device
 * acknowledgements they wait for).
 *
 * The device models with threads of their own are agents too: 3 the APU
 * (one frame a turn; xbox_SchedHostPause and the FE-trap hold), 4 the NV2A
 * thread (register acks and the pushbuffer executor, one pass a turn; the
 * raster pool is waited for inside it, and a notify waits for the timer
 * agent through xbox_SchedWaitUntil). A replay does not sleep between a
 * device model's turns.
 *
 * File: "BX2SCHED", u32 version, then tagged records (0x03 is written only
 * when it says something, so a file without it is what older builds wrote):
 *   0x03 varint flags                           bit 0: RECOMP_PORTABLE_MATH
 *                                               bit 1: file I/O lets go of the
 *                                               CPU (xbox_SchedIoLetGo)
 *   0x01 varint agent, varint count            a run
 *   0x02 varint agent, varint kind, zigzag-varint delta   a value (delta
 *        from the previous value of that agent and kind)
 * Off by default; off, every entry point here is a test and a return.
 */
#if defined(_WIN32)
#include <windows.h>
#endif
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "kernel.h"
#include "xbox_memory_layout.h"

#if defined(_WIN32)
#include <intrin.h>                    /* _ReturnAddress */
#include <dbghelp.h>                   /* the stuck report names host code */
#pragma comment(lib, "dbghelp.lib")
#pragma comment(lib, "synchronization.lib")   /* WaitOnAddress */
#endif

extern RECOMP_TLS uint32_t g_esp;
extern ptrdiff_t g_xbox_mem_offset;

#define SCHED_AGENTS     256
#define SCHED_KINDS      16

static int s_mode = -1;                 /* 0 off, 1 record, 2 replay, 3 virtual */

/* File I/O lets go of the guest CPU for the host read or write (bridge
 * NtReadFile/NtWriteFile), as a thread blocked on the disk does on the
 * console. Holding it, a slow disk stopped every other agent: a pack loader
 * held it for 3 s of 64 KB reads in a recorded boot, the USB control
 * transfer XPP was waiting for timed out meanwhile, and the boot stalled at
 * the logo movie. Recordings say whether they were made so (tag 0x03 bit 1);
 * older ones were not, and replay as made. */
static int s_io_letgo = 1;

/* Recording and replay only. In virtual mode time does not move during a
 * host read, so nothing times out, and a buffer still filling while another
 * agent hashes guest memory (network play's sync check) would read
 * differently on two PCs. (A recording's own RECOMP_INPUT_HASH checkpoints can
 * catch such a buffer half-filled: ~40 KB at an odd checkpoint, gone at the
 * next; the guest state itself replays exactly.) */
int xbox_SchedIoLetGo(void)
{
    return (s_mode == 1 || s_mode == 2) && s_io_letgo;
}
/* 2: every wait's outcome is recorded, not only a finite one's. */
#define SCHED_FILE_VER 2
static uint32_t s_file_ver = SCHED_FILE_VER;
static RECOMP_TLS int t_agent;          /* this host thread's agent, 0 = none */
static RECOMP_TLS int t_held;           /* holds the guest CPU */
static volatile LONG s_next_guest = XBOX_SCHED_AGENT_MAIN + 1;
static volatile int s_holder;           /* agent holding the CPU, diagnostics */

/* --- record ---------------------------------------------------------- */
static volatile LONG s_ticket_next, s_ticket_serving, s_waiters;
static volatile LONG64 s_wait_since;    /* QPC when the first waiter began */
static LARGE_INTEGER s_qpf;
static FILE *s_out;
static int s_last_agent = -1;
static uint32_t s_run;
static uint64_t s_prev_val[SCHED_AGENTS][SCHED_KINDS];
static uint64_t s_runs_written, s_vals_written;
static volatile DWORD s_agent_tid[256];      /* host thread of each agent */

/* --- replay ---------------------------------------------------------- */
typedef struct { uint8_t agent; uint32_t count; } SchedRun;
typedef struct { uint8_t *kind; uint64_t *val; uint32_t n, i; } SchedQueue;
static SchedRun  *s_runs;
static size_t     s_nruns, s_cur;
static uint32_t   s_left;
static volatile int s_turn;             /* agent of the current run, -1 free */
static int        s_busy;
static SRWLOCK    s_rlock = SRWLOCK_INIT;
static CONDITION_VARIABLE s_rcv = CONDITION_VARIABLE_INIT;
static SchedQueue s_q[SCHED_AGENTS];
/* Where each agent last let go of the CPU, for the stuck report: the host
 * code that blocked and the guest return address at the time. */
static void *volatile s_where[SCHED_AGENTS];
static volatile uint32_t s_where_ret[SCHED_AGENTS];
static volatile LONG s_blocked[SCHED_AGENTS];
static volatile LONG s_desync_reports;
typedef struct { uint64_t *n; uint32_t count, i; } SchedPreempts;
static SchedPreempts s_pre[SCHED_AGENTS];      /* replay: loop counts to yield at */
static void sched_arm_loop_check(void);
static uint64_t sched_value(int kind, uint64_t live, int released);

/* --- mode 3: deterministic without a recording (RECOMP_SCHED_VIRTUAL=1) ---
 *
 * The same one guest CPU, but nothing decided by the host: no recording to
 * follow and none made, and still the same run every time from the same
 * input. That is what lockstep multiplayer needs, and a replay that needs no
 * schedule file.
 *
 * - The CPU is a token passed round-robin among the agents in agent-id
 *   order. An agent joins the rotation when it is created (in its creator's
 *   turn, or during the single-threaded start-up), leaves when it ends, and is
 *   skipped while suspended -- suspension is the scheduler's own count here,
 *   not a host SuspendThread.
 * - A let-go names what the agent waits for (t_vw): nothing (a yield), a
 *   virtual deadline (a sleep, a device model's period), a condition tried
 *   while holding the token (an object a wait can take, a free lock, a
 *   serviced notify), or both. When the token reaches an agent whose
 *   condition does not hold, it passes it straight on. Objects are only ever
 *   taken in their waiter's turn, so no host thread's timing decides who gets
 *   one.
 * - Time is virtual (s_vt, ns): each turn adds its work (loop iterations,
 *   g_loop_n, and a fixed cost); a whole round of agents that all passed jumps
 *   it to the earliest deadline. Every clock the guest reads derives from it.
 *   The host only paces: it sleeps while virtual time is ahead of the wall
 *   clock (RECOMP_SCHED_VIRTUAL_FAST=1: not at all, for tests).
 * - Loop heads preempt every SCHED_LOOP_INTERVAL iterations, always. */
typedef struct {
    int (*fn)(void *ctx);       /* tried holding the token; may take the object */
    void *ctx;
    uint64_t deadline;          /* virtual ns, 0 none */
    int hit, timed_out;
    int spin;                   /* a yield in a spin: runs, but counts as idle */
} SchedVWait;
static volatile uint64_t s_vt;
static volatile LONG s_tok = XBOX_SCHED_AGENT_MAIN;
static volatile LONG s_alive[SCHED_AGENTS], s_susp[SCHED_AGENTS];
static volatile uint64_t s_dl[SCHED_AGENTS];
static HANDLE s_tev[SCHED_AGENTS];
static int s_idle_run;
static LARGE_INTEGER s_wall0;
static int s_vfast;
static RECOMP_TLS SchedVWait t_vw;
static RECOMP_TLS uint64_t t_loop0;
static RECOMP_TLS uint64_t t_vt_left;    /* virtual time at this agent's last let-go */
#define VT_LOOP_NS 10ull
#define VT_TURN_NS 1000ull
#define VT_BASE_FILETIME 134116992000000000ull   /* 2026-01-01 00:00 UTC */
static int s_sites;                     /* RECOMP_SCHED_SITES, see sched_value */
static volatile LONG s_site_reports;
static void sched_preempt_load(int agent, uint64_t n);

/* Pumps: run at every yield point, holding the CPU. A device model's
 * acknowledgement that a guest spin loop waits for -- with the CPU held by
 * the spinner, nothing else would ever get to do it. */
#define SCHED_PUMPS 8
static void (*s_pumps[SCHED_PUMPS])(void);
static int s_npumps;
/* Ticks: run by the timer agent at each of its turns (device models that
 * used to poll guest memory on threads of their own). */
#define SCHED_TICKS 8
static void (*s_ticks[SCHED_TICKS])(void);
static int s_nticks;

static void sched_put_varint(uint64_t v)
{
    while (v >= 0x80) {
        fputc((int)(v & 0x7F) | 0x80, s_out);
        v >>= 7;
    }
    fputc((int)v, s_out);
}

static int sched_get_varint(const uint8_t **p, const uint8_t *end, uint64_t *v)
{
    uint64_t r = 0;
    int shift = 0;
    while (*p < end) {
        uint8_t b = *(*p)++;
        r |= (uint64_t)(b & 0x7F) << shift;
        if (!(b & 0x80)) {
            *v = r;
            return 1;
        }
        shift += 7;
        if (shift > 63)
            return 0;
    }
    return 0;
}

static void sched_emit_run(void)
{
    if (s_last_agent < 0 || !s_run)
        return;
    fputc(0x01, s_out);
    sched_put_varint((uint64_t)s_last_agent);
    sched_put_varint(s_run);
    s_runs_written++;
}

void xbox_SchedFlush(void)
{
    if (s_mode == 1 && s_out)
        fflush(s_out);
}

static DWORD WINAPI sched_flusher(LPVOID unused)
{
    (void)unused;
    for (;;) {
        Sleep(1000);
        xbox_SchedFlush();
    }
}

void xbox_SchedClose(void);

static void sched_atexit(void)
{
    xbox_SchedClose();
}

/* Write the run in progress and flush: on every way out of the process
 * (atexit, the window's close, the watchdog). */
void xbox_SchedVtraceFlush(void);

void xbox_SchedClose(void)
{
    xbox_SchedVtraceFlush();
    if (s_mode == 1 && s_out) {
        sched_emit_run();
        s_run = 0;
        fflush(s_out);
        fprintf(stderr, "  [SCHED] recording closed: %llu runs, %llu values\n",
                (unsigned long long)s_runs_written,
                (unsigned long long)s_vals_written);
    }
}

/* The per-agent lists read from a schedule grow by doubling (1024, 2048, ...):
 * grown by 1024 at a time, a recording with tens of millions of values spent
 * minutes in realloc before the replay began (Windows copies on most of them). */
static int sched_grow_at(uint32_t n)
{
    return n == 0 || (n >= 1024 && !(n & (n - 1)));
}

static uint32_t sched_grown(uint32_t n)
{
    return n < 1024 ? 1024 : n * 2;
}

static int sched_load(const char *path)
{
    FILE *f = fopen(path, "rb");
    uint8_t *buf, *p, *end;
    long size;
    size_t cap = 1024;
    uint32_t ver;

    if (!f) {
        fprintf(stderr, "  [SCHED] cannot open %s; schedule replay off\n", path);
        return 0;
    }
    fseek(f, 0, SEEK_END);
    size = ftell(f);
    fseek(f, 0, SEEK_SET);
    buf = (uint8_t *)malloc(size > 0 ? (size_t)size : 1);
    if (!buf || fread(buf, 1, (size_t)size, f) != (size_t)size || size < 12
        || memcmp(buf, "BX2SCHED", 8)) {
        fprintf(stderr, "  [SCHED] %s is not a schedule; replay off\n", path);
        fclose(f);
        free(buf);
        return 0;
    }
    fclose(f);
    memcpy(&ver, buf + 8, 4);
    s_file_ver = ver;
    /* How the recording computed the x87 transcendentals; absent: the C
     * library's. A replay must match it whatever the environment says.
     * Likewise whether file I/O let go of the CPU: absent, it did not. */
    recomp_set_portable_math(0);
    s_io_letgo = 0;
    p = buf + 12;
    end = buf + size;
    s_runs = (SchedRun *)malloc(cap * sizeof(SchedRun));
    while (p < end) {
        uint8_t tag = *p++;
        uint64_t a, b, c;
        if (tag == 0x01) {
            if (!sched_get_varint((const uint8_t **)&p, end, &a)
                || !sched_get_varint((const uint8_t **)&p, end, &b))
                break;
            if (s_nruns == cap) {
                cap *= 2;
                s_runs = (SchedRun *)realloc(s_runs, cap * sizeof(SchedRun));
            }
            s_runs[s_nruns].agent = (uint8_t)a;
            s_runs[s_nruns].count = (uint32_t)b;
            s_nruns++;
        } else if (tag == 0x02) {
            SchedQueue *q;
            uint64_t prev;
            int64_t d;
            if (!sched_get_varint((const uint8_t **)&p, end, &a)
                || !sched_get_varint((const uint8_t **)&p, end, &b)
                || !sched_get_varint((const uint8_t **)&p, end, &c)
                || a >= SCHED_AGENTS || b >= SCHED_KINDS)
                break;
            q = &s_q[a];
            if (sched_grow_at(q->n)) {
                q->kind = (uint8_t *)realloc(q->kind, sched_grown(q->n));
                q->val = (uint64_t *)realloc(q->val, sched_grown(q->n) * sizeof(uint64_t));
            }
            d = (int64_t)((c >> 1) ^ (~(c & 1) + 1));     /* zigzag */
            prev = s_prev_val[a][b] + (uint64_t)d;
            s_prev_val[a][b] = prev;
            if (b == XBOX_SCHED_PREEMPT) {    /* their own list, see below */
                sched_preempt_load((int)a, prev);
                continue;
            }
            q->kind[q->n] = (uint8_t)b;
            q->val[q->n] = prev;
            q->n++;
        } else if (tag == 0x03) {
            if (!sched_get_varint((const uint8_t **)&p, end, &a))
                break;
            recomp_set_portable_math((int)(a & 1));
            s_io_letgo = (int)((a >> 1) & 1);
            if (a & 1)
                fprintf(stderr, "  [SCHED] recorded with RECOMP_PORTABLE_MATH: "
                        "portable math on for this replay\n");
        } else
            break;
    }
    free(buf);
    if (p < end)
        fprintf(stderr, "  [SCHED] %s: stopped reading at a bad record\n", path);
    {
        uint64_t vals = 0;
        int i;
        for (i = 0; i < SCHED_AGENTS; i++)
            vals += s_q[i].n;
        fprintf(stderr, "  [SCHED] replaying %s: %zu runs, %llu values\n", path,
                s_nruns, (unsigned long long)vals);
    }
    s_cur = 0;
    if (s_nruns) {
        s_turn = s_runs[0].agent;
        s_left = s_runs[0].count;
    } else
        s_turn = -1;
    return 1;
}

/* Once, before anyone sees a mode: two threads asking at start-up both loaded
 * the schedule into the same queues (and one saw "off" in the middle), which
 * corrupted the replayed values in about one run in three. */
static INIT_ONCE s_mode_once = INIT_ONCE_STATIC_INIT;
static BOOL CALLBACK sched_mode_init(PINIT_ONCE o, PVOID x, PVOID *c);

int xbox_SchedMode(void)
{
    if (s_mode < 0)
        InitOnceExecuteOnce(&s_mode_once, sched_mode_init, NULL, NULL);
    return s_mode;
}

static BOOL CALLBACK sched_mode_init(PINIT_ONCE o, PVOID x, PVOID *c)
{
    (void)o; (void)x; (void)c;
    {
        const char *r = getenv("RECOMP_SCHED_RECORD");
        const char *p = getenv("RECOMP_SCHED_REPLAY");
        const char *v = getenv("RECOMP_SCHED_VIRTUAL");
        int mode = 0;
        QueryPerformanceFrequency(&s_qpf);
        if (v && *v && *v != '0' && !(p && *p) && !(r && *r)) {
            QueryPerformanceCounter(&s_wall0);
            s_vfast = getenv("RECOMP_SCHED_VIRTUAL_FAST") != NULL;
            mode = 3;
            fprintf(stderr, "  [SCHED] virtual mode: deterministic guest CPU and "
                    "clocks, no recording%s\n", s_vfast ? " (unpaced)" : "");
        } else if (p && *p) {
            mode = sched_load(p) ? 2 : 0;
        } else if (r && *r) {
            s_out = fopen(r, "wb");
            if (s_out) {
                uint32_t ver = SCHED_FILE_VER;
                setvbuf(s_out, NULL, _IOFBF, 1 << 20);
                fwrite("BX2SCHED", 1, 8, s_out);
                fwrite(&ver, 4, 1, s_out);
                {   /* RECOMP_SCHED_IO_LETGO=0: record as older builds did */
                    const char *lg = getenv("RECOMP_SCHED_IO_LETGO");
                    if (lg && *lg == '0')
                        s_io_letgo = 0;
                }
                fputc(0x03, s_out);
                sched_put_varint((recomp_portable_math_enabled() ? 1u : 0u)
                                 | (s_io_letgo ? 2u : 0u));
                mode = 1;
                atexit(sched_atexit);
                CloseHandle(CreateThread(NULL, 0, sched_flusher, NULL, 0, NULL));
                /* record_blinx2.ps1 looks for this text in the exe to tell
                 * a build that records a replayable schedule. */
                fprintf(stderr, "  [SCHED] recording the guest schedule "
                        "(format %u) to %s\n", ver, r);
            } else
                fprintf(stderr, "  [SCHED] cannot write %s; recording off\n", r);
        }
        s_mode = mode;             /* published last */
    }
    return TRUE;
}

/* The schedule paces the device models (no host sleeps between their turns):
 * a replay, and virtual mode. */
int xbox_SchedReplaying(void)
{
    int m = xbox_SchedMode();
    return m == 2 || m == 3;
}

int xbox_SchedVirtual(void)
{
    return xbox_SchedMode() == 3;
}

uint64_t xbox_SchedVirtualNs(void)
{
    return s_vt;
}

static HANDLE sched_tev(int agent)
{
    if (!s_tev[agent]) {
        HANDLE e = CreateEventA(NULL, FALSE, FALSE, NULL);
        if (InterlockedCompareExchangePointer((PVOID *)&s_tev[agent], e, NULL))
            CloseHandle(e);
    }
    return s_tev[agent];
}

/* The host's performance counter and tick count -- or, in virtual mode, the
 * same in virtual time: what device models decide by (the vblank's cadence,
 * a timer's due time, the APU's hold). */
void xbox_SchedQpcHost(LARGE_INTEGER *q)
{
    if (xbox_SchedMode() == 3) {
        uint64_t t = s_vt, hz = (uint64_t)s_qpf.QuadPart;
        q->QuadPart = (LONGLONG)((t / 1000000000ull) * hz
                                 + (t % 1000000000ull) * hz / 1000000000ull);
        return;
    }
    QueryPerformanceCounter(q);
}

uint64_t xbox_SchedTickMs64(void)
{
    if (xbox_SchedMode() == 3)
        return 1000ull + s_vt / 1000000ull;
    return GetTickCount64();
}

/* Virtual mode: an agent joins the rotation. Called by whatever creates it,
 * at a point every run reaches in the same order (a guest turn, or start-up
 * before the guest runs); tid if known, for resumes and suspends by handle. */
void xbox_SchedExpect(int agent, int suspended, DWORD tid)
{
    if (xbox_SchedMode() != 3 || agent <= 0 || agent >= SCHED_AGENTS)
        return;
    sched_tev(agent);
    if (tid)
        s_agent_tid[agent] = tid;
    s_susp[agent] = suspended ? 1 : 0;
    InterlockedExchange(&s_alive[agent], 1);
}

/* Virtual mode: the next turn of this agent comes ns of virtual time from
 * now (a device model's period). A no-op otherwise. */
/* Virtual mode: the coming let-go is a spin's yield (see sched_acquire3). */
void xbox_SchedMarkSpin(void)
{
    if (s_mode == 3)
        t_vw.spin = 1;
}

void xbox_SchedVirtualPause(uint64_t ns)
{
    /* From the agent's own last turn, not from whatever the clock reads now:
     * a device model calls this between its turns, while another agent is
     * moving the clock, and how far it has got is host timing. */
    if (xbox_SchedMode() == 3)
        t_vw.deadline = (t_held ? s_vt : t_vt_left) + ns;
}

void xbox_SchedBind(int agent)
{
    t_agent = agent;
    if (agent > 0 && agent < SCHED_AGENTS)
        s_agent_tid[agent] = GetCurrentThreadId();
    if (agent == XBOX_SCHED_AGENT_MAIN)
        xbox_SchedExpect(agent, 0, 0);
}

int xbox_SchedAgent(void)
{
    return t_agent;
}

int xbox_SchedNewGuestAgent(void)
{
    if (!xbox_SchedMode())
        return 0;
    return (int)InterlockedIncrement(&s_next_guest) - 1;
}

/* --- the CPU itself -------------------------------------------------- */

/* --- a replay that cannot go on ------------------------------------------
 *
 * Once the run takes another path than the recording, the schedule's turns
 * and values no longer fit it, and sooner or later an agent waits for a turn
 * that never comes: a frozen game, with nothing on screen to say why. So the
 * first mismatch -- or a stall -- ends the replay: a line in the log, a
 * message box unless headless, exit code 4. The game sets xbox_sched_poll
 * (its input poll count) so the message can say where. */
volatile uint32_t xbox_sched_poll;
static volatile DWORD s_free_since;     /* when the CPU was last let go */

static DWORD sched_stall_ms(void)
{
    static DWORD ms;
    if (!ms) {
        const char *e = getenv("RECOMP_REPLAY_STALL_SECS");
        ms = (e && atoi(e) > 0 ? (DWORD)atoi(e) : 30) * 1000;
    }
    return ms;
}

static void sched_replay_fail(int agent, const char *why)
{
    static volatile LONG once;
    char msg[640];
    if (InterlockedExchange(&once, 1))
        return;                         /* another thread is ending it */
    snprintf(msg, sizeof msg,
             "The replay diverged from the recording at input poll %u "
             "(agent %d, schedule run %zu of %zu): %s.\n\n"
             "From here the game would no longer do what the recording did, "
             "so the replay has stopped. The log has the details.",
             xbox_sched_poll, agent, s_cur, s_nruns, why);
    fprintf(stderr, "[REPLAY] diverged at poll %u (agent %d, run %zu of %zu) "
            "- replay stopped: %s\n", xbox_sched_poll, agent, s_cur, s_nruns, why);
    fflush(stderr);
    if (!getenv("RECOMP_HEADLESS"))
        MessageBoxA(NULL, msg, "Blinx 2 replay",
                    MB_OK | MB_ICONWARNING | MB_TOPMOST | MB_SETFOREGROUND);
    fflush(stderr);
    TerminateProcess(GetCurrentProcess(), 4);
}

/* Where the asking agent is, by name: the lifted guest functions are host
 * functions (sub_XXXXXXXX), so the first divergence names the guest code that
 * took another path. */
static void sched_backtrace(void)
{
    static int sym = -1;
    void *fr[24];
    USHORT n, i;
    char buf[sizeof(SYMBOL_INFO) + 256];
    SYMBOL_INFO *si = (SYMBOL_INFO *)buf;
    if (sym < 0) {
        SymInitialize(GetCurrentProcess(), NULL, TRUE);
        sym = 1;
    }
    n = CaptureStackBackTrace(1, 24, fr, NULL);
    for (i = 0; i < n; i++) {
        DWORD64 off = 0;
        memset(buf, 0, sizeof buf);
        si->SizeOfStruct = sizeof(SYMBOL_INFO);
        si->MaxNameLen = 255;
        fprintf(stderr, "      #%u %s+0x%llx\n", i,
                SymFromAddr(GetCurrentProcess(), (DWORD64)(uintptr_t)fr[i], &off, si)
                    ? si->Name : "?", (unsigned long long)off);
    }
    fprintf(stderr, "      guest esp 0x%08X:", g_esp);
    if (g_esp && g_esp < 0x08000000u)
        for (i = 0; i < 8; i++)
            fprintf(stderr, " %08X", *(uint32_t *)(g_xbox_mem_offset + g_esp + 4 * i));
    fprintf(stderr, "\n");
}

/* Kind-0 entries (RECOMP_SCHED_SITES): a guest call site, a page hash, or
 * a let-go of the CPU tagged with the host code that made it. */
#define SCHED_LETGO_TAG 0xB10C000000000000ull

static void sched_describe(uint64_t v, char *buf, size_t n)
{
    if ((v & 0xFFFF000000000000ull) == SCHED_LETGO_TAG) {
        char sb[sizeof(SYMBOL_INFO) + 256];
        SYMBOL_INFO *si = (SYMBOL_INFO *)sb;
        DWORD64 off = 0;
        DWORD64 a = (DWORD64)(uintptr_t)GetModuleHandleA(NULL) + (v & 0xFFFFFFFFull);
        static int sym;
        if (!sym) {
            SymInitialize(GetCurrentProcess(), NULL, TRUE);
            sym = 1;
        }
        memset(sb, 0, sizeof sb);
        si->SizeOfStruct = sizeof(SYMBOL_INFO);
        si->MaxNameLen = 255;
        snprintf(buf, n, "a let-go of the CPU from %s+0x%llx",
                 SymFromAddr(GetCurrentProcess(), a, &off, si) ? si->Name : "?",
                 (unsigned long long)off);
    } else
        snprintf(buf, n, "0x%08llX", (unsigned long long)v);
}

static void sched_stuck(DWORD waited_ms)
{
    if (InterlockedIncrement(&s_desync_reports) > 20)
        return;
    if (s_turn > 0 && s_turn < SCHED_AGENTS) {
        /* Name the host code: the build carries a PDB. */
        static int sym = -1;
        char buf[sizeof(SYMBOL_INFO) + 256];
        SYMBOL_INFO *si = (SYMBOL_INFO *)buf;
        DWORD64 off = 0;
        const char *name = "?";
        if (sym < 0) {          /* main.c may have done it already: fine */
            SymInitialize(GetCurrentProcess(), NULL, TRUE);
            sym = 1;
        }
        memset(buf, 0, sizeof buf);
        si->SizeOfStruct = sizeof(SYMBOL_INFO);
        si->MaxNameLen = 255;
        if (sym && SymFromAddr(GetCurrentProcess(), (DWORD64)(uintptr_t)s_where[s_turn],
                               &off, si))
            name = si->Name;
        fprintf(stderr, "  [SCHED] agent %d %s; it last let go in %s+0x%llx "
                "(exe+0x%llx; guest [esp] 0x%08X)\n", s_turn,
                s_blocked[s_turn] ? "is blocked" : "has not come back",
                name, (unsigned long long)off,
                (unsigned long long)((char *)s_where[s_turn]
                                     - (char *)GetModuleHandleA(NULL)),
                s_where_ret[s_turn]);
    }
    if (s_mode == 2)
        fprintf(stderr, "  [SCHED] agent %d waited %lu ms for its turn; run %zu "
                "of %zu is agent %d (%u left), holder %d\n", t_agent,
                (unsigned long)waited_ms, s_cur, s_nruns, s_turn, s_left,
                s_holder);
    else
        fprintf(stderr, "  [SCHED] agent %d waited %lu ms for the CPU; agent %d "
                "holds it (spinning without a yield point?)\n", t_agent,
                (unsigned long)waited_ms, s_holder);
    fflush(stderr);
}

/* Whether the replay has handed this agent its turn (it has not taken it). */
static int sched_my_turn(void)
{
    return s_mode == 2 && s_turn == t_agent && !s_busy;
}

/* --- mode 3: the token ------------------------------------------------- */

static int sched_vw_met(void)
{
    if (t_vw.fn) {
        if (t_vw.fn(t_vw.ctx)) {
            t_vw.hit = 1;
            return 1;
        }
        if (t_vw.deadline && s_vt >= t_vw.deadline) {
            t_vw.timed_out = 1;
            return 1;
        }
        return 0;
    }
    return !t_vw.deadline || s_vt >= t_vw.deadline;
}

static int sched_next_agent(int me)
{
    int i;
    for (i = 1; i <= SCHED_AGENTS; i++) {
        int a = (me + i) % SCHED_AGENTS;
        if (a && s_alive[a] && !s_susp[a])
            return a;
    }
    return me;
}

static void sched_pass_token(int me)
{
    int n = sched_next_agent(me);
    InterlockedExchange(&s_tok, n);
    SetEvent(sched_tev(n));
}

static int sched_runnable_count(void)
{
    int a, n = 0;
    for (a = 1; a < SCHED_AGENTS; a++)
        if (s_alive[a] && !s_susp[a])
            n++;
    return n;
}

/* Every agent passed: nothing can happen before the earliest deadline, so
 * virtual time goes there -- and, paced, the host waits for the wall clock to
 * catch up, which is what makes the game run at its own speed. */
static void sched_vt_idle_advance(void)
{
    uint64_t m = ~0ull;
    int a;
    for (a = 1; a < SCHED_AGENTS; a++)
        if (s_alive[a] && !s_susp[a] && s_dl[a] > s_vt && s_dl[a] < m)
            m = s_dl[a];
    if (m == ~0ull) {
        static volatile LONG warned;
        if (InterlockedIncrement(&warned) <= 3) {
            fprintf(stderr, "  [SCHED] virtual: every agent waits with no deadline "
                    "(deadlock?); time moves on 1 ms\n");
            fflush(stderr);
        }
        m = s_vt + 1000000ull;
    }
    s_vt = m;
    if (!s_vfast) {
        LARGE_INTEGER now;
        uint64_t wall;
        QueryPerformanceCounter(&now);
        wall = (uint64_t)((now.QuadPart - s_wall0.QuadPart) * 1000000000.0
                          / (double)s_qpf.QuadPart);
        if (s_vt > wall + 1000000ull)
            Sleep((DWORD)((s_vt - wall) / 1000000ull));
    }
}

extern RECOMP_TLS uint64_t g_loop_n;      /* the loop heads, below */

/* RECOMP_SCHED_VTRACE=<file> (virtual mode, a debugging aid): every token
 * taken and every clock value handed out, as 24-byte records -- u8 type
 * (1 turn, 2 value), u8 agent, u16 kind, u32 guest [esp], u64 virtual time
 * or value, u64 loop count -- so two runs compared record by record
 * (scratch vtcmp.py) show where the first one went another way. */
static FILE *s_vtrace;
static int s_vtrace_on = -1;
static CRITICAL_SECTION s_vtrace_cs;

static void sched_vtrace(int type, int agent, int kind, uint64_t a, uint64_t b)
{
    struct { uint8_t t, ag; uint16_t k; uint32_t site; uint64_t a, b; } r;
    if (s_vtrace_on < 0) {
        const char *e = getenv("RECOMP_SCHED_VTRACE");
        InitializeCriticalSection(&s_vtrace_cs);
        s_vtrace = e && *e ? fopen(e, "wb") : NULL;
        if (s_vtrace)
            setvbuf(s_vtrace, NULL, _IOFBF, 1 << 22);
        s_vtrace_on = s_vtrace != NULL;
    }
    if (!s_vtrace_on)
        return;
    r.t = (uint8_t)type;
    r.ag = (uint8_t)agent;
    r.k = (uint16_t)kind;
    r.site = g_esp && g_esp < 0x08000000u ? *(uint32_t *)(g_xbox_mem_offset + g_esp) : 0;
    r.a = a;
    r.b = b;
    EnterCriticalSection(&s_vtrace_cs);
    fwrite(&r, sizeof r, 1, s_vtrace);
    {
        static unsigned n;
        if (!(++n & 0xFFFF))
            fflush(s_vtrace);           /* runs are often killed */
    }
    LeaveCriticalSection(&s_vtrace_cs);
}

void xbox_SchedVtraceFlush(void)
{
    if (s_vtrace_on > 0) {
        EnterCriticalSection(&s_vtrace_cs);
        fflush(s_vtrace);
        LeaveCriticalSection(&s_vtrace_cs);
    }
}

static void sched_acquire3(void)
{
    int me = t_agent;
    HANDLE ev = sched_tev(me);
    for (;;) {
        DWORD t0 = GetTickCount();
        int warned = 0;
        while (s_tok != me) {
            WaitForSingleObject(ev, 1000);
            if (!warned && s_tok != me && GetTickCount() - t0 > 10000) {
                int h = s_tok;
                warned = 1;
                fprintf(stderr, "  [SCHED] virtual: agent %d waited 10 s for the "
                        "token; agent %d has it%s\n", me, h,
                        h > 0 && h < SCHED_AGENTS && s_alive[h] ? "" : " (not alive)");
                if (h > 0 && h < SCHED_AGENTS && s_where[h])
                    sched_stuck(GetTickCount() - t0);
                fflush(stderr);
            }
        }
        t_held = 1;                         /* the condition may take an object */
        if (sched_vw_met()) {
            /* A spinner (a yield point, a zero sleep, a plain yield) runs
             * its next look at the flag, but makes no progress of its own:
             * a round of spinners and passers moves the clock on, or a
             * thread spinning on the vblank would hold time still. */
            if (t_vw.spin) {
                t_vw.spin = 0;
                if (++s_idle_run >= sched_runnable_count()) {
                    sched_vt_idle_advance();
                    s_idle_run = 0;
                }
            } else {
                s_idle_run = 0;
            }
            t_loop0 = g_loop_n;
            s_holder = me;
            sched_vtrace(1, me, 0, s_vt, g_loop_n);
            return;                         /* the caller keeps t_held */
        }
        t_held = 0;
        s_dl[me] = t_vw.deadline ? t_vw.deadline : ~0ull;
        if (++s_idle_run >= sched_runnable_count()) {
            sched_vt_idle_advance();
            s_idle_run = 0;
        }
        sched_pass_token(me);
    }
}

static void sched_release3(void)
{
    s_vt += (g_loop_n - t_loop0) * VT_LOOP_NS + VT_TURN_NS;
    t_vt_left = s_vt;
    s_holder = 0;
    sched_pass_token(t_agent);
}

static void sched_acquire(void)
{
    if (s_mode == 3) {
        sched_acquire3();
        return;
    }
    if (s_mode == 1) {
        LONG t = InterlockedIncrement(&s_ticket_next) - 1;
        DWORD t0 = 0;
        int spins = 0, warned = 0;
        while (s_ticket_serving != t) {
            LONG v = s_ticket_serving;
            if (v == t)
                break;
            if (++spins < 64) {
                YieldProcessor();
                continue;
            }
            if (!t0) {
                LARGE_INTEGER q;
                t0 = GetTickCount();
                QueryPerformanceCounter(&q);
                InterlockedCompareExchange64(&s_wait_since, q.QuadPart, 0);
            }
            InterlockedIncrement(&s_waiters);
            WaitOnAddress(&s_ticket_serving, &v, sizeof v, 2);
            InterlockedDecrement(&s_waiters);
            if (!warned && GetTickCount() - t0 > 3000) {
                warned = 1;
                sched_stuck(GetTickCount() - t0);
            }
        }
        if (!s_waiters)
            s_wait_since = 0;
        if (t_agent != s_last_agent) {
            sched_emit_run();
            s_last_agent = t_agent;
            s_run = 0;
        }
        s_run++;
        s_holder = t_agent;
        return;
    }
    /* replay */
    {
        DWORD t0 = GetTickCount();
        int warned = 0;
        AcquireSRWLockExclusive(&s_rlock);
        while (!((s_turn == t_agent || s_turn == -1) && !s_busy)) {
            if (!SleepConditionVariableSRW(&s_rcv, &s_rlock, 1000, 0)
                && !warned && GetTickCount() - t0 > 5000) {
                warned = 1;
                sched_stuck(GetTickCount() - t0);
            }
            /* Nobody holds the CPU and the agent whose turn it is has not
             * taken it: it is waiting for something this replay will not
             * produce. A busy holder (a long load) is not a stall. */
            if (!s_busy && s_turn != -1
                && GetTickCount() - s_free_since > sched_stall_ms()) {
                char why[128];
                int turn = s_turn;
                ReleaseSRWLockExclusive(&s_rlock);
                snprintf(why, sizeof why, "no progress for %lu s: agent %d's "
                         "turn never came", sched_stall_ms() / 1000, turn);
                sched_replay_fail(turn, why);
                AcquireSRWLockExclusive(&s_rlock);
            }
        }
        s_busy = 1;
        s_holder = t_agent;
        if (s_turn != -1)
            s_left--;
        ReleaseSRWLockExclusive(&s_rlock);
    }
}

static void sched_release(void)
{
    if (s_mode == 3) {
        sched_release3();
        return;
    }
    if (s_mode == 1) {
        s_holder = 0;
        InterlockedIncrement(&s_ticket_serving);
        if (s_waiters)
            WakeByAddressAll((PVOID)&s_ticket_serving);
        return;
    }
    AcquireSRWLockExclusive(&s_rlock);
    s_busy = 0;
    s_holder = 0;
    s_free_since = GetTickCount();
    if (s_turn == -1)
        WakeAllConditionVariable(&s_rcv);
    else if (s_left == 0) {
        if (++s_cur < s_nruns) {
            s_turn = s_runs[s_cur].agent;
            s_left = s_runs[s_cur].count;
        } else {
            s_turn = -1;
            fprintf(stderr, "  [SCHED] recorded schedule used up; running free\n");
            fflush(stderr);
        }
        WakeAllConditionVariable(&s_rcv);
    }
    ReleaseSRWLockExclusive(&s_rlock);
}

/* RECOMP_SCHED_SITES_PAGES=<va>[+<va>...] (with RECOMP_SCHED_SITES, both
 * runs): at every turn a hash of those 4 KB guest pages is recorded and, in
 * a replay, compared -- the first turn that differs follows the run that
 * wrote something else. A debugging aid. */
static void sched_turn_pages(void)
{
    static int n = -1;
    static uint32_t va[16];
    uint32_t h = 2166136261u;
    int i, k;
    if (n < 0) {
        const char *e = getenv("RECOMP_SCHED_SITES_PAGES");
        n = 0;
        while (e && *e && n < 16) {
            char *end;
            va[n++] = (uint32_t)strtoul(e, &end, 16) & ~0xFFFu;
            e = *end == '+' ? end + 1 : "";
        }
    }
    if (s_sites < 0)
        s_sites = getenv("RECOMP_SCHED_SITES") != NULL;
    if (!n || !s_sites || !t_agent || t_agent >= SCHED_AGENTS || s_mode == 3)
        return;
    for (i = 0; i < n; i++) {
        const uint32_t *p = (const uint32_t *)(g_xbox_mem_offset + va[i]);
        for (k = 0; k < 1024; k++)
            h = (h ^ p[k]) * 16777619u;
    }
    if (s_mode == 1) {
        sched_value(0, h, 0);
        return;
    }
    {
        SchedQueue *q = &s_q[t_agent];
        if (q->i < q->n && q->kind[q->i] == 0) {
            uint64_t rec = q->val[q->i++];
            if (rec != h && InterlockedIncrement(&s_site_reports) <= 6) {
                char d[300];
                sched_describe(rec, d, sizeof d);
                fprintf(stderr, "  [SCHED] agent %d turn at run %zu: pages differ "
                        "(recorded %s, now %08X); runs before: agent %d, %d, %d\n",
                        t_agent, s_cur, d, h,
                        s_cur >= 1 ? s_runs[s_cur - 1].agent : -1,
                        s_cur >= 2 ? s_runs[s_cur - 2].agent : -1,
                        s_cur >= 3 ? s_runs[s_cur - 3].agent : -1);
                fflush(stderr);
            }
        }
    }
}

int xbox_SchedEnter(void)
{
    if (!xbox_SchedMode() || !t_agent || t_held)
        return 0;
    sched_acquire();
    t_held = 1;
    sched_arm_loop_check();
    sched_turn_pages();
    if (s_mode == 3)
        memset(&t_vw, 0, sizeof t_vw);  /* a device model's pause is spent */
    return 1;
}

void xbox_SchedLeave(int entered)
{
    if (!entered)
        return;
    t_held = 0;
    sched_release();
}

int xbox_SchedBlockBegin(void)
{
    if (!t_held)
        return 0;
    if (s_sites < 0)
        s_sites = getenv("RECOMP_SCHED_SITES") != NULL;
    if (s_sites > 0 && t_agent > 0 && t_agent < SCHED_AGENTS && s_mode != 3) {
        /* RECOMP_SCHED_SITES: which host code let go, so a replay that does
         * not let go here (or does elsewhere) names it. */
        uint64_t tag = SCHED_LETGO_TAG | (uint64_t)(uint32_t)
            ((char *)_ReturnAddress() - (char *)GetModuleHandleA(NULL));
        if (s_mode == 1)
            sched_value(0, tag, 0);
        else {
            SchedQueue *q = &s_q[t_agent];
            if (q->i < q->n && q->kind[q->i] == 0) {
                uint64_t rec = q->val[q->i++];
                /* Named only while reports are left: symbolising both tags
                 * at every let-go made a replay of another build's (or
                 * another OS's) recording, whose tags all differ, crawl. */
                if (rec != tag && s_site_reports < 6) {
                    /* The same function counts as the same let-go: some
                     * (the loop check) let go from one branch per mode. */
                    char d1[300], d2[300];
                    const char *p1, *p2;
                    sched_describe(rec, d1, sizeof d1);
                    sched_describe(tag, d2, sizeof d2);
                    p1 = strrchr(d1, '+');
                    p2 = strrchr(d2, '+');
                    if (!(p1 && p2 && p1 - d1 == p2 - d2 && !strncmp(d1, d2, p1 - d1))
                        && InterlockedIncrement(&s_site_reports) <= 6) {
                        fprintf(stderr, "  [SCHED] agent %d value #%u: recorded "
                                "%s; replay: %s\n", t_agent, q->i, d1, d2);
                        fflush(stderr);
                    }
                }
            }
        }
    }
    if (t_agent > 0 && t_agent < SCHED_AGENTS) {
        s_where[t_agent] = _ReturnAddress();
        s_where_ret[t_agent] = g_esp && g_esp < 0x08000000u
            ? *(uint32_t *)(g_xbox_mem_offset + g_esp) : 0;
        s_blocked[t_agent] = 1;
    }
    t_held = 0;
    sched_release();
    return 1;
}

static RECOMP_TLS int t_no_pages;

void xbox_SchedBlockEnd(int released)
{
    if (!released)
        return;
    if (t_agent > 0 && t_agent < SCHED_AGENTS)
        s_blocked[t_agent] = 0;
    sched_acquire();
    t_held = 1;
    if (s_mode == 2)
        sched_arm_loop_check();     /* see xbox_SchedLoopCheck */
    if (!t_no_pages)
        sched_turn_pages();
}

/* xbox_SchedBlockEnd for a let-go whose outcome is a value: the recording
 * writes it after the CPU is back, a replay reads it before, so a per-turn
 * page hash (sched_turn_pages) between would be read in the wrong order. */
static void sched_block_end_valued(int b)
{
    t_no_pages = 1;
    xbox_SchedBlockEnd(b);
    t_no_pages = 0;
}

int xbox_SchedHeld(void)
{
    return t_held;
}

void xbox_SchedYield(uint32_t va)
{
    int i, b;
    (void)va;
    if (!t_held)
        return;
    for (i = 0; i < s_npumps; i++)
        s_pumps[i]();
    xbox_SchedMarkSpin();
    b = xbox_SchedBlockBegin();
    xbox_SchedBlockEnd(b);
}

/* A device model's pause between passes. In a replay its turns come when the
 * schedule says, and a host sleep only made everyone else wait out its
 * period at each of them: the agent goes straight to waiting for its turn. */
void xbox_SchedHostPause(DWORD ms)
{
    int m = xbox_SchedMode();
    if (m == 3)
        xbox_SchedVirtualPause((uint64_t)ms * 1000000ull);
    else if (m != 2)
        Sleep(ms);
}

void xbox_SchedAddPump(void (*fn)(void))
{
    if (s_npumps < SCHED_PUMPS)
        s_pumps[s_npumps++] = fn;
}

void xbox_SchedAddTick(void (*fn)(void))
{
    if (s_nticks < SCHED_TICKS)
        s_ticks[s_nticks++] = fn;
}

void xbox_SchedRunTicks(void)
{
    int i;
    for (i = 0; i < s_nticks; i++)
        s_ticks[i]();
}

/* --- values from outside --------------------------------------------- */

/* Values are recorded and replayed only by an agent holding the CPU: its
 * place in the order is then fixed. (sched_value_released is the exception,
 * for the waits and sleeps below, which take their recorded outcome in a
 * replay before they have the CPU back.) */
static int s_sites = -1;

uint64_t xbox_SchedValue(int kind, uint64_t live)
{
    return sched_value(kind, live, 0);
}

/* Virtual mode: every clock from virtual time, in the units the live value
 * has; decisions (vblank, timers firing, the APU's hold) are made from the
 * virtual clocks at their sites, so their live value already is virtual. */
static uint64_t sched_value_virtual2(int kind, uint64_t live);

static uint64_t sched_value_virtual(int kind, uint64_t live)
{
    uint64_t v = sched_value_virtual2(kind, live);
    sched_vtrace(2, t_agent, kind, v, g_loop_n);
    return v;
}

static uint64_t sched_value_virtual2(int kind, uint64_t live)
{
    uint64_t t = s_vt;
    switch (kind) {
    case XBOX_SCHED_QPC: {
        LARGE_INTEGER f = xbox_KeQueryPerformanceFrequency();
        uint64_t hz = (uint64_t)f.QuadPart;
        return (t / 1000000000ull) * hz + (t % 1000000000ull) * hz / 1000000000ull;
    }
    case XBOX_SCHED_SYSTIME:  return VT_BASE_FILETIME + t / 100;
    case XBOX_SCHED_INTTIME:  return t / 100;
    case XBOX_SCHED_RDTSC:    return (t / 1000ull) * 733333ull
                                     + (t % 1000ull) * 733333ull / 1000ull;
    case XBOX_SCHED_TICK:     return 1000ull + t / 1000000ull;
    case XBOX_SCHED_PAD:      return 0;          /* no host input: unchanged */
    case XBOX_SCHED_FILETIME: return VT_BASE_FILETIME;
    }
    return live;
}

static uint64_t sched_value(int kind, uint64_t live, int released)
{
    int a = t_agent;
    if (!xbox_SchedMode())
        return live;
    if (s_mode == 3)
        return sched_value_virtual(kind, live);
    if (!a || a >= SCHED_AGENTS || kind >= SCHED_KINDS || (!t_held && !released)) {
        /* A host value nobody records: if guest code asked for it, a replay
         * sees another one. Named, a few times, so the gap can be found. */
        static volatile LONG gaps;
        if (InterlockedIncrement(&gaps) <= 6) {
            fprintf(stderr, "  [SCHED] unrecorded value: kind %d asked by %s %d "
                    "(%s)\n", kind, a ? "agent" : "an unscheduled thread", a,
                    a && !t_held ? "not holding the CPU" : "no agent");
            sched_backtrace();
            fflush(stderr);
        }
        return live;
    }
    /* RECOMP_SCHED_SITES=1 (a debugging aid, in the recording and the replay
     * alike): each value is preceded by kind 0, the guest return address it
     * was asked from, so the first divergence names both sites. */
    if (s_sites < 0)
        s_sites = getenv("RECOMP_SCHED_SITES") != NULL;
    if (s_sites && kind != 0 && kind != XBOX_SCHED_PREEMPT) {
        uint32_t site = g_esp && g_esp < 0x08000000u
            ? *(uint32_t *)(g_xbox_mem_offset + g_esp) : 0;
        if (s_mode == 1)
            sched_value(0, site, released);
        else {
            SchedQueue *q = &s_q[a];
            if (q->i < q->n && q->kind[q->i] == 0) {
                uint64_t rec = q->val[q->i++];
                if (rec != site && InterlockedIncrement(&s_site_reports) <= 6) {
                    char d[300];
                    sched_describe(rec, d, sizeof d);
                    fprintf(stderr, "  [SCHED] agent %d value #%u (kind %d): "
                            "recorded %s, replay at guest 0x%08X\n",
                            a, q->i, kind, d, site);
                    if (s_site_reports <= 2)
                        sched_backtrace();
                    fflush(stderr);
                }
            }
        }
    }
    if (s_mode == 1) {
        int64_t d;
        uint64_t z;
        d = (int64_t)(live - s_prev_val[a][kind]);
        z = ((uint64_t)d << 1) ^ (uint64_t)(d >> 63);
        s_prev_val[a][kind] = live;
        fputc(0x02, s_out);
        sched_put_varint((uint64_t)a);
        sched_put_varint((uint64_t)kind);
        sched_put_varint(z);
        s_vals_written++;
        return live;
    }
    {
        SchedQueue *q = &s_q[a];
        if (q->i < q->n) {
            if (q->kind[q->i] != kind
                && InterlockedIncrement(&s_desync_reports) <= 20) {
                fprintf(stderr, "  [SCHED] agent %d value #%u: recorded kind %d, "
                        "asked for kind %d -- replay has diverged\n", a, q->i,
                        q->kind[q->i], kind);
                if (s_desync_reports <= 2)
                    sched_backtrace();
                fflush(stderr);
                if (!getenv("RECOMP_REPLAY_KEEP_GOING")) {
                    char why[96];
                    snprintf(why, sizeof why, "value #%u was recorded as kind %d, "
                             "the run asked for kind %d", q->i, q->kind[q->i], kind);
                    sched_replay_fail(a, why);
                }
            }
            return q->val[q->i++];
        }
        if (InterlockedIncrement(&s_desync_reports) <= 20) {
            fprintf(stderr, "  [SCHED] agent %d: recorded values used up (kind %d)\n",
                    a, kind);
            fflush(stderr);
        }
        return live;
    }
}

/* A host wait that may be waiting on another agent. Holding the CPU it is
 * first tried without blocking; only if that fails does the agent let go of
 * the CPU and block. A finite wait's outcome is host timing, so it is
 * recorded, and a replay gives it back: a recorded timeout returns at once
 * without waiting, anything else waits for as long as it takes. try_wait(ctx,
 * ms) does the wait and returns its result; timeout_code is what it returns
 * on a timeout. */
/* Virtual mode: one try of a wait's object (struct in xbox_SchedWaitFn). */
static int sched_vw_try_wait(void *p)
{
    struct { DWORD (*tw)(void *, DWORD); void *ctx; DWORD to, r; } *w = p;
    w->r = w->tw(w->ctx, 0);
    return w->r != w->to;
}

DWORD xbox_SchedWaitFn(DWORD (*try_wait)(void *ctx, DWORD ms), void *ctx,
                       DWORD ms, DWORD timeout_code)
{
    DWORD r;
    int b;

    if (!t_held)
        return try_wait(ctx, ms);
    r = try_wait(ctx, 0);
    if (r != timeout_code || ms == 0)
        return r;
    if (s_mode == 3) {
        /* Tried at each of this agent's turns, holding the token; the
         * object is taken there or the virtual deadline passes. */
        struct { DWORD (*tw)(void *, DWORD); void *ctx; DWORD to, r; } w;
        w.tw = try_wait; w.ctx = ctx; w.to = timeout_code; w.r = timeout_code;
        memset(&t_vw, 0, sizeof t_vw);
        t_vw.fn = sched_vw_try_wait;
        t_vw.ctx = &w;
        t_vw.deadline = ms == INFINITE ? 0 : s_vt + (uint64_t)ms * 1000000ull;
        b = xbox_SchedBlockBegin();
        xbox_SchedBlockEnd(b);
        r = t_vw.hit ? w.r : timeout_code;
        memset(&t_vw, 0, sizeof t_vw);
        return r;
    }
    b = xbox_SchedBlockBegin();
    if (s_mode == 2) {
        /* The object is taken by the host wait, as in the recording: taking
         * it at the agent's next turn instead counted two signals that
         * arrived in between as one (an auto-reset event), and replays went
         * another way from the first frame. */
        DWORD rec = timeout_code;
        if (ms != INFINITE || s_file_ver >= 2)
            rec = (DWORD)sched_value(XBOX_SCHED_WAIT, timeout_code, 1);
        if (rec == timeout_code && ms != INFINITE) {
            sched_block_end_valued(b);
            return rec;
        }
        if (s_file_ver < 2) {
            r = try_wait(ctx, INFINITE);
            sched_block_end_valued(b);
            return r;
        }
        /* The agent's next turn means the recording's wait was satisfied. A
         * notification event set and reset again in the setter's turn did
         * satisfy it (the host waiter woke in between), but a replay's host
         * waiter can miss that window and block for ever on its own turn:
         * once the turn has come, the recorded outcome stands. */
        for (;;) {
            r = try_wait(ctx, 1);
            if (r != timeout_code)
                break;
            if (sched_my_turn()) {
                r = try_wait(ctx, 0);
                if (r == timeout_code)
                    r = rec;
                break;
            }
        }
        sched_block_end_valued(b);
        return r;
    }
    r = try_wait(ctx, ms);
    sched_block_end_valued(b);
    xbox_SchedValue(XBOX_SCHED_WAIT, r);
    return r;
}

/* A lock another agent may hold, taken only while holding the CPU.
 *
 * Blocking on the host primitive took it outside the schedule: a thread
 * that released the lock and took it straight back before the waiter woke
 * (a critical section is not fair) left the waiter blocked at a turn the
 * recording gave it -- a replay deadlocked on CRI's ADX lock that way. So
 * the agent lets go of the CPU, waits until the lock is free without taking
 * it (recording) or not at all (replay: its next turn comes when the
 * recording's did), takes the CPU back and tries again. Each try is a turn,
 * the same number in both. */
int xbox_SchedTakeLock(int (*try_take)(void *ctx), void (*wait_free)(void *ctx),
                       void *ctx)
{
    if (try_take(ctx))
        return 1;
    if (t_held && s_mode == 3) {
        int b;
        memset(&t_vw, 0, sizeof t_vw);
        t_vw.fn = try_take;
        t_vw.ctx = ctx;
        b = xbox_SchedBlockBegin();
        xbox_SchedBlockEnd(b);
        memset(&t_vw, 0, sizeof t_vw);
        return 1;
    }
    if (!t_held) {
        do
            wait_free(ctx);
        while (!try_take(ctx));
        return 1;
    }
    for (;;) {
        int b = xbox_SchedBlockBegin();
        if (s_mode == 1)
            wait_free(ctx);
        xbox_SchedBlockEnd(b);
        if (try_take(ctx))
            return 1;
    }
}

static int cs_try(void *cs)
{
    return TryEnterCriticalSection((CRITICAL_SECTION *)cs) != 0;
}

/* Until it is free, without taking it: entering and leaving it here held it
 * for a moment while not holding the CPU, so an agent that tried it then
 * failed and let go -- a turn in the recording that a replay, which does not
 * wait here, never makes. That split replays a few minutes in. */
static void cs_wait_free(void *cs)
{
    while (((volatile CRITICAL_SECTION *)cs)->OwningThread)
        SwitchToThread();
}

/* A host critical section that another agent may hold. */
void xbox_SchedEnterCS(CRITICAL_SECTION *cs)
{
    if (!t_held) {
        /* RECOMP_SCHED_CS_TRACE: taken outside the schedule (who owns it
         * then is host timing) -- the first few, with where from. */
        static int tr = -1;
        static volatile LONG shown;
        if (tr < 0)
            tr = getenv("RECOMP_SCHED_CS_TRACE") != NULL;
        if (tr && xbox_SchedMode() && InterlockedIncrement(&shown) <= 8) {
            fprintf(stderr, "  [CSW] agent %d takes CS %p without the CPU\n",
                    t_agent, (void *)cs);
            sched_backtrace();
            fflush(stderr);
        }
        EnterCriticalSection(cs);
        return;
    }
    {
        static int tr = -1;
        static volatile LONG shown;
        if (tr < 0)
            tr = getenv("RECOMP_SCHED_CS_TRACE") != NULL;
        if (tr && !TryEnterCriticalSection(cs)) {
            if (InterlockedIncrement(&shown) <= 4000) {
                DWORD owner = (DWORD)(uintptr_t)cs->OwningThread;
                int a, oa = 0;
                for (a = 1; a < SCHED_AGENTS; a++)
                    if (s_agent_tid[a] == owner) { oa = a; break; }
                fprintf(stderr, "  [CSW] run %zu agent %d waits for CS %p held "
                        "by agent %d (tid %lu)\n", s_mode == 2 ? s_cur : (size_t)s_runs_written,
                        t_agent, (void *)cs, oa, (unsigned long)owner);
            }
        } else if (tr) {
            return;                     /* took it */
        }
    }
    xbox_SchedTakeLock(cs_try, cs_wait_free, cs);
}

/* A guest sleep: let go of the CPU for its length. A replay does not sleep
 * at all -- the schedule says when the thread runs again -- but returns what
 * the recorded sleep returned (an alertable one can end early). */
DWORD xbox_SchedSleep(DWORD ms, BOOL alertable)
{
    DWORD r;
    int b;
    if (!t_held)
        return SleepEx(ms, alertable);
    if (s_mode == 3) {
        memset(&t_vw, 0, sizeof t_vw);
        t_vw.deadline = ms ? s_vt + (uint64_t)ms * 1000000ull : 0;
        t_vw.spin = ms == 0;
        b = xbox_SchedBlockBegin();
        xbox_SchedBlockEnd(b);
        memset(&t_vw, 0, sizeof t_vw);
        return 0;
    }
    b = xbox_SchedBlockBegin();
    if (s_mode == 2) {
        r = (DWORD)sched_value(XBOX_SCHED_SLEEP, 0, 1);
        sched_block_end_valued(b);
        return r;
    }
    r = SleepEx(ms, alertable);
    sched_block_end_valued(b);
    xbox_SchedValue(XBOX_SCHED_SLEEP, r);
    return r;
}

/* A thread suspending itself. It lets go of the guest CPU first, so there is
 * a moment between that and the suspension when another agent can run --
 * and resume it before it has suspended, which would leave it suspended
 * for good (the CRI audio threads suspend themselves every tick and the
 * vblank resumes them). So the thread says it is about to suspend while it
 * still holds the CPU, and a resume of such a thread (xbox_SchedResume)
 * waits until the suspension has really happened. */
#define SCHED_SUSPENDERS 64
static volatile LONG s_suspending[SCHED_SUSPENDERS];

DWORD xbox_SchedSuspendSelf(HANDLE self)
{
    LONG tid = (LONG)GetCurrentThreadId();
    int i, slot = -1, b;
    DWORD prev;

    if (!t_held)
        return SuspendThread(self);
    if (s_mode == 3) {                      /* the scheduler's own count */
        prev = (DWORD)InterlockedIncrement(&s_susp[t_agent]) - 1;
        b = xbox_SchedBlockBegin();
        xbox_SchedBlockEnd(b);
        return prev;
    }
    for (i = 0; i < SCHED_SUSPENDERS && slot < 0; i++)
        if (InterlockedCompareExchange(&s_suspending[i], tid, 0) == 0)
            slot = i;
    b = xbox_SchedBlockBegin();
    prev = SuspendThread(self);
    if (slot >= 0)
        InterlockedExchange(&s_suspending[slot], 0);
    xbox_SchedBlockEnd(b);
    return prev;
}

/* Which agent a host thread is (0: none bound), for diagnostics. Agents bind
 * in every mode (xbox_SchedBind), so this answers in normal play too. */
int xbox_SchedAgentOfThread(DWORD tid)
{
    int a;
    if (!tid)
        return 0;
    for (a = 1; a < SCHED_AGENTS; a++)
        if (s_agent_tid[a] == tid)
            return a;
    return 0;
}

static int sched_agent_of(HANDLE h)
{
    DWORD tid = GetThreadId(h);
    int a;
    if (!tid)
        return 0;
    for (a = 1; a < SCHED_AGENTS; a++)
        if (s_alive[a] && s_agent_tid[a] == tid)
            return a;
    return 0;
}

/* Virtual mode: another agent suspended -- skipped by the token until
 * resumed. Elsewhere SuspendThread. */
DWORD xbox_SchedSuspendOther(HANDLE h)
{
    int a;
    if (xbox_SchedMode() == 3 && (a = sched_agent_of(h)))
        return (DWORD)InterlockedIncrement(&s_susp[a]) - 1;
    return SuspendThread(h);
}

DWORD xbox_SchedResume(HANDLE h)
{
    LONG tid = (LONG)GetThreadId(h);
    DWORD prev, t0;
    int i;

    if (xbox_SchedMode() == 3 && (i = sched_agent_of(h))) {
        prev = (DWORD)s_susp[i];
        if (prev)
            InterlockedDecrement(&s_susp[i]);
        return prev;
    }
    prev = ResumeThread(h);

    if (prev != 0 || !tid)
        return prev;
    for (i = 0; i < SCHED_SUSPENDERS; i++)
        if (s_suspending[i] == tid)
            break;
    if (i == SCHED_SUSPENDERS)
        return prev;
    /* Not suspended yet, but about to be: resume it once it is. This used to
     * give up after 2000 Sleep(0)s, which a loaded machine can spend before
     * the thread gets to SuspendThread: it then stayed suspended for good,
     * the guest got 0 for its previous suspend count instead of 1, and a
     * replay stuck on that thread's next turn. Its slot clears only once it
     * has been resumed, so wait for as long as that takes. */
    t0 = GetTickCount();
    while (s_suspending[i] == tid) {
        prev = ResumeThread(h);
        if (prev != 0)
            return prev;
        if (GetTickCount() - t0 > 10000) {
            fprintf(stderr, "  [SCHED] thread %ld has not suspended itself in "
                    "10 s; resume dropped\n", (long)tid);
            fflush(stderr);
            return prev;
        }
        SwitchToThread();
    }
    return prev;
}

static void sched_preempt_load(int agent, uint64_t n)
{
    SchedPreempts *p = &s_pre[agent];
    if (sched_grow_at(p->count))
        p->n = (uint64_t *)realloc(p->n, sched_grown(p->count) * sizeof(uint64_t));
    p->n[p->count++] = n;
}

/* --- preemption at loop heads -------------------------------------------
 *
 * A guest thread that spins in guest code -- waiting for a flag another
 * agent sets, with no kernel call in the loop -- never blocks, so it would
 * keep the CPU for ever. tools.recomp --loop-heads puts RECOMP_LOOP_HEAD()
 * at every loop head: a per-thread count of loop iterations (g_loop_n),
 * deterministic for a thread that runs the same code on the same data, and
 * a check every so often (g_loop_chk). Recording: if another agent has been
 * waiting for the CPU for more than 2 ms, the holder lets it have it, and
 * the count it yielded at is recorded. Replay: the thread yields when its
 * count reaches the next recorded one, exactly there. */
#define SCHED_LOOP_INTERVAL (1u << 16)
RECOMP_TLS uint64_t g_loop_n;
RECOMP_TLS uint64_t g_loop_chk = ~0ull;

static void sched_arm_loop_check(void)
{
    if (s_mode == 1 || s_mode == 3)
        g_loop_chk = g_loop_n + SCHED_LOOP_INTERVAL;
    else if (s_mode == 2) {
        SchedPreempts *p = &s_pre[t_agent];
        g_loop_chk = p->i < p->count ? p->n[p->i] : ~0ull;
    }
}

static uint64_t s_pre_next(void)
{
    SchedPreempts *p = &s_pre[t_agent];
    return p->i < p->count ? p->n[p->i] : ~0ull;
}

void xbox_SchedLoopCheck(void)
{
    if (!t_held || !t_agent) {
        /* No preemption without the CPU, in either mode. A replay used to
         * disarm here (~0) and re-arm only in xbox_SchedEnter, which a guest
         * thread seldom calls again (it lets go and takes the CPU back
         * through BlockBegin/BlockEnd): after one check made while not
         * holding, it never preempted again, ran on where the recording had
         * let go, and the replay diverged. Both now keep checking. */
        static volatile LONG shown;
        if (t_agent && s_mode == 2 && InterlockedIncrement(&shown) <= 3) {
            fprintf(stderr, "  [SCHED] agent %d: loop check without the CPU "
                    "(count %llu)\n", t_agent, (unsigned long long)g_loop_n);
            sched_backtrace();
            fflush(stderr);
        }
        if (s_mode == 1 || s_mode == 3 || !t_agent)
            g_loop_chk = g_loop_n + SCHED_LOOP_INTERVAL;
        else
            g_loop_chk = g_loop_n + 1 > s_pre_next() ? g_loop_n + 1 : s_pre_next();
        return;
    }
    if (s_mode == 3) {                      /* a fixed quantum */
        xbox_SchedBlockEnd(xbox_SchedBlockBegin());
        g_loop_chk = g_loop_n + SCHED_LOOP_INTERVAL;
        return;
    }
    if (s_mode == 1) {
        LONG64 since = s_wait_since;
        LARGE_INTEGER now;
        if (s_waiters && since) {
            QueryPerformanceCounter(&now);
            if (now.QuadPart - since > s_qpf.QuadPart / 500) {     /* 2 ms */
                xbox_SchedValue(XBOX_SCHED_PREEMPT, g_loop_n);
                xbox_SchedBlockEnd(xbox_SchedBlockBegin());
            }
        }
        g_loop_chk = g_loop_n + SCHED_LOOP_INTERVAL;
        return;
    }
    if (s_mode == 2) {
        SchedPreempts *p = &s_pre[t_agent];
        if (p->i < p->count && p->n[p->i] == g_loop_n) {
            p->i++;
            xbox_SchedBlockEnd(xbox_SchedBlockBegin());
        } else if (p->i < p->count && p->n[p->i] < g_loop_n
                   && !getenv("RECOMP_REPLAY_KEEP_GOING")) {
            char why[128];
            snprintf(why, sizeof why, "agent %d ran past the loop count %llu "
                     "where the recording let go of the CPU (now %llu)", t_agent,
                     (unsigned long long)p->n[p->i], (unsigned long long)g_loop_n);
            sched_replay_fail(t_agent, why);
        }
        sched_arm_loop_check();
    }
}

/* Thread completion, as the guest sees it.
 *
 * A guest thread ends by letting go of the CPU and then exiting the host
 * thread. Whether another agent's "has it ended?" -- XAPI GetExitCodeThread
 * through ObReferenceObjectByHandle's KTHREAD block, or a wait on the thread
 * handle -- sees the host thread gone depends on how far that exit got: host
 * timing, and the replay of a stage load went another way from there. So
 * under a schedule a thread counts as ended from the moment it says so while
 * it still holds the CPU (PsTerminateSystemThread, or its start routine
 * returning), recorded here by host thread id. */
#define SCHED_EXITED 256
static volatile LONG s_exited_tid[SCHED_EXITED];
static DWORD s_exited_code[SCHED_EXITED];

void xbox_SchedThreadStarting(void)
{
    LONG tid = (LONG)GetCurrentThreadId();
    int i;
    for (i = 0; i < SCHED_EXITED; i++)        /* a reused id is a new thread */
        InterlockedCompareExchange(&s_exited_tid[i], 0, tid);
}

void xbox_SchedThreadEnding(DWORD code)
{
    LONG tid = (LONG)GetCurrentThreadId();
    int i;
    if (!xbox_SchedMode())
        return;
    if (s_mode == 3 && t_agent > 0 && t_agent < SCHED_AGENTS)
        InterlockedExchange(&s_alive[t_agent], 0);  /* out of the rotation */
    for (i = 0; i < SCHED_EXITED; i++)
        if (s_exited_tid[i] == 0
            && InterlockedCompareExchange(&s_exited_tid[i], tid, 0) == 0) {
            s_exited_code[i] = code;
            return;
        }
}

/* 1 if the thread behind h has ended (code set), 0 if it runs, -1 if h is
 * not a thread or there is no schedule (ask the host). */
int xbox_SchedThreadEnded(HANDLE h, DWORD *code)
{
    LONG tid;
    int i;
    if (!xbox_SchedMode() || !(tid = (LONG)GetThreadId(h)))
        return -1;
    for (i = 0; i < SCHED_EXITED; i++)
        if (s_exited_tid[i] == tid) {
            if (code)
                *code = s_exited_code[i];
            return 1;
        }
    return 0;
}

/* Let go of the CPU and wait until done(ctx) holds or ms pass -- a device
 * model waiting for an agent to service it (the GPU stalled at a notify
 * until the timer agent has run the interrupt). Whether it was done in time
 * is host timing: recorded, and a replay waits only if the recording's wait
 * succeeded. Exactly one let-go either way. */
int xbox_SchedWaitUntil(int (*done)(void *ctx), void *ctx, DWORD ms)
{
    int b, ok;
    DWORD t0;
    if (!t_held) {
        t0 = GetTickCount();
        while (!(ok = done(ctx)) && GetTickCount() - t0 < ms)
            Sleep(0);
        return ok;
    }
    if (s_mode == 3) {
        memset(&t_vw, 0, sizeof t_vw);
        t_vw.fn = done;
        t_vw.ctx = ctx;
        t_vw.deadline = s_vt + (uint64_t)ms * 1000000ull;
        b = xbox_SchedBlockBegin();
        xbox_SchedBlockEnd(b);
        ok = t_vw.hit;
        memset(&t_vw, 0, sizeof t_vw);
        return ok;
    }
    b = xbox_SchedBlockBegin();
    if (s_mode == 2) {
        ok = (int)sched_value(XBOX_SCHED_WAIT, 1, 1);
        t0 = GetTickCount();
        while (ok && !done(ctx) && GetTickCount() - t0 < 60000)
            Sleep(0);
        sched_block_end_valued(b);
        return ok;
    }
    t0 = GetTickCount();
    while (!(ok = done(ctx)) && GetTickCount() - t0 < ms)
        Sleep(0);
    sched_block_end_valued(b);
    xbox_SchedValue(XBOX_SCHED_WAIT, (uint64_t)ok);
    return ok;
}
