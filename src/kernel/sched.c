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
 * File: "BX2SCHED", u32 version, then tagged records:
 *   0x01 varint agent, varint count            a run
 *   0x02 varint agent, varint kind, zigzag-varint delta   a value (delta
 *        from the previous value of that agent and kind)
 * Off by default; off, every entry point here is a test and a return.
 */
#include <windows.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "kernel.h"
#include "xbox_memory_layout.h"

#include <intrin.h>                    /* _ReturnAddress */
#include <dbghelp.h>                   /* the stuck report names host code */
#pragma comment(lib, "dbghelp.lib")
#pragma comment(lib, "synchronization.lib")   /* WaitOnAddress */

extern RECOMP_TLS uint32_t g_esp;
extern ptrdiff_t g_xbox_mem_offset;

#define SCHED_AGENTS     256
#define SCHED_KINDS      16

static int s_mode = -1;                 /* 0 off, 1 record, 2 replay */
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
void xbox_SchedClose(void)
{
    if (s_mode == 1 && s_out) {
        sched_emit_run();
        s_run = 0;
        fflush(s_out);
        fprintf(stderr, "  [SCHED] recording closed: %llu runs, %llu values\n",
                (unsigned long long)s_runs_written,
                (unsigned long long)s_vals_written);
    }
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
            if (!(q->n & 1023)) {
                q->kind = (uint8_t *)realloc(q->kind, q->n + 1024);
                q->val = (uint64_t *)realloc(q->val, (q->n + 1024) * sizeof(uint64_t));
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
        int mode = 0;
        QueryPerformanceFrequency(&s_qpf);
        if (p && *p) {
            mode = sched_load(p) ? 2 : 0;
        } else if (r && *r) {
            s_out = fopen(r, "wb");
            if (s_out) {
                uint32_t ver = SCHED_FILE_VER;
                setvbuf(s_out, NULL, _IOFBF, 1 << 20);
                fwrite("BX2SCHED", 1, 8, s_out);
                fwrite(&ver, 4, 1, s_out);
                mode = 1;
                atexit(sched_atexit);
                CloseHandle(CreateThread(NULL, 0, sched_flusher, NULL, 0, NULL));
                fprintf(stderr, "  [SCHED] recording the guest schedule to %s\n", r);
            } else
                fprintf(stderr, "  [SCHED] cannot write %s; recording off\n", r);
        }
        s_mode = mode;             /* published last */
    }
    return TRUE;
}

int xbox_SchedReplaying(void)
{
    return xbox_SchedMode() == 2;
}

void xbox_SchedBind(int agent)
{
    t_agent = agent;
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

static void sched_acquire(void)
{
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

int xbox_SchedEnter(void)
{
    if (!xbox_SchedMode() || !t_agent || t_held)
        return 0;
    sched_acquire();
    t_held = 1;
    sched_arm_loop_check();
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

void xbox_SchedBlockEnd(int released)
{
    if (!released)
        return;
    if (t_agent > 0 && t_agent < SCHED_AGENTS)
        s_blocked[t_agent] = 0;
    sched_acquire();
    t_held = 1;
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
    b = xbox_SchedBlockBegin();
    xbox_SchedBlockEnd(b);
}

/* A device model's pause between passes. In a replay its turns come when the
 * schedule says, and a host sleep only made everyone else wait out its
 * period at each of them: the agent goes straight to waiting for its turn. */
void xbox_SchedHostPause(DWORD ms)
{
    if (xbox_SchedMode() != 2)
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
static uint64_t sched_value(int kind, uint64_t live, int released);

uint64_t xbox_SchedValue(int kind, uint64_t live)
{
    return sched_value(kind, live, 0);
}

static uint64_t sched_value(int kind, uint64_t live, int released)
{
    int a = t_agent;
    if (!xbox_SchedMode() || !a || a >= SCHED_AGENTS || kind >= SCHED_KINDS)
        return live;
    if (!t_held && !released)
        return live;
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
                fflush(stderr);
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
            xbox_SchedBlockEnd(b);
            return rec;
        }
        if (s_file_ver < 2) {
            r = try_wait(ctx, INFINITE);
            xbox_SchedBlockEnd(b);
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
        xbox_SchedBlockEnd(b);
        return r;
    }
    r = try_wait(ctx, ms);
    xbox_SchedBlockEnd(b);
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

static void cs_wait_free(void *cs)
{
    EnterCriticalSection((CRITICAL_SECTION *)cs);
    LeaveCriticalSection((CRITICAL_SECTION *)cs);
}

/* A host critical section that another agent may hold. */
void xbox_SchedEnterCS(CRITICAL_SECTION *cs)
{
    if (!t_held) {
        EnterCriticalSection(cs);
        return;
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
    b = xbox_SchedBlockBegin();
    if (s_mode == 2) {
        r = (DWORD)sched_value(XBOX_SCHED_SLEEP, 0, 1);
        xbox_SchedBlockEnd(b);
        return r;
    }
    r = SleepEx(ms, alertable);
    xbox_SchedBlockEnd(b);
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

DWORD xbox_SchedResume(HANDLE h)
{
    LONG tid = (LONG)GetThreadId(h);
    DWORD prev = ResumeThread(h);
    int i, tries = 0;

    if (prev != 0 || !tid)
        return prev;
    for (i = 0; i < SCHED_SUSPENDERS; i++)
        if (s_suspending[i] == tid)
            break;
    if (i == SCHED_SUSPENDERS)
        return prev;
    /* Not suspended yet, but about to be: resume it once it is. */
    while (s_suspending[i] == tid && tries++ < 2000) {
        prev = ResumeThread(h);
        if (prev != 0)
            return prev;
        Sleep(0);
    }
    return prev;
}

static void sched_preempt_load(int agent, uint64_t n)
{
    SchedPreempts *p = &s_pre[agent];
    if (!(p->count & 1023))
        p->n = (uint64_t *)realloc(p->n, (p->count + 1024) * sizeof(uint64_t));
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
    if (s_mode == 1)
        g_loop_chk = g_loop_n + SCHED_LOOP_INTERVAL;
    else if (s_mode == 2) {
        SchedPreempts *p = &s_pre[t_agent];
        g_loop_chk = p->i < p->count ? p->n[p->i] : ~0ull;
    }
}

void xbox_SchedLoopCheck(void)
{
    if (!t_held || !t_agent) {
        g_loop_chk = s_mode == 1 ? g_loop_n + SCHED_LOOP_INTERVAL : ~0ull;
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
    b = xbox_SchedBlockBegin();
    if (s_mode == 2) {
        ok = (int)sched_value(XBOX_SCHED_WAIT, 1, 1);
        t0 = GetTickCount();
        while (ok && !done(ctx) && GetTickCount() - t0 < 60000)
            Sleep(0);
        xbox_SchedBlockEnd(b);
        return ok;
    }
    t0 = GetTickCount();
    while (!(ok = done(ctx)) && GetTickCount() - t0 < ms)
        Sleep(0);
    xbox_SchedBlockEnd(b);
    xbox_SchedValue(XBOX_SCHED_WAIT, (uint64_t)ok);
    return ok;
}
