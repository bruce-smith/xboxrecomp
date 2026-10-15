/*
 * The Win32 host primitives the runtime leans on, on a POSIX host.
 *
 * win32_compat.c used to accept these and do nothing: a vectored handler was
 * registered and never called, so every trapped device register (NV2A, APU,
 * USB) was a plain crash; SuspendThread only counted, so a thread that
 * suspended itself -- Blinx 2's audio init does -- ran straight on. Each check
 * here is a behaviour the kernel HLE and the device models depend on:
 *
 *   veh_read/veh_write   a fault on a PROT_NONE page reaches the handler with
 *                        the Win32 record (code, read/write, address), and
 *                        the edited CONTEXT is what execution resumes with
 *   veh_order            First=1 handlers run before earlier ones
 *   suspend_self         SuspendThread on yourself blocks until resumed
 *   suspend_other        another thread stops, its registers are readable,
 *                        counts nest, and ResumeThread restarts it
 *   suspend_main         the main thread (not created here) can be suspended
 *   srw_cv               SleepConditionVariableSRW wakes and times out
 *   wait_on_address      WaitOnAddress wakes on a change and times out
 *   open_thread          OpenThread finds a thread by id
 */
#include "platform/xbox_winnt.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>

static int failures = 0;

static void check(int ok, const char *what)
{
    printf("  %-56s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok) failures++;
}

/* ---- vectored handlers ---------------------------------------------- */

static uint8_t *s_trap_page;
static volatile ULONG_PTR s_seen_rw, s_seen_addr;
static volatile int s_order[4], s_norder;

static LONG trap_handler(PEXCEPTION_POINTERS ep)
{
    const uint8_t *ip = (const uint8_t *)(uintptr_t)ep->ContextRecord->Rip;
    ULONG_PTR a = ep->ExceptionRecord->ExceptionInformation[1];

    if (ep->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION
        || a < (ULONG_PTR)s_trap_page || a >= (ULONG_PTR)s_trap_page + 4096)
        return EXCEPTION_CONTINUE_SEARCH;
    s_order[s_norder++ & 3] = 2;
    s_seen_rw = ep->ExceptionRecord->ExceptionInformation[0];
    s_seen_addr = a;
    if (ip[0] == 0x8B && ip[1] == 0x02) {          /* mov eax, [rdx] */
        ep->ContextRecord->Rax = 0x12345678u;
        ep->ContextRecord->Rip += 2;
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    if (ip[0] == 0x89 && ip[1] == 0x02) {          /* mov [rdx], eax */
        ep->ContextRecord->Rip += 2;
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

static LONG first_handler(PEXCEPTION_POINTERS ep)
{
    (void)ep;
    s_order[s_norder++ & 3] = 1;
    return EXCEPTION_CONTINUE_SEARCH;
}

static void test_veh(void)
{
    uint32_t v = 0;
    PVOID h1, h2;

    s_trap_page = (uint8_t *)mmap(NULL, 4096, PROT_NONE,
                                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    h1 = AddVectoredExceptionHandler(0, trap_handler);
    h2 = AddVectoredExceptionHandler(1, first_handler);

    __asm__ volatile(".byte 0x8B, 0x02"            /* mov eax, [rdx] */
                     : "=a"(v) : "d"(s_trap_page + 0x10) : "memory");
    check(v == 0x12345678u && s_seen_rw == 0
          && s_seen_addr == (ULONG_PTR)(s_trap_page + 0x10),
          "veh_read: handler saw a read, its Rax was resumed with");

    __asm__ volatile(".byte 0x89, 0x02"            /* mov [rdx], eax */
                     : : "a"(7), "d"(s_trap_page + 0x20) : "memory");
    check(s_seen_rw == 1 && s_seen_addr == (ULONG_PTR)(s_trap_page + 0x20),
          "veh_write: handler saw a write at the right address");

    check(s_order[0] == 1 && s_order[1] == 2,
          "veh_order: First=1 handler runs first");

    check(RemoveVectoredExceptionHandler(h2) == 1
          && RemoveVectoredExceptionHandler(h1) == 1,
          "veh: handlers remove");
}

/* ---- suspension ------------------------------------------------------ */

static volatile LONG s_progress, s_spin, s_stop;

static DWORD WINAPI self_suspender(LPVOID p)
{
    (void)p;
    InterlockedExchange(&s_progress, 1);
    SuspendThread(GetCurrentThread());
    InterlockedExchange(&s_progress, 2);
    return 0;
}

static DWORD WINAPI spinner(LPVOID p)
{
    (void)p;
    while (!s_stop)
        InterlockedIncrement(&s_spin);
    return 0;
}

static void test_suspend(void)
{
    HANDLE t;
    LONG a, b;
    DWORD prev, tid = 0;
    CONTEXT ctx;
    int i;

    /* self */
    t = CreateThread(NULL, 0, self_suspender, NULL, 0, NULL);
    for (i = 0; i < 2000 && s_progress != 1; i++) Sleep(1);
    Sleep(50);
    a = s_progress;
    prev = ResumeThread(t);
    check(a == 1 && prev == 1 && WaitForSingleObject(t, 2000) == WAIT_OBJECT_0
          && s_progress == 2, "suspend_self: blocks until ResumeThread");
    CloseHandle(t);

    /* another thread */
    t = CreateThread(NULL, 0, spinner, NULL, 0, &tid);
    for (i = 0; i < 2000 && s_spin < 1000; i++) Sleep(1);
    prev = SuspendThread(t);
    a = s_spin;
    Sleep(50);
    b = s_spin;
    memset(&ctx, 0, sizeof ctx);
    ctx.ContextFlags = CONTEXT_FULL;
    check(prev == 0 && a == b, "suspend_other: the thread stops");
    check(GetThreadContext(t, &ctx) && ctx.Rip && ctx.Rsp,
          "suspend_other: GetThreadContext while stopped");
    check(SuspendThread(t) == 1 && ResumeThread(t) == 2,
          "suspend_other: counts nest");
    Sleep(20);
    check(s_spin == b, "suspend_other: still stopped at count 1");
    check(ResumeThread(t) == 1, "suspend_other: last resume returns 1");
    Sleep(20);
    check(s_spin != b, "suspend_other: runs again after resume");
    check(!GetThreadContext(t, &ctx), "suspend_other: no context while running");

    /* by id */
    {
        HANDLE o = OpenThread(THREAD_SUSPEND_RESUME, FALSE, tid);
        check(o && GetThreadId(o) == tid && GetThreadId(t) == tid,
              "open_thread: by id");
        if (o) CloseHandle(o);
        check(OpenThread(THREAD_SUSPEND_RESUME, FALSE, 0x7FFFFFF0u) == NULL,
              "open_thread: unknown id is NULL");
    }
    s_stop = 1;
    WaitForSingleObject(t, 2000);
    CloseHandle(t);
}

/* The main thread was not created by this layer. */
static HANDLE s_main;
static volatile LONG s_main_spin, s_main_ok, s_main_stop;

static DWORD WINAPI main_suspender(LPVOID p)
{
    LONG a, b;
    (void)p;
    while (s_main_spin < 1000) Sleep(1);
    if (SuspendThread(s_main) != 0) return 0;
    a = s_main_spin;
    Sleep(50);
    b = s_main_spin;
    ResumeThread(s_main);
    s_main_ok = (a == b) ? 1 : -1;
    s_main_stop = 1;
    return 0;
}

static void test_suspend_main(void)
{
    HANDLE t;
    DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(),
                    &s_main, 0, FALSE, DUPLICATE_SAME_ACCESS);
    t = CreateThread(NULL, 0, main_suspender, NULL, 0, NULL);
    while (!s_main_stop)
        InterlockedIncrement(&s_main_spin);
    WaitForSingleObject(t, 2000);
    CloseHandle(t);
    check(s_main_ok == 1, "suspend_main: the main thread stops and resumes");
}

/* ---- condition variables and address waits --------------------------- */

static SRWLOCK s_lock = SRWLOCK_INIT;
static CONDITION_VARIABLE s_cv = CONDITION_VARIABLE_INIT;
static volatile LONG s_ready, s_word;

static DWORD WINAPI producer(LPVOID p)
{
    (void)p;
    Sleep(30);
    AcquireSRWLockExclusive(&s_lock);
    s_ready = 1;
    ReleaseSRWLockExclusive(&s_lock);
    WakeAllConditionVariable(&s_cv);
    Sleep(30);
    InterlockedExchange(&s_word, 1);
    WakeByAddressAll((PVOID)&s_word);
    return 0;
}

static void test_waits(void)
{
    HANDLE t;
    BOOL r;
    LONG zero = 0;
    DWORD t0;

    AcquireSRWLockExclusive(&s_lock);
    t0 = GetTickCount();
    r = SleepConditionVariableSRW(&s_cv, &s_lock, 40, 0);
    check(!r && GetLastError() == 1460 && GetTickCount() - t0 >= 30,
          "srw_cv: times out with ERROR_TIMEOUT");
    t = CreateThread(NULL, 0, producer, NULL, 0, NULL);
    while (!s_ready)
        if (!SleepConditionVariableSRW(&s_cv, &s_lock, 2000, 0))
            break;
    check(s_ready == 1, "srw_cv: wakes on WakeAllConditionVariable");
    ReleaseSRWLockExclusive(&s_lock);

    while (s_word == 0)
        if (!WaitOnAddress(&s_word, &zero, sizeof zero, 2000))
            break;
    check(s_word == 1, "wait_on_address: wakes on a change");
    t0 = GetTickCount();
    r = WaitOnAddress(&s_word, (PVOID)&s_word, sizeof s_word, 40);
    check(!r && GetTickCount() - t0 >= 30, "wait_on_address: times out");
    WaitForSingleObject(t, 2000);
    CloseHandle(t);
}

static void test_interlocked(void)
{
    volatile LONGLONG q = 5;
    volatile LONG l = 0xF0;
    check(InterlockedIncrement64(&q) == 6 && InterlockedAdd64(&q, 4) == 10
          && InterlockedExchangeAdd64(&q, 1) == 10 && InterlockedExchange64(&q, 0) == 11
          && InterlockedDecrement64(&q) == -1,
          "interlocked64");
    check(InterlockedOr(&l, 0x0F) == 0xF0 && InterlockedAnd(&l, 0x3C) == 0xFF
          && l == 0x3C, "interlocked and/or return the old value");
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("POSIX host primitives:\n");
    test_veh();
    test_suspend();
    test_suspend_main();
    test_waits();
    test_interlocked();
    printf("%s (%d failure%s)\n", failures ? "FAILED" : "OK", failures,
           failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
