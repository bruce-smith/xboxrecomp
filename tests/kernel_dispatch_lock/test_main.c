/* DISPATCH_LEVEL is a lock: one thread at a time, DPCs included, ISRs not.
 *
 * On the single-CPU Xbox, raising to DISPATCH_LEVEL keeps every other thread
 * and every DPC off the processor, and XDK code relies on it -- XPP edits its
 * pending-transfer list under nothing else, and a DPC racing it gave Blinx 2 a
 * cyclic list and a frozen game. kernel_hal.c turns the crossing into a
 * process-wide lock. This checks the three properties that matter:
 *
 *   1. two threads raised to DISPATCH_LEVEL never overlap;
 *   2. a host-run DPC (xbox_IrqlEnterInterrupt(2)) waits for a raised thread;
 *   3. a host-delivered ISR (xbox_IrqlEnterInterrupt(16)) does not -- it
 *      preempts DISPATCH_LEVEL on the console.
 *
 * --expect-overlap (run with RECOMP_DISPATCH_LOCK=0) is the negative control:
 * without the lock, property 1 must visibly fail. */
#include "kernel.h"
#include "xbox_memory_layout.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void *recomp_lookup(ULONG address) { (void)address; abort(); }
void *recomp_lookup_manual(ULONG address) { (void)address; abort(); }
extern ptrdiff_t g_xbox_mem_offset;

static volatile LONG s_inside, s_overlaps, s_stop;

static DWORD WINAPI raiser(LPVOID unused)
{
    (void)unused;
    while (!s_stop) {
        KIRQL old = xbox_KeRaiseIrqlToDpcLevel();
        if (InterlockedIncrement(&s_inside) != 1)
            InterlockedIncrement(&s_overlaps);
        Sleep(0);                           /* widen the window */
        InterlockedDecrement(&s_inside);
        xbox_KfLowerIrql(old);
    }
    return 0;
}

/* Holds DISPATCH_LEVEL for hold_ms, signalling once it is raised. */
typedef struct { HANDLE raised; DWORD hold_ms; } Holder;

static DWORD WINAPI holder(LPVOID p)
{
    Holder *h = (Holder *)p;
    KIRQL old = xbox_KeRaiseIrqlToDpcLevel();
    SetEvent(h->raised);
    Sleep(h->hold_ms);
    xbox_KfLowerIrql(old);
    return 0;
}

static DWORD time_enter(int level, DWORD hold_ms)
{
    Holder h = { CreateEventA(NULL, TRUE, FALSE, NULL), hold_ms };
    HANDLE t = CreateThread(NULL, 0, holder, &h, 0, NULL);
    ULONGLONG t0;
    int saved;
    DWORD waited;

    WaitForSingleObject(h.raised, INFINITE);
    t0 = GetTickCount64();
    saved = xbox_IrqlEnterInterrupt(level);
    waited = (DWORD)(GetTickCount64() - t0);
    xbox_IrqlLeaveInterrupt(saved);
    WaitForSingleObject(t, INFINITE);
    CloseHandle(t);
    CloseHandle(h.raised);
    return waited;
}

int main(int argc, char **argv)
{
    int expect_overlap = argc > 1 && strcmp(argv[1], "--expect-overlap") == 0;
    HANDLE th[4];
    int i, failed = 0;
    DWORD dpc_wait, isr_wait;
    /* Every IRQL change is published to KPCR.Irql in guest memory, through
     * each thread's TIB (XBOX_TIB_MAIN until a guest thread gets its own). */
    uint8_t *memory = VirtualAlloc(NULL, 1024 * 1024, MEM_RESERVE | MEM_COMMIT,
                                   PAGE_READWRITE);
    if (!memory)
        return 10;
    g_xbox_mem_offset = (ptrdiff_t)memory;

    for (i = 0; i < 4; i++)
        th[i] = CreateThread(NULL, 0, raiser, NULL, 0, NULL);
    Sleep(500);
    s_stop = 1;
    WaitForMultipleObjects(4, th, TRUE, INFINITE);

    if (expect_overlap) {
        if (s_overlaps == 0) {
            fprintf(stderr, "FAIL negative control: no overlap without the lock\n");
            return 1;
        }
        printf("PASS negative control: %ld overlaps with RECOMP_DISPATCH_LOCK=0\n",
               (long)s_overlaps);
        return 0;
    }

    if (s_overlaps != 0) {
        fprintf(stderr, "FAIL %ld overlaps of threads at DISPATCH_LEVEL\n",
                (long)s_overlaps);
        failed = 1;
    }
    dpc_wait = time_enter(2, 300);
    if (dpc_wait < 200) {
        fprintf(stderr, "FAIL a DPC entered after %lu ms while a thread held "
                "DISPATCH_LEVEL for 300 ms\n", (unsigned long)dpc_wait);
        failed = 1;
    }
    isr_wait = time_enter(16, 300);
    if (isr_wait > 100) {
        fprintf(stderr, "FAIL an ISR waited %lu ms behind DISPATCH_LEVEL\n",
                (unsigned long)isr_wait);
        failed = 1;
    }
    if (xbox_IrqlRaisedCount() != 0) {
        fprintf(stderr, "FAIL depth %d left raised\n", xbox_IrqlRaisedCount());
        failed = 1;
    }
    if (!failed)
        printf("PASS dispatch lock: no overlap, DPC waited %lu ms, ISR %lu ms\n",
               (unsigned long)dpc_wait, (unsigned long)isr_wait);
    return failed;
}
