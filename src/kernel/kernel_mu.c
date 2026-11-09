/*
 * Memory units (the Xbox memory card), emulated at the XAPI level.
 *
 * A title talks to memory units through XAPI: XGetDevices with the MU device
 * type reports which controller slots hold one, XMountMU(port, slot, &drive)
 * gives it a drive letter, and saves then go through that letter exactly as
 * through U:. On a console XAPI's mass-storage driver and the kernel's FATX
 * stand behind it; here a title's overrides of those XAPI functions call the
 * functions below, and a memory unit is a host folder:
 *
 *     \Device\MU_<n>  ->  <save dir>\MU\<n>        n = port * 2 + slot
 *
 * mounted as F: (port 1 top), G: (port 1 bottom), H:, ... as XAPI assigns
 * them. Like XAPI, the drive is the title's own folder on the unit,
 * \Device\MU_<n>\UDATA\<title id>, taken from the title's U: link, so the
 * folder looks like a real unit's.
 *
 * Which slots hold a unit (the mask: bit port for the top slot, bit 16+port
 * for the bottom one, as XGetDevices reports them) is fixed from power-on:
 * - an exact replay: what the recording says (sched.c tag 0x04; a recording
 *   without it had none, so it replays as made);
 * - a network game (the scheduler's virtual mode): RECOMP_NET_MU, which the
 *   session sets the same on every PC; none without it;
 * - an input replay (RECOMP_INPUT_REPLAY) without RECOMP_MU: none, as the
 *   recordings made before memory units were;
 * - otherwise RECOMP_MU: "0" or "off" for none, or the controller ports
 *   (1-4) with a unit in the top slot, e.g. "1" or "1,2". Unset: port 1.
 */
#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int xbox_SchedMuMask(void);
BOOL xbox_path_mkdirs(const char* xbox_path);

static int s_mask = -1;

static int mu_parse(const char *v, int dflt)
{
    int mask = 0;
    if (!v || !*v)
        return dflt;
    if (!strcmp(v, "0") || !strcmp(v, "off"))
        return 0;
    for (; *v; v++)
        if (*v >= '1' && *v <= '4')
            mask |= 1 << (*v - '1');
    return mask;
}

/* sched_mode: the scheduler's mode. The recorder passes its own (1) while
 * it writes the schedule header: asking xbox_SchedMode() there would
 * re-enter the scheduler's one-time init and hang. */
int xbox_MuMaskFor(int sched_mode)
{
    if (s_mask < 0) {
        int rep = sched_mode == 2 ? xbox_SchedMuMask() : -1;
        const char *why;
        if (rep >= 0) {
            s_mask = rep;
            why = "as recorded";
        } else if (sched_mode == 3) {
            s_mask = mu_parse(getenv("RECOMP_NET_MU"), 0);
            why = "network game";
        } else if (getenv("RECOMP_INPUT_REPLAY") && !getenv("RECOMP_MU")) {
            s_mask = 0;
            why = "input replay";
        } else {
            s_mask = mu_parse(getenv("RECOMP_MU"), 1);
            why = getenv("RECOMP_MU") ? "RECOMP_MU" : "default";
        }
        fprintf(stderr, "  [MU] memory units: mask 0x%X (%s)\n", (unsigned)s_mask, why);
        fflush(stderr);
    }
    return s_mask;
}

int xbox_MuMask(void)
{
    return s_mask >= 0 ? s_mask : xbox_MuMaskFor(xbox_SchedMode());
}

static int mu_present(unsigned port, unsigned slot)
{
    return port < 4 && slot < 2 && (xbox_MuMask() >> (port + 16 * slot)) & 1;
}

/* The title's folder name under UDATA, from its U: link
 * (\Device\Harddisk0\Partition1\UDATA\<title id>). */
static void mu_title_dir(char *out, size_t n)
{
    const char *u = xbox_LookupSymbolicLink("\\??\\U:");
    const char *last = u ? strrchr(u, '\\') : NULL;
    snprintf(out, n, "%s", last && last[1] ? last + 1 : "4d530065");
}

static void mu_set_ansi(XBOX_ANSI_STRING *s, char *text)
{
    s->Length = (USHORT)strlen(text);
    s->MaximumLength = (USHORT)(s->Length + 1);
    s->Buffer = text;
}

/* XMountMU: 0 (ERROR_SUCCESS) and the drive letter, or a Win32 error. */
uint32_t xbox_MuMount(uint32_t port, uint32_t slot, char *drive)
{
    char link[16], target[128], title[64];
    XBOX_ANSI_STRING l, t;
    char letter = (char)('F' + port * 2 + slot);
    const char *have;

    if (!mu_present(port, slot))
        return 1167u;                         /* ERROR_DEVICE_NOT_CONNECTED */
    snprintf(link, sizeof link, "\\??\\%c:", letter);
    mu_title_dir(title, sizeof title);
    snprintf(target, sizeof target, "\\Device\\MU_%X\\UDATA\\%s",
             (unsigned)(port * 2 + slot), title);
    have = xbox_LookupSymbolicLink(link);
    if (!have || strcmp(have, target)) {
        if (have) {
            mu_set_ansi(&l, link);
            xbox_IoDeleteSymbolicLink(&l);
        }
        xbox_path_mkdirs(target);
        mu_set_ansi(&l, link);
        mu_set_ansi(&t, target);
        xbox_IoCreateSymbolicLink(&l, &t);
        fprintf(stderr, "  [MU] port %u %s slot mounted as %c: (%s)\n", port + 1,
                slot ? "bottom" : "top", letter, target);
        fflush(stderr);
    }
    if (drive)
        *drive = letter;
    return 0;
}

/* XUnmountMU. */
uint32_t xbox_MuUnmount(uint32_t port, uint32_t slot)
{
    char link[16];
    XBOX_ANSI_STRING l;
    if (port >= 4 || slot >= 2)
        return 87u;                           /* ERROR_INVALID_PARAMETER */
    snprintf(link, sizeof link, "\\??\\%c:", (char)('F' + port * 2 + slot));
    if (xbox_LookupSymbolicLink(link)) {
        mu_set_ansi(&l, link);
        xbox_IoDeleteSymbolicLink(&l);
    }
    return 0;
}
