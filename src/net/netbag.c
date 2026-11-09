/*
 * netbag.c -- Journey Bags for network games.
 *
 * Blinx 2 brings a player's own team to another console as a "Journey Bag":
 * Versus -> Prepare for Battle -> Create Journey Bag writes one to a memory
 * unit, the other console plays with it, and Return Journey Bag Data puts
 * the progress back into the home save. A network game does the same for
 * every joiner, automatically, with the game's own code doing the create and
 * the return: a headless child of this exe walks those menus (the title's
 * RECOMP_JOURNEYBAG driver, blinx2-recomp journeybag.c) and exits 0 or 5.
 *
 * Everything happens on a work copy, <root>/jbag, never on the real saves:
 * - prepare: copy the real UDATA/TDATA there, note a hash of every real save
 *   folder (baseline.txt), and create the bag on the work copy's memory unit
 *   (MU/0);
 * - finish (a clean end of the session, every peer's hashes equal): put the
 *   session's bag back on the work copy's unit, return it there, verify the
 *   result loads, and only then commit: the one changed save folder replaces
 *   the real one, after a backup (<root>/backups/<time>/), and is read back;
 *   if anything differs the backup goes back at once. A real folder that
 *   changed since prepare is never overwritten.
 * The same commit keeps the host's progress (xbox_NetBagHostCommit; on
 * unless RECOMP_NET_KEEP_HOST=0). Every step is written to
 * netplay_journeybag.log.
 *
 * <root> is <RECOMP_SAVE_DIR>/netplay when a save dir is given, else
 * <data>/netplay (np_data_dir). The child passes run in <root>/jbag/run
 * (RECOMP_RUN_DIR), so their logs never rotate the player's.
 */
#include "net_os.h"
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "netplay.h"
#include "net_tree.h"
#include "../kernel/kernel.h"       /* the Win32 vocabulary on every host */

#if !defined(_WIN32)
#include <signal.h>
#include <sys/wait.h>
#endif

#define BAG_TIMEOUT_MS  (300u * 1000u)
#define BAG_BACKUPS     20

/* RECOMP_NET_TEST_BAGFAIL=return|verify|commit (testing the safety paths): that
 * step fails -- the return pass, the verify pass, or the commit's read-back
 * (one real file is spoiled after the copy, so the backup must go back). */
static int test_fail(const char *step)
{
    const char *t = getenv("RECOMP_NET_TEST_BAGFAIL");
    return t && !strcmp(t, step);
}

/* ---- the log ------------------------------------------------------------------ */

static void bag_log(const char *fmt, ...)
{
    char line[1024];
    va_list ap;
    time_t t = time(NULL);
    struct tm *tm = localtime(&t);
    FILE *f;
    va_start(ap, fmt);
    vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);
    fprintf(stderr, "[JBAG] %s\n", line);
    fflush(stderr);
    f = fopen("netplay_journeybag.log", "a");
    if (f) {
        fprintf(f, "%04d-%02d-%02d %02d:%02d:%02d  %s\n", tm->tm_year + 1900,
                tm->tm_mon + 1, tm->tm_mday, tm->tm_hour, tm->tm_min, tm->tm_sec, line);
        fclose(f);
    }
}

/* ---- paths -------------------------------------------------------------------- */

void xbox_NetBagRoot(const char *explicit_save_dir, char *out, size_t n)
{
    if (explicit_save_dir && *explicit_save_dir)
        snprintf(out, n, "%s%cnetplay", explicit_save_dir, NP_SEP);
    else {
        char base[MAX_PATH];
        np_data_dir(base, sizeof base);
        snprintf(out, n, "%s%cnetplay", base, NP_SEP);
    }
}

static void join_path(char *out, size_t n, const char *a, const char *b)
{
    snprintf(out, n, "%s%c%s", a, NP_SEP, b);
}

/* ---- hashing a folder (sorted, so the same on every OS) ------------------------ */

typedef struct { char rel[260]; uint64_t size; } BagFile;

static int bag_cmp(const void *a, const void *b)
{
    return strcmp(((const BagFile *)a)->rel, ((const BagFile *)b)->rel);
}

static void list_files(const char *root, const char *rel, BagFile *out, int *n, int max)
{
    char dir[MAX_PATH];
    NpDir d;
    snprintf(dir, sizeof dir, "%s%s%s", root, rel[0] ? "/" : "", rel);
    np_native_path(dir + strlen(root));
    if (!np_dir_open(&d, dir))
        return;
    while (np_dir_next(&d)) {
        char r[260];
        snprintf(r, sizeof r, "%s%s%s", rel, rel[0] ? "/" : "", d.name);
        if (d.is_dir)
            list_files(root, r, out, n, max);
        else if (*n < max) {
            snprintf(out[*n].rel, sizeof out[*n].rel, "%s", r);
            out[*n].size = d.size;
            (*n)++;
        }
    }
    np_dir_close(&d);
}

static uint64_t fnv64(uint64_t h, const void *p, size_t n)
{
    const unsigned char *b = (const unsigned char *)p;
    while (n--)
        h = (h ^ *b++) * 1099511628211ull;
    return h;
}

/* Names and contents of every file under dir; 0 for an empty or missing one. */
uint64_t xbox_NetBagHashTree(const char *dir)
{
    static BagFile files[512];
    int n = 0, i;
    uint64_t h = 1469598103934665603ull;
    list_files(dir, "", files, &n, 512);
    if (!n)
        return 0;
    qsort(files, (size_t)n, sizeof files[0], bag_cmp);
    for (i = 0; i < n; i++) {
        char full[MAX_PATH], buf[65536];
        size_t k;
        FILE *f;
        snprintf(full, sizeof full, "%s/%s", dir, files[i].rel);
        np_native_path(full + strlen(dir));
        h = fnv64(h, files[i].rel, strlen(files[i].rel) + 1);
        f = fopen(full, "rb");
        if (!f)
            continue;
        while ((k = fread(buf, 1, sizeof buf, f)) > 0)
            h = fnv64(h, buf, k);
        fclose(f);
    }
    return h;
}

/* ---- the child pass ------------------------------------------------------------- */

/* Variables a pass must not inherit: it is a plain single-player run. */
static int bag_env_dropped(const char *kv)
{
    static const char *const prefix[] = {
        "RECOMP_NET", "RECOMP_SCHED_", "RECOMP_INPUT_", "RECOMP_PAD_", "RECOMP_FB_",
        "RECOMP_CHEAT", "RECOMP_JOURNEYBAG", "RECOMP_TEST_", "RECOMP_VK_",
        "RECOMP_USB_PADS", "RECOMP_MU", "RECOMP_PORTABLE_MATH", "RECOMP_CAPTURE",
        "RECOMP_SAVE_DIR", "RECOMP_RUN_DIR", "RECOMP_HEADLESS", "RECOMP_GPU",
        "RECOMP_AUDIO", "RECOMP_USB=", "RECOMP_WATCHDOG", NULL
    };
    int i;
    for (i = 0; prefix[i]; i++) {
        size_t k, n = strlen(prefix[i]);
        for (k = 0; k < n; k++) {
            char c = kv[k];
            if (c >= 'a' && c <= 'z')
                c = (char)(c - 32);                 /* Windows names ignore case */
            if (c != prefix[i][k])
                break;
        }
        if (k == n)
            return 1;
    }
    return 0;
}

/* Whether the pass's log already says "[JBAG] failed": the driver may report
 * a failure and not exit (an older driver, an unknown job), and waiting out
 * the timeout then costs minutes for nothing. */
static int log_says_failed(const char *logp)
{
    char line[512];
    int failed = 0;
    FILE *f = fopen(logp, "r");
    if (!f)
        return 0;
    while (!failed && fgets(line, sizeof line, f))
        failed = strstr(line, "[JBAG] failed: ") != NULL;
    fclose(f);
    return failed;
}

/* Runs this exe headless with `job` (RECOMP_JOURNEYBAG) on save dir `work`;
 * its stderr goes to <work>/run/<name>.log. 0 done, 5 the driver failed,
 * -1 it could not run, timed out or crashed. */
static int bag_child(const char *work, const char *job, const char *name)
{
    char exe[MAX_PATH], run[MAX_PATH], logp[MAX_PATH];
    char kv[11][MAX_PATH + 40];
    int nkv = 0, i;
    np_exe_path(exe, sizeof exe);
    join_path(run, sizeof run, work, "run");
    np_tree_mk(run);
    snprintf(logp, sizeof logp, "%s%c%s.log", run, NP_SEP, name);
    snprintf(kv[nkv++], sizeof kv[0], "RECOMP_HEADLESS=1");
    snprintf(kv[nkv++], sizeof kv[0], "RECOMP_USB=1");
    snprintf(kv[nkv++], sizeof kv[0], "RECOMP_AUDIO=0");
    snprintf(kv[nkv++], sizeof kv[0], "RECOMP_GPU=software");
    snprintf(kv[nkv++], sizeof kv[0], "RECOMP_MU=1");
    snprintf(kv[nkv++], sizeof kv[0], "RECOMP_INPUT_RECORD=0");
    snprintf(kv[nkv++], sizeof kv[0], "RECOMP_SAVE_DIR=%s", work);
    snprintf(kv[nkv++], sizeof kv[0], "RECOMP_RUN_DIR=%s", run);
    snprintf(kv[nkv++], sizeof kv[0], "RECOMP_JOURNEYBAG=%s", job);
    snprintf(kv[nkv++], sizeof kv[0], "RECOMP_JOURNEYBAG_EXIT=1");
    bag_log("pass %s: %s (log %s)", name, job, logp);
#if defined(_WIN32)
    {
        char *cur = GetEnvironmentStringsA(), *p, *blk, *w;
        size_t size = 2;
        STARTUPINFOA si;
        PROCESS_INFORMATION pi;
        SECURITY_ATTRIBUTES sa;
        HANDLE lf;
        DWORD code = 1, r;
        for (p = cur; *p; p += strlen(p) + 1)
            size += strlen(p) + 1;
        for (i = 0; i < nkv; i++)
            size += strlen(kv[i]) + 1;
        blk = w = (char *)malloc(size);
        if (!blk) {
            FreeEnvironmentStringsA(cur);
            return -1;
        }
        for (p = cur; *p; p += strlen(p) + 1)
            if (!bag_env_dropped(p)) {
                strcpy(w, p);
                w += strlen(p) + 1;
            }
        for (i = 0; i < nkv; i++) {
            strcpy(w, kv[i]);
            w += strlen(kv[i]) + 1;
        }
        *w = 0;
        FreeEnvironmentStringsA(cur);
        memset(&sa, 0, sizeof sa);
        sa.nLength = sizeof sa;
        sa.bInheritHandle = TRUE;
        lf = CreateFileA(logp, GENERIC_WRITE, FILE_SHARE_READ, &sa, CREATE_ALWAYS,
                         FILE_ATTRIBUTE_NORMAL, NULL);
        memset(&si, 0, sizeof si);
        si.cb = sizeof si;
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdInput = NULL;
        si.hStdOutput = lf;
        si.hStdError = lf;
        if (!CreateProcessA(exe, NULL, NULL, NULL, TRUE, CREATE_NO_WINDOW, blk, run, &si, &pi)) {
            bag_log("pass %s: cannot start %s (error %lu)", name, exe,
                    (unsigned long)GetLastError());
            free(blk);
            if (lf != INVALID_HANDLE_VALUE)
                CloseHandle(lf);
            return -1;
        }
        free(blk);
        if (lf != INVALID_HANDLE_VALUE)
            CloseHandle(lf);
        {
            unsigned waited = 0, failed_at = 0;
            while ((r = WaitForSingleObject(pi.hProcess, 1000)) == WAIT_TIMEOUT
                   && waited < BAG_TIMEOUT_MS) {
                waited += 1000;
                if (!failed_at && log_says_failed(logp))
                    failed_at = waited;
                if (failed_at && waited - failed_at >= 3000)
                    break;                        /* it said so and did not exit */
            }
            if (r != WAIT_OBJECT_0) {
                TerminateProcess(pi.hProcess, 99);    /* our own child */
                WaitForSingleObject(pi.hProcess, 5000);
                if (failed_at)
                    bag_log("pass %s: the driver failed and did not exit; stopped", name);
                else
                    bag_log("pass %s: no result in %u s; stopped", name,
                            BAG_TIMEOUT_MS / 1000u);
                code = failed_at ? 5 : (DWORD)-1;
            } else
                GetExitCodeProcess(pi.hProcess, &code);
        }
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
        bag_log("pass %s: exit %ld", name, (long)(int)code);
        return (int)code == 0 ? 0 : (int)code == 5 ? 5 : -1;
    }
#else
    {
        pid_t pid = fork();
        int st = 0;
        unsigned waited = 0, failed_at = 0;
        if (pid < 0)
            return -1;
        if (pid == 0) {
            extern char **environ;
            char **e;
            FILE *lf;
            /* Drop every inherited variable on the list, then set ours. */
            for (;;) {
                int dropped = 0;
                for (e = environ; *e; e++)
                    if (bag_env_dropped(*e)) {
                        char key[256];
                        const char *eq = strchr(*e, '=');
                        size_t k = eq ? (size_t)(eq - *e) : strlen(*e);
                        if (k >= sizeof key)
                            k = sizeof key - 1;
                        memcpy(key, *e, k);
                        key[k] = 0;
                        unsetenv(key);
                        dropped = 1;
                        break;
                    }
                if (!dropped)
                    break;
            }
            for (i = 0; i < nkv; i++)
                putenv(kv[i]);
            lf = freopen(logp, "w", stderr);
            (void)lf;
            if (chdir(run) != 0) { /* RECOMP_RUN_DIR chdirs too */ }
            execl(exe, exe, (char *)NULL);
            _exit(127);
        }
        while (waitpid(pid, &st, WNOHANG) == 0) {
            if (waited % 1000 == 0 && !failed_at && log_says_failed(logp))
                failed_at = waited + 1;
            if (failed_at && waited + 1 - failed_at >= 3000) {
                kill(pid, SIGKILL);               /* it said so and did not exit */
                waitpid(pid, &st, 0);
                bag_log("pass %s: the driver failed and did not exit; stopped", name);
                return 5;
            }
            if (waited >= BAG_TIMEOUT_MS) {
                kill(pid, SIGKILL);               /* our own child */
                waitpid(pid, &st, 0);
                bag_log("pass %s: no result in %u s; stopped", name, BAG_TIMEOUT_MS / 1000u);
                return -1;
            }
            usleep(100000);
            waited += 100;
        }
        if (!WIFEXITED(st)) {
            bag_log("pass %s: ended by signal %d", name, WIFSIGNALED(st) ? WTERMSIG(st) : 0);
            return -1;
        }
        bag_log("pass %s: exit %d", name, WEXITSTATUS(st));
        return WEXITSTATUS(st) == 0 ? 0 : WEXITSTATUS(st) == 5 ? 5 : -1;
    }
#endif
}

/* The driver's own reason, from the pass's log ("[JBAG] failed: ..."). */
static void bag_reason(const char *work, const char *name, char *why, size_t n)
{
    char p[MAX_PATH], line[512];
    FILE *f;
    snprintf(p, sizeof p, "%s%crun%c%s.log", work, NP_SEP, NP_SEP, name);
    f = fopen(p, "r");
    if (!f)
        return;
    while (fgets(line, sizeof line, f)) {
        char *s = strstr(line, "[JBAG] failed: ");
        if (s) {
            s += 15;
            s[strcspn(s, "\r\n")] = 0;
            snprintf(why, n, "%s", s);
        }
    }
    fclose(f);
}

/* The return pass's "[JBAG] summary: ..." (what came home), if it printed one. */
static void bag_summary(const char *work, char *out, size_t n)
{
    char p[MAX_PATH], line[512];
    FILE *f;
    out[0] = 0;
    snprintf(p, sizeof p, "%s%crun%creturn.log", work, NP_SEP, NP_SEP);
    f = fopen(p, "r");
    if (!f)
        return;
    while (fgets(line, sizeof line, f)) {
        char *s = strstr(line, "[JBAG] summary: ");
        if (s) {
            s += 16;
            s[strcspn(s, "\r\n")] = 0;
            snprintf(out, n, "%s", s);
        }
    }
    fclose(f);
}

/* ---- save folders: <saves>/UDATA/<title>/<id> ------------------------------------- */

typedef struct { char id[64]; uint64_t h; } SaveSum;

/* Every save folder under <saves>/UDATA (any title), with a hash of each. */
static int save_sums(const char *saves, SaveSum *out, int max)
{
    char udata[MAX_PATH], tdir[MAX_PATH], sdir[MAX_PATH];
    NpDir t, s;
    int n = 0;
    join_path(udata, sizeof udata, saves, "UDATA");
    if (!np_dir_open(&t, udata))
        return 0;
    while (np_dir_next(&t)) {
        if (!t.is_dir)
            continue;
        join_path(tdir, sizeof tdir, udata, t.name);
        if (!np_dir_open(&s, tdir))
            continue;
        while (np_dir_next(&s))
            if (s.is_dir && n < max) {
                join_path(sdir, sizeof sdir, tdir, s.name);
                snprintf(out[n].id, sizeof out[n].id, "%s%c%s", t.name, NP_SEP, s.name);
                out[n].h = xbox_NetBagHashTree(sdir);
                n++;
            }
        np_dir_close(&s);
    }
    np_dir_close(&t);
    return n;
}

static const SaveSum *find_sum(const SaveSum *a, int n, const char *id)
{
    int i;
    for (i = 0; i < n; i++)
        if (!strcmp(a[i].id, id))
            return &a[i];
    return NULL;
}

static int write_baseline(const char *path, const char *saves, int slot)
{
    SaveSum s[64];
    int n = save_sums(saves, s, 64), i;
    FILE *f = fopen(path, "w");
    if (!f)
        return 0;
    fprintf(f, "slot %d\nsrc %s\n", slot, saves);
    for (i = 0; i < n; i++)
        fprintf(f, "save %s %016llX\n", s[i].id, (unsigned long long)s[i].h);
    fclose(f);
    return 1;
}

static int read_baseline(const char *path, SaveSum *out, int max, int *slot)
{
    char line[600];
    int n = 0;
    FILE *f = fopen(path, "r");
    if (!f)
        return -1;
    while (fgets(line, sizeof line, f)) {
        char id[300];
        unsigned long long h;
        if (sscanf(line, "slot %d", slot) == 1)
            continue;
        if (n < max && sscanf(line, "save %299s %llx", id, &h) == 2) {
            snprintf(out[n].id, sizeof out[n].id, "%s", id);
            out[n].h = (uint64_t)h;
            n++;
        }
    }
    fclose(f);
    return n;
}

static void stamp(char *out, size_t n)
{
    time_t t = time(NULL);
    struct tm *tm = localtime(&t);
    snprintf(out, n, "%04d%02d%02d-%02d%02d%02d", tm->tm_year + 1900, tm->tm_mon + 1,
             tm->tm_mday, tm->tm_hour, tm->tm_min, tm->tm_sec);
}

/* Keep the newest BAG_BACKUPS folders under <root>/backups. */
static void prune_backups(const char *root)
{
    char dir[MAX_PATH], names[256][64], p[MAX_PATH];
    NpDir d;
    int n = 0, i, j;
    join_path(dir, sizeof dir, root, "backups");
    if (!np_dir_open(&d, dir))
        return;
    while (np_dir_next(&d))
        if (d.is_dir && n < 256)
            snprintf(names[n++], sizeof names[0], "%s", d.name);
    np_dir_close(&d);
    for (i = 0; i < n; i++)                       /* oldest first (names are times) */
        for (j = i + 1; j < n; j++)
            if (strcmp(names[j], names[i]) < 0) {
                char t[64];
                memcpy(t, names[i], 64);
                memcpy(names[i], names[j], 64);
                memcpy(names[j], t, 64);
            }
    for (i = 0; i + BAG_BACKUPS < n; i++) {
        join_path(p, sizeof p, dir, names[i]);
        np_tree_rm(p);
    }
}

/* Replace the real save folders that `changed` lists with the work copy's,
 * after backing each up; read back, and restore every one on any mismatch. */
static int commit(const char *root, const char *real, const char *work,
                  char ids[][64], int n, char *msg, size_t mn)
{
    char bdir[MAX_PATH], st[32], a[MAX_PATH], b[MAX_PATH], c[MAX_PATH], tmp[MAX_PATH];
    int i, ok = 1;
    stamp(st, sizeof st);
    join_path(tmp, sizeof tmp, root, "backups");
    join_path(bdir, sizeof bdir, tmp, st);
    for (i = 0; i < n; i++) {                       /* 1: back up */
        snprintf(a, sizeof a, "%s%cUDATA%c%s", real, NP_SEP, NP_SEP, ids[i]);
        snprintf(c, sizeof c, "%s%cUDATA%c%s", bdir, NP_SEP, NP_SEP, ids[i]);
        np_tree_copy(a, c);
        if (xbox_NetBagHashTree(a) != xbox_NetBagHashTree(c)) {
            snprintf(msg, mn, "could not back up the save (%s); nothing changed", a);
            bag_log("commit refused: backup of %s did not read back", a);
            return 0;
        }
    }
    bag_log("backup: %s", bdir);
    for (i = 0; i < n && ok; i++) {                 /* 2: replace, read back */
        snprintf(a, sizeof a, "%s%cUDATA%c%s", real, NP_SEP, NP_SEP, ids[i]);
        snprintf(b, sizeof b, "%s%cUDATA%c%s", work, NP_SEP, NP_SEP, ids[i]);
        np_tree_rm(a);
        np_tree_copy(b, a);
        if (test_fail("commit")) {
            char spoil[MAX_PATH];
            FILE *f;
            snprintf(spoil, sizeof spoil, "%s%cSaveMeta.xbx", a, NP_SEP);
            f = fopen(spoil, "ab");
            if (f) {
                fputc(0x55, f);
                fclose(f);
            }
            bag_log("test: spoiled %s after the copy", spoil);
        }
        if (xbox_NetBagHashTree(a) != xbox_NetBagHashTree(b)) {
            bag_log("commit: %s did not read back; restoring the backup", a);
            ok = 0;
        } else
            bag_log("commit: %s replaced", a);
    }
    if (!ok) {                                      /* 3: put everything back */
        for (i = 0; i < n; i++) {
            snprintf(a, sizeof a, "%s%cUDATA%c%s", real, NP_SEP, NP_SEP, ids[i]);
            snprintf(c, sizeof c, "%s%cUDATA%c%s", bdir, NP_SEP, NP_SEP, ids[i]);
            np_tree_rm(a);
            np_tree_copy(c, a);
            bag_log("restored %s from the backup (%s)", a,
                    xbox_NetBagHashTree(a) == xbox_NetBagHashTree(c) ? "ok" : "FAILED");
        }
        snprintf(msg, mn, "writing the save failed; it was restored from the backup in %s",
                 bdir);
        return 0;
    }
    prune_backups(root);
    return 1;
}

/* RECOMP_NET_JBAG_TEST=1 (test scripts): never commit to the default save
 * place, so a test that forgot RECOMP_SAVE_DIR cannot touch a player's saves. */
static int commit_allowed(const char *real, int explicit_dir)
{
    const char *t = getenv("RECOMP_NET_JBAG_TEST");
    if (t && *t == '1' && !explicit_dir) {
        bag_log("commit refused: RECOMP_NET_JBAG_TEST=1 and no RECOMP_SAVE_DIR (%s)", real);
        return 0;
    }
    return 1;
}

/* ---- prepare (a joiner, before joining) ------------------------------------------- */

int xbox_NetBagPrepare(const char *saves, const char *root, int slot, char *why, size_t wn)
{
    char work[MAX_PATH], a[MAX_PATH], b[MAX_PATH], ready[MAX_PATH];
    int r;
    why[0] = 0;
    join_path(work, sizeof work, root, "jbag");
    np_tree_rm(work);
    np_tree_mk(work);
    join_path(a, sizeof a, saves, "UDATA");
    join_path(b, sizeof b, work, "UDATA");
    np_tree_copy(a, b);
    join_path(a, sizeof a, saves, "TDATA");
    join_path(b, sizeof b, work, "TDATA");
    np_tree_copy(a, b);
    join_path(a, sizeof a, work, "MU");
    join_path(b, sizeof b, a, "0");
    np_tree_mk(b);
    join_path(a, sizeof a, work, "baseline.txt");
    if (!write_baseline(a, saves, slot)) {
        snprintf(why, wn, "cannot write %s", a);
        return 0;
    }
    bag_log("prepare: save %d from %s, work copy %s", slot, saves, work);
    {
        char job[32];
        snprintf(job, sizeof job, "create:%d", slot);
        r = bag_child(work, job, "create");
    }
    if (r != 0) {
        snprintf(why, wn, r == 5 ? "the game could not create the Journey Bag"
                              : "the Journey Bag pass did not finish");
        bag_reason(work, "create", why, wn);
        bag_log("prepare failed: %s", why);
        return 0;
    }
    join_path(a, sizeof a, work, "MU");
    join_path(b, sizeof b, a, "0");
    if (!xbox_NetBagHashTree(b)) {
        snprintf(why, wn, "the game said the Journey Bag was saved, but none is on the unit");
        bag_log("prepare failed: %s", why);
        return 0;
    }
    join_path(ready, sizeof ready, work, "ready");
    {
        FILE *f = fopen(ready, "w");
        if (f) {
            fprintf(f, "%016llX\n", (unsigned long long)xbox_NetBagHashTree(b));
            fclose(f);
        }
    }
    bag_log("prepare: Journey Bag ready (%s)", b);
    return 1;
}

int xbox_NetBagReady(const char *root, char *bagdir, size_t n)
{
    char work[MAX_PATH], ready[MAX_PATH], mu[MAX_PATH];
    FILE *f;
    join_path(work, sizeof work, root, "jbag");
    join_path(ready, sizeof ready, work, "ready");
    f = fopen(ready, "r");
    if (!f)
        return 0;
    fclose(f);
    join_path(mu, sizeof mu, work, "MU");
    snprintf(bagdir, n, "%s%c0", mu, NP_SEP);
    return xbox_NetBagHashTree(bagdir) != 0;
}

/* ---- verify ----------------------------------------------------------------------- */

/* The returned save loads: the driver's verify pass when it has one, else the
 * files' shape (a home save of the game's three files, the data file whole). */
static int bag_verify(const char *work, int slot, char *why, size_t wn)
{
    char job[32];
    int r;
    if (test_fail("verify")) {
        snprintf(why, wn, "test: the verify pass fails");
        return 0;
    }
    snprintf(job, sizeof job, "verify:%d", slot);
    r = bag_child(work, job, "verify");
    if (r == 0)
        return 1;
    {
        char reason[256] = "";
        bag_reason(work, "verify", reason, sizeof reason);
        if (strstr(reason, "RECOMP_JOURNEYBAG=") && strstr(reason, "want")) {
            bag_log("verify: this build's driver has no verify mode; checking the files");
            return 2;                               /* caller checks the files */
        }
        snprintf(why, wn, "the returned save did not load in the game%s%s",
                 reason[0] ? ": " : "", reason);
    }
    return 0;
}

static int files_ok(const char *work, const char *id)
{
    char p[MAX_PATH];
    FILE *f;
    long size;
    snprintf(p, sizeof p, "%s%cUDATA%c%s%cblinx2data.bin", work, NP_SEP, NP_SEP, id, NP_SEP);
    f = fopen(p, "rb");
    if (!f)
        return 1;                                   /* another title: no rule */
    fseek(f, 0, SEEK_END);
    size = ftell(f);
    fclose(f);
    if (size != 81920)
        return 0;
    snprintf(p, sizeof p, "%s%cUDATA%c%s%cSaveMeta.xbx", work, NP_SEP, NP_SEP, id, NP_SEP);
    f = fopen(p, "rb");
    if (!f)
        return 0;
    fclose(f);
    return 1;
}

/* ---- finish (a joiner, after a clean end) ------------------------------------------ */

/* The work copy's save folders that differ from the real ones as they were
 * at prepare; refuses when a real one changed since, or one appeared. */
static int changed_folders(const char *real, const char *work, const SaveSum *base, int nb,
                           char ids[][64], int max, char *why, size_t wn)
{
    SaveSum w[64], r[64];
    int nw = save_sums(work, w, 64), nr = save_sums(real, r, 64), i, n = 0;
    for (i = 0; i < nw; i++) {
        const SaveSum *b = find_sum(base, nb, w[i].id), *cur = find_sum(r, nr, w[i].id);
        if (!b) {
            snprintf(why, wn, "the work copy has a save the real saves did not (%s)", w[i].id);
            return -1;
        }
        if (b->h == w[i].h)
            continue;
        if (!cur || cur->h != b->h) {
            snprintf(why, wn, "save %s changed since the Journey Bag was made; it was "
                     "left alone", w[i].id);
            return -1;
        }
        if (n < max)
            snprintf(ids[n++], 64, "%s", w[i].id);
    }
    return n;
}

int xbox_NetBagFinish(const char *saves, const char *root, int explicit_dir,
                      const char *session_bag, char *msg, size_t mn)
{
    char work[MAX_PATH], a[MAX_PATH], mu0[MAX_PATH], why[300] = "";
    char ids[8][64];
    SaveSum base[64];
    int nb, slot = 1, r, n, i;
    msg[0] = 0;
    join_path(work, sizeof work, root, "jbag");
    join_path(a, sizeof a, work, "baseline.txt");
    nb = read_baseline(a, base, 64, &slot);
    if (nb < 0) {
        snprintf(msg, mn, "no Journey Bag work copy (%s)", a);
        bag_log("finish refused: %s", msg);
        return 0;
    }
    bag_log("finish: the session's bag %s (hash %016llX)", session_bag,
            (unsigned long long)xbox_NetBagHashTree(session_bag));
    join_path(a, sizeof a, work, "MU");
    join_path(mu0, sizeof mu0, a, "0");
    np_tree_rm(mu0);
    np_tree_copy(session_bag, mu0);
    if (xbox_NetBagHashTree(mu0) != xbox_NetBagHashTree(session_bag)) {
        snprintf(msg, mn, "could not copy the session's Journey Bag");
        bag_log("finish failed: %s", msg);
        return 0;
    }
    r = bag_child(work, "return", "return");
    if (test_fail("return"))
        r = 5;
    if (r != 0) {
        snprintf(why, sizeof why, "the game could not return the Journey Bag");
        bag_reason(work, "return", why, sizeof why);
        snprintf(msg, mn, "%s; your save was not changed", why);
        bag_log("finish failed: %s", msg);
        return 0;
    }
    n = changed_folders(saves, work, base, nb, ids, 8, why, sizeof why);
    if (n < 0) {
        snprintf(msg, mn, "%s; your save was not changed", why);
        bag_log("finish refused: %s", msg);
        return 0;
    }
    if (n != 1) {
        snprintf(msg, mn, "the return changed %d saves (expected 1); your save was not "
                 "changed", n);
        bag_log("finish refused: %s", msg);
        return 0;
    }
    r = bag_verify(work, slot, why, sizeof why);
    if (r == 2)
        for (i = 0, r = 1; i < n; i++)
            if (!files_ok(work, ids[i])) {
                snprintf(why, sizeof why, "the returned save %s is not whole", ids[i]);
                r = 0;
            }
    if (!r) {
        snprintf(msg, mn, "%s; your save was not changed", why);
        bag_log("finish refused: %s", msg);
        return 0;
    }
    if (!commit_allowed(saves, explicit_dir)) {
        snprintf(msg, mn, "test run: the save was not committed");
        return 0;
    }
    if (!commit(root, saves, work, ids, n, msg, mn))
        return 0;
    {
        char sum[300];
        bag_summary(work, sum, sizeof sum);
        if (sum[0])
            bag_log("returned: %s", sum);
        snprintf(msg, mn, "Your Journey Bag came home: save %d now has this game's progress%s%s%s "
                 "(backup in %s%cbackups).", slot, sum[0] ? " (" : "", sum, sum[0] ? ")" : "",
                 root, NP_SEP);
    }
    join_path(a, sizeof a, work, "ready");
    np_unlink(a);
    bag_log("finish: done");
    return 1;
}

/* ---- the host's own progress (on unless RECOMP_NET_KEEP_HOST=0) --------------------- */

int xbox_NetBagHostBaseline(const char *saves, const char *root)
{
    char p[MAX_PATH];
    np_tree_mk(root);
    join_path(p, sizeof p, root, "host_baseline.txt");
    return write_baseline(p, saves, 0);
}

int xbox_NetBagHostCommit(const char *saves, const char *root, int explicit_dir,
                          const char *session, char *msg, size_t mn)
{
    char p[MAX_PATH], work[MAX_PATH], a[MAX_PATH], b[MAX_PATH], why[300] = "";
    char ids[8][64];
    SaveSum base[64];
    int nb, slot = 0, n, r, i;
    msg[0] = 0;
    join_path(p, sizeof p, root, "host_baseline.txt");
    nb = read_baseline(p, base, 64, &slot);
    if (nb < 0) {
        snprintf(msg, mn, "no record of the saves at the start");
        return 0;
    }
    /* RECOMP_NET_TEST_HOSTSESSION=<dir> (testing): take the session's saves
     * from there, e.g. a copy the game really changed. */
    if (getenv("RECOMP_NET_TEST_HOSTSESSION") && *getenv("RECOMP_NET_TEST_HOSTSESSION")) {
        session = getenv("RECOMP_NET_TEST_HOSTSESSION");
        bag_log("test: the host's session saves from %s", session);
    }
    /* A copy of the session's saves to verify and commit from. */
    join_path(work, sizeof work, root, "hostwork");
    np_tree_rm(work);
    join_path(a, sizeof a, session, "UDATA");
    join_path(b, sizeof b, work, "UDATA");
    np_tree_copy(a, b);
    join_path(a, sizeof a, session, "TDATA");
    join_path(b, sizeof b, work, "TDATA");
    np_tree_copy(a, b);
    n = changed_folders(saves, work, base, nb, ids, 8, why, sizeof why);
    if (n < 0) {
        snprintf(msg, mn, "%s; your save was not changed", why);
        bag_log("host commit refused: %s", msg);
        return 0;
    }
    if (!n) {
        snprintf(msg, mn, "no save changed in this game");
        bag_log("host commit: nothing changed");
        return 1;
    }
    {   /* Every home save must load in the game (the driver's slots 1 and 2),
         * not only the changed one: which slot that is, the game decides. */
        SaveSum all[64];
        int ns = save_sums(work, all, 64), k;
        if (ns > 2)
            ns = 2;
        for (k = 1, r = 1; k <= ns && r; k++) {
            r = bag_verify(work, k, why, sizeof why);
            if (r == 2)
                for (i = 0, r = 1; i < n; i++)
                    if (!files_ok(work, ids[i]))
                        r = 0;
        }
    }
    if (!r) {
        snprintf(msg, mn, "%s; your save was not changed", why[0] ? why : "verify failed");
        bag_log("host commit refused: %s", msg);
        return 0;
    }
    if (!commit_allowed(saves, explicit_dir)) {
        snprintf(msg, mn, "test run: the save was not committed");
        return 0;
    }
    if (!commit(root, saves, work, ids, n, msg, mn))
        return 0;
    snprintf(msg, mn, "This game's progress was kept in your save (backup in %s%cbackups).",
             root, NP_SEP);
    bag_log("host commit: done (%d save(s))", n);
    return 1;
}

/* ---- undo ----------------------------------------------------------------------------- */

int xbox_NetBagRestoreLast(const char *saves, const char *root, char *msg, size_t mn)
{
    char dir[MAX_PATH], newest[64] = "", bdir[MAX_PATH], ud[MAX_PATH];
    char undo[MAX_PATH], st[32], tmp[MAX_PATH];
    NpDir d, t, s;
    int n = 0;
    join_path(dir, sizeof dir, root, "backups");
    if (np_dir_open(&d, dir)) {
        while (np_dir_next(&d))
            if (d.is_dir && !strstr(d.name, "-undo") && strcmp(d.name, newest) > 0)
                snprintf(newest, sizeof newest, "%s", d.name);
        np_dir_close(&d);
    }
    if (!newest[0]) {
        snprintf(msg, mn, "There is no backup to restore.");
        return 0;
    }
    join_path(bdir, sizeof bdir, dir, newest);
    stamp(st, sizeof st);
    snprintf(tmp, sizeof tmp, "%s-undo", st);
    join_path(undo, sizeof undo, dir, tmp);
    join_path(ud, sizeof ud, bdir, "UDATA");
    if (np_dir_open(&t, ud)) {
        while (np_dir_next(&t)) {
            char tdir[MAX_PATH];
            if (!t.is_dir)
                continue;
            join_path(tdir, sizeof tdir, ud, t.name);
            if (!np_dir_open(&s, tdir))
                continue;
            while (np_dir_next(&s)) {
                char from[MAX_PATH], to[MAX_PATH], keep[MAX_PATH];
                if (!s.is_dir)
                    continue;
                snprintf(from, sizeof from, "%s%c%s", tdir, NP_SEP, s.name);
                snprintf(to, sizeof to, "%s%cUDATA%c%s%c%s", saves, NP_SEP, NP_SEP,
                         t.name, NP_SEP, s.name);
                snprintf(keep, sizeof keep, "%s%cUDATA%c%s%c%s", undo, NP_SEP, NP_SEP,
                         t.name, NP_SEP, s.name);
                np_tree_copy(to, keep);            /* the current one, in case */
                np_tree_rm(to);
                np_tree_copy(from, to);
                bag_log("restore: %s from %s (%s)", to, bdir,
                        xbox_NetBagHashTree(to) == xbox_NetBagHashTree(from) ? "ok" : "FAILED");
                n++;
            }
            np_dir_close(&s);
        }
        np_dir_close(&t);
    }
    if (n)
        snprintf(msg, mn, "Restored %d save(s) from the backup of %s. The replaced ones "
                 "are in %s.", n, newest, undo);
    else
        snprintf(msg, mn, "The backup %s was empty.", newest);
    return n > 0;
}
