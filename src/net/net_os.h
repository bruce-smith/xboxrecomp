/*
 * net_os.h -- the little netplay.c needs from the OS beyond the Win32
 * vocabulary (which kernel.h brings on both hosts: threads, SRW locks,
 * GetTickCount, MessageBoxA, ...): sockets, a directory walk, paths, the
 * environment. Winsock on Windows, BSD sockets and POSIX elsewhere, so a
 * Windows player and a Linux / Steam Deck player can share a game.
 *
 * Paths travel between peers with '/' (np_wire_path / np_native_path):
 * each side stores them with its own separator.
 */
#ifndef XBOX_NET_OS_H
#define XBOX_NET_OS_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)

#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#pragma comment(lib, "ws2_32.lib")

#define NP_SEP        '\\'
#define NP_SENDFLAGS  0

static int np_net_init(void)
{
    WSADATA wd;
    return WSAStartup(MAKEWORD(2, 2), &wd) == 0;
}

static void np_rcvtimeo(SOCKET s, unsigned ms)
{
    DWORD to = ms;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char *)&to, sizeof to);
}

#define np_sockerr()  WSAGetLastError()

static void np_setenv(const char *k, const char *v) { _putenv_s(k, v); }

/* Per-user data: %LOCALAPPDATA%\xboxrecomp (as the Z: cache). */
static void np_data_dir(char *out, size_t n)
{
    char base[MAX_PATH];
    if (!GetEnvironmentVariableA("LOCALAPPDATA", base, MAX_PATH))
        snprintf(base, sizeof base, ".");
    snprintf(out, n, "%s\\xboxrecomp", base);
}

static void np_exe_path(char *out, size_t n)
{
    if (!GetModuleFileNameA(NULL, out, (DWORD)n))
        out[0] = 0;
}

static void np_mkdir(const char *p)  { CreateDirectoryA(p, NULL); }
static void np_rmdir(const char *p)  { RemoveDirectoryA(p); }
static void np_unlink(const char *p)
{
    SetFileAttributesA(p, FILE_ATTRIBUTE_NORMAL);
    DeleteFileA(p);
}

typedef struct {
    HANDLE h;
    WIN32_FIND_DATAA fd;
    int first;
    char name[MAX_PATH];
    int is_dir;
    uint64_t size;
} NpDir;

static int np_dir_open(NpDir *d, const char *dir)
{
    char pat[MAX_PATH];
    snprintf(pat, sizeof pat, "%s\\*", dir);
    d->h = FindFirstFileA(pat, &d->fd);
    d->first = 1;
    return d->h != INVALID_HANDLE_VALUE;
}

/* The next entry other than . and ..; 0 at the end. */
static int np_dir_next(NpDir *d)
{
    for (;;) {
        if (!d->first && !FindNextFileA(d->h, &d->fd))
            return 0;
        d->first = 0;
        if (!strcmp(d->fd.cFileName, ".") || !strcmp(d->fd.cFileName, ".."))
            continue;
        snprintf(d->name, sizeof d->name, "%s", d->fd.cFileName);
        d->is_dir = (d->fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
        d->size = ((uint64_t)d->fd.nFileSizeHigh << 32) | d->fd.nFileSizeLow;
        return 1;
    }
}

static void np_dir_close(NpDir *d) { FindClose(d->h); }

#else   /* POSIX */

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <dirent.h>
#include <errno.h>
#include <unistd.h>

typedef int SOCKET;
#define INVALID_SOCKET  (-1)
#define closesocket     close
#define SD_SEND         SHUT_WR
#define NP_SEP          '/'
#ifdef MSG_NOSIGNAL
#define NP_SENDFLAGS    MSG_NOSIGNAL    /* a joiner that went away is not a SIGPIPE */
#else
#define NP_SENDFLAGS    0
#endif

static int np_net_init(void) { return 1; }

static void np_rcvtimeo(SOCKET s, unsigned ms)
{
    struct timeval tv;
    tv.tv_sec = (time_t)(ms / 1000u);
    tv.tv_usec = (suseconds_t)((ms % 1000u) * 1000u);
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
}

#define np_sockerr()  errno

static void np_setenv(const char *k, const char *v) { setenv(k, v, 1); }

/* Per-user data: $XDG_DATA_HOME/xboxrecomp, else ~/.local/share/xboxrecomp
 * (as kernel_path.c's saves). */
static void np_data_dir(char *out, size_t n)
{
    const char *xdg = getenv("XDG_DATA_HOME"), *home = getenv("HOME");
    if (xdg && *xdg)
        snprintf(out, n, "%s/xboxrecomp", xdg);
    else
        snprintf(out, n, "%s/.local/share/xboxrecomp", home && *home ? home : ".");
}

static void np_exe_path(char *out, size_t n)
{
    ssize_t k = readlink("/proc/self/exe", out, n - 1);
    out[k > 0 ? k : 0] = 0;
}

static void np_mkdir(const char *p)  { mkdir(p, 0755); }
static void np_rmdir(const char *p)  { rmdir(p); }
static void np_unlink(const char *p) { unlink(p); }

typedef struct {
    DIR *h;
    char dir[4096];
    char name[256];
    int is_dir;
    uint64_t size;
} NpDir;

static int np_dir_open(NpDir *d, const char *dir)
{
    snprintf(d->dir, sizeof d->dir, "%s", dir);
    d->h = opendir(dir);
    return d->h != NULL;
}

static int np_dir_next(NpDir *d)
{
    struct dirent *e;
    while ((e = readdir(d->h)) != NULL) {
        char full[4096 + 256];
        struct stat st;
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, ".."))
            continue;
        snprintf(full, sizeof full, "%s/%s", d->dir, e->d_name);
        if (stat(full, &st))
            continue;
        snprintf(d->name, sizeof d->name, "%s", e->d_name);
        d->is_dir = S_ISDIR(st.st_mode);
        d->size = (uint64_t)st.st_size;
        return 1;
    }
    return 0;
}

static void np_dir_close(NpDir *d) { closedir(d->h); }

#endif

/* A relative path as it travels ('/') and as it is stored (NP_SEP). */
static void np_wire_path(char *p)
{
    for (; *p; p++)
        if (*p == '\\')
            *p = '/';
}

static void np_native_path(char *p)
{
    for (; *p; p++)
        if (*p == '/' || *p == '\\')
            *p = NP_SEP;
}

static int np_copy_file(const char *from, const char *to)
{
    char buf[65536];
    size_t n;
    FILE *a = fopen(from, "rb"), *b;
    if (!a)
        return 0;
    b = fopen(to, "wb");
    if (!b) {
        fclose(a);
        return 0;
    }
    while ((n = fread(buf, 1, sizeof buf, a)) > 0)
        fwrite(buf, 1, n, b);
    fclose(a);
    fclose(b);
    return 1;
}

#endif /* XBOX_NET_OS_H */
