/**
 * The Vulkan backend's platform file for POSIX hosts (Linux, SteamOS):
 * the window-system surface through SDL2 (the framebuffer window,
 * video/fb_present_sdl.c, is an SDL window: X11 or Wayland alike), and
 * pthreads for the queue lock, the completion thread, its events and mutexes.
 *
 * The surface extensions are asked for only when RECOMP_VK_PRESENT is set and
 * there is a window (not RECOMP_HEADLESS): rendering offscreen needs none,
 * and a headless machine may have no display to ask SDL about.
 */
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <time.h>
#include "vk_internal.h"
#include <SDL.h>
#include <SDL_vulkan.h>

const char *const *vk_surface_extensions(void)
{
    static const char *ext[16];
    static int done;
    unsigned n = 0;

    if (done)
        return ext;
    done = 1;
    if (!getenv("RECOMP_VK_PRESENT") || getenv("RECOMP_HEADLESS"))
        return ext;
    SDL_SetHint(SDL_HINT_NO_SIGNAL_HANDLERS, "1");
    if (SDL_InitSubSystem(SDL_INIT_VIDEO) != 0
        || SDL_Vulkan_LoadLibrary(NULL) != 0
        || !SDL_Vulkan_GetInstanceExtensions(NULL, &n, NULL)
        || n == 0 || n >= sizeof ext / sizeof ext[0]
        || !SDL_Vulkan_GetInstanceExtensions(NULL, &n, ext)) {
        fprintf(stderr, "  [VK] no window-system extensions from SDL: %s\n",
                SDL_GetError());
        ext[0] = NULL;
        return ext;
    }
    ext[n] = NULL;
    return ext;
}

/* native_window: the framebuffer window's SDL_Window, made with
 * SDL_WINDOW_VULKAN. */
VkSurfaceKHR vk_surface_create(VkInstance inst, void *native_window)
{
    VkSurfaceKHR s = VK_NULL_HANDLE;
    if (!native_window
        || !SDL_Vulkan_CreateSurface((SDL_Window *)native_window, inst, &s)) {
        fprintf(stderr, "  [VK] SDL_Vulkan_CreateSurface: %s\n", SDL_GetError());
        return VK_NULL_HANDLE;
    }
    return s;
}

static pthread_mutex_t s_lock = PTHREAD_MUTEX_INITIALIZER;

void vk_lock(void)   { pthread_mutex_lock(&s_lock); }
void vk_unlock(void) { pthread_mutex_unlock(&s_lock); }

/* Threads, events and mutexes for the backend's completion thread. */
typedef struct { void (*fn)(void *); void *arg; } ThreadStart;

static void *thread_main(void *p)
{
    ThreadStart t = *(ThreadStart *)p;
    free(p);
    t.fn(t.arg);
    return NULL;
}

int vk_thread_start(void (*fn)(void *), void *arg)
{
    ThreadStart *t = (ThreadStart *)malloc(sizeof *t);
    pthread_t th;
    if (!t)
        return 0;
    t->fn = fn;
    t->arg = arg;
    if (pthread_create(&th, NULL, thread_main, t) != 0) {
        free(t);
        return 0;
    }
    pthread_detach(th);
    return 1;
}

/* An auto-reset event, as CreateEventA(NULL, FALSE, FALSE, NULL). */
typedef struct {
    pthread_mutex_t m;
    pthread_cond_t  c;
    int             set;
} Event;

void *vk_event_new(void)
{
    Event *e = (Event *)calloc(1, sizeof *e);
    pthread_condattr_t a;
    if (!e)
        return NULL;
    pthread_mutex_init(&e->m, NULL);
    pthread_condattr_init(&a);
    pthread_condattr_setclock(&a, CLOCK_MONOTONIC);
    pthread_cond_init(&e->c, &a);
    pthread_condattr_destroy(&a);
    return e;
}

void vk_event_set(void *p)
{
    Event *e = (Event *)p;
    pthread_mutex_lock(&e->m);
    e->set = 1;
    pthread_cond_signal(&e->c);
    pthread_mutex_unlock(&e->m);
}

void vk_event_wait(void *p, uint32_t ms)
{
    Event *e = (Event *)p;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    ts.tv_sec += ms / 1000;
    ts.tv_nsec += (long)(ms % 1000) * 1000000L;
    if (ts.tv_nsec >= 1000000000L) {
        ts.tv_sec++;
        ts.tv_nsec -= 1000000000L;
    }
    pthread_mutex_lock(&e->m);
    while (!e->set)
        if (pthread_cond_timedwait(&e->c, &e->m, &ts) == ETIMEDOUT)
            break;
    e->set = 0;
    pthread_mutex_unlock(&e->m);
}

void *vk_mutex_new(void)
{
    pthread_mutex_t *m = (pthread_mutex_t *)malloc(sizeof *m);
    if (m)
        pthread_mutex_init(m, NULL);
    return m;
}
void vk_mutex_lock(void *m)   { pthread_mutex_lock((pthread_mutex_t *)m); }
void vk_mutex_unlock(void *m) { pthread_mutex_unlock((pthread_mutex_t *)m); }
void vk_yield(void)           { sched_yield(); }
void vk_store_fence(void)     { __atomic_thread_fence(__ATOMIC_SEQ_CST); }
