/**
 * The one platform file of the Vulkan backend (Win32): a window (HWND) to a
 * VkSurfaceKHR, and the queue lock. Other platforms get a sibling (Xlib,
 * Wayland, or an SDL window; pthreads) exporting the same functions.
 */
#include <windows.h>
#include <stdlib.h>
#include "vk_internal.h"

const char *const *vk_surface_extensions(void)
{
    static const char *const ext[] = {
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
        NULL
    };
    return ext;
}

VkSurfaceKHR vk_surface_create(VkInstance inst, void *native_window)
{
    VkWin32SurfaceCreateInfoKHR ci = { VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR };
    VkSurfaceKHR s = VK_NULL_HANDLE;
    VkResult r_;

    ci.hinstance = GetModuleHandleA(NULL);
    ci.hwnd = (HWND)native_window;
    if (!VK_OK(vkCreateWin32SurfaceKHR(inst, &ci, NULL, &s)))
        return VK_NULL_HANDLE;
    return s;
}

static SRWLOCK s_lock = SRWLOCK_INIT;

void vk_lock(void)   { AcquireSRWLockExclusive(&s_lock); }
void vk_unlock(void) { ReleaseSRWLockExclusive(&s_lock); }

/* Threads, events and mutexes for the backend's completion thread. */
typedef struct { void (*fn)(void *); void *arg; } ThreadStart;

static DWORD WINAPI thread_main(LPVOID p)
{
    ThreadStart t = *(ThreadStart *)p;
    free(p);
    t.fn(t.arg);
    return 0;
}

int vk_thread_start(void (*fn)(void *), void *arg)
{
    ThreadStart *t = (ThreadStart *)malloc(sizeof *t);
    HANDLE h;
    if (!t)
        return 0;
    t->fn = fn;
    t->arg = arg;
    h = CreateThread(NULL, 0, thread_main, t, 0, NULL);
    if (!h) {
        free(t);
        return 0;
    }
    CloseHandle(h);
    return 1;
}

void *vk_event_new(void)            { return CreateEventA(NULL, FALSE, FALSE, NULL); }
void  vk_event_set(void *e)         { SetEvent((HANDLE)e); }
void  vk_event_wait(void *e, uint32_t ms) { WaitForSingleObject((HANDLE)e, ms); }

void *vk_mutex_new(void)
{
    SRWLOCK *l = (SRWLOCK *)malloc(sizeof *l);
    if (l)
        InitializeSRWLock(l);
    return l;
}
void vk_mutex_lock(void *m)   { AcquireSRWLockExclusive((SRWLOCK *)m); }
void vk_mutex_unlock(void *m) { ReleaseSRWLockExclusive((SRWLOCK *)m); }
void vk_yield(void)           { Sleep(0); }
