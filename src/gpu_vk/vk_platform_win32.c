/**
 * The one platform file of the Vulkan backend (Win32): a window (HWND) to a
 * VkSurfaceKHR, and the queue lock. Other platforms get a sibling (Xlib,
 * Wayland, or an SDL window; pthreads) exporting the same functions.
 */
#include <windows.h>
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
