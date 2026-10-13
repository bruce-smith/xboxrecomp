/**
 * Vulkan GPU backend: public entry points.
 *
 * The software pushbuffer executor (src/kernel/nv2a_pb_exec.c) stays the
 * reference renderer. This library is the GPU path behind it, switched on
 * per run:
 *
 *   RECOMP_VK_PRESENT=1   software rendering, Vulkan presentation: the
 *                         framebuffer window shows its frames through a
 *                         Vulkan swapchain instead of GDI (milestone 1).
 *   RECOMP_VK_DEVICE=<n|name>  pick a physical device by index or by a
 *                         substring of its name (default: the first discrete
 *                         GPU, else the first device).
 *   RECOMP_VK_VSYNC=1     FIFO presentation (default MAILBOX, else
 *                         IMMEDIATE: the title paces itself on its vblank,
 *                         and the window must never hold it back).
 *
 * Platform-neutral: the native window is opaque here (an HWND on Windows),
 * and only vk_surface_*.c knows what it is.
 */
#ifndef XBOXRECOMP_XBOX_VK_H
#define XBOXRECOMP_XBOX_VK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Vulkan presentation on `native_window`. 1 on success; 0 if Vulkan is not
 * usable here (no loader, no device, no surface support), already logged,
 * and the caller keeps its own presentation path. */
int  xbox_vk_present_init(void *native_window);

/* Show a w x h frame of 32-bit pixels, 0x00RRGGBB (as the framebuffer
 * window converts every surface format to), scaled to the window with its
 * aspect ratio kept and black bars around it. `linear`: bilinear scaling,
 * else nearest. Returns 0 when nothing was shown (window minimised, device
 * lost); the caller simply tries again with the next frame. */
int  xbox_vk_present_frame(const uint32_t *px, uint32_t w, uint32_t h, int linear);

void xbox_vk_present_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif
