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

/* Overlay hook: an in-game menu (Dear ImGui) drawn over each presented
 * frame, before present, whether the frame came from the GPU renderer or
 * from the CPU. The callback runs inside frame submission, on the thread
 * that presents, while `command_buffer` is recording: record into it with
 * dynamic rendering (vkCmdBeginRendering) on `image_view`, which is in
 * COLOR_ATTACHMENT_OPTIMAL layout; leave it in that layout. Handles are
 * the Vulkan ones (VkInstance, ...) as opaque values, so this header needs
 * no Vulkan headers; `get_instance_proc_addr` is vkGetInstanceProcAddr (the
 * library loads Vulkan at run time: give it to ImGui_ImplVulkan_LoadFunctions).
 * frame_x/y/w/h: where the game's frame is in the window (the rest is the
 * black bars). Only with Vulkan presentation (RECOMP_VK_PRESENT=1). */
typedef struct {
    void *get_instance_proc_addr;
    void *instance, *physical_device, *device, *queue;
    uint32_t queue_family;
    void *command_buffer;
    uint64_t image, image_view;
    uint32_t format;                /* VkFormat of the swapchain */
    uint32_t width, height;         /* the window's client area */
    uint32_t image_count, image_index;
    uint64_t frame;
    uint32_t frame_x, frame_y, frame_w, frame_h;
} XboxVkOverlay;

typedef void (*XboxVkOverlayFn)(void *user, const XboxVkOverlay *o);

/* Install (or with NULL remove) the overlay. Thread-safe. */
void xbox_vk_overlay_set(XboxVkOverlayFn fn, void *user);

#ifdef __cplusplus
}
#endif

#endif
