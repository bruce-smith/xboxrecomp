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

/* 1 if this machine has a Vulkan device the renderer can use (no device is
 * created). Network play asks before joining a game that renders with it. */
int  xbox_vk_usable(void);

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

/* Image-quality settings of the GPU renderer, for an in-game menu; the
 * environment gives the start values (RECOMP_VK_SCALE=1|3|5,
 * RECOMP_VK_AA=fxaa). A change applies at the next flip. Nothing the game
 * can see changes with them: frames read back into its memory stay at its
 * own resolution and visibility tests count native pixels, so exact
 * replays stay exact. Full-resolution output needs Vulkan presentation
 * (RECOMP_VK_PRESENT=1); with the GDI window the supersampled frame is shown
 * filtered down to 640x480. */
typedef struct {
    uint32_t render_scale;      /* 1 (native, default), 3 or 5; even -> next odd */
    uint32_t post_aa;           /* 0 off (default), 1 FXAA */
} XboxVkSettings;

void xbox_vk_settings_get(XboxVkSettings *s);
void xbox_vk_settings_set(const XboxVkSettings *s);

#ifdef __cplusplus
}
#endif

#endif
