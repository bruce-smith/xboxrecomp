/**
 * Vulkan backend internals shared by the files of src/gpu_vk.
 *
 * One device for the process. It is created with the presentation surface
 * when there is a window (the queue must be able to present to it), or
 * without one for headless runs.
 */
#ifndef XBOXRECOMP_VK_INTERNAL_H
#define XBOXRECOMP_VK_INTERNAL_H

#include <stddef.h>
#include <stdint.h>
#include "volk.h"

typedef struct {
    int              ready;
    VkInstance       instance;
    VkPhysicalDevice phys;
    VkDevice         device;
    VkQueue          queue;          /* graphics + transfer (+ present) */
    uint32_t         qfam;
    VkPhysicalDeviceProperties       props;
    VkPhysicalDeviceMemoryProperties mem;
    VkCommandPool    pool;           /* window thread's command buffers */
    int              swapchain;      /* VK_KHR_swapchain enabled */
    int              nan_preserve;   /* shaders may keep NaN/inf exactly */
    volatile int     lost;           /* VK_ERROR_DEVICE_LOST seen */
    int              wsi;            /* window-system extensions enabled */
} XboxVk;

extern XboxVk g_vk;

/* Log a failed Vulkan call once per call site and return 0. */
int vk_fail(const char *what, VkResult r);
#define VK_OK(call) ((r_ = (call)) == VK_SUCCESS || vk_fail(#call, r_))

/* Instance, device and queue. `surface` may be VK_NULL_HANDLE (headless);
 * when given, the chosen queue family can present to it. The instance comes
 * first because the surface is made from it: vk_instance_init, then the
 * surface, then vk_device_init. */
int  vk_instance_init(void);
int  vk_device_init(VkSurfaceKHR surface);

/* A memory type with all of `want` among those `bits` allows; -1 if none. */
int  vk_memory_type(uint32_t bits, VkMemoryPropertyFlags want);

/* Buffers and images with their own allocation (few and long-lived so far;
 * a sub-allocator comes with the per-draw resources). */
int  vk_buffer_create(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags props, VkBuffer *buf,
                      VkDeviceMemory *mem, void **mapped);
int  vk_image_create(uint32_t w, uint32_t h, VkFormat fmt,
                     VkImageUsageFlags usage, VkImage *img,
                     VkDeviceMemory *mem);
/* Layers (6 for a cube map) and create flags (CUBE_COMPATIBLE). */
int  vk_image_create_ex(uint32_t w, uint32_t h, VkFormat fmt,
                        VkImageUsageFlags usage, uint32_t layers,
                        VkImageCreateFlags flags, VkImage *img,
                        VkDeviceMemory *mem);

/* GLSL (#version 450) to SPIR-V with glslang, and a module for it, cached
 * by source for the life of the process. VK_NULL_HANDLE: did not compile
 * (logged with the source). */
int  vk_glsl_to_spirv(int fragment, const char *src, uint32_t **words,
                      size_t *nwords);
VkShaderModule vk_shader_module(int fragment, const char *src);

/* The platform layer, vk_platform_<os>.c: */
/* the native window to a VkSurfaceKHR, */
VkSurfaceKHR vk_surface_create(VkInstance inst, void *native_window);
/* the instance extensions that needs (NULL-terminated), */
const char *const *vk_surface_extensions(void);
/* and one process-wide lock: the queue is shared by the executor thread
 * (rendering) and the window thread (presentation), and Vulkan wants
 * vkQueueSubmit / vkQueuePresentKHR externally synchronised. Also held
 * while the device is created, whichever thread gets there first. */
void vk_lock(void);
void vk_unlock(void);
/* A thread, an auto-reset event (wait with a timeout in ms), a mutex and a
 * yield: the backend's completion thread and its state. */
int   vk_thread_start(void (*fn)(void *), void *arg);
void *vk_event_new(void);
void  vk_event_set(void *e);
void  vk_event_wait(void *e, uint32_t ms);
void *vk_mutex_new(void);
void  vk_mutex_lock(void *m);
void  vk_mutex_unlock(void *m);
void  vk_yield(void);
/* A full memory barrier: stores before it are visible to other CPUs before
 * stores after it. */
void  vk_store_fence(void);
/* Vulkan presentation (vk_present.c), for the renderer: is the window's
 * swapchain up, and show the device image `img` (GENERAL layout, its top-left
 * w x h) in it -- from then on the window's own CPU frames are not shown. */
int   vk_present_ready(void);
int   vk_present_image(VkImage img, uint32_t w, uint32_t h);
/* The window's swapchain size (0 before the first frame). */
int   vk_present_extent(uint32_t *w, uint32_t *h);
/* Presentation frame-rate cap in Hz, 0 = none (RECOMP_VK_FPS_CAP). */
void  vk_present_set_cap(uint32_t hz);

/* The on-disk shader and pipeline cache (vk_cache.c). */
void        vk_cache_init(void);
/* The VkPipelineCache data saved by an earlier run on this device and
 * driver, or NULL. */
const void *vk_cache_pipeline_data(size_t *n);
/* SPIR-V for a GLSL source compiled before (malloc'd copy), 0 if none. */
int         vk_cache_spirv_get(int fragment, const char *src, uint32_t **words,
                               size_t *nwords);
void        vk_cache_spirv_put(int fragment, const char *src, const uint32_t *words,
                               size_t nwords);
/* At a flip: write what is new, at most every 10 s and not in the middle of
 * a burst of new pipelines (force: now). */
void        vk_cache_save(VkPipelineCache pc, unsigned new_pipelines, int force);
double      vk_cache_now_ms(void);
/* GLSL compiles in this process since the last call: compiled, answered by
 * the cache, and ms spent in glslang. */
void        vk_shader_stats(unsigned *compiled, unsigned *cached, double *ms);
/* The platform's part: the cache folder (made if missing; 0 if there is
 * none), replacing one file by another, and this process's id. */
int         vk_cache_dir(char *buf, size_t n);
int         vk_file_replace(const char *from, const char *to);
unsigned    vk_process_id(void);

#endif
