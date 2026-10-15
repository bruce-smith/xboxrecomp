/**
 * Present frames through a Vulkan swapchain.
 *
 * Milestone 1 of the GPU backend: the software executor still draws every
 * pixel, and the framebuffer window hands each finished frame here instead of
 * to GDI's StretchDIBits. A copy into a device image, a scaled blit into the
 * swapchain image with black bars, present. It proves the device, surface,
 * swapchain and resize plumbing the GPU renderer will present through.
 *
 * Called from the window thread only. The swapchain follows the window's
 * client size; a minimised window (0 x 0) simply shows nothing.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vk_internal.h"
#include "xbox_vk.h"

#define VKP_FRAMES 2        /* frames in flight */
#define VKP_IMAGES 8

static struct {
    int ok;
    VkCommandPool pool;
    VkSurfaceKHR surface;
    VkSwapchainKHR chain;
    VkFormat fmt;
    VkExtent2D extent;
    VkPresentModeKHR mode;
    uint32_t nimg;
    VkImage img[VKP_IMAGES];
    VkImageView view[VKP_IMAGES];       /* for the overlay's rendering */
    VkSemaphore done[VKP_IMAGES];       /* per image: rendering finished */
    int stale;                          /* SUBOPTIMAL: rebuild next frame */
    struct {
        VkCommandBuffer cb;
        VkFence fence;
        VkSemaphore acquired;
        VkBuffer staging;
        VkDeviceMemory smem;
        void *map;
        VkDeviceSize size;
    } f[VKP_FRAMES];
    uint32_t fi;
    /* The frame as a device image, the blit's source. */
    VkImage src;
    VkDeviceMemory srcmem;
    uint32_t sw, sh;
} s;

static void chain_destroy_images(void)
{
    uint32_t i;
    for (i = 0; i < s.nimg; i++) {
        if (s.done[i]) {
            vkDestroySemaphore(g_vk.device, s.done[i], NULL);
            s.done[i] = VK_NULL_HANDLE;
        }
        if (s.view[i]) {
            vkDestroyImageView(g_vk.device, s.view[i], NULL);
            s.view[i] = VK_NULL_HANDLE;
        }
    }
    s.nimg = 0;
}

/* (Re)build the swapchain at the window's current size. 0: nothing to show
 * now (minimised) or failed. */
static int chain_build(void)
{
    VkSurfaceCapabilitiesKHR caps;
    VkSwapchainCreateInfoKHR ci = { VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
    VkSwapchainKHR old = s.chain;
    VkSemaphoreCreateInfo sci = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    uint32_t n, i;
    VkResult r_;

    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g_vk.phys, s.surface, &caps) != VK_SUCCESS)
        return 0;
    if (caps.currentExtent.width == 0 || caps.currentExtent.height == 0)
        return 0;
    vk_lock();
    vkDeviceWaitIdle(g_vk.device);   /* the queue is shared: see vk_lock */
    vk_unlock();
    chain_destroy_images();

    ci.surface = s.surface;
    ci.minImageCount = caps.minImageCount + 1 < 3 ? 3 : caps.minImageCount + 1;
    if (caps.maxImageCount && ci.minImageCount > caps.maxImageCount)
        ci.minImageCount = caps.maxImageCount;
    ci.imageFormat = s.fmt;
    ci.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    ci.imageExtent = caps.currentExtent;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT
                  | (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
                  | (caps.supportedUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)
                      ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
                      : (VkCompositeAlphaFlagBitsKHR)(caps.supportedCompositeAlpha
                                                      & -(int)caps.supportedCompositeAlpha);
    ci.presentMode = s.mode;
    ci.clipped = VK_TRUE;
    ci.oldSwapchain = old;
    if (!VK_OK(vkCreateSwapchainKHR(g_vk.device, &ci, NULL, &s.chain))) {
        s.chain = VK_NULL_HANDLE;
        if (old)
            vkDestroySwapchainKHR(g_vk.device, old, NULL);
        return 0;
    }
    if (old)
        vkDestroySwapchainKHR(g_vk.device, old, NULL);
    s.extent = caps.currentExtent;
    n = VKP_IMAGES;
    vkGetSwapchainImagesKHR(g_vk.device, s.chain, &n, s.img);
    s.nimg = n;
    for (i = 0; i < n; i++) {
        if (!VK_OK(vkCreateSemaphore(g_vk.device, &sci, NULL, &s.done[i])))
            return 0;
        if (caps.supportedUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) {
            VkImageViewCreateInfo vi = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
            vi.image = s.img[i];
            vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vi.format = s.fmt;
            vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            vi.subresourceRange.levelCount = 1;
            vi.subresourceRange.layerCount = 1;
            if (!VK_OK(vkCreateImageView(g_vk.device, &vi, NULL, &s.view[i])))
                s.view[i] = VK_NULL_HANDLE;
        }
    }
    s.stale = 0;
    return 1;
}

int xbox_vk_present_init(void *native_window)
{
    VkSurfaceFormatKHR fmts[64];
    VkPresentModeKHR modes[16];
    uint32_t n, i;
    VkCommandBufferAllocateInfo ai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkSemaphoreCreateInfo sci = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    VkResult r_;

    if (s.ok)
        return 1;
    if (!vk_instance_init() || !g_vk.wsi)
        return 0;
    s.surface = vk_surface_create(g_vk.instance, native_window);
    if (!s.surface || !vk_device_init(s.surface))
        return 0;
    {
        /* The device may have been made for the renderer, before there was
         * a window: check it can present to this one. */
        VkBool32 can = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(g_vk.phys, g_vk.qfam, s.surface, &can);
        if (!can || !g_vk.swapchain) {
            fprintf(stderr, "  [VK] the device cannot present to this window\n");
            return 0;
        }
    }
    /* Command buffers from a pool of this thread's own: pools are not
     * thread-safe, and the renderer records on the executor thread. */
    {
        VkCommandPoolCreateInfo pci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = g_vk.qfam;
        if (!VK_OK(vkCreateCommandPool(g_vk.device, &pci, NULL, &s.pool)))
            return 0;
    }

    /* UNORM, so the guest's bytes go to the screen unchanged: an sRGB
     * swapchain would encode them a second time and wash the image out. */
    n = 64;
    vkGetPhysicalDeviceSurfaceFormatsKHR(g_vk.phys, s.surface, &n, fmts);
    s.fmt = VK_FORMAT_UNDEFINED;
    for (i = 0; i < n && s.fmt == VK_FORMAT_UNDEFINED; i++)
        if (fmts[i].format == VK_FORMAT_B8G8R8A8_UNORM)
            s.fmt = fmts[i].format;
    for (i = 0; i < n && s.fmt == VK_FORMAT_UNDEFINED; i++)
        if (fmts[i].format == VK_FORMAT_R8G8B8A8_UNORM)
            s.fmt = fmts[i].format;
    if (s.fmt == VK_FORMAT_UNDEFINED) {
        fprintf(stderr, "  [VK] surface offers no 8-bit UNORM format\n");
        return 0;
    }

    /* The title paces itself on its own vblank counter; presentation must
     * never hold the window thread for a display refresh. */
    n = 16;
    vkGetPhysicalDeviceSurfacePresentModesKHR(g_vk.phys, s.surface, &n, modes);
    s.mode = VK_PRESENT_MODE_FIFO_KHR;
    if (!getenv("RECOMP_VK_VSYNC")) {
        for (i = 0; i < n; i++)
            if (modes[i] == VK_PRESENT_MODE_IMMEDIATE_KHR)
                s.mode = modes[i];
        for (i = 0; i < n; i++)
            if (modes[i] == VK_PRESENT_MODE_MAILBOX_KHR)
                s.mode = modes[i];
    }

    ai.commandPool = s.pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (i = 0; i < VKP_FRAMES; i++) {
        if (!VK_OK(vkAllocateCommandBuffers(g_vk.device, &ai, &s.f[i].cb))
            || !VK_OK(vkCreateFence(g_vk.device, &fci, NULL, &s.f[i].fence))
            || !VK_OK(vkCreateSemaphore(g_vk.device, &sci, NULL, &s.f[i].acquired)))
            return 0;
    }
    fprintf(stderr, "  [VK] presenting through Vulkan (format %d, present mode %d)\n",
            (int)s.fmt, (int)s.mode);
    fflush(stderr);
    s.ok = 1;
    return 1;
}

static void barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from,
                    VkImageLayout to, VkAccessFlags src_access,
                    VkAccessFlags dst_access, VkPipelineStageFlags src_stage,
                    VkPipelineStageFlags dst_stage)
{
    VkImageMemoryBarrier b = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    b.srcAccessMask = src_access;
    b.dstAccessMask = dst_access;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b.subresourceRange.levelCount = 1;
    b.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(cb, src_stage, dst_stage, 0, 0, NULL, 0, NULL, 1, &b);
}

/* The source image at w x h; recreated when the frame size changes. */
static int src_image(uint32_t w, uint32_t h)
{
    if (s.src && s.sw == w && s.sh == h)
        return 1;
    vk_lock();
    vkDeviceWaitIdle(g_vk.device);   /* the queue is shared: see vk_lock */
    vk_unlock();
    if (s.src) {
        vkDestroyImage(g_vk.device, s.src, NULL);
        vkFreeMemory(g_vk.device, s.srcmem, NULL);
        s.src = VK_NULL_HANDLE;
    }
    if (!vk_image_create(w, h, VK_FORMAT_B8G8R8A8_UNORM,
                         VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                         &s.src, &s.srcmem))
        return 0;
    s.sw = w;
    s.sh = h;
    return 1;
}

/* Frames that could not be shown, by reason: each reason is logged the
 * first three times, and RECOMP_VK_STATS prints the counts every 600 calls.
 * (A window that stays blank with no error logged was the reason.) */
static unsigned s_shown, s_drop[7];
static const char *const k_drop[7] = {
    "not ready", "no surface capabilities", "minimised (zero extent)",
    "swapchain rebuild failed", "swapchain out of date",
    "no source image", "no staging buffer"
};
static int drop(int why)
{
    if (s_drop[why]++ < 3)
        fprintf(stderr, "  [VK] present: frame dropped: %s\n", k_drop[why]);
    return 0;
}

/* The overlay (xbox_vk_overlay_set): drawn over the frame before present. */
static XboxVkOverlayFn s_overlay;
static void *s_overlay_user;
static uint64_t s_frame_no;
/* One presenter at a time: the window thread with CPU frames, or the
 * renderer with its own image at the flip. Once the renderer presents, the
 * window thread's frames are skipped (vk_present_image). */
static void *s_mtx;
static volatile int s_backend_presents;
static volatile int s_linear;

/* RECOMP_VK_PRESENT_DUMP=<n>: the swapchain image of the nth present, as
 * it goes to the screen (bars and overlay included), written to
 * vkpresent_<n>.bmp next to the exe -- what was presented, independent of
 * screen capture (GDI capture does not see Vulkan swapchains here). */
static VkBuffer s_dump_buf;
static VkDeviceMemory s_dump_mem;
static void *s_dump_map;

static int dump_record(VkCommandBuffer cb, VkImage img)
{
    static long at = -2;
    VkBufferImageCopy c;
    if (at == -2) {
        const char *e = getenv("RECOMP_VK_PRESENT_DUMP");
        at = e ? atol(e) : -1;
    }
    if (at < 0 || (long)s_frame_no != at)
        return 0;
    if (!s_dump_buf
        && !vk_buffer_create((VkDeviceSize)s.extent.width * s.extent.height * 4,
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                             | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                             &s_dump_buf, &s_dump_mem, &s_dump_map))
        return 0;
    barrier(cb, img, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);
    memset(&c, 0, sizeof c);
    c.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    c.imageSubresource.layerCount = 1;
    c.imageExtent.width = s.extent.width;
    c.imageExtent.height = s.extent.height;
    c.imageExtent.depth = 1;
    vkCmdCopyImageToBuffer(cb, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s_dump_buf, 1, &c);
    barrier(cb, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_ACCESS_TRANSFER_READ_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    return 1;
}

static void dump_write(uint32_t w, uint32_t h)
{
    char path[64];
    FILE *f;
    uint8_t hdr[54];
    uint32_t img = w * h * 4, total = 54 + img, y, x;
    int32_t sh = -(int32_t)h;
    snprintf(path, sizeof path, "vkpresent_%llu.bmp", (unsigned long long)s_frame_no);
    f = fopen(path, "wb");
    if (!f)
        return;
    memset(hdr, 0, sizeof hdr);
    hdr[0] = 'B'; hdr[1] = 'M';
    memcpy(hdr + 2, &total, 4);
    hdr[10] = 54; hdr[14] = 40;
    memcpy(hdr + 18, &w, 4);
    memcpy(hdr + 22, &sh, 4);
    hdr[26] = 1; hdr[28] = 32;
    memcpy(hdr + 34, &img, 4);
    fwrite(hdr, 1, sizeof hdr, f);
    for (y = 0; y < h; y++) {
        uint8_t *row = (uint8_t *)s_dump_map + (size_t)y * w * 4;
        if (s.fmt == VK_FORMAT_R8G8B8A8_UNORM)        /* BMP wants B G R */
            for (x = 0; x < w; x++) {
                uint8_t t = row[x * 4];
                row[x * 4] = row[x * 4 + 2];
                row[x * 4 + 2] = t;
            }
        fwrite(row, 4, w, f);
    }
    fclose(f);
    fprintf(stderr, "  [VK] present %llu written to %s\n", (unsigned long long)s_frame_no, path);
}

/* Show a frame: `px` (w x h, 0x00RRGGBB, uploaded) or the device image
 * `img` (in GENERAL layout, its top-left w x h), scaled to the window with
 * its aspect kept and black bars, the overlay over it, presented. */
static int present(const uint32_t *px, VkImage img, uint32_t w, uint32_t h, int linear)
{
    VkSurfaceCapabilitiesKHR caps;
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    VkPresentInfoKHR pi = { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkBufferImageCopy copy;
    VkImageBlit blit;
    VkClearColorValue black;
    VkImageSubresourceRange range;
    VkDeviceSize bytes = (VkDeviceSize)w * h * 4;
    uint32_t idx, dw, dh, dx, dy;
    VkResult r, r_;
    VkCommandBuffer cb;
    VkImage src;
    VkImageLayout src_layout;
    int dumping;

    {
        static int stats = -1;
        unsigned calls = s_shown;
        int i;
        if (stats < 0)
            stats = getenv("RECOMP_VK_STATS") != NULL;
        for (i = 0; i < 7; i++)
            calls += s_drop[i];
        if (stats && calls && calls % 600 == 0)
            fprintf(stderr, "[VKPRESENT] %u shown; dropped: %u not ready, %u caps, %u minimised,"
                    " %u rebuild, %u out of date, %u source, %u staging\n", s_shown,
                    s_drop[0], s_drop[1], s_drop[2], s_drop[3], s_drop[4], s_drop[5], s_drop[6]);
    }
    if (!s.ok || (!px && !img) || !w || !h)
        return drop(0);
    vkWaitForFences(g_vk.device, 1, &s.f[s.fi].fence, VK_TRUE, UINT64_MAX);

    /* Follow the window: a size change makes the old chain out of date. */
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g_vk.phys, s.surface, &caps) != VK_SUCCESS)
        return drop(1);
    if (caps.currentExtent.width == 0 || caps.currentExtent.height == 0)
        return drop(2);                             /* minimised */
    if (!s.chain || s.stale || caps.currentExtent.width != s.extent.width
        || caps.currentExtent.height != s.extent.height)
        if (!chain_build())
            return drop(3);

    r = vkAcquireNextImageKHR(g_vk.device, s.chain, UINT64_MAX,
                              s.f[s.fi].acquired, VK_NULL_HANDLE, &idx);
    if (r == VK_ERROR_OUT_OF_DATE_KHR) {
        s.stale = 1;
        return drop(4);
    }
    if (r == VK_SUBOPTIMAL_KHR)
        s.stale = 1;
    else if (r != VK_SUCCESS)
        return vk_fail("vkAcquireNextImageKHR", r);

    if (px) {
        if (!src_image(w, h))
            return drop(5);
        if (s.f[s.fi].size < bytes) {
            if (s.f[s.fi].staging) {
                vkDestroyBuffer(g_vk.device, s.f[s.fi].staging, NULL);
                vkFreeMemory(g_vk.device, s.f[s.fi].smem, NULL);
            }
            if (!vk_buffer_create(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                  | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                  &s.f[s.fi].staging, &s.f[s.fi].smem, &s.f[s.fi].map)) {
                s.f[s.fi].size = 0;
                s.f[s.fi].staging = VK_NULL_HANDLE;
                return drop(6);
            }
            s.f[s.fi].size = bytes;
        }
        memcpy(s.f[s.fi].map, px, (size_t)bytes);
    }

    /* The frame scaled to fit with its aspect kept, centred. */
    if ((uint64_t)s.extent.width * h >= (uint64_t)s.extent.height * w) {
        dh = s.extent.height;
        dw = (uint32_t)((uint64_t)s.extent.height * w / h);
    } else {
        dw = s.extent.width;
        dh = (uint32_t)((uint64_t)s.extent.width * h / w);
    }
    dx = (s.extent.width - dw) / 2;
    dy = (s.extent.height - dh) / 2;

    cb = s.f[s.fi].cb;
    vkResetFences(g_vk.device, 1, &s.f[s.fi].fence);
    vkResetCommandBuffer(cb, 0);
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);

    if (px) {
        barrier(cb, s.src, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        memset(&copy, 0, sizeof copy);
        copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.imageSubresource.layerCount = 1;
        copy.imageExtent.width = w;
        copy.imageExtent.height = h;
        copy.imageExtent.depth = 1;
        vkCmdCopyBufferToImage(cb, s.f[s.fi].staging, s.src,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        barrier(cb, s.src, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT);
        src = s.src;
        src_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    } else {
        /* The renderer's image, written by work submitted before this on
         * the same queue: make those writes visible to the blit. */
        VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        mb.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT
                         | VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
        src = img;
        src_layout = VK_IMAGE_LAYOUT_GENERAL;
    }

    barrier(cb, s.img[idx], VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    memset(&black, 0, sizeof black);
    memset(&range, 0, sizeof range);
    range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    range.levelCount = 1;
    range.layerCount = 1;
    vkCmdClearColorImage(cb, s.img[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         &black, 1, &range);
    barrier(cb, s.img[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);
    memset(&blit, 0, sizeof blit);
    blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.srcSubresource.layerCount = 1;
    blit.srcOffsets[1].x = (int32_t)w;
    blit.srcOffsets[1].y = (int32_t)h;
    blit.srcOffsets[1].z = 1;
    blit.dstSubresource = blit.srcSubresource;
    blit.dstOffsets[0].x = (int32_t)dx;
    blit.dstOffsets[0].y = (int32_t)dy;
    blit.dstOffsets[1].x = (int32_t)(dx + dw);
    blit.dstOffsets[1].y = (int32_t)(dy + dh);
    blit.dstOffsets[1].z = 1;
    vkCmdBlitImage(cb, src, src_layout, s.img[idx],
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                   linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
    if (s_overlay && s.view[idx]) {
        XboxVkOverlay o;
        barrier(cb, s.img[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
        memset(&o, 0, sizeof o);
        o.get_instance_proc_addr = (void *)vkGetInstanceProcAddr;
        o.instance = (void *)g_vk.instance;
        o.physical_device = (void *)g_vk.phys;
        o.device = (void *)g_vk.device;
        o.queue = (void *)g_vk.queue;
        o.queue_family = g_vk.qfam;
        o.command_buffer = (void *)cb;
        o.image = (uint64_t)(uintptr_t)s.img[idx];
        o.image_view = (uint64_t)(uintptr_t)s.view[idx];
        o.format = (uint32_t)s.fmt;
        o.width = s.extent.width;
        o.height = s.extent.height;
        o.image_count = s.nimg;
        o.image_index = idx;
        o.frame = s_frame_no;
        o.frame_x = dx;
        o.frame_y = dy;
        o.frame_w = dw;
        o.frame_h = dh;
        s_overlay(s_overlay_user, &o);
        barrier(cb, s.img[idx], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    } else {
        barrier(cb, s.img[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_WRITE_BIT, 0,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    }
    dumping = dump_record(cb, s.img[idx]);
    vkEndCommandBuffer(cb);

    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &s.f[s.fi].acquired;
    si.pWaitDstStageMask = &wait_stage;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &s.done[idx];
    vk_lock();
    if (!VK_OK(vkQueueSubmit(g_vk.queue, 1, &si, s.f[s.fi].fence))) {
        vk_unlock();
        return 0;
    }

    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &s.done[idx];
    pi.swapchainCount = 1;
    pi.pSwapchains = &s.chain;
    pi.pImageIndices = &idx;
    r = vkQueuePresentKHR(g_vk.queue, &pi);
    vk_unlock();
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR)
        s.stale = 1;
    else if (r != VK_SUCCESS)
        vk_fail("vkQueuePresentKHR", r);
    s.fi = (s.fi + 1) % VKP_FRAMES;
    if (dumping) {
        vkWaitForFences(g_vk.device, 1, &s.f[(s.fi + VKP_FRAMES - 1) % VKP_FRAMES].fence,
                        VK_TRUE, UINT64_MAX);
        dump_write(s.extent.width, s.extent.height);
    }
    s_shown++;
    s_frame_no++;
    return 1;
}

static void present_lock(void)
{
    if (!s_mtx) {
        vk_lock();
        if (!s_mtx)
            s_mtx = vk_mutex_new();
        vk_unlock();
    }
    vk_mutex_lock(s_mtx);
}

int xbox_vk_present_frame(const uint32_t *px, uint32_t w, uint32_t h, int linear)
{
    int r;
    s_linear = linear;                  /* F9 applies to the renderer's frames too */
    if (s_backend_presents)
        return 1;                       /* the renderer shows its own frames */
    present_lock();
    r = present(px, VK_NULL_HANDLE, w, h, linear);
    vk_mutex_unlock(s_mtx);
    return r;
}

int vk_present_ready(void)
{
    return s.ok;
}

int vk_present_image(VkImage img, uint32_t w, uint32_t h)
{
    int r;
    if (!s.ok)
        return 0;
    s_backend_presents = 1;
    present_lock();
    r = present(NULL, img, w, h, s_linear);
    vk_mutex_unlock(s_mtx);
    return r;
}

void xbox_vk_overlay_set(XboxVkOverlayFn fn, void *user)
{
    present_lock();
    s_overlay_user = user;
    s_overlay = fn;
    vk_mutex_unlock(s_mtx);
}

void xbox_vk_present_shutdown(void)
{
    uint32_t i;
    if (!s.ok)
        return;
    vk_lock();
    vkDeviceWaitIdle(g_vk.device);   /* the queue is shared: see vk_lock */
    vk_unlock();
    chain_destroy_images();
    for (i = 0; i < VKP_FRAMES; i++) {
        vkDestroyFence(g_vk.device, s.f[i].fence, NULL);
        vkDestroySemaphore(g_vk.device, s.f[i].acquired, NULL);
        if (s.f[i].staging) {
            vkDestroyBuffer(g_vk.device, s.f[i].staging, NULL);
            vkFreeMemory(g_vk.device, s.f[i].smem, NULL);
        }
    }
    if (s.src) {
        vkDestroyImage(g_vk.device, s.src, NULL);
        vkFreeMemory(g_vk.device, s.srcmem, NULL);
    }
    if (s.chain)
        vkDestroySwapchainKHR(g_vk.device, s.chain, NULL);
    vkDestroyCommandPool(g_vk.device, s.pool, NULL);
    vkDestroySurfaceKHR(g_vk.instance, s.surface, NULL);
    memset(&s, 0, sizeof s);
}
