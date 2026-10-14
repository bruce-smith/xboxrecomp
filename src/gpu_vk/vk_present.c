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
    for (i = 0; i < s.nimg; i++)
        if (s.done[i]) {
            vkDestroySemaphore(g_vk.device, s.done[i], NULL);
            s.done[i] = VK_NULL_HANDLE;
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
    ci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
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
    for (i = 0; i < n; i++)
        if (!VK_OK(vkCreateSemaphore(g_vk.device, &sci, NULL, &s.done[i])))
            return 0;
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
    if (!vk_instance_init())
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

int xbox_vk_present_frame(const uint32_t *px, uint32_t w, uint32_t h, int linear)
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
    if (!s.ok || !px || !w || !h)
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
    vkCmdBlitImage(cb, s.src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s.img[idx],
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                   linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
    barrier(cb, s.img[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_WRITE_BIT, 0,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
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
    s_shown++;
    return 1;
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
