/**
 * Vulkan instance, physical device, logical device and queue.
 *
 * Vulkan 1.3 is required (dynamic rendering and extended dynamic state keep
 * the pipeline count down later). Both GPUs on the development machine (an
 * RTX 4090 and an AMD iGPU) report 1.4.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vk_internal.h"

XboxVk g_vk;

int vk_fail(const char *what, VkResult r)
{
    fprintf(stderr, "  [VK] %s failed: VkResult %d\n", what, (int)r);
    fflush(stderr);
    return 0;
}

int vk_instance_init(void)
{
    VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    VkInstanceCreateInfo ci = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    const char *const *want = vk_surface_extensions();
    const char *ext[8];
    uint32_t n = 0, have_n = 0, i, j, version = 0;
    VkExtensionProperties *have;
    VkResult r_;

    if (g_vk.instance)
        return 1;
    if (volkInitialize() != VK_SUCCESS) {
        fprintf(stderr, "  [VK] no Vulkan loader on this machine\n");
        return 0;
    }
    if (vkEnumerateInstanceVersion)
        vkEnumerateInstanceVersion(&version);
    if (version < VK_API_VERSION_1_3) {
        fprintf(stderr, "  [VK] instance version %u.%u < 1.3\n",
                VK_API_VERSION_MAJOR(version), VK_API_VERSION_MINOR(version));
        return 0;
    }
    vkEnumerateInstanceExtensionProperties(NULL, &have_n, NULL);
    have = (VkExtensionProperties *)calloc(have_n ? have_n : 1, sizeof *have);
    vkEnumerateInstanceExtensionProperties(NULL, &have_n, have);
    for (i = 0; want && want[i]; i++) {
        for (j = 0; j < have_n && strcmp(have[j].extensionName, want[i]); j++)
            ;
        if (j == have_n) {
            fprintf(stderr, "  [VK] instance extension %s missing\n", want[i]);
            free(have);
            return 0;
        }
        ext[n++] = want[i];
    }
    free(have);

    app.pApplicationName = "xboxrecomp";
    app.pEngineName = "xboxrecomp";
    app.apiVersion = VK_API_VERSION_1_3;
    ci.pApplicationInfo = &app;
    ci.enabledExtensionCount = n;
    ci.ppEnabledExtensionNames = ext;
    /* RECOMP_VK_VALIDATION=1: the Khronos validation layer, if the machine
     * has one (it comes with the SDK; this one does not). */
    if (getenv("RECOMP_VK_VALIDATION")) {
        static const char *const layer = "VK_LAYER_KHRONOS_validation";
        ci.enabledLayerCount = 1;
        ci.ppEnabledLayerNames = &layer;
    }
    if (!VK_OK(vkCreateInstance(&ci, NULL, &g_vk.instance))) {
        g_vk.instance = VK_NULL_HANDLE;
        return 0;
    }
    volkLoadInstance(g_vk.instance);
    return 1;
}

/* A queue family that does graphics (and so transfer) and, given a surface,
 * can present to it; -1 if none. */
static int pick_queue(VkPhysicalDevice pd, VkSurfaceKHR surface)
{
    VkQueueFamilyProperties qf[16];
    uint32_t n = 16, i;

    vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, qf);
    for (i = 0; i < n; i++) {
        VkBool32 present = VK_TRUE;
        if (!(qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT))
            continue;
        if (surface)
            vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, surface, &present);
        if (present)
            return (int)i;
    }
    return -1;
}

static int device_init_locked(VkSurfaceKHR surface);

int vk_device_init(VkSurfaceKHR surface)
{
    int ok;
    vk_lock();
    ok = device_init_locked(surface);
    vk_unlock();
    return ok;
}

static int device_init_locked(VkSurfaceKHR surface)
{
    VkPhysicalDevice pds[8];
    uint32_t n = 8, i;
    int best = -1, bestq = -1, score = -1;
    const char *sel = getenv("RECOMP_VK_DEVICE");
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    VkPhysicalDeviceVulkan13Features f13 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    VkPhysicalDeviceVulkan12Features f12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    VkPhysicalDeviceFeatures2 f2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    VkPhysicalDeviceFeatures have;
    VkExtensionProperties dexts[512];
    uint32_t ndext = 512;
    int swapchain = 0;
    VkCommandPoolCreateInfo pci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    static const char *const dext[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
    VkResult r_;

    if (g_vk.ready)
        return 1;
    if (vkEnumeratePhysicalDevices(g_vk.instance, &n, pds) < 0 || !n) {
        fprintf(stderr, "  [VK] no physical devices\n");
        return 0;
    }
    for (i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p;
        int q, s;
        vkGetPhysicalDeviceProperties(pds[i], &p);
        q = pick_queue(pds[i], surface);
        fprintf(stderr, "  [VK] device %u: %s (type %d, API %u.%u)%s\n", i,
                p.deviceName, (int)p.deviceType, VK_API_VERSION_MAJOR(p.apiVersion),
                VK_API_VERSION_MINOR(p.apiVersion), q < 0 ? ", no usable queue" : "");
        if (q < 0 || p.apiVersion < VK_API_VERSION_1_3)
            continue;
        if (sel) {
            char *end;
            unsigned long k = strtoul(sel, &end, 10);
            s = (*end == 0) ? (k == i) * 10 : (strstr(p.deviceName, sel) != NULL) * 10;
        } else {
            s = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 3
              : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2 : 1;
        }
        if (s > score) {
            score = s;
            best = (int)i;
            bestq = q;
        }
    }
    if (best < 0 || (sel && score < 10)) {
        fprintf(stderr, "  [VK] no suitable device%s%s\n",
                sel ? " matching RECOMP_VK_DEVICE=" : "", sel ? sel : "");
        return 0;
    }
    g_vk.phys = pds[best];
    g_vk.qfam = (uint32_t)bestq;
    vkGetPhysicalDeviceProperties(g_vk.phys, &g_vk.props);
    vkGetPhysicalDeviceMemoryProperties(g_vk.phys, &g_vk.mem);

    qci.queueFamilyIndex = g_vk.qfam;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    /* What the backend relies on: dynamic rendering, near-plane clipping by
     * clip distance with depth clamped instead of clipped (the software
     * path clips at the near plane only), exact visibility counts, and
     * query pools reset from the host. All standard on desktop GPUs. */
    vkGetPhysicalDeviceFeatures(g_vk.phys, &have);
    if (!have.depthClamp || !have.shaderClipDistance || !have.occlusionQueryPrecise) {
        fprintf(stderr, "  [VK] %s lacks depthClamp/shaderClipDistance/"
                "occlusionQueryPrecise\n", g_vk.props.deviceName);
        return 0;
    }
    f2.features.depthClamp = VK_TRUE;
    f2.features.shaderClipDistance = VK_TRUE;
    f2.features.occlusionQueryPrecise = VK_TRUE;
    f2.features.samplerAnisotropy = have.samplerAnisotropy;
    f2.pNext = &f12;
    f12.hostQueryReset = VK_TRUE;
    f12.pNext = &f13;
    f13.dynamicRendering = VK_TRUE;
    f13.synchronization2 = VK_TRUE;
    /* The swapchain extension whenever the device has it, so the window
     * can present on the device the renderer made headless-first. */
    if (vkEnumerateDeviceExtensionProperties(g_vk.phys, NULL, &ndext, dexts) >= 0)
        for (i = 0; i < ndext; i++)
            if (!strcmp(dexts[i].extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME))
                swapchain = 1;
    dci.pNext = &f2;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = swapchain ? 1 : 0;
    dci.ppEnabledExtensionNames = dext;
    if (!VK_OK(vkCreateDevice(g_vk.phys, &dci, NULL, &g_vk.device)))
        return 0;
    volkLoadDevice(g_vk.device);
    vkGetDeviceQueue(g_vk.device, g_vk.qfam, 0, &g_vk.queue);
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = g_vk.qfam;
    if (!VK_OK(vkCreateCommandPool(g_vk.device, &pci, NULL, &g_vk.pool)))
        return 0;
    fprintf(stderr, "  [VK] using %s, queue family %u\n", g_vk.props.deviceName,
            g_vk.qfam);
    fflush(stderr);
    g_vk.swapchain = swapchain;
    g_vk.ready = 1;
    return 1;
}

int vk_memory_type(uint32_t bits, VkMemoryPropertyFlags want)
{
    uint32_t i;
    for (i = 0; i < g_vk.mem.memoryTypeCount; i++)
        if ((bits & (1u << i))
            && (g_vk.mem.memoryTypes[i].propertyFlags & want) == want)
            return (int)i;
    return -1;
}

static int vk_alloc(VkMemoryRequirements *req, VkMemoryPropertyFlags props,
                    VkDeviceMemory *mem)
{
    VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    int t = vk_memory_type(req->memoryTypeBits, props);
    VkResult r_;

    if (t < 0) {
        fprintf(stderr, "  [VK] no memory type for flags 0x%X\n", (unsigned)props);
        return 0;
    }
    ai.allocationSize = req->size;
    ai.memoryTypeIndex = (uint32_t)t;
    return VK_OK(vkAllocateMemory(g_vk.device, &ai, NULL, mem));
}

int vk_buffer_create(VkDeviceSize size, VkBufferUsageFlags usage,
                     VkMemoryPropertyFlags props, VkBuffer *buf,
                     VkDeviceMemory *mem, void **mapped)
{
    VkBufferCreateInfo ci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    VkMemoryRequirements req;
    VkResult r_;

    ci.size = size;
    ci.usage = usage;
    if (!VK_OK(vkCreateBuffer(g_vk.device, &ci, NULL, buf)))
        return 0;
    vkGetBufferMemoryRequirements(g_vk.device, *buf, &req);
    if (!vk_alloc(&req, props, mem))
        return 0;
    if (!VK_OK(vkBindBufferMemory(g_vk.device, *buf, *mem, 0)))
        return 0;
    if (mapped && !VK_OK(vkMapMemory(g_vk.device, *mem, 0, VK_WHOLE_SIZE, 0, mapped)))
        return 0;
    return 1;
}

int vk_image_create(uint32_t w, uint32_t h, VkFormat fmt,
                    VkImageUsageFlags usage, VkImage *img, VkDeviceMemory *mem)
{
    return vk_image_create_ex(w, h, fmt, usage, 1, 0, img, mem);
}

int vk_image_create_ex(uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage,
                       uint32_t layers, VkImageCreateFlags flags, VkImage *img,
                       VkDeviceMemory *mem)
{
    VkImageCreateInfo ci = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    VkMemoryRequirements req;
    VkResult r_;

    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = fmt;
    ci.extent.width = w;
    ci.extent.height = h;
    ci.extent.depth = 1;
    ci.flags = flags;
    ci.mipLevels = 1;
    ci.arrayLayers = layers;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = usage;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (!VK_OK(vkCreateImage(g_vk.device, &ci, NULL, img)))
        return 0;
    vkGetImageMemoryRequirements(g_vk.device, *img, &req);
    if (!vk_alloc(&req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, mem))
        return 0;
    return VK_OK(vkBindImageMemory(g_vk.device, *img, *mem, 0));
}
