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
    if (r == VK_ERROR_DEVICE_LOST)
        g_vk.lost = 1;                   /* the renderer hands back to software */
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
    /* The window-system extensions only matter for presenting through
     * Vulkan: without them the renderer still works (headless, or into the
     * usual window), so they are asked for only when all are there. */
    g_vk.wsi = 1;
    for (i = 0; want && want[i]; i++) {
        for (j = 0; j < have_n && strcmp(have[j].extensionName, want[i]); j++)
            ;
        if (j == have_n) {
            fprintf(stderr, "  [VK] instance extension %s missing: no Vulkan presentation\n",
                    want[i]);
            g_vk.wsi = 0;
        }
    }
    for (i = 0; g_vk.wsi && want && want[i]; i++)
        ext[n++] = want[i];
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

/* Why `pd` cannot run the renderer, or NULL: Vulkan 1.3, a graphics queue,
 * and the features device_init_locked enables. */
static const char *unsuitable(VkPhysicalDevice pd, const VkPhysicalDeviceProperties *p, int q)
{
    VkPhysicalDeviceVulkan13Features f13 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    VkPhysicalDeviceVulkan12Features f12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    VkPhysicalDeviceFeatures2 f2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    if (p->apiVersion < VK_API_VERSION_1_3)
        return "Vulkan 1.3 needed";
    if (q < 0)
        return "no usable queue";
    f2.pNext = &f12;
    f12.pNext = &f13;
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    if (!f2.features.depthClamp || !f2.features.shaderClipDistance
        || !f2.features.occlusionQueryPrecise)
        return "lacks depthClamp/shaderClipDistance/occlusionQueryPrecise";
    if (!f12.hostQueryReset || !f12.timelineSemaphore)
        return "lacks hostQueryReset/timelineSemaphore";
    if (!f13.dynamicRendering || !f13.synchronization2)
        return "lacks dynamicRendering/synchronization2";
    return NULL;
}

static int device_init_locked(VkSurfaceKHR surface);

/* Whether this machine has a device the backend could use (the rules of
 * device_init_locked: suitable, and not a CPU implementation unless
 * RECOMP_VK_DEVICE names it), without creating one. For network play: a
 * PC joining a game that renders with Vulkan says whether it can
 * (src/net/netplay.c). The instance it creates is the one the backend
 * keeps using. */
int xbox_vk_usable(void)
{
    VkPhysicalDevice pds[16];
    uint32_t n = 16, i;
    const char *sel = getenv("RECOMP_VK_DEVICE");
    if (!vk_instance_init())
        return 0;
    if (vkEnumeratePhysicalDevices(g_vk.instance, &n, pds) < 0 || !n)
        return 0;
    for (i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p;
        int named = 0;
        vkGetPhysicalDeviceProperties(pds[i], &p);
        if (unsuitable(pds[i], &p, pick_queue(pds[i], VK_NULL_HANDLE)))
            continue;
        if (sel && *sel) {
            char *end;
            unsigned long k = strtoul(sel, &end, 10);
            named = (*end == 0) ? k == i : strstr(p.deviceName, sel) != NULL;
        }
        if (p.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU || named)
            return 1;
    }
    return 0;
}

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
    /* The choice: RECOMP_VK_DEVICE=<index|part of the name> if it names a
     * suitable device; otherwise a discrete GPU, then an integrated one, then
     * a virtual one. A CPU implementation (lavapipe, SwiftShader) only when
     * named: it would be slower than the software renderer. */
    for (i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p;
        const char *why;
        int q, s, named = 0;
        vkGetPhysicalDeviceProperties(pds[i], &p);
        q = pick_queue(pds[i], surface);
        why = unsuitable(pds[i], &p, q);
        if (sel) {
            char *end;
            unsigned long k = strtoul(sel, &end, 10);
            named = (*end == 0) ? k == i : strstr(p.deviceName, sel) != NULL;
        }
        fprintf(stderr, "  [VK] device %u: %s (type %d, API %u.%u)%s%s\n", i,
                p.deviceName, (int)p.deviceType, VK_API_VERSION_MAJOR(p.apiVersion),
                VK_API_VERSION_MINOR(p.apiVersion), why ? ": " : "", why ? why : "");
        if (why)
            continue;
        s = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 4
          : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 3
          : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU ? 2
          : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU ? 0 : 1;
        if (named)
            s = 10;
        if (s > score) {
            score = s;
            best = (int)i;
            bestq = q;
        }
    }
    if (sel && score < 10)
        fprintf(stderr, "  [VK] no suitable device matches RECOMP_VK_DEVICE=%s;"
                " choosing automatically\n", sel);
    if (best < 0 || score <= 0) {
        fprintf(stderr, "  [VK] no suitable GPU%s\n",
                best >= 0 ? " (only a CPU implementation; name it in RECOMP_VK_DEVICE"
                            " to use it anyway)" : "");
        return 0;
    }
    g_vk.phys = pds[best];
    g_vk.qfam = (uint32_t)bestq;
    vkGetPhysicalDeviceProperties(g_vk.phys, &g_vk.props);
    {
        VkPhysicalDeviceVulkan12Properties p12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES };
        VkPhysicalDeviceProperties2 p2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
        p2.pNext = &p12;
        vkGetPhysicalDeviceProperties2(g_vk.phys, &p2);
        g_vk.nan_preserve = p12.shaderSignedZeroInfNanPreserveFloat32
                         && !getenv("RECOMP_VK_FAST_FLOAT");
    }
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
    f12.timelineSemaphore = VK_TRUE;
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
