/**
 * The Vulkan rendering backend (RECOMP_GPU=vulkan): the pushbuffer
 * executor's draws, clears, visibility reports and flips on the GPU.
 *
 * Geometry arrives already through the vertex program (the executor runs
 * it on the CPU, as the software path does), so this draws Nv2aVshOutput
 * vertices. Pixels go through the title's register combiners, translated to
 * a fragment shader per combiner program (vk_psh.c), with the four texture
 * stages in front of them. Depth, stencil, blend, colour mask, culling,
 * scissor and fog follow the software path's semantics.
 *
 * Model:
 *  - One command buffer is recorded on the executor thread at a time
 *    (a "slot"); it is submitted at the flip, when a visibility result is
 *    asked for, or when a per-slot resource runs out. Two slots alternate.
 *  - Guest memory is read only at record time (vertices into the slot's
 *    upload ring, textures decoded and uploaded), so D3D's fences may pass
 *    as soon as the commands are read -- the executor's rule.
 *  - Guest colour surfaces are GPU images keyed by address; a texture whose
 *    address is a surface's samples that image (render to texture). At the
 *    flip the presented surface is read back into guest memory, so the
 *    framebuffer window, frame dumps and anything else reading the frame
 *    see it there (milestone 2; direct presentation is later).
 *  - Every image stays in VK_IMAGE_LAYOUT_GENERAL; ordering is global
 *    memory barriers at render-pass and upload boundaries.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vk_internal.h"
#include "nv2a_backend.h"
#include "vk_psh.h"
#include "vk_vsh.h"
#include "xbox_vk.h"
#if defined(XBOXRECOMP_HAVE_SETTINGS)
#include "xbox_ui.h"                     /* xbox_setting_* */
#endif

extern ptrdiff_t xbox_GetMemoryOffset(void);
extern int xbox_SchedMode(void);          /* kernel: 0 off, 1 record, 2 replay */

#define VKB_SLOTS     2
#define VKB_RING      (64u << 20)        /* upload bytes per slot */
#define VKB_QUERIES   8192               /* occlusion queries per slot */
#define VKB_SETS      8192               /* descriptor sets per slot */
#define VKB_SETCACHE  1024               /* of them found again by textures */
#define VKB_SURFS     64
#define VKB_TEXS      1024
#define VKB_TEXMAP    4096               /* texture lookup hash slots */
#define VKB_PIPES     4096               /* hash table slots */
#define VKB_FSH       2048               /* fragment shaders by combiner key */
#define VKB_VSH       1024               /* vertex programs */
#define VKB_ZEPOCHS   64
#define VKB_REPORTS   32                 /* visibility answers in flight */
#define VKB_ZDIM      1024               /* depth images: the software zbuf's size */

#define NV_PRIM_TRIANGLES      5
#define NV_PRIM_TRIANGLE_STRIP 6
#define NV_PRIM_TRIANGLE_FAN   7
#define NV_PRIM_QUADS          8
#define NV_PRIM_QUAD_STRIP     9
#define NV_PRIM_POLYGON        10

typedef struct {
    VkCommandBuffer cb;
    VkFence fence;
    int pending;                         /* submitted, fence not yet seen */
    uint64_t seq;                        /* its value on B.timeline */
    VkBuffer ring;
    VkDeviceMemory ringmem;
    uint8_t *ringmap;
    VkDeviceSize ringpos;
    VkDescriptorPool dpool;
    uint32_t nsets;
    VkQueryPool qpool;
    uint32_t nq;
    uint32_t qepoch[VKB_QUERIES];        /* which visibility test each counts for */
} Slot;

typedef struct {
    uint32_t addr, depth;                /* key: address, colour or zeta */
    uint32_t bpp;                        /* colour: 2 or 4 */
    uint32_t w, h;                       /* in guest pixels */
    uint32_t sc;                         /* its image is sc times larger (render scale) */
    VkFormat fmt;
    VkImage img;
    VkDeviceMemory mem;
    VkImageView view;
    int dirty;                           /* written since the last barrier */
    uint32_t used;                       /* frame last used */
    /* What the guest memory under the surface looked like when the GPU
     * last wrote it (surf_sig): if it no longer does, the CPU has written
     * there since, and a texture at this address reads guest memory. */
    uint64_t sig;
    uint32_t sig_frame, gbytes;
} Surf;

typedef struct {
    uint32_t addr, w, h, color, palette, cube, pitch, sgn;
    uint64_t hash;
    uint32_t checked;                    /* frame the hash was last verified */
    VkImage img;
    VkDeviceMemory mem;
    VkImageView view;
    uint32_t used;
} Tex;

/* A host-visible buffer a frame is read back into, and when it is done. */
typedef struct {
    VkBuffer buf;
    VkDeviceMemory mem;
    uint8_t *map;
    VkDeviceSize size;
    uint64_t seq;                        /* submission that fills it */
    uint32_t w, h;                       /* what it holds */
    int pending;                         /* filled, not yet written out */
} Readback;

typedef struct {
    uint64_t key;
    VkShaderModule vs, fs;
    VkPipeline pipe;
} Pipe;

static struct {
    int ok;
    Slot slot[VKB_SLOTS];
    int cur;
    int recording;
    VkCommandPool pool;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;
    VkPipelineCache pcache;
    VkShaderModule vs;
    Pipe pipes[VKB_PIPES];
    int npipes;
    uint32_t new_pipes, new_vsh;          /* created since the last stats line */
    uint32_t frame_pipes;                 /* created since the last flip */
    double   pipe_ms;                     /* in vkCreateGraphicsPipelines */
    uint32_t zdraws, nreports;            /* counted draws, GET_REPORTs */
    struct { VkPshKey key; VkShaderModule fs; int used; } fsh[VKB_FSH];
    /* Vertex programs, by their instruction words. */
    struct { uint64_t h; uint32_t n; uint32_t (*w)[4]; VkShaderModule vs; } vsh[VKB_VSH];
    /* The constants last uploaded, and where, valid while slot_gen holds. */
    float cst[192][4];
    VkDeviceSize cst_off;
    uint32_t cst_gen, slot_gen;
    /* Descriptor sets of the slot being recorded, by the textures they hold
     * (the buffers in them are dynamic, so draws with the same textures share
     * one). Valid while slot_gen and view_gen hold: a destroyed view's handle
     * may come back for another image. */
    struct {
        uint32_t gen, vgen;
        VkImageView v[4];
        VkSampler s[4];
        VkDescriptorSet set;
    } sets[VKB_SETCACHE];
    uint32_t view_gen;
    int rebind;    /* RECOMP_VK_REBIND=1: a new set and all state each draw (A/B) */
    /* The last draw's uniforms and where they are, valid while u_gen ==
     * slot_gen: an identical draw uses that copy. */
    VkDrawUniforms u_last;
    VkDeviceSize u_off;
    uint32_t u_gen;
    /* What the command buffer being recorded has bound and set, so a draw
     * sends only what changed. Valid while gen == slot_gen; anything else
     * binding a pipeline into it (down, fx, area passes) clears gen. */
    struct {
        uint32_t gen;
        VkPipeline pipe;
        VkDescriptorSet set;
        uint32_t dyn[3];
        VkViewport view;
        VkRect2D sc;
        float bc[4];
        uint32_t cull, front, dte, dwe, dop, ste, sop[4], fmask, wmask, ref;
    } bound;
    VkSampler samplers[2][8][8];         /* [linear][addr_u][addr_v] */
    Surf surf[VKB_SURFS];
    Tex tex[VKB_TEXS];
    /* Index into tex[] (+1, 0 empty) by a hash of the texture's key: most
     * lookups hit here instead of scanning every slot. A stale entry fails
     * the key comparison and falls back to the scan. */
    uint16_t texmap[VKB_TEXMAP];
    Tex white, white_cube;               /* stage with nothing bound */
    VkFormat zfmt;
    /* Rendering state. */
    int rendering;
    Surf *rt, *zt;
    int pending_barrier;                 /* something written must be visible */
    uint32_t frame;
    /* Visibility tests. Queries are summed per test (epoch) as their
     * submissions complete; a GET_REPORT is answered by the completion thread
     * once the GPU has passed the submission that ends it. `mtx` guards
     * slot completion, the sums and the queue. */
    uint32_t epoch;
    uint64_t epoch_count[VKB_ZEPOCHS];
    VkSemaphore timeline;                /* signalled with each submit's seq */
    uint64_t seq;
    void *mtx, *rep_event;
    struct { uint32_t at, epoch, force; uint64_t stamp, seq; } rep[VKB_REPORTS];
    int rep_head, nrep;
    /* Read-back of the presented frame: rb for the synchronous kinds
     * (compare mode, frame dumps, RECOMP_VK_WRITEBACK=1), arb[] the two
     * a flip alternates between when it does not wait. */
    Readback rb, arb[2];
    int arbi;
    int compare_mode;                    /* RECOMP_VK_COMPARE */
    VkCommandBuffer sink;                /* after a loss: recorded, never submitted */
    int sink_begun;
    /* Supersampling: every surface image is `scale` times the guest's size
     * (1-4); a requested change applies at the next flip. */
    uint32_t scale, scale_req;
    VkImage down;                        /* native-size copy for read-backs */
    VkDeviceMemory downmem;
    uint32_t down_w, down_h;
    VkFormat down_fmt;
    /* Stats. */
    uint32_t draws, tris, uploads, submits;
} B;

/* ---------------------------------------------------------------------- */
/* Shaders (milestone 2: generic). */

static const char *k_vs =
"#version 450\n"
"layout(location=0) in vec4 ipos;\n"
"layout(location=1) in vec4 id0;\n"
"layout(location=2) in vec4 id1;\n"
"layout(location=3) in vec4 ifog;\n"
"layout(location=4) in vec4 it0;\n"
"layout(location=5) in vec4 it1;\n"
"layout(location=6) in vec4 it2;\n"
"layout(location=7) in vec4 it3;\n"
"layout(std140, set=0, binding=0) uniform U {\n"
"    vec4 vp, k, o, fogp, fogc;\n"
"    vec4 c0[8], c1[8];\n"
"    vec4 fc0, fc1;\n"
"    vec4 tsz[4], bm[4], bl[4];\n"
"    ivec4 at;\n"
"} u;\n"
"layout(location=0) CQ out vec4 d0;\n"
"layout(location=1) CQ out vec4 d1;\n"
"layout(location=2) CQ out float fogf;\n"
"layout(location=3) out vec4 tc0;\n"
"layout(location=4) out vec4 tc1;\n"
"layout(location=5) out vec4 tc2;\n"
"layout(location=6) out vec4 tc3;\n"
"out gl_PerVertex { vec4 gl_Position; float gl_ClipDistance[2]; };\n"
"/* The fog unit per vertex (geo_fog), from oFog.x. */\n"
"float fog_factor(float d) {\n"
"    float f, px = u.fogp.z, py = u.fogp.w;\n"
"    int mode = int(u.fogp.y);\n"
"    if (u.fogp.x == 0.0) return 1.0;\n"
"    if (mode == 0x800 || mode == 0x802) f = px + exp2(d * py * 16.0) - 1.5;\n"
"    else if (mode == 0x801 || mode == 0x803) f = px + exp2(-d * d * py * py * 32.0) - 1.5;\n"
"    else f = px + d * py - 1.0;\n"
"    if (mode == 0x802 || mode == 0x803 || mode == 0x804) f = abs(f);\n"
"    if (isnan(f)) f = 1.0;\n"
"    return clamp(f, 0.0, 1.0);\n"
"}\n"
"void main() {\n"
"    /* oPos is in screen space (the D3D epilogue divided by w and applied\n"
"     * the viewport) with w kept: back to clip space for Vulkan, so it\n"
"     * clips and interpolates perspective-correctly. */\n"
"    float w = ipos.w;\n"
"    gl_Position = vec4((ipos.x * u.vp.x - 1.0) * w, (ipos.y * u.vp.y - 1.0) * w,\n"
"                       ipos.z * u.vp.z * w, w);\n"
"    /* The near plane is D3D's z = 0 in clip space, and a w guard, as the\n"
"     * software path clips (geo_clipped); depth itself is clamped. */\n"
"    float zc = u.k.z != 0.0 ? (ipos.z - u.o.z) / u.k.z * w : w;\n"
"    gl_ClipDistance[0] = zc;\n"
"    gl_ClipDistance[1] = w - 1e-3;\n"
"    d0 = id0;\n"
"    d1 = id1;\n"
"    fogf = fog_factor(ifog.x);\n"
"    tc0 = it0; tc1 = it1; tc2 = it2; tc3 = it3;\n"
"}\n";

/* ---------------------------------------------------------------------- */
/* Slots. */

/* RECOMP_VK_SIMULATE_LOST: submissions are swallowed from now on. */
static int s_gpu_silent;

/* Waits never block for ever: a lost device (driver reset) may never
 * signal anything, and a thread stuck here would leave the title polling a
 * visibility report that is never answered. They wait in 100 ms steps and
 * give up once the device is lost, or after 2 s, which then counts as lost.
 * 1 when the wait completed. */
static void mark_lost(const char *why)
{
    if (!g_vk.lost)
        fprintf(stderr, "  [VK] device lost (%s): the GPU renderer stops\n", why);
    g_vk.lost = 1;
}

/* How long a wait may take before the device counts as lost, in 100 ms
 * steps: RECOMP_VK_LOST_MS, else 2 s on a GPU and 60 s on a CPU
 * implementation (llvmpipe, SwiftShader), which can take seconds for one
 * heavy frame without anything being wrong. */
static int wait_steps(void)
{
    static int steps;
    if (!steps) {
        const char *e = getenv("RECOMP_VK_LOST_MS");
        int ms = e ? atoi(e)
               : g_vk.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU ? 60000 : 2000;
        steps = ms < 100 ? 1 : ms / 100;
    }
    return steps;
}

static int fence_wait(VkFence f)
{
    int i;
    for (i = 0; i < wait_steps() && !g_vk.lost; i++) {
        VkResult r = vkWaitForFences(g_vk.device, 1, &f, VK_TRUE, 100000000ull);
        if (r == VK_SUCCESS)
            return 1;
        if (r != VK_TIMEOUT) {
            mark_lost("a fence wait failed");
            return 0;
        }
    }
    if (!g_vk.lost)
        mark_lost("a fence did not signal in time (RECOMP_VK_LOST_MS)");
    return 0;
}

static int timeline_wait(uint64_t value)
{
    VkSemaphoreWaitInfo wi = { VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
    int i;
    wi.semaphoreCount = 1;
    wi.pSemaphores = &B.timeline;
    wi.pValues = &value;
    for (i = 0; i < wait_steps() && !g_vk.lost; i++) {
        VkResult r = vkWaitSemaphores(g_vk.device, &wi, 100000000ull);
        if (r == VK_SUCCESS)
            return 1;
        if (r != VK_TIMEOUT) {
            mark_lost("a timeline wait failed");
            return 0;
        }
    }
    if (!g_vk.lost)
        mark_lost("the GPU did not finish in time (RECOMP_VK_LOST_MS)");
    return 0;
}

static void slot_complete(Slot *s)
{
    uint32_t i;
    vk_mutex_lock(B.mtx);
    if (!s->pending) {
        vk_mutex_unlock(B.mtx);
        return;
    }
    if (!fence_wait(s->fence)) {
        s->pending = 0;                  /* lost: its queries count nothing */
        vk_mutex_unlock(B.mtx);
        return;
    }
    s->pending = 0;
    if (s->nq) {
        uint64_t *res = (uint64_t *)malloc(sizeof(uint64_t) * s->nq);
        if (res && vkGetQueryPoolResults(g_vk.device, s->qpool, 0, s->nq,
                                         sizeof(uint64_t) * s->nq, res,
                                         sizeof(uint64_t),
                                         VK_QUERY_RESULT_64_BIT
                                         | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS)
            for (i = 0; i < s->nq; i++)
                B.epoch_count[s->qepoch[i] % VKB_ZEPOCHS] += res[i];
        free(res);
    }
    vk_mutex_unlock(B.mtx);
}

/* After a loss nothing the device may still be executing is touched again
 * (a "lost" device may only be late: llvmpipe went on running the
 * submission and crashed on a reset descriptor pool). Commands recorded
 * from then on go to this command buffer, which is never submitted. */
static VkCommandBuffer sink_cb(void)
{
    if (!B.sink_begun) {
        VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(B.sink, &bi);
        B.sink_begun = 1;
    }
    return B.sink;
}

static void slot_begin(void)
{
    Slot *s = &B.slot[B.cur];
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };

    slot_complete(s);
    if (g_vk.lost)
        return;                           /* the slot may still be in use */
    vkResetFences(g_vk.device, 1, &s->fence);
    vkResetCommandBuffer(s->cb, 0);
    vkResetDescriptorPool(g_vk.device, s->dpool, 0);
    vkResetQueryPool(g_vk.device, s->qpool, 0, VKB_QUERIES);
    s->ringpos = 0;
    s->nsets = 0;
    s->nq = 0;
    B.slot_gen++;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(s->cb, &bi);
    B.recording = 1;
}

static VkCommandBuffer cmd(void)
{
    if (!B.recording && !g_vk.lost)
        slot_begin();
    if (g_vk.lost && !B.recording)
        return sink_cb();
    return B.slot[B.cur].cb;
}

static void end_rendering(void)
{
    if (!B.rendering)
        return;
    vkCmdEndRendering(B.slot[B.cur].cb);
    B.rendering = 0;
}

/* Everything written so far visible to everything after: attachments to
 * sampling, transfers and later attachments. Global, coarse and correct. */
static void full_barrier(void)
{
    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    int i;
    end_rendering();
    mb.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                     | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
                     | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT
                     | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT
                     | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                     | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                     | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
                     | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd(),
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                         | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT
                         | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT
                         | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                         | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                         | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT
                         | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT
                         | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);
    for (i = 0; i < VKB_SURFS; i++)
        B.surf[i].dirty = 0;
    B.pending_barrier = 0;
}

/* Submit what is recorded; `wait`: and wait for it (results, read-back). */
static void submit(int wait)
{
    Slot *s = &B.slot[B.cur];
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    VkResult r_;

    if (!B.recording) {
        if (wait)
            slot_complete(&B.slot[(B.cur + VKB_SLOTS - 1) % VKB_SLOTS]);
        return;
    }
    end_rendering();
    vkEndCommandBuffer(s->cb);
    si.commandBufferCount = 1;
    si.pCommandBuffers = &s->cb;
    {
        VkTimelineSemaphoreSubmitInfo ti = { VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO };
        uint64_t v = B.seq + 1;
        ti.signalSemaphoreValueCount = 1;
        ti.pSignalSemaphoreValues = &v;
        si.pNext = &ti;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &B.timeline;
        {
            /* RECOMP_VK_FAIL_AT=<n>: the nth submission is never sent and
             * the device counts as lost -- the timeline then never reaches
             * it, as after a real loss (testing the fall-back). */
            static long fail_at = -2;
            int ok = 0;
            if (fail_at == -2) {
                const char *e = getenv("RECOMP_VK_FAIL_AT");
                fail_at = e ? atol(e) : -1;
            }
            if (fail_at >= 0 && (long)B.submits + 1 >= fail_at) {
                mark_lost("RECOMP_VK_FAIL_AT");
            } else if (s_gpu_silent && !g_vk.lost) {
                ok = 1;                  /* "sent", never signals (SIMULATE_LOST) */
            } else if (!g_vk.lost) {
                vk_lock();
                ok = VK_OK(vkQueueSubmit(g_vk.queue, 1, &si, s->fence));
                vk_unlock();
                if (!ok)
                    mark_lost("a submission failed");
            }
            s->seq = ++B.seq;
            if (!ok) {
                B.recording = 0;
                B.cur = (B.cur + 1) % VKB_SLOTS;
                return;
            }
        }
    }
    s->pending = 1;
    B.recording = 0;
    B.submits++;
    /* The next command buffer starts with nothing bound and nothing
     * rendering; a barrier at its start orders it after this one. */
    B.rt = B.zt = NULL;
    B.pending_barrier = 1;
    if (wait)
        slot_complete(s);
    B.cur = (B.cur + 1) % VKB_SLOTS;
}

/* Nothing in flight: before destroying anything a slot may reference. */
static void wait_idle(void)
{
    int i;
    submit(1);
    for (i = 0; i < VKB_SLOTS; i++)
        slot_complete(&B.slot[i]);
}

/* Upload space in the current slot; submits and moves on when full. */
static VkDeviceSize ring_alloc(VkDeviceSize n, VkDeviceSize align, void **p)
{
    Slot *s;
    VkDeviceSize at;
    if (n > VKB_RING)
        return (VkDeviceSize)-1;
    cmd();
    s = &B.slot[B.cur];
    at = (s->ringpos + align - 1) & ~(align - 1);
    if (at + n > VKB_RING) {
        submit(0);
        cmd();
        s = &B.slot[B.cur];
        at = 0;
    }
    s->ringpos = at + n;
    *p = s->ringmap + at;
    return at;
}

/* ---------------------------------------------------------------------- */
/* Surfaces. */

static VkFormat color_format(uint32_t surface_format, uint32_t bpp)
{
    if (bpp == 2)
        return (surface_format & 0xF) == 0x3 ? VK_FORMAT_R5G6B5_UNORM_PACK16
                                             : VK_FORMAT_A1R5G5B5_UNORM_PACK16;
    return VK_FORMAT_B8G8R8A8_UNORM;
}

static void surf_destroy(Surf *s)
{
    if (g_vk.lost) {                      /* may still be in use: leaked */
        memset(s, 0, sizeof *s);
        return;
    }
    if (s->view) vkDestroyImageView(g_vk.device, s->view, NULL);
    B.view_gen++;                        /* cached descriptor sets name it */
    if (s->img) vkDestroyImage(g_vk.device, s->img, NULL);
    if (s->mem) vkFreeMemory(g_vk.device, s->mem, NULL);
    memset(s, 0, sizeof *s);
}

static int surf_create(Surf *s, uint32_t w, uint32_t h)
{
    VkImageViewCreateInfo vi = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    VkImageUsageFlags u;
    VkResult r_;
    VkImageMemoryBarrier b = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };

    u = s->depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                 : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    u |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    s->sc = B.scale ? B.scale : 1;
    if (!vk_image_create(w * s->sc, h * s->sc, s->fmt, u, &s->img, &s->mem))
        return 0;
    s->w = w;
    s->h = h;
    vi.image = s->img;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = s->fmt;
    vi.subresourceRange.aspectMask = s->depth
        ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT
        : VK_IMAGE_ASPECT_COLOR_BIT;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.layerCount = 1;
    if (!VK_OK(vkCreateImageView(g_vk.device, &vi, NULL, &s->view)))
        return 0;
    /* To GENERAL, where every image here lives, and to known contents. */
    end_rendering();
    b.srcAccessMask = 0;
    b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = s->img;
    b.subresourceRange = vi.subresourceRange;
    vkCmdPipelineBarrier(cmd(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &b);
    if (s->depth) {
        VkClearDepthStencilValue z = { 1.0f, 0 };
        VkImageSubresourceRange r = vi.subresourceRange;
        vkCmdClearDepthStencilImage(cmd(), s->img, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &r);
    } else {
        VkClearColorValue c;
        VkImageSubresourceRange r = vi.subresourceRange;
        memset(&c, 0, sizeof c);
        vkCmdClearColorImage(cmd(), s->img, VK_IMAGE_LAYOUT_GENERAL, &c, 1, &r);
    }
    B.pending_barrier = 1;
    return 1;
}

/* The image for a guest surface, at least w x h. A surface seen bigger
 * than before (a wider clip later in the frame) is regrown with its
 * contents copied. */
static Surf *surf_get(uint32_t addr, int depth, uint32_t bpp, VkFormat fmt,
                      uint32_t w, uint32_t h)
{
    Surf *s, *free_s = NULL, *lru = NULL;
    int i;

    for (i = 0; i < VKB_SURFS; i++) {
        s = &B.surf[i];
        if (!s->img) {
            if (!free_s)
                free_s = s;
            continue;
        }
        if (s->addr == addr && s->depth == (uint32_t)depth
            && (depth || (s->bpp == bpp && s->fmt == fmt))) {
            if (s->w < w || s->h < h) {
                Surf old = *s, n;
                VkImageCopy c;
                memset(&n, 0, sizeof n);
                n.addr = addr; n.depth = (uint32_t)depth; n.bpp = bpp; n.fmt = fmt;
                if (!surf_create(&n, w > old.w ? w : old.w, h > old.h ? h : old.h))
                    return s;
                full_barrier();
                memset(&c, 0, sizeof c);
                c.srcSubresource.aspectMask = depth
                    ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT
                    : VK_IMAGE_ASPECT_COLOR_BIT;
                c.srcSubresource.layerCount = 1;
                c.dstSubresource = c.srcSubresource;
                c.extent.width = old.w * old.sc;
                c.extent.height = old.h * old.sc;
                c.extent.depth = 1;
                vkCmdCopyImage(cmd(), old.img, VK_IMAGE_LAYOUT_GENERAL, n.img,
                               VK_IMAGE_LAYOUT_GENERAL, 1, &c);
                B.pending_barrier = 1;
                wait_idle();                /* the old image is still referenced */
                surf_destroy(s);
                *s = n;
            }
            s->used = B.frame;
            return s;
        }
        if (!lru || s->used < lru->used)
            lru = s;
    }
    s = free_s;
    if (!s) {
        wait_idle();
        surf_destroy(lru);
        s = lru;
    }
    s->addr = addr;
    s->depth = (uint32_t)depth;
    s->bpp = bpp;
    s->fmt = fmt;
    if (!surf_create(s, w, h)) {
        surf_destroy(s);
        return NULL;
    }
    s->used = B.frame;
    return s;
}

/* The colour surface a draw or clear targets. */
/* 64 dwords sampled across the guest bytes under a surface. */
static uint64_t surf_sig(const Surf *s)
{
    const uint8_t *mem = (const uint8_t *)xbox_GetMemoryOffset();
    uint64_t h = 1469598103934665603ull;
    uint32_t i, v, n = s->gbytes & ~3u;
    if (n < 4)
        return 0;
    for (i = 0; i < 64; i++) {
        memcpy(&v, mem + s->addr + ((uint64_t)n * i / 64 & ~3ull), 4);
        h = (h ^ v) * 1099511628211ull;
    }
    return h;
}

static Surf *target_color(const Nv2aDrawState *st)
{
    Surf *s;
    uint32_t w, h;
    if (!st->color_addr || (st->bpp != 2 && st->bpp != 4))
        return NULL;
    if (st->swz_w) {
        w = st->swz_w;
        h = st->swz_h;
    } else {
        w = st->pitch / st->bpp;
        h = st->clip_y + st->clip_h;
        if (w < st->clip_x + st->clip_w)
            w = st->clip_x + st->clip_w;
    }
    if (!w || !h || w > 4096 || h > 4096)
        return NULL;
    s = surf_get(st->color_addr, 0, st->bpp, color_format(st->surface_format, st->bpp),
                 w, h);
    /* The GPU writes it now: what guest memory holds under it is from
     * before (taken once a frame). */
    if (s && s->sig_frame != B.frame + 1) {
        s->gbytes = st->swz_w ? w * h * st->bpp : st->pitch * h;
        s->sig = surf_sig(s);
        s->sig_frame = B.frame + 1;
    }
    return s;
}

static Surf *target_zeta(const Nv2aDrawState *st)
{
    return surf_get(st->zeta_addr, 1, 0, B.zfmt, VKB_ZDIM, VKB_ZDIM);
}

static void begin_rendering(Surf *rt, Surf *zt)
{
    VkRenderingInfo ri = { VK_STRUCTURE_TYPE_RENDERING_INFO };
    VkRenderingAttachmentInfo ca = { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    VkRenderingAttachmentInfo da = { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };

    if (B.rendering && B.rt == rt && B.zt == zt && !B.pending_barrier)
        return;
    end_rendering();
    if (B.pending_barrier || (rt && rt->dirty) || (zt && zt->dirty) || B.rt != rt)
        full_barrier();
    ca.imageView = rt->view;
    ca.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    ca.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    da.imageView = zt ? zt->view : VK_NULL_HANDLE;
    da.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    da.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    da.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    ri.renderArea.extent.width = rt->w * rt->sc;
    ri.renderArea.extent.height = rt->h * rt->sc;
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &ca;
    ri.pDepthAttachment = zt ? &da : NULL;
    ri.pStencilAttachment = zt ? &da : NULL;
    vkCmdBeginRendering(cmd(), &ri);
    B.rendering = 1;
    B.rt = rt;
    B.zt = zt;
}

/* ---------------------------------------------------------------------- */
/* Textures. */

static uint64_t hash_bytes(const uint8_t *p, size_t n, uint64_t h)
{
    size_t i;
    const uint64_t *q = (const uint64_t *)p;
    for (i = 0; i + 8 <= n; i += 8)
        h = (h ^ q[i / 8]) * 1099511628211ull;
    for (; i < n; i++)
        h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

/* Bytes of guest memory a texture reads (all faces, all levels). */
static size_t tex_bytes(const Nv2aTexState *t)
{
    if (!t->normalized)
        return (size_t)t->pitch * t->height;
    return (size_t)t->face_stride * (t->cube ? 6u : 1u);
}

static uint64_t tex_hash(const Nv2aTexState *t)
{
    const uint8_t *mem = (const uint8_t *)xbox_GetMemoryOffset();
    uint64_t h = 1469598103934665603ull;
    h = hash_bytes(mem + t->offset, tex_bytes(t), h);
    if (t->palette && (t->color == 0x0B))
        h = hash_bytes(mem + t->palette, 1024, h);
    return h;
}

static void tex_destroy(Tex *x)
{
    if (g_vk.lost) {
        memset(x, 0, sizeof *x);
        return;
    }
    if (x->view) vkDestroyImageView(g_vk.device, x->view, NULL);
    B.view_gen++;
    if (x->img) vkDestroyImage(g_vk.device, x->img, NULL);
    if (x->mem) vkFreeMemory(g_vk.device, x->mem, NULL);
    memset(x, 0, sizeof *x);
}

/* Decode level 0 (face 0) with the executor's own texel reader, upload. */
/* IEEE half from float (round to nearest even; no NaN payloads needed). */
static uint16_t f2h(float f)
{
    uint32_t x, sign, e, m;
    memcpy(&x, &f, 4);
    sign = (x >> 16) & 0x8000u;
    e = (x >> 23) & 0xFF;
    m = x & 0x7FFFFFu;
    if (e == 0xFF)
        return (uint16_t)(sign | 0x7C00u | (m ? 0x200u : 0));
    if (e > 142)
        return (uint16_t)(sign | 0x7C00u);
    if (e < 113) {                              /* subnormal or zero */
        if (e < 103)
            return (uint16_t)sign;
        m |= 0x800000u;
        {
            uint32_t sh = 126 - e, h = m >> sh, rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
            if (rem > half || (rem == half && (h & 1)))
                h++;
            return (uint16_t)(sign | h);
        }
    }
    {
        uint32_t h = ((e - 112) << 10) | (m >> 13), rem = m & 0x1FFFu;
        if (rem > 0x1000u || (rem == 0x1000u && (h & 1)))
            h++;
        return (uint16_t)(sign | h);
    }
}

/* Decode every face of level 0 with the executor's own texel reader
 * (exactly the software path's texels), upload. A texture with signed
 * channels (the filter's bits 28-31, bump maps) goes up as half floats
 * already in -1..1, so the GPU's filtering blends signed values as the
 * software path does (texel_unpack converts before filtering). */
static int tex_upload(Tex *x, const Nv2aTexState *t)
{
    VkImageViewCreateInfo vi = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    VkImageMemoryBarrier b = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    VkBufferImageCopy c;
    uint32_t u, v, face, faces = x->cube ? 6u : 1u, bpt = x->sgn ? 8u : 4u;
    uint8_t *px;
    VkDeviceSize at, face_bytes;
    VkFormat fmt = x->sgn ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_B8G8R8A8_UNORM;
    VkResult r_;
    int fresh = !x->img;

    if (!t->width || !t->height || t->width > 4096 || t->height > 4096)
        return 0;
    if (fresh) {
        if (!vk_image_create_ex(t->width, t->height, fmt,
                                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                faces, x->cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0,
                                &x->img, &x->mem))
            return 0;
        vi.image = x->img;
        vi.viewType = x->cube ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D;
        vi.format = fmt;
        vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vi.subresourceRange.levelCount = 1;
        vi.subresourceRange.layerCount = faces;
        if (!VK_OK(vkCreateImageView(g_vk.device, &vi, NULL, &x->view)))
            return 0;
    }
    face_bytes = (VkDeviceSize)t->width * t->height * bpt;
    at = ring_alloc(face_bytes * faces, 16, (void **)&px);
    if (at == (VkDeviceSize)-1)
        return 0;
    for (face = 0; face < faces; face++) {
        uint8_t *f = px + face_bytes * face;
        for (v = 0; v < t->height; v++)
            for (u = 0; u < t->width; u++) {
                uint32_t argb;
                size_t i = (size_t)v * t->width + u;
                if (!nv2a_exec_texel(t, face * t->face_stride, u, v, &argb))
                    argb = 0xFFFFFFFFu;        /* undecodable: white, as soft */
                if (!x->sgn) {
                    memcpy(f + i * 4, &argb, 4);
                } else {
                    /* R G B A halves; bit 28 + k signs A R G B (texel_unpack). */
                    static const int sh[4] = {16, 8, 0, 24};     /* R G B A */
                    static const int sb[4] = {29, 30, 31, 28};
                    uint16_t *h = (uint16_t *)(f + i * 8);
                    int k;
                    for (k = 0; k < 4; k++) {
                        uint32_t byte = (argb >> sh[k]) & 0xFF;
                        float val;
                        if ((x->sgn << 28) >> sb[k] & 1) {
                            val = (float)(int8_t)byte / 127.0f;
                            if (val < -1.0f) val = -1.0f;
                        } else {
                            val = (float)byte / 255.0f;
                        }
                        h[k] = f2h(val);
                    }
                }
            }
    }
    end_rendering();
    if (fresh) {
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = x->img;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.layerCount = faces;
        vkCmdPipelineBarrier(cmd(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &b);
    } else {
        /* Earlier draws of this slot may still sample the old texels. */
        VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        mb.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd(), VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    }
    memset(&c, 0, sizeof c);
    c.bufferOffset = at;
    c.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    c.imageSubresource.layerCount = faces;
    c.imageExtent.width = t->width;
    c.imageExtent.height = t->height;
    c.imageExtent.depth = 1;
    vkCmdCopyBufferToImage(cmd(), B.slot[B.cur].ring, x->img, VK_IMAGE_LAYOUT_GENERAL,
                           1, &c);
    B.pending_barrier = 1;
    B.uploads++;
    return 1;
}

/* The view to sample for stage `t`: a render target's own image when the
 * texture is one (render to texture), else the decoded texture. */
/* The view to sample for a stage and the size of the image behind it: a
 * render target's own image when the texture is one (render to texture),
 * else the decoded texture. `cube`: the shader samples a cube map. */
static VkImageView tex_view(const Nv2aTexState *t, int cube, uint32_t *iw, uint32_t *ih)
{
    Tex *x, *free_x = NULL, *lru = NULL;
    uint32_t sgn = (t->filter >> 28) & 0xF, mslot;
    int i;

    *iw = *ih = 1;
    if (!t->valid)
        return cube ? B.white_cube.view : B.white.view;
    if (!cube) {
        for (i = 0; i < VKB_SURFS; i++) {
            Surf *s = &B.surf[i];
            if (s->img && !s->depth && s->addr == t->offset) {
                /* RECOMP_VK_RTT_TRACE=1: each distinct render target /
                 * texture pairing once -- how the title samples what it
                 * rendered (sizes, formats, layout). */
                static int trace = -1;
                if (trace < 0)
                    trace = getenv("RECOMP_VK_RTT_TRACE") != NULL;
                if (trace) {
                    static uint32_t seen[256][4];
                    static unsigned nseen;
                    uint32_t key[4] = { t->offset, t->width << 16 | t->height,
                                        t->color << 8 | (uint32_t)t->normalized,
                                        s->w << 16 | s->h };
                    unsigned k;
                    for (k = 0; k < nseen && memcmp(seen[k], key, sizeof key); k++)
                        ;
                    if (k == nseen && nseen < 256) {
                        memcpy(seen[nseen++], key, sizeof key);
                        fprintf(stderr, "[VKRTT] %08X: texture %ux%u fmt %02X %s pitch %u"
                                " <- surface %ux%u bpp %u fmt %d (frame %u)\n",
                                t->offset, t->width, t->height, t->color,
                                t->normalized ? "swizzled" : "linear", t->pitch,
                                s->w, s->h, s->bpp, (int)s->fmt, B.frame);
                    }
                }
                /* The CPU wrote the memory since the GPU last rendered here
                 * (a freed render target reused for a texture): the texels
                 * are guest memory's, not the old image's. Not in compare
                 * mode, where the software renderer writes every target. */
                if (s->sig_frame && !B.compare_mode && surf_sig(s) != s->sig) {
                    static int said;
                    if (said++ < 5)
                        fprintf(stderr, "  [VK] %08X: the CPU rewrote a render target's memory;"
                                " sampling it as a texture\n", s->addr);
                    break;
                }
                if (s->dirty)
                    B.pending_barrier = 1;
                *iw = s->w;
                *ih = s->h;
                return s->view;
            }
            /* A depth buffer sampled as a texture (shadow maps and the
             * like) reads guest memory, which the GPU never writes: say so
             * once, under the trace. */
            if (s->img && s->depth && s->addr == t->offset) {
                static int said;
                if (!said++ && getenv("RECOMP_VK_RTT_TRACE"))
                    fprintf(stderr, "[VKRTT] %08X: texture at a depth surface's address,"
                            " read from guest memory (frame %u)\n", t->offset, B.frame);
            }
        }
    }
#define TEX_MATCH(x) ((x)->img && (x)->addr == t->offset && (x)->w == t->width \
        && (x)->h == t->height && (x)->color == t->color && (x)->palette == t->palette \
        && (x)->pitch == t->pitch && (x)->cube == (uint32_t)cube && (x)->sgn == sgn)
    {
        uint32_t k = t->offset * 0x9E3779B1u ^ (t->width * 0x85EBCA77u + t->height)
                   ^ (t->color * 0xC2B2AE3Du) ^ t->palette ^ (t->pitch << 7)
                   ^ ((uint32_t)cube << 31) ^ (sgn << 27);
        mslot = (k ^ (k >> 15)) % VKB_TEXMAP;
    }
    x = B.texmap[mslot] ? &B.tex[B.texmap[mslot] - 1] : NULL;
    if (!x || !TEX_MATCH(x)) {
        x = NULL;
        for (i = 0; i < VKB_TEXS; i++) {
            Tex *y = &B.tex[i];
            if (!y->img) {
                if (!free_x)
                    free_x = y;
                continue;
            }
            if (TEX_MATCH(y)) {
                x = y;
                B.texmap[mslot] = (uint16_t)(i + 1);
                break;
            }
            if (!lru || y->used < lru->used)
                lru = y;
        }
    }
#undef TEX_MATCH
    if (x) {
        if (x->checked != B.frame) {
            uint64_t h = tex_hash(t);
            x->checked = B.frame;
            if (h != x->hash) {
                x->hash = h;
                tex_upload(x, t);
            }
        }
        x->used = B.frame;
        *iw = x->w;
        *ih = x->h;
        return x->view;
    }
    x = free_x;
    if (!x) {
        wait_idle();
        tex_destroy(lru);
        x = lru;
    }
    x->addr = t->offset;
    x->w = t->width;
    x->h = t->height;
    x->color = t->color;
    x->palette = t->palette;
    x->pitch = t->pitch;
    x->cube = (uint32_t)cube;
    x->sgn = sgn;
    x->hash = tex_hash(t);
    x->checked = B.frame;
    x->used = B.frame;
    if (!tex_upload(x, t)) {
        tex_destroy(x);
        return cube ? B.white_cube.view : B.white.view;
    }
    B.texmap[mslot] = (uint16_t)(x - B.tex + 1);
    *iw = x->w;
    *ih = x->h;
    return x->view;
}

static VkSampler sampler(const Nv2aTexState *t)
{
    uint32_t mag = (t->filter >> 24) & 0xF, min = (t->filter >> 16) & 0xFF;
    int lin = mag == 2 || min == 2 || min == 4 || min == 6;
    uint32_t au = t->addr_u & 7, av = t->addr_v & 7;
    VkSampler *s = &B.samplers[lin][au][av];

    if (!*s) {
        VkSamplerCreateInfo ci = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        VkResult r_;
        ci.magFilter = ci.minFilter = lin ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        /* As the software path: 1 wraps, every other mode clamps. */
        ci.addressModeU = au == 1 ? VK_SAMPLER_ADDRESS_MODE_REPEAT
                                  : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ci.addressModeV = av == 1 ? VK_SAMPLER_ADDRESS_MODE_REPEAT
                                  : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ci.maxLod = 0.0f;                     /* level 0 only, as the software path */
        VK_OK(vkCreateSampler(g_vk.device, &ci, NULL, s));
    }
    return *s;
}

/* ---------------------------------------------------------------------- */
/* Pipelines. */

static VkBlendFactor blend_factor(uint32_t f)
{
    switch (f) {
    case 0x0000: return VK_BLEND_FACTOR_ZERO;
    case 0x0001: return VK_BLEND_FACTOR_ONE;
    case 0x0300: return VK_BLEND_FACTOR_SRC_COLOR;
    case 0x0301: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    case 0x0302: return VK_BLEND_FACTOR_SRC_ALPHA;
    case 0x0303: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case 0x0304: return VK_BLEND_FACTOR_DST_ALPHA;
    case 0x0305: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    case 0x0306: return VK_BLEND_FACTOR_DST_COLOR;
    case 0x0307: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
    case 0x0308: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
    case 0x8001: return VK_BLEND_FACTOR_CONSTANT_COLOR;
    case 0x8002: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
    case 0x8003: return VK_BLEND_FACTOR_CONSTANT_ALPHA;
    case 0x8004: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
    default:     return f ? VK_BLEND_FACTOR_ONE : VK_BLEND_FACTOR_ZERO;
    }
}

static VkBlendOp blend_op(uint32_t e)
{
    switch (e) {
    case 0x800A: return VK_BLEND_OP_SUBTRACT;
    case 0x800B: return VK_BLEND_OP_REVERSE_SUBTRACT;
    case 0x8007: return VK_BLEND_OP_MIN;
    case 0x8008: return VK_BLEND_OP_MAX;
    default:     return VK_BLEND_OP_ADD;
    }
}

/* The fragment shader for a combiner program, generated and compiled once
 * per distinct key. */
static VkShaderModule frag_shader(const VkPshKey *k)
{
    uint64_t h = hash_bytes((const uint8_t *)k, sizeof *k, 1469598103934665603ull);
    unsigned i = (unsigned)(h % VKB_FSH), probes;
    VkShaderModule m;
    char *src;

    for (probes = 0; probes < VKB_FSH; probes++, i = (i + 1) % VKB_FSH) {
        if (!B.fsh[i].used)
            break;
        if (!memcmp(&B.fsh[i].key, k, sizeof *k))
            return B.fsh[i].fs;
    }
    src = vk_psh_glsl(k);
    m = src ? vk_shader_module(1, src) : VK_NULL_HANDLE;
    if (getenv("RECOMP_VK_SHADER_DUMP") && src)
        fprintf(stderr, "[VKPSH] ---\n%s", src);
    free(src);
    if (m && probes < VKB_FSH) {
        B.fsh[i].key = *k;
        B.fsh[i].fs = m;
        B.fsh[i].used = 1;
    }
    return m;
}

/* vs == VK_NULL_HANDLE: the CPU-transformed vertex shader with its vertex
 * inputs; otherwise a program shader that pulls its own vertices. */
static VkPipeline pipeline(const Nv2aDrawState *st, VkFormat cfmt, VkShaderModule vs,
                           VkShaderModule fs)
{
    uint32_t blend = st->blend_enable && !(st->blend_sfactor == 1 && st->blend_dfactor == 0);
    uint32_t mask = (st->color_mask >> 16 & 1) | (st->color_mask >> 7 & 2)
                  | (st->color_mask << 2 & 4) | (st->color_mask >> 21 & 8);
    uint64_t key = (uint64_t)cfmt | (uint64_t)mask << 8 | (uint64_t)blend << 12, h;
    VkGraphicsPipelineCreateInfo ci = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    VkPipelineShaderStageCreateInfo ss[2];
    VkVertexInputBindingDescription vb;
    VkVertexInputAttributeDescription va[8];
    VkPipelineVertexInputStateCreateInfo vis = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    VkPipelineInputAssemblyStateCreateInfo ias = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    VkPipelineViewportStateCreateInfo vps = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    VkPipelineRasterizationStateCreateInfo rs = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    VkPipelineMultisampleStateCreateInfo ms = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    VkPipelineDepthStencilStateCreateInfo ds = { VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    VkPipelineColorBlendAttachmentState cba;
    VkPipelineColorBlendStateCreateInfo cbs = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    VkPipelineDynamicStateCreateInfo dyn = { VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    VkPipelineRenderingCreateInfo pr = { VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    static const VkDynamicState dstates[] = {
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_BLEND_CONSTANTS,
        VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
        VK_DYNAMIC_STATE_STENCIL_REFERENCE,
        VK_DYNAMIC_STATE_CULL_MODE, VK_DYNAMIC_STATE_FRONT_FACE,
        VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE, VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
        VK_DYNAMIC_STATE_DEPTH_COMPARE_OP, VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE,
        VK_DYNAMIC_STATE_STENCIL_OP,
    };
    VkPipeline p = VK_NULL_HANDLE;
    unsigned slot, probes;
    int i;
    double t0;
    VkResult r_;

    if (blend)
        key |= (uint64_t)(st->blend_sfactor & 0xFFFF) << 16
             | (uint64_t)(st->blend_dfactor & 0xFFFF) << 32
             | (uint64_t)(st->blend_equation & 0xFFFF) << 48;
    /* Pipelines by (fragment shader, fixed-function key). */
    h = (key * 0x9E3779B97F4A7C15ull) ^ (uint64_t)(uintptr_t)fs
      ^ ((uint64_t)(uintptr_t)vs * 0xC2B2AE3D27D4EB4Full);
    slot = (unsigned)((h ^ (h >> 29)) % VKB_PIPES);
    for (probes = 0; probes < VKB_PIPES; probes++, slot = (slot + 1) % VKB_PIPES) {
        if (!B.pipes[slot].pipe)
            break;
        if (B.pipes[slot].key == key && B.pipes[slot].fs == fs && B.pipes[slot].vs == vs)
            return B.pipes[slot].pipe;
    }

    memset(ss, 0, sizeof ss);
    ss[0].sType = ss[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ss[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    ss[0].module = vs ? vs : B.vs;
    ss[0].pName = "main";
    ss[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    ss[1].module = fs;
    ss[1].pName = "main";
    vb.binding = 0;
    vb.stride = sizeof(Nv2aVshOutput);
    vb.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    for (i = 0; i < 8; i++) {
        va[i].location = (uint32_t)i;
        va[i].binding = 0;
        va[i].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        va[i].offset = (uint32_t)(i * 16);    /* pos d0 d1 fog tex0..3 */
    }
    if (!vs) {
        vis.vertexBindingDescriptionCount = 1;
        vis.pVertexBindingDescriptions = &vb;
        vis.vertexAttributeDescriptionCount = 8;
        vis.pVertexAttributeDescriptions = va;
    }
    ias.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    vps.viewportCount = 1;
    vps.scissorCount = 1;
    rs.depthClampEnable = VK_TRUE;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.lineWidth = 1.0f;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    memset(&cba, 0, sizeof cba);
    cba.blendEnable = blend ? VK_TRUE : VK_FALSE;
    cba.srcColorBlendFactor = cba.srcAlphaBlendFactor = blend_factor(st->blend_sfactor);
    cba.dstColorBlendFactor = cba.dstAlphaBlendFactor = blend_factor(st->blend_dfactor);
    cba.colorBlendOp = cba.alphaBlendOp = blend_op(st->blend_equation);
    cba.colorWriteMask = mask;
    cbs.attachmentCount = 1;
    cbs.pAttachments = &cba;
    dyn.dynamicStateCount = sizeof dstates / sizeof dstates[0];
    dyn.pDynamicStates = dstates;
    pr.colorAttachmentCount = 1;
    pr.pColorAttachmentFormats = &cfmt;
    pr.depthAttachmentFormat = B.zfmt;
    pr.stencilAttachmentFormat = B.zfmt;
    ci.pNext = &pr;
    ci.stageCount = 2;
    ci.pStages = ss;
    ci.pVertexInputState = &vis;
    ci.pInputAssemblyState = &ias;
    ci.pViewportState = &vps;
    ci.pRasterizationState = &rs;
    ci.pMultisampleState = &ms;
    ci.pDepthStencilState = &ds;
    ci.pColorBlendState = &cbs;
    ci.pDynamicState = &dyn;
    ci.layout = B.layout;
    t0 = vk_cache_now_ms();
    if (!VK_OK(vkCreateGraphicsPipelines(g_vk.device, B.pcache, 1, &ci, NULL, &p)))
        return VK_NULL_HANDLE;
    B.pipe_ms += vk_cache_now_ms() - t0;
    B.new_pipes++;
    B.frame_pipes++;
    if (probes < VKB_PIPES && B.npipes < VKB_PIPES * 3 / 4) {
        B.pipes[slot].key = key;
        B.pipes[slot].fs = fs;
        B.pipes[slot].vs = vs;
        B.pipes[slot].pipe = p;
        B.npipes++;
    }
    return p;
}

/* A D3DCOLOR (0xAARRGGBB) as r g b a. */
static void unpack_argb(uint32_t c, float o[4])
{
    o[0] = (float)((c >> 16) & 0xFF) / 255.0f;
    o[1] = (float)((c >> 8) & 0xFF) / 255.0f;
    o[2] = (float)(c & 0xFF) / 255.0f;
    o[3] = (float)(c >> 24) / 255.0f;
}

static VkCompareOp compare_op(uint32_t f)
{
    return (f >= 0x200 && f <= 0x207) ? (VkCompareOp)(f - 0x200) : VK_COMPARE_OP_ALWAYS;
}

static VkStencilOp stencil_op(uint32_t op)
{
    switch (op) {
    case 0x0000: return VK_STENCIL_OP_ZERO;
    case 0x1E01: return VK_STENCIL_OP_REPLACE;
    case 0x1E02: return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
    case 0x1E03: return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
    case 0x150A: return VK_STENCIL_OP_INVERT;
    case 0x8507: return VK_STENCIL_OP_INCREMENT_AND_WRAP;
    case 0x8508: return VK_STENCIL_OP_DECREMENT_AND_WRAP;
    default:     return VK_STENCIL_OP_KEEP;
    }
}

/* ---------------------------------------------------------------------- */
/* Vertex programs on the GPU (raw draws). */

/* The shader for a program, by its instruction words. */
static VkShaderModule vert_shader(const uint32_t (*w)[4], uint32_t n)
{
    uint64_t h = hash_bytes((const uint8_t *)w, (size_t)n * 16, 1469598103934665603ull);
    unsigned i = (unsigned)(h % VKB_VSH), probes;
    VkShaderModule m;
    char *src;

    for (probes = 0; probes < VKB_VSH; probes++, i = (i + 1) % VKB_VSH) {
        if (!B.vsh[i].vs)
            break;
        if (B.vsh[i].h == h && B.vsh[i].n == n && !memcmp(B.vsh[i].w, w, (size_t)n * 16))
            return B.vsh[i].vs;
    }
    if (!vk_vsh_supported(w, n))
        return VK_NULL_HANDLE;
    src = vk_vsh_glsl(w, n, NULL);
    m = src ? vk_shader_module(0, src) : VK_NULL_HANDLE;
    B.new_vsh++;
    if (getenv("RECOMP_VK_SHADER_DUMP") && src)
        fprintf(stderr, "[VKVSH] ---\n%s", src);
    free(src);
    if (m && probes < VKB_VSH) {
        B.vsh[i].h = h;
        B.vsh[i].n = n;
        B.vsh[i].w = (uint32_t (*)[4])malloc((size_t)n * 16);
        if (B.vsh[i].w) {
            memcpy(B.vsh[i].w, w, (size_t)n * 16);
            B.vsh[i].vs = m;
        }
    }
    return m;
}

/* Bytes one element of an attribute takes (fetch_attr's reads). */
static uint32_t elem_bytes(uint32_t type, uint32_t size)
{
    uint32_t n = size < 4 ? size : 4;
    switch (type) {
    case 0: return 4;                         /* D3DCOLOR, whatever the size */
    case 2: return 4 * n;
    case 4: return n;
    case 1: case 5: return 2 * n;
    case 6: return 4;
    default: return 4;
    }
}

/* Room for n more bytes in this slot's ring, moving to the next slot now if
 * there is not: the allocations that follow then cannot. */
static void ring_reserve(VkDeviceSize n)
{
    cmd();
    if (B.slot[B.cur].ringpos + n + 1024 > VKB_RING)
        submit(0);
    cmd();
}

/* ---------------------------------------------------------------------- */
/* The interface. */

static float zmax_of(const Nv2aDrawState *st)
{
    return st->zeta_format == 1 ? 65535.0f : 16777215.0f;
}

static void scissor_of(const Nv2aDrawState *st, const Surf *rt, VkRect2D *r)
{
    uint32_t x0 = st->clip_x, y0 = st->clip_y;
    uint32_t x1 = st->clip_x + st->clip_w, y1 = st->clip_y + st->clip_h;
    if (x1 > rt->w) x1 = rt->w;
    if (y1 > rt->h) y1 = rt->h;
    if (x0 > x1) x0 = x1;
    if (y0 > y1) y0 = y1;
    r->offset.x = (int32_t)(x0 * rt->sc);
    r->offset.y = (int32_t)(y0 * rt->sc);
    r->extent.width = (x1 - x0) * rt->sc;
    r->extent.height = (y1 - y0) * rt->sc;
}

static void ds_clear(const Nv2aDrawState *st);   /* RECOMP_VK_DRAWSTATS */
static void ds_report(void);

static void vk_clear(const Nv2aDrawState *st, uint32_t what)
{
    Surf *rt, *zt;
    ds_clear(st);
    VkClearAttachment ca[2];
    VkClearRect cr;
    uint32_t n = 0;

    if (!B.ok || g_vk.lost)
        return;
    rt = target_color(st);
    if (!rt)
        return;
    zt = target_zeta(st);
    begin_rendering(rt, zt);
    memset(ca, 0, sizeof ca);
    if (what & 0xF0) {
        uint32_t c = st->clear_color;
        ca[n].aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ca[n].colorAttachment = 0;
        ca[n].clearValue.color.float32[0] = (float)((c >> 16) & 0xFF) / 255.0f;
        ca[n].clearValue.color.float32[1] = (float)((c >> 8) & 0xFF) / 255.0f;
        ca[n].clearValue.color.float32[2] = (float)(c & 0xFF) / 255.0f;
        ca[n].clearValue.color.float32[3] = (float)(c >> 24) / 255.0f;
        n++;
    }
    if ((what & 3) && zt) {
        uint32_t zc = st->zeta_format == 1 ? (st->zstencil_clear & 0xFFFF)
                                           : (st->zstencil_clear >> 8);
        ca[n].aspectMask = ((what & 1) ? VK_IMAGE_ASPECT_DEPTH_BIT : 0)
                         | ((what & 2) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
        ca[n].clearValue.depthStencil.depth = (float)zc / zmax_of(st);
        ca[n].clearValue.depthStencil.stencil = st->zstencil_clear & 0xFF;
        if (ca[n].clearValue.depthStencil.depth > 1.0f)
            ca[n].clearValue.depthStencil.depth = 1.0f;
        n++;
    }
    if (!n)
        return;
    scissor_of(st, rt, &cr.rect);
    cr.baseArrayLayer = 0;
    cr.layerCount = 1;
    if (!cr.rect.extent.width || !cr.rect.extent.height)
        return;
    vkCmdClearAttachments(cmd(), n, ca, 1, &cr);
    rt->dirty = 1;
    if (zt)
        zt->dirty = 1;
}

/* Triangle-list indices for a primitive, in the software path's order and
 * winding (geo_assemble). Returns the count. */
static uint32_t prim_indices(uint32_t prim, uint32_t n, uint32_t *ix)
{
    uint32_t i, m = 0;
    switch (prim) {
    case NV_PRIM_TRIANGLES:
        for (i = 0; i + 2 < n; i += 3) { ix[m++] = i; ix[m++] = i + 1; ix[m++] = i + 2; }
        break;
    case NV_PRIM_TRIANGLE_STRIP:
        for (i = 0; i + 2 < n; i++) {
            if (i & 1) { ix[m++] = i + 1; ix[m++] = i; }
            else       { ix[m++] = i; ix[m++] = i + 1; }
            ix[m++] = i + 2;
        }
        break;
    case NV_PRIM_TRIANGLE_FAN:
    case NV_PRIM_POLYGON:
        for (i = 1; i + 1 < n; i++) { ix[m++] = 0; ix[m++] = i; ix[m++] = i + 1; }
        break;
    case NV_PRIM_QUADS:
        for (i = 0; i + 3 < n; i += 4) {
            ix[m++] = i; ix[m++] = i + 1; ix[m++] = i + 2;
            ix[m++] = i; ix[m++] = i + 2; ix[m++] = i + 3;
        }
        break;
    case NV_PRIM_QUAD_STRIP:
        for (i = 0; i + 3 < n; i += 2) {
            ix[m++] = i; ix[m++] = i + 1; ix[m++] = i + 3;
            ix[m++] = i; ix[m++] = i + 3; ix[m++] = i + 2;
        }
        break;
    default:
        break;
    }
    return m;
}

/* The draw's descriptor set: the textures (views and samplers) it samples,
 * the ring's buffers at dynamic offsets. Found again when an earlier draw in
 * the slot had the same textures; made otherwise. */
static VkDescriptorSet draw_set(Slot *s, const Nv2aDrawState *st, const VkImageView tv[4])
{
    VkSampler sm[4];
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo ai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    VkDescriptorBufferInfo bi[3];
    VkDescriptorImageInfo ii[4];
    VkWriteDescriptorSet w[4];
    uint32_t h = 2166136261u;
    unsigned i, k, at, free_at = VKB_SETCACHE;
    const uint8_t *b;

    for (i = 0; i < 4; i++)
        sm[i] = sampler(&st->tex[i]);
    b = (const uint8_t *)tv;
    for (i = 0; i < 4 * sizeof tv[0]; i++)
        h = (h ^ b[i]) * 16777619u;
    b = (const uint8_t *)sm;
    for (i = 0; i < sizeof sm; i++)
        h = (h ^ b[i]) * 16777619u;
    at = h % VKB_SETCACHE;
    for (k = 0; k < 8 && !B.rebind; k++, at = (at + 1) % VKB_SETCACHE) {
        if (B.sets[at].gen != B.slot_gen || B.sets[at].vgen != B.view_gen) {
            free_at = at;                /* from an earlier slot: free */
            break;
        }
        if (!memcmp(B.sets[at].v, tv, sizeof B.sets[at].v)
            && !memcmp(B.sets[at].s, sm, sizeof sm))
            return B.sets[at].set;
    }

    ai.descriptorPool = s->dpool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &B.dsl;
    if (vkAllocateDescriptorSets(g_vk.device, &ai, &set) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    s->nsets++;
    for (i = 0; i < 3; i++) {
        bi[i].buffer = s->ring;
        bi[i].offset = 0;
    }
    bi[0].range = sizeof(VkDrawUniforms);
    bi[1].range = sizeof B.cst;
    bi[2].range = VK_WHOLE_SIZE;
    for (i = 0; i < 4; i++) {
        ii[i].sampler = sm[i];
        ii[i].imageView = tv[i];
        ii[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    }
    memset(w, 0, sizeof w);
    for (i = 0; i < 4; i++) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = set;
        w[i].descriptorCount = 1;
    }
    w[0].dstBinding = 0;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    w[0].pBufferInfo = &bi[0];
    w[1].dstBinding = 1;                 /* rolls over bindings 1-4 */
    w[1].descriptorCount = 4;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w[1].pImageInfo = ii;
    w[2].dstBinding = 5;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    w[2].pBufferInfo = &bi[1];
    w[3].dstBinding = 6;
    w[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
    w[3].pBufferInfo = &bi[2];
    vkUpdateDescriptorSets(g_vk.device, 4, w, 0, NULL);
    if (free_at < VKB_SETCACHE) {
        B.sets[free_at].gen = B.slot_gen;
        B.sets[free_at].vgen = B.view_gen;
        memcpy(B.sets[free_at].v, tv, sizeof B.sets[free_at].v);
        memcpy(B.sets[free_at].s, sm, sizeof sm);
        B.sets[free_at].set = set;
    }
    return set;
}

static void draw_one(const Nv2aDrawState *st, const Nv2aDrawGeom *g)
{
    Surf *rt, *zt;
    VkPipeline p;
    VkShaderModule fs;
    VkPshKey key;
    VkDeviceSize voff, ioff, uoff;
    VkDeviceSize ualign = g_vk.props.limits.minUniformBufferOffsetAlignment;
    void *vp;
    uint32_t *ix, nix, maxix;
    VkCommandBuffer cb;
    VkDrawUniforms *U, *Uw;
    static VkDrawUniforms uh;
    int32_t vo = 0;                      /* vertexOffset */
    VkViewport view;
    VkRect2D sc;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkImageView tv[4];
    uint32_t tw[4], th[4];
    int i;
    Slot *s;
    /* Raw draws: the program runs here (vs), vertices are pulled from the
     * blob, constants come from cst_off. */
    int raw = g->v == NULL;
    VkShaderModule vs = VK_NULL_HANDLE;
    VkDeviceSize salign = g_vk.props.limits.minStorageBufferOffsetAlignment;
    VkDeviceSize boff = 0, bbytes = 0;
    uint32_t imin = 0, imax = 0, blob_lo = 0;
    uint32_t aoff[16];

    if (!B.ok || g_vk.lost || g->n < 3)
        return;
    /* Per-slot limits first, so nothing below has to move to a new slot
     * half-way through a draw. */
    cmd();
    s = &B.slot[B.cur];
    if (s->nsets >= VKB_SETS || s->nq >= VKB_QUERIES)
        submit(0);
    rt = target_color(st);
    if (!rt)
        return;
    zt = target_zeta(st);

    if (raw) {
        uint32_t k;
        vs = vert_shader(g->prog, g->prog_n);
        if (!vs)
            return;
        imin = imax = g->idx[0];
        for (k = 1; k < g->n; k++) {
            if (g->idx[k] < imin) imin = g->idx[k];
            if (g->idx[k] > imax) imax = g->idx[k];
        }
    }

    /* The combiner program's shader, and the stages it samples. */
    vk_psh_key(st, &key);
    fs = frag_shader(&key);
    if (!fs)
        return;
    for (i = 0; i < 4; i++) {
        int used = st->rc_seen ? ((st->rc.stage_program >> (i * 5)) & 0x1F) != 0 : i == 0;
        tw[i] = th[i] = 1;
        tv[i] = used ? tex_view(&st->tex[i], key.cube[i], &tw[i], &th[i])
                     : (key.cube[i] ? B.white_cube.view : B.white.view);
    }

    /* Vertices (or the raw vertex data), indices and the uniforms, in one
     * slot: room is reserved first, so no allocation below moves on. */
    maxix = (g->n + 2) * 6;
    if (salign < 16)
        salign = 16;
    if (ualign < 16)
        ualign = 16;
    if (raw) {
        /* The span of guest memory the draw's attributes read, from element
         * imin to imax: one copy when they interleave (the usual case),
         * attribute by attribute when they are far apart. */
        const uint8_t *src = g->inline_data ? g->inline_data
                                            : (const uint8_t *)xbox_GetMemoryOffset();
        uint32_t k, lo = 0xFFFFFFFFu, hi = 0, sum = 0;
        int whole;
        for (k = 0; k < 16; k++) {
            const Nv2aAttrDesc *a = &g->attr[k];
            uint32_t l, h;
            aoff[k] = 0xFFFFFFFFu;
            if (!a->size || !a->stride || (!g->inline_data && !a->offset))
                continue;
            l = a->offset + imin * a->stride;
            h = a->offset + imax * a->stride + elem_bytes(a->type, a->size);
            if (l < lo) lo = l;
            if (h > hi) hi = h;
            sum += h - l;
        }
        whole = hi > lo && hi - lo <= 4 * sum + 65536;
        bbytes = hi <= lo ? 16 : whole ? (hi - (lo & ~3u)) + 8 : sum + 16 * 4 + 8;
        if (bbytes > (32u << 20)) {
            static int said;
            if (!said++)
                fprintf(stderr, "  [VK] draw reads %u bytes of vertex data; skipped\n",
                        (unsigned)bbytes);
            return;
        }
        ring_reserve(bbytes + salign + (VkDeviceSize)maxix * 4 + ualign
                     + sizeof(VkDrawUniforms) + ualign + sizeof B.cst + ualign);
        {
            uint8_t *bp;
            boff = ring_alloc(bbytes, salign, (void **)&bp);
            if (boff == (VkDeviceSize)-1)
                return;
            /* rd32 may read one word on; an inline payload is cut at its end,
             * where fetch_attr reads zeros, so all of it starts zeroed. */
            if (g->inline_data)
                memset(bp, 0, bbytes);
            else
                memset(bp + bbytes - 8, 0, 8);
            if (whole) {
                uint32_t base = lo & ~3u, n = hi - base;
                if (g->inline_data)                  /* fetch_attr's bound */
                    n = base < g->inline_bytes ? (n < g->inline_bytes - base ? n
                                                  : g->inline_bytes - base) : 0;
                memcpy(bp, src + base, n);
                for (k = 0; k < 16; k++)
                    if (g->attr[k].size && g->attr[k].stride
                        && (g->inline_data || g->attr[k].offset))
                        aoff[k] = g->attr[k].offset + imin * g->attr[k].stride - base;
            } else if (hi > lo) {
                uint32_t at = 0;
                for (k = 0; k < 16; k++) {
                    const Nv2aAttrDesc *a = &g->attr[k];
                    uint32_t l, h;
                    if (!a->size || !a->stride || (!g->inline_data && !a->offset))
                        continue;
                    l = (a->offset + imin * a->stride) & ~3u;
                    h = a->offset + imax * a->stride + elem_bytes(a->type, a->size);
                    if (g->inline_data && h > g->inline_bytes)
                        h = g->inline_bytes > l ? g->inline_bytes : l;
                    memcpy(bp + at, src + l, h - l);
                    aoff[k] = at + ((a->offset + imin * a->stride) & 3u);
                    at = (at + h - l + 3) & ~3u;
                }
            }
        }
        {
            uint8_t *ip;
            ioff = ring_alloc((VkDeviceSize)maxix * 4, 4, (void **)&ip);
            ix = (uint32_t *)ip;
        }
        {
            uint8_t *up;
            uoff = ring_alloc(sizeof(VkDrawUniforms), ualign, (void **)&up);
            U = (VkDrawUniforms *)up;
        }
        /* Constants: re-sent only when they changed or the slot did. */
        if (B.cst_gen != B.slot_gen || memcmp(B.cst, g->constants, sizeof B.cst)) {
            uint8_t *cp;
            B.cst_off = ring_alloc(sizeof B.cst, ualign, (void **)&cp);
            memcpy(B.cst, g->constants, sizeof B.cst);
            memcpy(cp, B.cst, sizeof B.cst);
            B.cst_gen = B.slot_gen;
        }
        voff = 0;
        vp = NULL;
        /* Built and remapped in host memory, then copied: the ring is
         * write-combined, and reading it back costs a bus round trip per
         * word (half the executor's time in a 4000-draw frame). */
        {
            static uint32_t *tmp;
            static uint32_t cap;
            uint32_t k;
            if (maxix > cap) {
                uint32_t *t = (uint32_t *)realloc(tmp, (size_t)maxix * 4);
                if (!t)
                    return;
                tmp = t;
                cap = maxix;
            }
            nix = prim_indices(g->prim, g->n, tmp);
            for (k = 0; k < nix; k++)
                tmp[k] = g->idx[tmp[k]];
            memcpy(ix, tmp, (size_t)nix * 4);
        }
    } else {
        /* The uniforms' dynamic offset must be a multiple of ualign. */
        VkDeviceSize vbytes = ((VkDeviceSize)g->n * sizeof(Nv2aVshOutput) + ualign - 1)
                            & ~(ualign - 1);
        VkDeviceSize ibytes = ((VkDeviceSize)maxix * 4 + ualign - 1) & ~(ualign - 1);
        uint8_t *base;
        voff = ring_alloc(vbytes + ibytes + sizeof(VkDrawUniforms), ualign, (void **)&base);
        if (voff == (VkDeviceSize)-1)
            return;
        vp = base;
        ix = (uint32_t *)(base + vbytes);
        ioff = voff + vbytes;
        uoff = ioff + ibytes;
        U = (VkDrawUniforms *)(base + vbytes + ibytes);
        memcpy(vp, g->v, (size_t)g->n * sizeof(Nv2aVshOutput));
        nix = prim_indices(g->prim, g->n, ix);
    }
    if (!nix)
        return;

    /* The uniforms, built in host memory (the ring is write-combined). */
    Uw = U;
    U = &uh;
    memset(U, 0, sizeof *U);
    U->vp[0] = 2.0f / (float)rt->w;
    U->vp[1] = 2.0f / (float)rt->h;
    U->vp[2] = 1.0f / zmax_of(st);
    for (i = 0; i < 3; i++) {
        U->k[i] = st->vp_scale[i];
        U->o[i] = st->vp_offset[i];
    }
    U->fogp[0] = st->fog_enable ? 1.0f : 0.0f;
    U->fogp[1] = (float)st->fog_mode;
    U->fogp[2] = st->fog_param[0];
    U->fogp[3] = st->fog_param[1];
    /* FOG_COLOR is R in bits 0-7, the reverse of a D3DCOLOR. */
    U->fogc[0] = (float)(st->fog_color & 0xFF) / 255.0f;
    U->fogc[1] = (float)((st->fog_color >> 8) & 0xFF) / 255.0f;
    U->fogc[2] = (float)((st->fog_color >> 16) & 0xFF) / 255.0f;
    U->fogc[3] = (float)(st->fog_color >> 24) / 255.0f;
    {
        uint32_t flags = st->rc.control >> 8;
        for (i = 0; i < 8; i++) {
            unpack_argb(st->rc.factor0[(flags & 0x010) ? i : 0], U->c0[i]);
            unpack_argb(st->rc.factor1[(flags & 0x100) ? i : 0], U->c1[i]);
        }
        unpack_argb(st->rc.final_c0, U->fc0);
        unpack_argb(st->rc.final_c1, U->fc1);
    }
    for (i = 0; i < 4; i++) {
        const Nv2aTexState *t = &st->tex[i];
        float w = (float)(t->width ? t->width : 1), h = (float)(t->height ? t->height : 1);
        /* Coordinates to the sampled image's [0,1]: normalised ones scale by
         * texture/image (equal unless a render target is larger), texel ones
         * divide by the image. */
        if (t->normalized) {
            U->tsz[i][0] = w / (float)tw[i];
            U->tsz[i][1] = h / (float)th[i];
        } else {
            U->tsz[i][0] = 1.0f / (float)tw[i];
            U->tsz[i][1] = 1.0f / (float)th[i];
        }
        U->tsz[i][2] = t->normalized ? 1.0f : 0.0f;
        U->tsz[i][3] = t->valid ? 1.0f : 0.0f;
        memcpy(U->bm[i], t->bump_mat, sizeof U->bm[i]);
        U->bl[i][0] = t->bump_scale;
        U->bl[i][1] = t->bump_offset;
    }
    U->at[0] = (int32_t)(st->alpha_test != 0);
    U->at[1] = (int32_t)st->alpha_func;
    U->at[2] = (int32_t)(st->alpha_ref & 0xFF);
    if (raw) {
        for (i = 0; i < 16; i++) {
            const Nv2aAttrDesc *a = &g->attr[i];
            U->attr[i][0] = aoff[i] == 0xFFFFFFFFu ? 0 : aoff[i];
            U->attr[i][1] = a->stride;
            U->attr[i][2] = (a->type & 0xF) | (a->size & 0xF) << 4
                          | (a->size && aoff[i] == 0xFFFFFFFFu ? 0x100u : 0u);
            memcpy(U->imm[i], g->imm[i], sizeof U->imm[i]);
        }
        /* The element index minus imin is the vertex's place in the copied
         * data: the draw's vertexOffset (-imin) subtracts it, so uniforms of
         * draws with the same layout are the same. */
        U->imin[0] = 0;
        vo = -(int32_t)imin;
    }
    /* The same uniforms as the last draw in this command buffer: its copy
     * again (nothing written, and the bound set's offsets stay). */
    if (B.u_gen == B.slot_gen && !B.rebind && !memcmp(U, &B.u_last, sizeof *U)) {
        uoff = B.u_off;
    } else {
        memcpy(Uw, U, sizeof *U);
        B.u_last = *U;
        B.u_off = uoff;
        B.u_gen = B.slot_gen;
    }

    s = &B.slot[B.cur];
    set = draw_set(s, st, tv);
    if (!set)
        return;

    p = pipeline(st, rt->fmt, vs, fs);
    if (!p)
        return;
    begin_rendering(rt, zt);
    cb = cmd();
    if (B.bound.gen != B.slot_gen || B.rebind) {
        memset(&B.bound, 0xFF, sizeof B.bound);     /* matches nothing */
        B.bound.gen = B.slot_gen;
    }
    if (B.bound.pipe != p) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
        B.bound.pipe = p;
    }
    {
        /* Bindings 0, 5 and 6: the uniforms, the constants, the raw data. */
        uint32_t dyn[3];
        dyn[0] = (uint32_t)uoff;
        dyn[1] = raw ? (uint32_t)B.cst_off : 0;
        dyn[2] = raw ? (uint32_t)boff : 0;
        if (B.bound.set != set || memcmp(B.bound.dyn, dyn, sizeof dyn)) {
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, B.layout, 0, 1, &set,
                                    3, dyn);
            B.bound.set = set;
            memcpy(B.bound.dyn, dyn, sizeof dyn);
        }
    }
    if (!raw)
        vkCmdBindVertexBuffers(cb, 0, 1, &s->ring, &voff);
    vkCmdBindIndexBuffer(cb, s->ring, ioff, VK_INDEX_TYPE_UINT32);

    /* Render scales are odd (1, 3, 5): the middle sub-pixel of each block is
     * the native pixel centre, which the counting pass below counts, with no
     * shift of the picture. (An even scale would need the picture moved by
     * half a sub-pixel, which loses coverage along pixel-aligned edges.) */
    memset(&view, 0, sizeof view);
    view.width = (float)(rt->w * rt->sc);
    view.height = (float)(rt->h * rt->sc);
    view.maxDepth = 1.0f;
    if (memcmp(&B.bound.view, &view, sizeof view)) {
        vkCmdSetViewport(cb, 0, 1, &view);
        B.bound.view = view;
    }
    memset(&sc, 0, sizeof sc);
    scissor_of(st, rt, &sc);
    if (!sc.extent.width || !sc.extent.height)
        return;
    if (memcmp(&B.bound.sc, &sc, sizeof sc)) {
        vkCmdSetScissor(cb, 0, 1, &sc);
        B.bound.sc = sc;
    }
    {
        float bc[4];
        unpack_argb(st->blend_color, bc);
        if (memcmp(B.bound.bc, bc, sizeof bc)) {
            vkCmdSetBlendConstants(cb, bc);
            memcpy(B.bound.bc, bc, sizeof bc);
        }
    }
/* The dynamic state the draw wants, sent only when it differs from what the
 * command buffer has. */
#define SET_DYN(field, value, call) do {                                    \
        uint32_t v_ = (uint32_t)(value);                                    \
        if (B.bound.field != v_) {                                          \
            B.bound.field = v_;                                             \
            call;                                                           \
        }                                                                   \
    } while (0)
    /* Culling, judged as the software path does (geo_cull): its winding is
     * the screen-space area's sign, flipped for a y-up viewport, and Vulkan's
     * framebuffer space is the same screen space. */
    {
        int yup = st->vp_scale[1] > 0.0f;
        int ccw = (st->front_face == 0x901) ^ yup;
        VkCullModeFlags cm = VK_CULL_MODE_NONE;
        VkFrontFace ff = ccw ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
        if (st->cull_enable) {
            switch (st->cull_face) {
            case 0x404: cm = VK_CULL_MODE_FRONT_BIT; break;
            case 0x408: cm = VK_CULL_MODE_FRONT_AND_BACK; break;
            default:    cm = VK_CULL_MODE_BACK_BIT; break;
            }
        }
        SET_DYN(cull, cm, vkCmdSetCullMode(cb, cm));
        SET_DYN(front, ff, vkCmdSetFrontFace(cb, ff));
    }
    /* Depth: the software path writes depth whenever the mask is set, test
     * or no test. */
    {
        VkBool32 dte = (st->depth_test || st->depth_mask) && zt;
        VkBool32 dwe = st->depth_mask != 0;
        VkCompareOp dop = st->depth_test ? compare_op(st->depth_func) : VK_COMPARE_OP_ALWAYS;
        VkBool32 ste = st->st_test && zt;
        VkStencilOp f = stencil_op(st->st_fail), zp = stencil_op(st->st_zpass);
        VkStencilOp zf = stencil_op(st->st_zfail);
        VkCompareOp sf = compare_op(st->st_func);
        SET_DYN(dte, dte, vkCmdSetDepthTestEnable(cb, dte));
        SET_DYN(dwe, dwe, vkCmdSetDepthWriteEnable(cb, dwe));
        SET_DYN(dop, dop, vkCmdSetDepthCompareOp(cb, dop));
        SET_DYN(ste, ste, vkCmdSetStencilTestEnable(cb, ste));
        if (B.bound.sop[0] != (uint32_t)f || B.bound.sop[1] != (uint32_t)zp
            || B.bound.sop[2] != (uint32_t)zf || B.bound.sop[3] != (uint32_t)sf) {
            vkCmdSetStencilOp(cb, VK_STENCIL_FACE_FRONT_AND_BACK, f, zp, zf, sf);
            B.bound.sop[0] = (uint32_t)f;
            B.bound.sop[1] = (uint32_t)zp;
            B.bound.sop[2] = (uint32_t)zf;
            B.bound.sop[3] = (uint32_t)sf;
        }
        SET_DYN(fmask, st->st_fmask & 0xFF,
                vkCmdSetStencilCompareMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, v_));
        SET_DYN(wmask, st->st_wmask & 0xFF,
                vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, v_));
        SET_DYN(ref, st->st_ref & 0xFF,
                vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_AND_BACK, v_));
    }
#undef SET_DYN

    /* Compare and shadow modes: software answers every visibility test (the
     * executor sends no reports here), so nothing is counted. */
    if (st->zpass_enable && !B.compare_mode && rt->sc > 1) {
        /* Supersampled: the count must be the native one (the title steers
         * by it, and a replay must see the same numbers at any scale). The
         * draw runs once more first, writing nothing, with a shader that
         * keeps only the sub-pixel at each native pixel centre, under the
         * query; then the visible draw, uncounted. */
        Nv2aDrawState cs = *st;
        VkPshKey ck = key;
        VkPipeline cp;
        B.zdraws++;
        cs.color_mask = 0;
        cs.blend_enable = 0;
        ck.count_scale = (uint8_t)rt->sc;
        cp = pipeline(&cs, rt->fmt, vs, frag_shader(&ck));
        if (cp) {
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, cp);
            vkCmdSetDepthWriteEnable(cb, VK_FALSE);
            vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, 0);
            vkCmdBeginQuery(cb, s->qpool, s->nq, VK_QUERY_CONTROL_PRECISE_BIT);
            vkCmdDrawIndexed(cb, nix, 1, 0, vo, 0);
            vkCmdEndQuery(cb, s->qpool, s->nq);
            s->qepoch[s->nq++] = B.epoch;
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
            vkCmdSetDepthWriteEnable(cb, st->depth_mask != 0);
            vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, st->st_wmask & 0xFF);
        }
        vkCmdDrawIndexed(cb, nix, 1, 0, vo, 0);
    } else if (st->zpass_enable && !B.compare_mode) {
        B.zdraws++;
        vkCmdBeginQuery(cb, s->qpool, s->nq, VK_QUERY_CONTROL_PRECISE_BIT);
        vkCmdDrawIndexed(cb, nix, 1, 0, vo, 0);
        vkCmdEndQuery(cb, s->qpool, s->nq);
        s->qepoch[s->nq++] = B.epoch;
    } else {
        vkCmdDrawIndexed(cb, nix, 1, 0, vo, 0);
    }
    rt->dirty = 1;
    if (zt)
        zt->dirty = 1;
    B.draws++;
    B.tris += nix / 3;
}

/* RECOMP_VK_DRAWSTATS=<n>: for a frame with at least n draws (at most one
 * a second), what the title drew: per render target, the commonest draws
 * by signature (vertex program, texture 0, primitive, vertex count, target),
 * exact repeats (same signature, vertex data address and constants), the
 * spread of vertex counts, clears and visibility reports. For finding why a
 * scene submits tens of thousands of draws. */
#define DS_RTS  32
#define DS_SIGS 4096
static struct {
    long min;                            /* -1: off */
    uint32_t draws, tris, zdraws, clears, reports, repeats, small;
    uint32_t nbin[8];                    /* vertex counts: 3, <=6, <=16, ... */
    struct { uint32_t addr, w, h, draws, tris, zdraws, clears; } rt[DS_RTS];
    uint32_t nrt;
    struct {
        uint64_t key;
        uint32_t count, tris, n, prim, tex0, rt, prog_n, zdraws, repeats;
        uint32_t last_attr;
        uint64_t last_cst;
    } sig[DS_SIGS];
    double last_print;
} DS = { -2 };

static int ds_on(void)
{
    if (DS.min == -2) {
        const char *e = getenv("RECOMP_VK_DRAWSTATS");
        DS.min = e ? atol(e) : -1;
    }
    return DS.min >= 0;
}

static unsigned ds_rt(const Nv2aDrawState *st)
{
    unsigned i;
    for (i = 0; i < DS.nrt; i++)
        if (DS.rt[i].addr == st->color_addr)
            return i;
    if (DS.nrt == DS_RTS)
        return DS_RTS - 1;
    DS.rt[DS.nrt].addr = st->color_addr;
    DS.rt[DS.nrt].w = st->clip_x + st->clip_w;
    DS.rt[DS.nrt].h = st->clip_y + st->clip_h;
    return DS.nrt++;
}

static uint32_t ds_tris(uint32_t prim, uint32_t n)
{
    switch (prim) {
    case NV_PRIM_TRIANGLES:      return n / 3;
    case NV_PRIM_TRIANGLE_STRIP:
    case NV_PRIM_TRIANGLE_FAN:
    case NV_PRIM_POLYGON:        return n >= 3 ? n - 2 : 0;
    case NV_PRIM_QUADS:          return n / 4 * 2;
    case NV_PRIM_QUAD_STRIP:     return n >= 4 ? (n - 2) / 2 * 2 : 0;
    default:                     return 0;
    }
}

static void ds_print(const char *when);

static void ds_draw(const Nv2aDrawState *st, const Nv2aDrawGeom *g)
{
    uint64_t pk, key, cst;
    uint32_t tris = ds_tris(g->prim, g->n), b, attr;
    unsigned r, i, k;
    if (!ds_on())
        return;
    DS.draws++;
    DS.tris += tris;
    if (st->zpass_enable)
        DS.zdraws++;
    if (g->n <= 4)
        DS.small++;
    for (b = 0, k = 3; k < g->n && b < 7; b++)
        k = b ? k * 4 : 6;
    DS.nbin[b]++;
    r = ds_rt(st);
    DS.rt[r].draws++;
    DS.rt[r].tris += tris;
    if (st->zpass_enable)
        DS.rt[r].zdraws++;
    pk = g->prog ? hash_bytes((const uint8_t *)g->prog, (size_t)g->prog_n * 16,
                              1469598103934665603ull) : 0;
    key = pk ^ ((uint64_t)st->tex[0].offset << 1) ^ ((uint64_t)g->prim << 40)
        ^ ((uint64_t)g->n << 44) ^ ((uint64_t)st->color_addr * 0x9E3779B97F4A7C15ull);
    attr = g->v ? 0 : g->attr[0].offset;
    cst = g->constants ? hash_bytes((const uint8_t *)g->constants, NV2A_VSH_CONSTANTS * 16,
                                    1469598103934665603ull) : 0;
    i = (unsigned)(key % DS_SIGS);
    for (k = 0; k < DS_SIGS; k++, i = (i + 1) % DS_SIGS) {
        if (!DS.sig[i].count || DS.sig[i].key == key)
            break;
    }
    if (k == DS_SIGS)
        return;
    if (DS.sig[i].count) {
        if (DS.sig[i].last_attr == attr && DS.sig[i].last_cst == cst) {
            DS.sig[i].repeats++;
            DS.repeats++;
        }
    } else {
        DS.sig[i].key = key;
        DS.sig[i].n = g->n;
        DS.sig[i].prim = g->prim;
        DS.sig[i].tex0 = st->tex[0].valid ? st->tex[0].offset : 0;
        DS.sig[i].rt = st->color_addr;
        DS.sig[i].prog_n = g->prog ? g->prog_n : 0;
    }
    DS.sig[i].count++;
    DS.sig[i].tris += tris;
    if (st->zpass_enable)
        DS.sig[i].zdraws++;
    DS.sig[i].last_attr = attr;
    DS.sig[i].last_cst = cst;
    /* A frame that does not end: say so every 2 s, from what it drew. */
    if ((DS.draws & 1023) == 0 && DS.draws >= (uint32_t)DS.min) {
        double now = vk_cache_now_ms();
        if (now - DS.last_print >= 2000.0) {
            DS.last_print = now;
            ds_print("not flipped yet");
        }
    }
}

static void ds_clear(const Nv2aDrawState *st)
{
    if (!ds_on())
        return;
    DS.clears++;
    DS.rt[ds_rt(st)].clears++;
}

static void ds_report(void)
{
    if (ds_on())
        DS.reports++;
}

static int ds_cmp(const void *a, const void *b)
{
    uint32_t x = DS.sig[*(const unsigned *)a].count, y = DS.sig[*(const unsigned *)b].count;
    return x < y ? 1 : x > y ? -1 : 0;
}

static void ds_print(const char *when)
{
    static unsigned idx[DS_SIGS];
    unsigned i, n = 0;
    {
        fprintf(stderr, "[VKDRAWS] frame %u (%s): %u draws, %u tris, %u counted, %u reports,"
                " %u clears, %u exact repeats, %u with <= 4 vertices;"
                " vertex counts 3/6/24/96/384/1536/6144/more: %u %u %u %u %u %u %u %u\n",
                B.frame, when, DS.draws, DS.tris, DS.zdraws, DS.reports, DS.clears, DS.repeats,
                DS.small, DS.nbin[0], DS.nbin[1], DS.nbin[2], DS.nbin[3], DS.nbin[4],
                DS.nbin[5], DS.nbin[6], DS.nbin[7]);
        for (i = 0; i < DS.nrt; i++)
            fprintf(stderr, "[VKDRAWS]   target %08X %ux%u: %u draws, %u tris, %u counted,"
                    " %u clears\n", DS.rt[i].addr, DS.rt[i].w, DS.rt[i].h, DS.rt[i].draws,
                    DS.rt[i].tris, DS.rt[i].zdraws, DS.rt[i].clears);
        for (i = 0; i < DS_SIGS; i++)
            if (DS.sig[i].count)
                idx[n++] = i;
        qsort(idx, n, sizeof idx[0], ds_cmp);
        fprintf(stderr, "[VKDRAWS]   %u signatures; commonest:\n", n);
        for (i = 0; i < n && i < 15; i++) {
            const unsigned j = idx[i];
            fprintf(stderr, "[VKDRAWS]     %6u x prim %u n %u (%u tris) prog %u words tex0 %08X"
                    " -> %08X; %u counted, %u exact repeats\n", DS.sig[j].count,
                    DS.sig[j].prim, DS.sig[j].n, DS.sig[j].tris, DS.sig[j].prog_n,
                    DS.sig[j].tex0, DS.sig[j].rt, DS.sig[j].zdraws, DS.sig[j].repeats);
        }
    }
    fflush(stderr);
}

static void ds_flip(void)
{
    double now;
    if (!ds_on())
        return;
    now = vk_cache_now_ms();
    if (DS.draws >= (uint32_t)DS.min && now - DS.last_print >= 1000.0) {
        DS.last_print = now;
        ds_print("at its flip");
    }
    {
        long min = DS.min;
        double lp = DS.last_print;
        memset(&DS, 0, sizeof DS);
        DS.min = min;
        DS.last_print = lp;
    }
}

/* RECOMP_VK_STATS: the executor thread's CPU cycles in draws (the counter
 * costs a little per call, so only then). */
static int s_stats = -1;
static uint64_t s_draw_cycles;

static void vk_draw(const Nv2aDrawState *st, const Nv2aDrawGeom *g)
{
    uint64_t c0;
    ds_draw(st, g);
    if (s_stats < 0)
        s_stats = getenv("RECOMP_VK_STATS") != NULL;
    if (!s_stats) {
        draw_one(st, g);
        return;
    }
    c0 = vk_thread_cycles();
    draw_one(st, g);
    s_draw_cycles += vk_thread_cycles() - c0;
}

static void vk_report_clear(void)
{
    if (!B.ok || g_vk.lost)
        return;
    vk_mutex_lock(B.mtx);
    B.epoch++;
    B.epoch_count[B.epoch % VKB_ZEPOCHS] = 0;
    vk_mutex_unlock(B.mtx);
}

/* The 16-byte report: time stamp, count, then 0 for "done", last. */
static void report_write(uint32_t at, uint64_t stamp, uint32_t count)
{
    uint8_t *mem = (uint8_t *)xbox_GetMemoryOffset();
    uint32_t zero = 0;
    memcpy(mem + at, &stamp, 8);
    memcpy(mem + at + 8, &count, 4);
    /* The title polls the last word from another CPU: everything before it
     * must be visible first (rq_report_write does the same). */
    vk_store_fence();
    memcpy(mem + at + 12, &zero, 4);
}

/* Answers visibility tests as the GPU finishes their draws. */
static void report_thread(void *unused)
{
    (void)unused;
    for (;;) {
        vk_event_wait(B.rep_event, 100);
        for (;;) {
            uint64_t seq;
            int i, k, done;
            vk_mutex_lock(B.mtx);
            if (!B.nrep) {
                vk_mutex_unlock(B.mtx);
                break;
            }
            seq = B.rep[B.rep_head].seq;
            vk_mutex_unlock(B.mtx);
            done = timeline_wait(seq);
            /* Every submission up to seq is done: count its queries. */
            for (i = 0; done && i < VKB_SLOTS; i++)
                if (B.slot[i].pending && B.slot[i].seq <= seq)
                    slot_complete(&B.slot[i]);
            vk_mutex_lock(B.mtx);
            k = B.rep_head;
            {
                /* A lost device answers "visible": drawing what might have
                 * been hidden beats a title waiting for ever. */
                uint64_t n = B.epoch_count[B.rep[k].epoch % VKB_ZEPOCHS];
                uint32_t c = n > 0xFFFFFFFFu ? 0xFFFFFFFFu : (uint32_t)n;
                report_write(B.rep[k].at, B.rep[k].stamp,
                             B.rep[k].force || !done ? 100000u : c);
            }
            B.rep_head = (B.rep_head + 1) % VKB_REPORTS;
            B.nrep--;
            vk_mutex_unlock(B.mtx);
        }
    }
}

/* GET_REPORT: queued, answered when the GPU gets there (as the software
 * path's asynchronous mode does: the title polls the slot's last word). */
static void vk_report(uint32_t at, uint64_t stamp, int force)
{
    int k;
    ds_report();
    if (!B.ok)
        return;
    B.nreports++;
    if (g_vk.lost) {                  /* nothing will ever count it: visible */
        report_write(at, stamp, 100000u);
        return;
    }
    submit(0);                        /* the queries it counts go to the GPU */
    for (;;) {                        /* never more than VKB_REPORTS behind */
        vk_mutex_lock(B.mtx);
        if (B.nrep < VKB_REPORTS)
            break;
        vk_mutex_unlock(B.mtx);
        vk_event_set(B.rep_event);
        vk_yield();
    }
    k = (B.rep_head + B.nrep) % VKB_REPORTS;
    B.rep[k].at = at;
    B.rep[k].stamp = stamp;
    B.rep[k].epoch = B.epoch;
    B.rep[k].force = (uint32_t)force;
    B.rep[k].seq = B.seq;
    B.nrep++;
    vk_mutex_unlock(B.mtx);
    vk_event_set(B.rep_event);
    /* Answered before GET_REPORT returns under a guest-CPU schedule (an
     * exact recording or replay: the write must land in the NV2A agent's
     * turn, as the software path's drain makes its own) and with
     * RECOMP_REPORT_SYNC=1, the software path's diagnostic. */
    {
        static int sync = -1;
        if (sync < 0)
            sync = xbox_SchedMode() || getenv("RECOMP_REPORT_SYNC") != NULL;
        while (sync) {
            int left;
            vk_mutex_lock(B.mtx);
            left = B.nrep;
            vk_mutex_unlock(B.mtx);
            if (!left)
                break;
            vk_event_set(B.rep_event);
            vk_yield();
        }
    }
}

/* ---------------------------------------------------------------------- */
/* Box-filter downsampling of a supersampled surface: each native pixel the
 * plain average of its s x s sub-pixels -- the supersampling result. Used
 * for read-backs into guest memory and for windows too small to show the
 * full-resolution image well. (vkCmdBlitImage cannot do it: a linear blit
 * from 3x lands on the middle sub-pixel and point-samples.) */

static const char k_down_vs[] =
"#version 450\n"
"void main() {\n"
"    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);\n"
"    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
"}\n";

static const char k_down_fs[] =
"#version 450\n"
"layout(set=0, binding=0) uniform sampler2D src;\n"
"layout(push_constant) uniform P { int s; } pc;\n"
"layout(location=0) out vec4 o;\n"
"void main() {\n"
"    ivec2 b = ivec2(gl_FragCoord.xy) * pc.s;\n"
"    vec4 a = vec4(0.0);\n"
"    for (int y = 0; y < pc.s; y++)\n"
"        for (int x = 0; x < pc.s; x++)\n"
"            a += texelFetch(src, b + ivec2(x, y), 0);\n"
"    o = a / float(pc.s * pc.s);\n"
"}\n";

static struct {
    int ready;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;
    VkSampler smp;
    VkShaderModule vs, fs;
    struct { VkFormat fmt; VkPipeline p; } pipe[4];
    int npipe;
    VkImageView view;                   /* of B.down */
} D;

static int down_init(void)
{
    VkDescriptorSetLayoutBinding lb;
    VkDescriptorSetLayoutCreateInfo dli = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    VkPipelineLayoutCreateInfo pli = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    VkPushConstantRange pr;
    VkSamplerCreateInfo sci = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    VkResult r_;

    if (D.ready)
        return D.ready > 0;
    D.ready = -1;
    memset(&lb, 0, sizeof lb);
    lb.binding = 0;
    lb.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    lb.descriptorCount = 1;
    lb.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    dli.bindingCount = 1;
    dli.pBindings = &lb;
    pr.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pr.offset = 0;
    pr.size = 4;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &D.dsl;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pr;
    sci.magFilter = sci.minFilter = VK_FILTER_NEAREST;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (!VK_OK(vkCreateDescriptorSetLayout(g_vk.device, &dli, NULL, &D.dsl))
        || !VK_OK(vkCreatePipelineLayout(g_vk.device, &pli, NULL, &D.layout))
        || !VK_OK(vkCreateSampler(g_vk.device, &sci, NULL, &D.smp)))
        return 0;
    D.vs = vk_shader_module(0, k_down_vs);
    D.fs = vk_shader_module(1, k_down_fs);
    if (!D.vs || !D.fs)
        return 0;
    D.ready = 1;
    return 1;
}

static VkPipeline down_pipeline(VkFormat fmt)
{
    VkGraphicsPipelineCreateInfo ci = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    VkPipelineShaderStageCreateInfo ss[2];
    VkPipelineVertexInputStateCreateInfo vis = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    VkPipelineInputAssemblyStateCreateInfo ias = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    VkPipelineViewportStateCreateInfo vps = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    VkPipelineRasterizationStateCreateInfo rs = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    VkPipelineMultisampleStateCreateInfo ms = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    VkPipelineColorBlendAttachmentState cba;
    VkPipelineColorBlendStateCreateInfo cbs = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    VkDynamicState dyn_s[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dyn = { VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    VkPipelineRenderingCreateInfo pr = { VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    VkPipeline p;
    VkResult r_;
    int i;

    for (i = 0; i < D.npipe; i++)
        if (D.pipe[i].fmt == fmt)
            return D.pipe[i].p;
    if (D.npipe == 4)
        return VK_NULL_HANDLE;
    memset(ss, 0, sizeof ss);
    ss[0].sType = ss[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ss[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    ss[0].module = D.vs;
    ss[0].pName = "main";
    ss[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    ss[1].module = D.fs;
    ss[1].pName = "main";
    ias.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    vps.viewportCount = vps.scissorCount = 1;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.lineWidth = 1.0f;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    memset(&cba, 0, sizeof cba);
    cba.colorWriteMask = 0xF;
    cbs.attachmentCount = 1;
    cbs.pAttachments = &cba;
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates = dyn_s;
    pr.colorAttachmentCount = 1;
    pr.pColorAttachmentFormats = &fmt;
    ci.pNext = &pr;
    ci.stageCount = 2;
    ci.pStages = ss;
    ci.pVertexInputState = &vis;
    ci.pInputAssemblyState = &ias;
    ci.pViewportState = &vps;
    ci.pRasterizationState = &rs;
    ci.pMultisampleState = &ms;
    ci.pColorBlendState = &cbs;
    ci.pDynamicState = &dyn;
    ci.layout = D.layout;
    if (!VK_OK(vkCreateGraphicsPipelines(g_vk.device, B.pcache, 1, &ci, NULL, &p)))
        return VK_NULL_HANDLE;
    D.pipe[D.npipe].fmt = fmt;
    D.pipe[D.npipe].p = p;
    D.npipe++;
    return p;
}

/* The top-left w x h guest pixels of colour surface s as an image at the
 * guest's resolution, in GENERAL layout: s's own image at render scale 1,
 * else the box-filtered B.down (read-backs into guest memory stay native
 * whatever the scale). */
static VkImage downsampled(Surf *s, uint32_t w, uint32_t h)
{
    VkPipeline p;
    VkDescriptorSet set;
    VkDescriptorSetAllocateInfo ai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    VkDescriptorImageInfo ii;
    VkWriteDescriptorSet wr = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    VkRenderingInfo ri = { VK_STRUCTURE_TYPE_RENDERING_INFO };
    VkRenderingAttachmentInfo ca = { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    VkViewport vp;
    VkRect2D sc;
    VkCommandBuffer cb;
    int32_t scale = (int32_t)s->sc;
    VkResult r_;

    if (s->sc <= 1 || g_vk.lost || !down_init())
        return s->img;
    if (!B.down || B.down_w < w || B.down_h < h || B.down_fmt != s->fmt) {
        VkImageMemoryBarrier ib = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        VkImageViewCreateInfo vi = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        if (B.down) {
            wait_idle();
            vkDestroyImageView(g_vk.device, D.view, NULL);
            vkDestroyImage(g_vk.device, B.down, NULL);
            vkFreeMemory(g_vk.device, B.downmem, NULL);
            B.down = VK_NULL_HANDLE;
        }
        if (!vk_image_create(w, h, s->fmt, VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                             | VK_IMAGE_USAGE_TRANSFER_DST_BIT
                             | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &B.down, &B.downmem))
            return s->img;
        vi.image = B.down;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = s->fmt;
        vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vi.subresourceRange.levelCount = 1;
        vi.subresourceRange.layerCount = 1;
        if (!VK_OK(vkCreateImageView(g_vk.device, &vi, NULL, &D.view))) {
            vkDestroyImage(g_vk.device, B.down, NULL);
            vkFreeMemory(g_vk.device, B.downmem, NULL);
            B.down = VK_NULL_HANDLE;
            return s->img;
        }
        B.down_w = w;
        B.down_h = h;
        B.down_fmt = s->fmt;
        end_rendering();
        ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ib.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        ib.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ib.image = B.down;
        ib.subresourceRange = vi.subresourceRange;
        vkCmdPipelineBarrier(cmd(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL,
                             0, NULL, 1, &ib);
    }
    p = down_pipeline(s->fmt);
    if (!p)
        return s->img;
    end_rendering();
    full_barrier();                       /* the surface's writes, visible to sampling */
    cb = cmd();
    ai.descriptorPool = B.slot[B.cur].dpool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &D.dsl;
    if (vkAllocateDescriptorSets(g_vk.device, &ai, &set) != VK_SUCCESS) {
        submit(0);                        /* the slot's pool is full: the next one */
        cb = cmd();
        full_barrier();
        ai.descriptorPool = B.slot[B.cur].dpool;
        if (!VK_OK(vkAllocateDescriptorSets(g_vk.device, &ai, &set)))
            return s->img;
    }
    ii.sampler = D.smp;
    ii.imageView = s->view;
    ii.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    wr.dstSet = set;
    wr.dstBinding = 0;
    wr.descriptorCount = 1;
    wr.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wr.pImageInfo = &ii;
    vkUpdateDescriptorSets(g_vk.device, 1, &wr, 0, NULL);
    ca.imageView = D.view;
    ca.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    ca.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    ri.renderArea.extent.width = w;
    ri.renderArea.extent.height = h;
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &ca;
    vkCmdBeginRendering(cb, &ri);
    B.bound.gen = 0;                     /* a draw binds and sets everything again */
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, D.layout, 0, 1, &set, 0, NULL);
    vkCmdPushConstants(cb, D.layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4, &scale);
    vp.x = vp.y = 0.0f;
    vp.width = (float)w;
    vp.height = (float)h;
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;
    vkCmdSetViewport(cb, 0, 1, &vp);
    sc.offset.x = sc.offset.y = 0;
    sc.extent.width = w;
    sc.extent.height = h;
    vkCmdSetScissor(cb, 0, 1, &sc);
    vkCmdDraw(cb, 3, 1, 0, 0);
    vkCmdEndRendering(cb);
    mb.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);
    B.pending_barrier = 1;
    return B.down;
}

/* The colour surface at `addr` read back into r->map (rows of w * bpp
 * bytes): recorded and submitted; with `wait`, also finished and visible
 * to the CPU on return. 0 if there is no such surface. */
static int readback_into(Readback *r, uint32_t addr, uint32_t bpp, uint32_t *pw,
                         uint32_t *ph, int wait)
{
    Surf *s = NULL;
    int i;
    VkBufferImageCopy c;
    VkDeviceSize need;
    uint32_t w = *pw, h = *ph;

    for (i = 0; i < VKB_SURFS; i++)
        if (B.surf[i].img && !B.surf[i].depth && B.surf[i].addr == addr
            && B.surf[i].bpp == bpp)
            s = &B.surf[i];
    if (!s)
        return 0;
    if (w > s->w) w = s->w;
    if (h > s->h) h = s->h;
    need = (VkDeviceSize)w * h * bpp;
    if (need > r->size) {
        if (g_vk.lost)
            return 0;
        wait_idle();
        if (r->buf) {
            vkDestroyBuffer(g_vk.device, r->buf, NULL);
            vkFreeMemory(g_vk.device, r->mem, NULL);
        }
        if (!vk_buffer_create(need, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                              | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                              &r->buf, &r->mem, (void **)&r->map)
            && !vk_buffer_create(need, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                 | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                 &r->buf, &r->mem, (void **)&r->map)) {
            memset(r, 0, sizeof *r);
            return 0;
        }
        r->size = need;
    }
    full_barrier();
    memset(&c, 0, sizeof c);
    c.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    c.imageSubresource.layerCount = 1;
    c.imageExtent.width = w;
    c.imageExtent.height = h;
    c.imageExtent.depth = 1;
    vkCmdCopyImageToBuffer(cmd(), downsampled(s, w, h), VK_IMAGE_LAYOUT_GENERAL, r->buf,
                           1, &c);
    {
        VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd(), VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    }
    submit(wait);
    r->seq = B.seq;
    r->w = w;
    r->h = h;
    if (wait) {
        VkMappedMemoryRange mr = { VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE };
        mr.memory = r->mem;
        mr.size = VK_WHOLE_SIZE;
        vkInvalidateMappedMemoryRanges(g_vk.device, 1, &mr);
    }
    *pw = w;
    *ph = h;
    return 1;
}

static int readback(uint32_t addr, uint32_t bpp, uint32_t *pw, uint32_t *ph)
{
    return readback_into(&B.rb, addr, bpp, pw, ph, 1);
}

/* An asynchronous read-back finished: wait for its submission (normally
 * long done) and make it visible to the CPU. 0 if it never will. */
static int readback_finish(Readback *r)
{
    VkMappedMemoryRange mr = { VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE };
    if (!timeline_wait(r->seq))
        return 0;
    mr.memory = r->mem;
    mr.size = VK_WHOLE_SIZE;
    vkInvalidateMappedMemoryRanges(g_vk.device, 1, &mr);
    return 1;
}

static void frame_stats(void)
{
    if (getenv("RECOMP_VK_STATS") && B.frame % 60 == 0) {
        unsigned compiled, cached;
        double glsl_ms;
        static uint64_t last;
        uint64_t now = vk_thread_cycles();
        vk_shader_stats(&compiled, &cached, &glsl_ms);
        if (last)
            fprintf(stderr, "[VKCPU] frame %u: executor %.1f Mcycles a frame, %.1f in draws"
                    " (%.2f kcycles a draw)\n", B.frame, (double)(now - last) / 60e6,
                    (double)s_draw_cycles / 60e6,
                    B.draws ? (double)s_draw_cycles / 1e3 / B.draws : 0.0);
        last = now;
        s_draw_cycles = 0;
        fprintf(stderr, "[VKSTATS] frame %u: %u draws, %u tris, %u uploads, %u submits,"
                " %u new pipelines (%d kept, %.0f ms), %u new vertex shaders,"
                " GLSL %u compiled (%.0f ms) %u from the cache, %u counted draws,"
                " %u visibility reports (last 60 frames)\n",
                B.frame, B.draws, B.tris, B.uploads, B.submits, B.new_pipes, B.npipes,
                B.pipe_ms, B.new_vsh, compiled, glsl_ms, cached, B.zdraws, B.nreports);
        B.pipe_ms = 0;
        B.draws = B.tris = B.uploads = B.submits = B.new_pipes = B.new_vsh = 0;
        B.zdraws = B.nreports = 0;
    }
}

/* ---------------------------------------------------------------------- */
/* Settings (xbox_vk_settings_*): applied between frames, at a flip. */

static volatile uint32_t s_post_aa_req, s_post_aa;

/* The menu's Graphics options (src/ui settings: shown in the menu, kept in
 * blinx2_settings.ini, an environment variable wins), or the environment
 * alone where the settings store is not linked (standalone tests). Values
 * are option indices: supersampling {1, 3, 5}, AA {off, fxaa}. */
static const char *const k_scale_names[] = { "1", "3", "5" };
static const char *const k_aa_names[] = { "off", "fxaa" };
static int s_set_scale = -1, s_set_aa = -1;  /* last values read */

static uint32_t odd_scale(uint32_t sc);

static void settings_declare(void)
{
    int i, a;
#if defined(XBOXRECOMP_HAVE_SETTINGS)
    i = xbox_setting_choice("graphics", "vk_scale", "Supersampling (Vulkan)", k_scale_names,
                            3, 0, "RECOMP_VK_SCALE", 0);
    a = xbox_setting_choice("graphics", "vk_aa", "Anti-aliasing (Vulkan)", k_aa_names,
                            2, 0, "RECOMP_VK_AA", 0);
#else
    const char *e = getenv("RECOMP_VK_SCALE");
    int sc = e ? atoi(e) : 1;
    i = (int)(odd_scale(sc < 1 ? 1u : (uint32_t)sc) / 2);
    e = getenv("RECOMP_VK_AA");
    a = e && !strcmp(e, "fxaa");
#endif
    s_set_scale = i < 0 ? 0 : i > 2 ? 2 : i;
    s_set_aa = a != 0;
    B.scale = B.scale_req = 1u + 2u * (uint32_t)s_set_scale;
    s_post_aa = s_post_aa_req = (uint32_t)s_set_aa;
    if (B.scale > 1 || s_post_aa)
        fprintf(stderr, "  [VK] supersampling %ux, FXAA %s\n", B.scale, s_post_aa ? "on" : "off");
}

/* At each flip: a value the menu changed becomes a request (edge-triggered,
 * so xbox_vk_settings_set still works where nothing changes in the menu). */
static void settings_poll(void)
{
#if defined(XBOXRECOMP_HAVE_SETTINGS)
    int i = xbox_setting_get("graphics", "vk_scale", s_set_scale);
    int a = xbox_setting_get("graphics", "vk_aa", s_set_aa);
    if (i != s_set_scale && i >= 0 && i <= 2) {
        s_set_scale = i;
        B.scale_req = 1u + 2u * (uint32_t)i;
    }
    if (a != s_set_aa) {
        s_set_aa = a;
        s_post_aa_req = a != 0;
    }
#endif
}

/* Render scales are odd, 1 to 5: an even request goes to the next odd one. */
static uint32_t odd_scale(uint32_t sc)
{
    if (sc < 1)
        sc = 1;
    if (sc > 5)
        sc = 5;
    return sc | 1u;
}

/* Every surface again at render scale B.scale_req, its colour carried over
 * (filtered); depth starts cleared (the title clears it every frame). Visibility
 * counts stay native at any scale (the counting pass), so nothing the guest
 * sees changes. */
static void apply_settings(void)
{
    Surf old[VKB_SURFS];
    int i, n = 0;
    settings_poll();
    if (s_post_aa != s_post_aa_req) {
        s_post_aa = s_post_aa_req;
        fprintf(stderr, "  [VK] post-process AA: %s\n", s_post_aa ? "FXAA" : "off");
    }
    if (B.scale_req == B.scale)
        return;
    end_rendering();
    wait_idle();
    B.scale = B.scale_req;
    B.rt = B.zt = NULL;
    for (i = 0; i < VKB_SURFS; i++) {
        Surf *s = &B.surf[i], fresh;
        if (!s->img)
            continue;
        memset(&fresh, 0, sizeof fresh);
        fresh.addr = s->addr;
        fresh.depth = s->depth;
        fresh.bpp = s->bpp;
        fresh.fmt = s->fmt;
        fresh.used = s->used;
        if (!surf_create(&fresh, s->w, s->h)) {
            surf_destroy(&fresh);
            continue;                     /* keeps the old one */
        }
        if (!s->depth) {
            VkImageBlit b;
            full_barrier();
            memset(&b, 0, sizeof b);
            b.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            b.srcSubresource.layerCount = 1;
            b.dstSubresource = b.srcSubresource;
            b.srcOffsets[1].x = (int32_t)(s->w * s->sc);
            b.srcOffsets[1].y = (int32_t)(s->h * s->sc);
            b.srcOffsets[1].z = 1;
            b.dstOffsets[1].x = (int32_t)(fresh.w * fresh.sc);
            b.dstOffsets[1].y = (int32_t)(fresh.h * fresh.sc);
            b.dstOffsets[1].z = 1;
            vkCmdBlitImage(cmd(), s->img, VK_IMAGE_LAYOUT_GENERAL, fresh.img,
                           VK_IMAGE_LAYOUT_GENERAL, 1, &b, VK_FILTER_LINEAR);
        }
        fresh.sig = s->sig;
        fresh.sig_frame = s->sig_frame;
        fresh.gbytes = s->gbytes;
        old[n++] = *s;
        *s = fresh;
    }
    B.pending_barrier = 1;
    wait_idle();
    for (i = 0; i < n; i++)
        surf_destroy(&old[i]);
    fprintf(stderr, "  [VK] render scale %ux (%u surfaces rescaled)\n", B.scale, (unsigned)n);
}

void xbox_vk_settings_get(XboxVkSettings *out)
{
    memset(out, 0, sizeof *out);
    out->render_scale = B.scale_req ? B.scale_req : 1;
    out->post_aa = s_post_aa_req;
}

void xbox_vk_settings_set(const XboxVkSettings *in)
{
    B.scale_req = odd_scale(in->render_scale);
    s_post_aa_req = in->post_aa ? 1u : 0u;
}

/* ---------------------------------------------------------------------- */
/* Post-process anti-aliasing (RECOMP_VK_AA=fxaa, xbox_vk_settings): an
 * FXAA-style edge filter, written from the published description of the
 * technique (luma contrast test, edge direction from the 3x3 luma
 * neighbourhood, a 2- and 4-tap blend along it, the 4-tap one rejected if it
 * leaves the local luma range). Display only: it filters the image being
 * presented, never anything read back into guest memory. */

static const char k_fx_fs[] =
"#version 450\n"
"layout(set=0, binding=0) uniform sampler2D src;\n"
"layout(location=0) out vec4 o;\n"
"float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }\n"
"vec3 at(vec2 p) { return texture(src, p).rgb; }\n"
"void main() {\n"
"    vec2 t = 1.0 / vec2(textureSize(src, 0));\n"
"    vec2 uv = gl_FragCoord.xy * t;\n"
"    vec3 cM = at(uv);\n"
"    float lM = luma(cM);\n"
"    float lNW = luma(at(uv + vec2(-1.0, -1.0) * t));\n"
"    float lNE = luma(at(uv + vec2( 1.0, -1.0) * t));\n"
"    float lSW = luma(at(uv + vec2(-1.0,  1.0) * t));\n"
"    float lSE = luma(at(uv + vec2( 1.0,  1.0) * t));\n"
"    float lMin = min(lM, min(min(lNW, lNE), min(lSW, lSE)));\n"
"    float lMax = max(lM, max(max(lNW, lNE), max(lSW, lSE)));\n"
"    if (lMax - lMin < max(0.0312, lMax * 0.125)) { o = vec4(cM, 1.0); return; }\n"
"    vec2 dir = vec2(-((lNW + lNE) - (lSW + lSE)), (lNW + lSW) - (lNE + lSE));\n"
"    float reduce = max((lNW + lNE + lSW + lSE) * 0.03125, 1.0 / 128.0);\n"
"    float k = 1.0 / (min(abs(dir.x), abs(dir.y)) + reduce);\n"
"    dir = clamp(dir * k, vec2(-8.0), vec2(8.0)) * t;\n"
"    vec3 a = 0.5 * (at(uv + dir * (1.0 / 3.0 - 0.5)) + at(uv + dir * (2.0 / 3.0 - 0.5)));\n"
"    vec3 b = a * 0.5 + 0.25 * (at(uv - dir * 0.5) + at(uv + dir * 0.5));\n"
"    float lB = luma(b);\n"
"    o = vec4((lB < lMin || lB > lMax) ? a : b, 1.0);\n"
"}\n";

static struct {
    int ready;
    VkShaderModule fs;
    VkPipeline p;
    VkSampler smp;
    VkImage img;
    VkDeviceMemory mem;
    VkImageView view;
    uint32_t w, h;
} FX;

/* The FXAA'd copy (B8G8R8A8, w x h) of view `src` (whose top-left w x h is
 * the frame), or VK_NULL_HANDLE. */
static VkImage fx_pass(VkImageView src, uint32_t w, uint32_t h)
{
    VkDescriptorSet set;
    VkDescriptorSetAllocateInfo ai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    VkDescriptorImageInfo ii;
    VkWriteDescriptorSet wr = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    VkRenderingInfo ri = { VK_STRUCTURE_TYPE_RENDERING_INFO };
    VkRenderingAttachmentInfo ca = { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    VkViewport vp;
    VkRect2D sc;
    VkCommandBuffer cb;
    VkResult r_;
    static const VkFormat fmt = VK_FORMAT_B8G8R8A8_UNORM;

    if (!down_init())
        return VK_NULL_HANDLE;
    if (!FX.ready) {
        VkSamplerCreateInfo sci = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        VkGraphicsPipelineCreateInfo ci = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
        VkPipelineShaderStageCreateInfo ss[2];
        VkPipelineVertexInputStateCreateInfo vis = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
        VkPipelineInputAssemblyStateCreateInfo ias = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
        VkPipelineViewportStateCreateInfo vps = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
        VkPipelineRasterizationStateCreateInfo rs = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
        VkPipelineMultisampleStateCreateInfo ms = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
        VkPipelineColorBlendAttachmentState cba;
        VkPipelineColorBlendStateCreateInfo cbs = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
        VkDynamicState dyn_s[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dyn = { VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
        VkPipelineRenderingCreateInfo pr = { VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
        FX.ready = -1;
        FX.fs = vk_shader_module(1, k_fx_fs);
        sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
        sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (!FX.fs || !VK_OK(vkCreateSampler(g_vk.device, &sci, NULL, &FX.smp)))
            return VK_NULL_HANDLE;
        memset(ss, 0, sizeof ss);
        ss[0].sType = ss[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ss[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        ss[0].module = D.vs;
        ss[0].pName = "main";
        ss[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        ss[1].module = FX.fs;
        ss[1].pName = "main";
        ias.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        vps.viewportCount = vps.scissorCount = 1;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.lineWidth = 1.0f;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        memset(&cba, 0, sizeof cba);
        cba.colorWriteMask = 0xF;
        cbs.attachmentCount = 1;
        cbs.pAttachments = &cba;
        dyn.dynamicStateCount = 2;
        dyn.pDynamicStates = dyn_s;
        pr.colorAttachmentCount = 1;
        pr.pColorAttachmentFormats = &fmt;
        ci.pNext = &pr;
        ci.stageCount = 2;
        ci.pStages = ss;
        ci.pVertexInputState = &vis;
        ci.pInputAssemblyState = &ias;
        ci.pViewportState = &vps;
        ci.pRasterizationState = &rs;
        ci.pMultisampleState = &ms;
        ci.pColorBlendState = &cbs;
        ci.pDynamicState = &dyn;
        ci.layout = D.layout;              /* one sampler binding, as the downsample */
        if (!VK_OK(vkCreateGraphicsPipelines(g_vk.device, B.pcache, 1, &ci, NULL, &FX.p)))
            return VK_NULL_HANDLE;
        FX.ready = 1;
    }
    if (FX.ready < 0)
        return VK_NULL_HANDLE;
    if (g_vk.lost)
        return VK_NULL_HANDLE;
    if (!FX.img || FX.w != w || FX.h != h) {
        VkImageMemoryBarrier ib = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        VkImageViewCreateInfo vi = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        if (FX.img) {
            wait_idle();
            vkDestroyImageView(g_vk.device, FX.view, NULL);
            vkDestroyImage(g_vk.device, FX.img, NULL);
            vkFreeMemory(g_vk.device, FX.mem, NULL);
            FX.img = VK_NULL_HANDLE;
        }
        if (!vk_image_create(w, h, fmt, VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                             | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &FX.img, &FX.mem))
            return VK_NULL_HANDLE;
        vi.image = FX.img;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = fmt;
        vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vi.subresourceRange.levelCount = 1;
        vi.subresourceRange.layerCount = 1;
        if (!VK_OK(vkCreateImageView(g_vk.device, &vi, NULL, &FX.view)))
            return VK_NULL_HANDLE;
        FX.w = w;
        FX.h = h;
        end_rendering();
        ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ib.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        ib.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ib.image = FX.img;
        ib.subresourceRange = vi.subresourceRange;
        vkCmdPipelineBarrier(cmd(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL,
                             0, NULL, 1, &ib);
    }
    end_rendering();
    full_barrier();
    cb = cmd();
    ai.descriptorPool = B.slot[B.cur].dpool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &D.dsl;
    if (vkAllocateDescriptorSets(g_vk.device, &ai, &set) != VK_SUCCESS) {
        submit(0);
        cb = cmd();
        full_barrier();
        ai.descriptorPool = B.slot[B.cur].dpool;
        if (!VK_OK(vkAllocateDescriptorSets(g_vk.device, &ai, &set)))
            return VK_NULL_HANDLE;
    }
    ii.sampler = FX.smp;
    ii.imageView = src;
    ii.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    wr.dstSet = set;
    wr.dstBinding = 0;
    wr.descriptorCount = 1;
    wr.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wr.pImageInfo = &ii;
    vkUpdateDescriptorSets(g_vk.device, 1, &wr, 0, NULL);
    ca.imageView = FX.view;
    ca.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    ca.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    ri.renderArea.extent.width = w;
    ri.renderArea.extent.height = h;
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &ca;
    vkCmdBeginRendering(cb, &ri);
    B.bound.gen = 0;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, FX.p);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, D.layout, 0, 1, &set, 0, NULL);
    vp.x = vp.y = 0.0f;
    vp.width = (float)w;
    vp.height = (float)h;
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;
    vkCmdSetViewport(cb, 0, 1, &vp);
    sc.offset.x = sc.offset.y = 0;
    sc.extent.width = w;
    sc.extent.height = h;
    vkCmdSetScissor(cb, 0, 1, &sc);
    vkCmdDraw(cb, 3, 1, 0, 0);
    vkCmdEndRendering(cb);
    mb.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    B.pending_barrier = 1;
    return FX.img;
}

/* ---------------------------------------------------------------------- */
/* Area-filtered resize of a supersampled frame to the window's size: each
 * window pixel averages its whole footprint in the image (N x N bilinear
 * taps, N = 2 * ceil(ratio), at most 8). A plain bilinear blit from 1920 to
 * 1280 lines (3x in a 2x window) skips texels and turns thin lines -- the
 * rain -- into dots. */

static const char k_area_fs[] =
"#version 450\n"
"layout(set=0, binding=0) uniform sampler2D src;\n"
"layout(push_constant) uniform P { vec2 region; vec2 dst; } pc;\n"
"layout(location=0) out vec4 o;\n"
"void main() {\n"
"    vec2 tex = vec2(textureSize(src, 0));\n"
"    vec2 r = pc.region / pc.dst;\n"
"    vec2 x0 = (gl_FragCoord.xy - 0.5) * r;\n"
"    int n = int(min(8.0, 2.0 * ceil(max(r.x, r.y))));\n"
"    vec4 a = vec4(0.0);\n"
"    for (int j = 0; j < n; j++)\n"
"        for (int i = 0; i < n; i++)\n"
"            a += texture(src, (x0 + (vec2(i, j) + 0.5) / float(n) * r) / tex);\n"
"    o = a / float(n * n);\n"
"}\n";

static struct {
    int ready;
    VkPipelineLayout layout;
    VkPipeline p;
    VkSampler smp;
    VkImage img;
    VkDeviceMemory mem;
    VkImageView view;
    uint32_t w, h;
} AR;

/* The w x h guest-pixel frame of colour surface s, area-resized to dw x dh
 * (B8G8R8A8, GENERAL layout), or VK_NULL_HANDLE. */
static VkImage area_resize(Surf *s, uint32_t w, uint32_t h, uint32_t dw, uint32_t dh,
                           VkImageView *out_view)
{
    VkDescriptorSet set;
    VkDescriptorSetAllocateInfo ai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    VkDescriptorImageInfo ii;
    VkWriteDescriptorSet wr = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    VkRenderingInfo ri = { VK_STRUCTURE_TYPE_RENDERING_INFO };
    VkRenderingAttachmentInfo ca = { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    VkViewport vp;
    VkRect2D sc;
    VkCommandBuffer cb;
    float pc[4];
    VkResult r_;
    static const VkFormat fmt = VK_FORMAT_B8G8R8A8_UNORM;

    if (!down_init())
        return VK_NULL_HANDLE;
    if (!AR.ready) {
        VkPipelineLayoutCreateInfo pli = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        VkPushConstantRange pr;
        VkGraphicsPipelineCreateInfo ci = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
        VkPipelineShaderStageCreateInfo ss[2];
        VkPipelineVertexInputStateCreateInfo vis = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
        VkPipelineInputAssemblyStateCreateInfo ias = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
        VkPipelineViewportStateCreateInfo vps = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
        VkPipelineRasterizationStateCreateInfo rs = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
        VkPipelineMultisampleStateCreateInfo ms = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
        VkPipelineColorBlendAttachmentState cba;
        VkPipelineColorBlendStateCreateInfo cbs = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
        VkDynamicState dyn_s[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dyn = { VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
        VkPipelineRenderingCreateInfo prc = { VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
        VkShaderModule fs;
        AR.ready = -1;
        pr.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pr.offset = 0;
        pr.size = 16;
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &D.dsl;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pr;
        fs = vk_shader_module(1, k_area_fs);
        {
            VkSamplerCreateInfo sci = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
            sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
            sci.addressModeU = sci.addressModeV = sci.addressModeW =
                VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            if (!VK_OK(vkCreateSampler(g_vk.device, &sci, NULL, &AR.smp)))
                return VK_NULL_HANDLE;
        }
        if (!fs || !VK_OK(vkCreatePipelineLayout(g_vk.device, &pli, NULL, &AR.layout)))
            return VK_NULL_HANDLE;
        memset(ss, 0, sizeof ss);
        ss[0].sType = ss[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ss[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        ss[0].module = D.vs;
        ss[0].pName = "main";
        ss[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        ss[1].module = fs;
        ss[1].pName = "main";
        ias.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        vps.viewportCount = vps.scissorCount = 1;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.lineWidth = 1.0f;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        memset(&cba, 0, sizeof cba);
        cba.colorWriteMask = 0xF;
        cbs.attachmentCount = 1;
        cbs.pAttachments = &cba;
        dyn.dynamicStateCount = 2;
        dyn.pDynamicStates = dyn_s;
        prc.colorAttachmentCount = 1;
        prc.pColorAttachmentFormats = &fmt;
        ci.pNext = &prc;
        ci.stageCount = 2;
        ci.pStages = ss;
        ci.pVertexInputState = &vis;
        ci.pInputAssemblyState = &ias;
        ci.pViewportState = &vps;
        ci.pRasterizationState = &rs;
        ci.pMultisampleState = &ms;
        ci.pColorBlendState = &cbs;
        ci.pDynamicState = &dyn;
        ci.layout = AR.layout;
        if (!VK_OK(vkCreateGraphicsPipelines(g_vk.device, B.pcache, 1, &ci, NULL, &AR.p)))
            return VK_NULL_HANDLE;
        AR.ready = 1;
    }
    if (AR.ready < 0)
        return VK_NULL_HANDLE;
    if (g_vk.lost)
        return VK_NULL_HANDLE;
    if (!AR.img || AR.w != dw || AR.h != dh) {
        VkImageMemoryBarrier ib = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        VkImageViewCreateInfo vi = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        if (AR.img) {
            wait_idle();
            vkDestroyImageView(g_vk.device, AR.view, NULL);
            vkDestroyImage(g_vk.device, AR.img, NULL);
            vkFreeMemory(g_vk.device, AR.mem, NULL);
            AR.img = VK_NULL_HANDLE;
        }
        if (!vk_image_create(dw, dh, fmt, VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                             | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                             | VK_IMAGE_USAGE_SAMPLED_BIT, &AR.img, &AR.mem))
            return VK_NULL_HANDLE;
        vi.image = AR.img;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = fmt;
        vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vi.subresourceRange.levelCount = 1;
        vi.subresourceRange.layerCount = 1;
        if (!VK_OK(vkCreateImageView(g_vk.device, &vi, NULL, &AR.view)))
            return VK_NULL_HANDLE;
        AR.w = dw;
        AR.h = dh;
        end_rendering();
        ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ib.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        ib.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ib.image = AR.img;
        ib.subresourceRange = vi.subresourceRange;
        vkCmdPipelineBarrier(cmd(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL,
                             0, NULL, 1, &ib);
    }
    end_rendering();
    full_barrier();
    cb = cmd();
    ai.descriptorPool = B.slot[B.cur].dpool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &D.dsl;
    if (vkAllocateDescriptorSets(g_vk.device, &ai, &set) != VK_SUCCESS) {
        submit(0);
        cb = cmd();
        full_barrier();
        ai.descriptorPool = B.slot[B.cur].dpool;
        if (!VK_OK(vkAllocateDescriptorSets(g_vk.device, &ai, &set)))
            return VK_NULL_HANDLE;
    }
    ii.sampler = AR.smp;
    ii.imageView = s->view;
    ii.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    wr.dstSet = set;
    wr.dstBinding = 0;
    wr.descriptorCount = 1;
    wr.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wr.pImageInfo = &ii;
    vkUpdateDescriptorSets(g_vk.device, 1, &wr, 0, NULL);
    ca.imageView = AR.view;
    ca.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    ca.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    ri.renderArea.extent.width = dw;
    ri.renderArea.extent.height = dh;
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &ca;
    vkCmdBeginRendering(cb, &ri);
    B.bound.gen = 0;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, AR.p);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, AR.layout, 0, 1, &set, 0, NULL);
    pc[0] = (float)(w * s->sc);
    pc[1] = (float)(h * s->sc);
    pc[2] = (float)dw;
    pc[3] = (float)dh;
    vkCmdPushConstants(cb, AR.layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16, pc);
    vp.x = vp.y = 0.0f;
    vp.width = (float)dw;
    vp.height = (float)dh;
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;
    vkCmdSetViewport(cb, 0, 1, &vp);
    sc.offset.x = sc.offset.y = 0;
    sc.extent.width = dw;
    sc.extent.height = dh;
    vkCmdSetScissor(cb, 0, 1, &sc);
    vkCmdDraw(cb, 3, 1, 0, 0);
    vkCmdEndRendering(cb);
    mb.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);
    B.pending_barrier = 1;
    *out_view = AR.view;
    return AR.img;
}

/* The finished frame (the colour surface at `addr`, its top-left w x h) to
 * the window's Vulkan swapchain, when there is one. */
static void flip_present(uint32_t addr, uint32_t w, uint32_t h)
{
    int i;
    if (!vk_present_ready())
        return;
    for (i = 0; i < VKB_SURFS; i++) {
        Surf *s = &B.surf[i];
        if (s->img && !s->depth && s->addr == addr) {
            uint32_t fw = w < s->w ? w : s->w, fh = h < s->h ? h : s->h, ww, wh;
            VkImage img = s->img;
            VkImageView view = s->view;
            uint32_t pw = fw * s->sc, ph = fh * s->sc;
            /* Supersampled, in a window that shows the frame smaller than
             * the image: area-filtered to exactly the size it is shown at
             * (a blit would skip texels); otherwise the full image. */
            if (s->sc > 1 && vk_present_extent(&ww, &wh)) {
                uint32_t dw, dh;
                if ((uint64_t)ww * ph >= (uint64_t)wh * pw) {
                    dh = wh;
                    dw = (uint32_t)((uint64_t)wh * pw / ph);
                } else {
                    dw = ww;
                    dh = (uint32_t)((uint64_t)ww * ph / pw);
                }
                if (dw && dh && dh < ph) {
                    VkImageView v;
                    VkImage r = area_resize(s, fw, fh, dw, dh, &v);
                    if (r) {
                        img = r;
                        view = v;
                        pw = dw;
                        ph = dh;
                    }
                }
            }
            if (s_post_aa) {                /* FXAA, display only */
                VkImage fx = fx_pass(view, pw, ph);
                if (fx)
                    img = fx;
            }
            submit(0);
            vk_present_image(img, pw, ph);
            return;
        }
    }
}

/* We wrote the surface at `addr` into guest memory ourselves: that is not
 * the CPU rewriting it (surf_sig). */
static void surf_resign(uint32_t addr)
{
    int i;
    for (i = 0; i < VKB_SURFS; i++)
        if (B.surf[i].img && !B.surf[i].depth && B.surf[i].addr == addr && B.surf[i].sig_frame)
            B.surf[i].sig = surf_sig(&B.surf[i]);
}

static void flip_frame(uint32_t addr, uint32_t pitch, uint32_t bpp, uint32_t w, uint32_t h)
{
    uint8_t *mem = (uint8_t *)xbox_GetMemoryOffset();
    uint32_t y, rowb;

    /* Who needs the frame in guest memory, and how soon:
     *  2  exactly this frame, now (RECOMP_VK_WRITEBACK=1, frame dumps): the
     *     flip waits for the GPU;
     *  1  the framebuffer window: frame N is copied out at its flip and
     *     written into guest memory at flip N+1 -- into the buffer being
     *     presented then, so the window shows every frame, one late, and
     *     the flip never waits for the GPU;
     *  0  nobody (headless, no dumps): no read-back at all.
     * Nothing in the title is known to read its own front buffer.
     * RECOMP_VK_READBACK=0|1|2 picks one regardless (testing). */
    static int mode = -1;
    Readback *r, *prev;

    if (!B.ok || g_vk.lost)
        return;
    if (mode < 0) {
        const char *e = getenv("RECOMP_VK_READBACK");
        if (e && *e >= '0' && *e <= '2')
            mode = *e - '0';
        else if (getenv("RECOMP_VK_WRITEBACK") || getenv("RECOMP_FB_DUMP")
                 || getenv("RECOMP_FB_DUMP_FLIPS"))
            mode = 2;
        else
            mode = getenv("RECOMP_HEADLESS") ? 0 : 1;
    }
    B.frame++;
    /* With Vulkan presentation the window shows this image itself (at the
     * renderer's resolution): no read-back is needed for it. */
    if (!pitch || mode == 0 || (mode == 1 && vk_present_ready())) {
        submit(0);
        flip_present(addr, w, h);
        frame_stats();
        return;
    }
    if (mode == 2) {
        if (!readback(addr, bpp, &w, &h)) {
            submit(0);
            return;
        }
        rowb = w * bpp;
        for (y = 0; y < h; y++)
            memcpy(mem + addr + (size_t)y * pitch, B.rb.map + (size_t)y * rowb, rowb);
        surf_resign(addr);
        flip_present(addr, w, h);
        frame_stats();
        return;
    }
    r = &B.arb[B.arbi];
    prev = &B.arb[B.arbi ^ 1];
    if (readback_into(r, addr, bpp, &w, &h, 0)) {
        r->pending = 1;
        B.arbi ^= 1;
    } else {
        submit(0);
    }
    if (prev->pending) {
        uint32_t pw = prev->w < w ? prev->w : w, ph = prev->h < h ? prev->h : h;
        int ready = readback_finish(prev);
        rowb = prev->w * bpp;
        for (y = 0; ready && y < ph; y++)
            memcpy(mem + addr + (size_t)y * pitch, prev->map + (size_t)y * rowb,
                   (size_t)pw * bpp);
        prev->pending = 0;
        surf_resign(addr);
    }
    frame_stats();
}

/* RECOMP_VK_COMPARE: this frame against the software path's, already in
 * guest memory. A pixel differs when any channel is off by more than
 * RECOMP_VK_COMPARE_TOL (default 8). One [VKCMP] line per frame that has
 * differing pixels, and a summary every 300 flips; with
 * RECOMP_VK_COMPARE_DUMP=<prefix>, the worst frames and a sample of the rest
 * (up to 60) are written as
 * <prefix>NNNNN_{vk,sw,diff}.bmp (RECOMP_VK_COMPARE_DUMP_FROM=<flip>: none
 * before that flip); with RECOMP_VK_COMPARE_DUMP_MIN=<percent>,
 * only frames with at least that share of differing pixels, the first 60. */
static void write_bmp32(const char *path, const uint32_t *px, uint32_t w, uint32_t h)
{
    FILE *f = fopen(path, "wb");
    uint8_t hdr[54];
    uint32_t img = w * h * 4, total = 54 + img, y;
    int32_t sh = -(int32_t)h;
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
    for (y = 0; y < h; y++)
        fwrite(px + (size_t)y * w, 4, w, f);
    fclose(f);
}

static void compare_frame(uint32_t addr, uint32_t pitch, uint32_t bpp, uint32_t w, uint32_t h,
                       uint32_t flip)
{
    static int tol = -1, dumps;
    static const char *dump;
    static uint64_t frames, bad_frames, px_total, px_bad;
    static double worst = 0.0, dump_min = -1.0;
    static long dump_from;
    const uint8_t *mem = (const uint8_t *)xbox_GetMemoryOffset();
    uint32_t x, y, maxd = 0, nbad = 0;
    uint64_t sumd = 0;

    if (!B.ok || g_vk.lost)
        return;
    if (tol < 0) {
        const char *t = getenv("RECOMP_VK_COMPARE_TOL");
        tol = t ? atoi(t) : 8;
        dump = getenv("RECOMP_VK_COMPARE_DUMP");
        t = getenv("RECOMP_VK_COMPARE_DUMP_MIN");
        dump_min = t ? atof(t) / 100.0 : -1.0;
        t = getenv("RECOMP_VK_COMPARE_DUMP_FROM");
        dump_from = t ? atol(t) : 0;
    }
    B.frame++;
    if (bpp != 4 || !pitch || !readback(addr, bpp, &w, &h)) {
        submit(0);
        return;
    }
    for (y = 0; y < h; y++) {
        const uint32_t *vk = (const uint32_t *)(B.rb.map + (size_t)y * w * 4);
        const uint32_t *sw = (const uint32_t *)(mem + addr + (size_t)y * pitch);
        for (x = 0; x < w; x++) {
            uint32_t a = vk[x], b = sw[x], d = 0, k;
            for (k = 0; k < 24; k += 8) {           /* RGB: alpha is not shown */
                int dd = abs((int)((a >> k) & 0xFF) - (int)((b >> k) & 0xFF));
                if ((uint32_t)dd > d)
                    d = (uint32_t)dd;
            }
            sumd += d;
            if (d > maxd)
                maxd = d;
            if (d > (uint32_t)tol)
                nbad++;
        }
    }
    frames++;
    px_total += (uint64_t)w * h;
    px_bad += nbad;
    if (nbad) {
        double frac = (double)nbad / ((double)w * h);
        bad_frames++;
        fprintf(stderr, "[VKCMP] flip %u: %u px (%.3f%%) off by > %d, max %u, mean %.3f\n",
                flip, nbad, 100.0 * frac, tol, maxd, (double)sumd / ((double)w * h));
        /* The worst frames so far, and every 150th differing frame so the
         * ordinary ones are seen too. */
        if (dump && dumps < 60 && (long)flip >= dump_from
            && (dump_min >= 0.0 ? frac >= dump_min
                                : frac > worst * 0.5 || bad_frames % 150 == 0)) {
            char path[512];
            uint32_t *vkp = (uint32_t *)malloc((size_t)w * h * 4);
            uint32_t *swp = (uint32_t *)malloc((size_t)w * h * 4);
            uint32_t *dfp = (uint32_t *)malloc((size_t)w * h * 4);
            if (vkp && swp && dfp) {
                for (y = 0; y < h; y++)
                    for (x = 0; x < w; x++) {
                        uint32_t a = ((const uint32_t *)(B.rb.map + (size_t)y * w * 4))[x];
                        uint32_t b = ((const uint32_t *)(mem + addr + (size_t)y * pitch))[x];
                        uint32_t d = 0, k;
                        for (k = 0; k < 24; k += 8) {
                            uint32_t dd = (uint32_t)abs((int)((a >> k) & 0xFF) - (int)((b >> k) & 0xFF));
                            d = dd > d ? dd : d;
                        }
                        vkp[(size_t)y * w + x] = a | 0xFF000000u;
                        swp[(size_t)y * w + x] = b | 0xFF000000u;
                        /* diff: grey below tolerance, red above, scaled */
                        dfp[(size_t)y * w + x] = d > (uint32_t)tol
                            ? 0xFF000000u | (0x80u + (d > 127 ? 127 : d)) << 16
                            : 0xFF000000u | (d * 8 > 255 ? 255 : d * 8) * 0x010101u;
                    }
                snprintf(path, sizeof path, "%s%05u_vk.bmp", dump, flip);
                write_bmp32(path, vkp, w, h);
                snprintf(path, sizeof path, "%s%05u_sw.bmp", dump, flip);
                write_bmp32(path, swp, w, h);
                snprintf(path, sizeof path, "%s%05u_diff.bmp", dump, flip);
                write_bmp32(path, dfp, w, h);
                dumps++;
                if (frac > worst)
                    worst = frac;
            }
            free(vkp); free(swp); free(dfp);
        }
    }
    if (frames % 300 == 0)
        fprintf(stderr, "[VKCMP] summary: %llu frames, %llu with differences;"
                " %.4f%% of pixels off by > %d\n", (unsigned long long)frames,
                (unsigned long long)bad_frames,
                px_total ? 100.0 * (double)px_bad / (double)px_total : 0.0, tol);
    frame_stats();
}

/* ---------------------------------------------------------------------- */
/* Setup. */

/* 1x1 white, 2D and cube: what an unbound stage samples (the software path
 * reads white for a texture it cannot sample). */
static int make_white_one(Tex *x, int cube)
{
    VkImageViewCreateInfo vi = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    VkImageMemoryBarrier b = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    VkClearColorValue c;
    uint32_t layers = cube ? 6u : 1u;
    VkResult r_;

    if (!vk_image_create_ex(1, 1, VK_FORMAT_B8G8R8A8_UNORM,
                            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                            layers, cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0,
                            &x->img, &x->mem))
        return 0;
    vi.image = x->img;
    vi.viewType = cube ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D;
    vi.format = VK_FORMAT_B8G8R8A8_UNORM;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.layerCount = layers;
    if (!VK_OK(vkCreateImageView(g_vk.device, &vi, NULL, &x->view)))
        return 0;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = x->img;
    b.subresourceRange = vi.subresourceRange;
    vkCmdPipelineBarrier(cmd(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &b);
    c.float32[0] = c.float32[1] = c.float32[2] = c.float32[3] = 1.0f;
    vkCmdClearColorImage(cmd(), x->img, VK_IMAGE_LAYOUT_GENERAL, &c, 1,
                         &vi.subresourceRange);
    B.pending_barrier = 1;
    return 1;
}

static int make_white(void)
{
    return make_white_one(&B.white, 0) && make_white_one(&B.white_cube, 1);
}

static int backend_init(void)
{
    VkCommandPoolCreateInfo pci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    VkCommandBufferAllocateInfo ai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkDescriptorSetLayoutBinding lb[7];
    VkDescriptorSetLayoutCreateInfo dli = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    VkPipelineLayoutCreateInfo pli = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    VkDescriptorPoolSize ps[3];
    VkDescriptorPoolCreateInfo dpi = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    VkQueryPoolCreateInfo qpi = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
    VkFormatProperties fp;
    int i;
    VkResult r_;

    if (!vk_instance_init() || !vk_device_init(VK_NULL_HANDLE))
        return 0;

    /* Depth + stencil: D32_SFLOAT_S8 keeps the software path's float depth;
     * D24S8 where it is all there is. */
    B.zfmt = VK_FORMAT_D32_SFLOAT_S8_UINT;
    vkGetPhysicalDeviceFormatProperties(g_vk.phys, B.zfmt, &fp);
    if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT))
        B.zfmt = VK_FORMAT_D24_UNORM_S8_UINT;

    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = g_vk.qfam;
    if (!VK_OK(vkCreateCommandPool(g_vk.device, &pci, NULL, &B.pool)))
        return 0;
    ai.commandPool = B.pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    ps[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    ps[0].descriptorCount = VKB_SETS * 4;
    ps[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    ps[1].descriptorCount = VKB_SETS * 2;
    ps[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
    ps[2].descriptorCount = VKB_SETS;
    dpi.maxSets = VKB_SETS;
    dpi.poolSizeCount = 3;
    dpi.pPoolSizes = ps;
    qpi.queryType = VK_QUERY_TYPE_OCCLUSION;
    qpi.queryCount = VKB_QUERIES;
    if (!VK_OK(vkAllocateCommandBuffers(g_vk.device, &ai, &B.sink)))
        return 0;
    for (i = 0; i < VKB_SLOTS; i++) {
        Slot *s = &B.slot[i];
        if (!VK_OK(vkAllocateCommandBuffers(g_vk.device, &ai, &s->cb))
            || !VK_OK(vkCreateFence(g_vk.device, &fci, NULL, &s->fence))
            || !VK_OK(vkCreateDescriptorPool(g_vk.device, &dpi, NULL, &s->dpool))
            || !VK_OK(vkCreateQueryPool(g_vk.device, &qpi, NULL, &s->qpool))
            || !vk_buffer_create(VKB_RING, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
                                 | VK_BUFFER_USAGE_INDEX_BUFFER_BIT
                                 | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
                                 | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                 | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                 | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                 &s->ring, &s->ringmem, (void **)&s->ringmap))
            return 0;
        vkResetQueryPool(g_vk.device, s->qpool, 0, VKB_QUERIES);
    }

    /* Binding 0: the draw's uniforms (both stages); 1-4: texture stages. */
    memset(lb, 0, sizeof lb);
    lb[0].binding = 0;
    lb[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    lb[0].descriptorCount = 1;
    lb[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    for (i = 1; i <= 4; i++) {          /* one each: 2D or cube per shader */
        lb[i].binding = (uint32_t)i;
        lb[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        lb[i].descriptorCount = 1;
        lb[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    lb[5].binding = 5;                  /* vertex-program constants */
    lb[5].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    lb[5].descriptorCount = 1;
    lb[5].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    lb[6].binding = 6;                  /* the draw's vertex data */
    lb[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
    lb[6].descriptorCount = 1;
    lb[6].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    dli.bindingCount = 7;
    dli.pBindings = lb;
    if (!VK_OK(vkCreateDescriptorSetLayout(g_vk.device, &dli, NULL, &B.dsl)))
        return 0;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &B.dsl;
    if (!VK_OK(vkCreatePipelineLayout(g_vk.device, &pli, NULL, &B.layout)))
        return 0;
    {
        /* With what an earlier run left (vk_cache.c); a driver that does not
         * take the data back starts empty instead. */
        VkPipelineCacheCreateInfo pc = { VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO };
        vk_cache_init();
        pc.pInitialData = vk_cache_pipeline_data(&pc.initialDataSize);
        if (vkCreatePipelineCache(g_vk.device, &pc, NULL, &B.pcache) != VK_SUCCESS) {
            pc.pInitialData = NULL;
            pc.initialDataSize = 0;
            if (!VK_OK(vkCreatePipelineCache(g_vk.device, &pc, NULL, &B.pcache)))
                return 0;
        }
    }
    B.vs = vk_shader_module(0, k_vs);
    if (!B.vs)
        return 0;
    {
        VkSemaphoreTypeCreateInfo tci = { VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
        VkSemaphoreCreateInfo sci = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        tci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        sci.pNext = &tci;
        if (!VK_OK(vkCreateSemaphore(g_vk.device, &sci, NULL, &B.timeline)))
            return 0;
    }
    B.compare_mode = getenv("RECOMP_VK_COMPARE") != NULL || getenv("RECOMP_VK_SHADOW") != NULL;
    {
        const char *e = getenv("RECOMP_VK_REBIND");
        B.rebind = e && *e == '1';
    }
    settings_declare();
    B.mtx = vk_mutex_new();
    B.rep_event = vk_event_new();
    if (!B.mtx || !B.rep_event || !make_white())
        return 0;
    /* Last: nothing may fail after the thread is running (it would be left
     * working for a backend that never came up). */
    if (!vk_thread_start(report_thread, NULL))
        return 0;
    fprintf(stderr, "  [VK] rendering backend up (%s, depth format %d)\n",
            g_vk.props.deviceName, (int)B.zfmt);
    fflush(stderr);
    return 1;
}

/* The device is gone (driver reset, GPU removed): the executor renders in
 * software from the next frame. RECOMP_VK_SIMULATE_LOST=<flips>: from that
 * flip on the GPU goes silent -- submissions are accepted but never run, so
 * nothing signals and nothing reports an error, the hardest case: only the
 * 2 s wait limits notice, and the title must not hang meanwhile (testing). */
static int vk_lost(void)
{
    static int after = -2;
    if (after == -2) {
        const char *e = getenv("RECOMP_VK_SIMULATE_LOST");
        after = e ? atoi(e) : -1;
    }
    if (after >= 0 && B.frame >= (uint32_t)after && !s_gpu_silent) {
        fprintf(stderr, "  [VK] RECOMP_VK_SIMULATE_LOST: the GPU goes silent at flip %u\n",
                B.frame);
        s_gpu_silent = 1;
    }
    return g_vk.lost;
}

/* FLIP_STALL: the frame, then any settings change (between frames). */
/* Between two frames: new settings, and the shader cache to disk. */
static void between_frames(void)
{
    ds_flip();
    if (!B.ok || g_vk.lost)
        return;
    apply_settings();
    vk_cache_save(B.pcache, B.frame_pipes, 0);
    B.frame_pipes = 0;
}

static void vk_flip(uint32_t addr, uint32_t pitch, uint32_t bpp, uint32_t w, uint32_t h)
{
    flip_frame(addr, pitch, bpp, w, h);
    between_frames();
}

static void vk_compare(uint32_t addr, uint32_t pitch, uint32_t bpp, uint32_t w, uint32_t h,
                       uint32_t flip)
{
    /* RECOMP_VK_SHADOW=1 (netplay): the software renderer is authoritative
     * for everything the guest sees -- it renders into guest memory and
     * answers the visibility tests, exactly as with RECOMP_GPU=software --
     * and this renderer only draws what is shown (supersampling, FXAA and
     * the menu included). Nothing from it reaches guest memory; no diff. */
    static int shadow = -1;
    if (shadow < 0)
        shadow = getenv("RECOMP_VK_SHADOW") != NULL;
    if (shadow) {
        if (!B.ok || g_vk.lost)
            return;
        B.frame++;
        submit(0);
        flip_present(addr, w, h);
        frame_stats();
        between_frames();
        return;
    }
    compare_frame(addr, pitch, bpp, w, h, flip);
    between_frames();
}

static const Nv2aBackend k_backend = {
    "vulkan", 1, vk_clear, vk_draw, vk_report_clear, vk_report, vk_flip, vk_compare,
    vk_lost
};

const Nv2aBackend *xbox_vk_backend(void)
{
    if (!B.ok)
        B.ok = backend_init();
    if (!B.ok)
        fprintf(stderr, "  [VK] rendering backend unavailable\n");
    return B.ok ? &k_backend : NULL;
}
