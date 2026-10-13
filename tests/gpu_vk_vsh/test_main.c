/* Vertex programs: GLSL (vk_vsh.c) against nv2a_vsh_run. See README.md. */
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "vk_internal.h"
#include "vk_psh.h"
#include "vk_vsh.h"
#include "nv2a_backend.h"
#include "nv2a_vsh_interp.h"

extern const Nv2aBackend *xbox_vk_backend(void);

/* The backend's links into the executor, unused here. */
ptrdiff_t xbox_GetMemoryOffset(void) { return 0; }
int nv2a_exec_texel(const Nv2aTexState *t, uint32_t f, uint32_t u, uint32_t v, uint32_t *c)
{
    (void)t; (void)f; (void)u; (void)v;
    *c = 0xFFFFFFFFu;
    return 1;
}

#define NV 64                 /* vertices per program */
#define NOUT 8                /* oPos d0 d1 fog t0-t3 */

static uint32_t rng = 777;
static uint32_t rnd(void)
{
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return rng;
}
static float frand(float lo, float hi)
{
    return lo + (hi - lo) * (float)(rnd() & 0xFFFFFF) / 16777215.0f;
}

/* ---- the vertex layout: one attribute of each format, interleaved -------- */

typedef struct { uint32_t type, size, off; } Attr;
static Attr g_attr[16];
static uint32_t g_stride;
static uint8_t g_blob[NV * 256];
static float g_imm[16][4];

static void layout_attrs(void)
{
    /* type, size per input; size 0 = inline value */
    static const uint32_t ts[16][2] = {
        {2, 4}, {2, 3}, {0, 4}, {4, 4}, {1, 3}, {5, 2}, {6, 1}, {2, 1},
        {2, 2}, {4, 2}, {1, 4}, {5, 4}, {2, 4}, {0, 0}, {2, 4}, {2, 0},
    };
    uint32_t k, at = 0;
    for (k = 0; k < 16; k++) {
        uint32_t t = ts[k][0], n = ts[k][1], b;
        g_attr[k].type = t;
        g_attr[k].size = n;
        g_attr[k].off = at;
        b = t == 0 || t == 6 ? 4 : t == 2 ? 4 * n : t == 4 ? n : 2 * n;
        if (n)
            at += (b + 3) & ~3u;
    }
    g_stride = at;
}

/* fetch_attr, for the reference inputs. */
static void fetch(uint32_t k, uint32_t v, float out[4])
{
    const uint8_t *p = g_blob + (size_t)v * g_stride + g_attr[k].off;
    uint32_t i, n = g_attr[k].size;
    out[0] = out[1] = out[2] = 0.0f;
    out[3] = 1.0f;
    if (!n) {
        memcpy(out, g_imm[k], 16);
        return;
    }
    switch (g_attr[k].type) {
    case 0:
        out[0] = p[2] / 255.0f; out[1] = p[1] / 255.0f;
        out[2] = p[0] / 255.0f; out[3] = p[3] / 255.0f;
        break;
    case 2: for (i = 0; i < n; i++) memcpy(&out[i], p + 4 * i, 4); break;
    case 4: for (i = 0; i < n; i++) out[i] = p[i] / 255.0f; break;
    case 1: for (i = 0; i < n; i++) out[i] = ((const int16_t *)p)[i] / 32767.0f; break;
    case 5: for (i = 0; i < n; i++) out[i] = (float)((const int16_t *)p)[i]; break;
    case 6: {
        uint32_t w;
        memcpy(&w, p, 4);
        out[0] = (float)((int32_t)(w << 21) >> 21) / 1023.0f;
        out[1] = (float)((int32_t)(w << 10) >> 21) / 1023.0f;
        out[2] = (float)((int32_t)w >> 22) / 511.0f;
        break;
    }
    }
}

static void fill_vertices(void)
{
    uint32_t v, k, i;
    for (v = 0; v < NV; v++)
        for (k = 0; k < 16; k++) {
            uint8_t *p = g_blob + (size_t)v * g_stride + g_attr[k].off;
            switch (g_attr[k].type) {
            case 2:
                for (i = 0; i < g_attr[k].size; i++) {
                    float f = frand(-4.0f, 4.0f);
                    memcpy(p + 4 * i, &f, 4);
                }
                break;
            default:
                for (i = 0; i < 8; i++)
                    p[i] = (uint8_t)rnd();
                break;
            }
        }
    for (k = 0; k < 16; k++)
        for (i = 0; i < 4; i++)
            g_imm[k][i] = frand(-2.0f, 2.0f);
}

/* ---- random programs ----------------------------------------------------- */

static void put_field(uint32_t *ins, int dw, int lo, int width, uint32_t v)
{
    ins[dw] |= (v & ((1u << width) - 1u)) << lo;
}

static int g_imprecise;       /* the program uses EXP/LOG/LIT/RSQ */

static uint32_t random_program(uint32_t (*w)[4])
{
    static const uint32_t outs[] = {0, 3, 4, 5, 9, 10, 11, 12};
    uint32_t n = 2 + rnd() % 24, i, k;
    g_imprecise = 0;
    for (i = 0; i < n; i++) {
        uint32_t *ins = w[i];
        uint32_t mac = rnd() % 14, ilu = rnd() % 8;
        memset(ins, 0, 16);
        if (rnd() % 3 == 0) ilu = 0;
        if (rnd() % 4 == 0 && ilu) mac = 0;
        if (ilu >= 4) g_imprecise = 1;
        put_field(ins, 1, 25, 3, ilu);
        put_field(ins, 1, 21, 4, mac);
        put_field(ins, 1, 13, 8, rnd() % 192);
        put_field(ins, 1, 9, 4, rnd() % 16);
        for (k = 0; k < 3; k++) {
            uint32_t mux = 1 + rnd() % 3, reg = rnd() % 13, neg = rnd() & 1, sw = rnd() & 0xFF;
            if (k == 0) {
                put_field(ins, 1, 8, 1, neg); put_field(ins, 1, 0, 8, sw);
                put_field(ins, 2, 28, 4, reg); put_field(ins, 2, 26, 2, mux);
            } else if (k == 1) {
                put_field(ins, 2, 25, 1, neg); put_field(ins, 2, 17, 8, sw);
                put_field(ins, 2, 13, 4, reg); put_field(ins, 2, 11, 2, mux);
            } else {
                put_field(ins, 2, 10, 1, neg); put_field(ins, 2, 2, 8, sw);
                put_field(ins, 2, 0, 2, reg >> 2); put_field(ins, 3, 30, 2, reg & 3);
                put_field(ins, 3, 28, 2, mux);
            }
        }
        put_field(ins, 3, 24, 4, rnd());                 /* MAC temp mask */
        put_field(ins, 3, 20, 4, rnd() % 13);            /* temp dest */
        put_field(ins, 3, 16, 4, rnd());                 /* ILU temp mask */
        if (rnd() & 1) {
            put_field(ins, 3, 12, 4, rnd());             /* out mask */
            put_field(ins, 3, 11, 1, 1);                 /* to o[], never c[] */
            put_field(ins, 3, 3, 8, outs[rnd() % 8]);
            put_field(ins, 3, 2, 1, rnd() & 1);
        }
        put_field(ins, 3, 1, 1, (rnd() % 4) == 0);       /* a0-relative */
        if (i == n - 1)
            put_field(ins, 3, 0, 1, 1);
    }
    return n;
}

/* ---- the GPU side -------------------------------------------------------- */

static const char k_store[] =
"layout(std430, set=0, binding=7) buffer Out { vec4 r[]; } outb;\n"
"void test_store(uint v, vec4 p, vec4 a, vec4 b, vec4 f, vec4 t0, vec4 t1, vec4 t2, vec4 t3) {\n"
"    uint i = v * 8u;\n"
"    outb.r[i] = p; outb.r[i + 1u] = a; outb.r[i + 2u] = b; outb.r[i + 3u] = f;\n"
"    outb.r[i + 4u] = t0; outb.r[i + 5u] = t1; outb.r[i + 6u] = t2; outb.r[i + 7u] = t3;\n"
"}\n";

static VkDescriptorSetLayout dsl;
static VkPipelineLayout layout;
static VkDescriptorPool dpool;
static VkCommandPool cpool;
static VkCommandBuffer cb;
static VkFence fence;
static VkBuffer ubuf, cbuf, bbuf, obuf;
static VkDeviceMemory umem, cmem, bmem, omem;
static void *umap, *cmap, *bmap, *omap;

static int gpu_setup(void)
{
    VkDescriptorSetLayoutBinding lb[4];
    VkDescriptorSetLayoutCreateInfo dli = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    VkPipelineLayoutCreateInfo pli = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    VkDescriptorPoolSize ps[2];
    VkDescriptorPoolCreateInfo dpi = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    VkCommandPoolCreateInfo cpi = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    VkCommandBufferAllocateInfo cai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    static const uint32_t bind[4] = {0, 5, 6, 7};
    VkMemoryPropertyFlags hv = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    int i;

    memset(lb, 0, sizeof lb);
    for (i = 0; i < 4; i++) {
        lb[i].binding = bind[i];
        lb[i].descriptorType = i < 2 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                     : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        lb[i].descriptorCount = 1;
        lb[i].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    }
    dli.bindingCount = 4;
    dli.pBindings = lb;
    if (vkCreateDescriptorSetLayout(g_vk.device, &dli, NULL, &dsl)) return 0;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &dsl;
    if (vkCreatePipelineLayout(g_vk.device, &pli, NULL, &layout)) return 0;
    ps[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; ps[0].descriptorCount = 2;
    ps[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; ps[1].descriptorCount = 2;
    dpi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpi.maxSets = 1;
    dpi.poolSizeCount = 2;
    dpi.pPoolSizes = ps;
    if (vkCreateDescriptorPool(g_vk.device, &dpi, NULL, &dpool)) return 0;
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = g_vk.qfam;
    if (vkCreateCommandPool(g_vk.device, &cpi, NULL, &cpool)) return 0;
    cai.commandPool = cpool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(g_vk.device, &cai, &cb)) return 0;
    if (vkCreateFence(g_vk.device, &fci, NULL, &fence)) return 0;
    return vk_buffer_create(sizeof(VkDrawUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, hv, &ubuf, &umem, &umap)
        && vk_buffer_create(192 * 16, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, hv, &cbuf, &cmem, &cmap)
        && vk_buffer_create(sizeof g_blob + 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hv, &bbuf, &bmem, &bmap)
        && vk_buffer_create(NV * NOUT * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hv, &obuf, &omem, &omap);
}

/* Run program w on the NV vertices; outputs into omap. */
static int gpu_run(const uint32_t (*w)[4], uint32_t n)
{
    char *src = vk_vsh_glsl(w, n, k_store);
    VkShaderModule vs = vk_shader_module(0, src);
    VkGraphicsPipelineCreateInfo ci = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    VkPipelineShaderStageCreateInfo ss = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    VkPipelineVertexInputStateCreateInfo vis = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    VkPipelineInputAssemblyStateCreateInfo ias = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    VkPipelineRasterizationStateCreateInfo rs = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    VkPipelineRenderingCreateInfo pr = { VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    VkPipeline pipe;
    VkDescriptorSetAllocateInfo ai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    VkDescriptorSet set;
    VkDescriptorBufferInfo bi[4];
    VkWriteDescriptorSet wr[4];
    VkCommandBufferBeginInfo cbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    VkRenderingInfo ri = { VK_STRUCTURE_TYPE_RENDERING_INFO };
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    static const uint32_t bind[4] = {0, 5, 6, 7};
    VkBuffer bufs[4];
    int i;

    free(src);
    if (!vs)
        return 0;
    ss.stage = VK_SHADER_STAGE_VERTEX_BIT;
    ss.module = vs;
    ss.pName = "main";
    ias.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    rs.rasterizerDiscardEnable = VK_TRUE;
    rs.lineWidth = 1.0f;
    ci.pNext = &pr;
    ci.stageCount = 1;
    ci.pStages = &ss;
    ci.pVertexInputState = &vis;
    ci.pInputAssemblyState = &ias;
    ci.pRasterizationState = &rs;
    ci.layout = layout;
    if (vkCreateGraphicsPipelines(g_vk.device, VK_NULL_HANDLE, 1, &ci, NULL, &pipe))
        return 0;

    ai.descriptorPool = dpool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &dsl;
    if (vkAllocateDescriptorSets(g_vk.device, &ai, &set))
        return 0;
    bufs[0] = ubuf; bufs[1] = cbuf; bufs[2] = bbuf; bufs[3] = obuf;
    memset(wr, 0, sizeof wr);
    for (i = 0; i < 4; i++) {
        bi[i].buffer = bufs[i];
        bi[i].offset = 0;
        bi[i].range = VK_WHOLE_SIZE;
        wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet = set;
        wr[i].dstBinding = bind[i];
        wr[i].descriptorCount = 1;
        wr[i].descriptorType = i < 2 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                     : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(g_vk.device, 4, wr, 0, NULL);

    vkResetCommandBuffer(cb, 0);
    vkBeginCommandBuffer(cb, &cbi);
    ri.renderArea.extent.width = ri.renderArea.extent.height = 1;
    ri.layerCount = 1;
    vkCmdBeginRendering(cb, &ri);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &set, 0, NULL);
    vkCmdDraw(cb, NV, 1, 0, 0);
    vkCmdEndRendering(cb);
    {
        VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                             0, 1, &mb, 0, NULL, 0, NULL);
    }
    vkEndCommandBuffer(cb);
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    vkResetFences(g_vk.device, 1, &fence);
    vkQueueSubmit(g_vk.queue, 1, &si, fence);
    vkWaitForFences(g_vk.device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkFreeDescriptorSets(g_vk.device, dpool, 1, &set);
    vkDestroyPipeline(g_vk.device, pipe, NULL);
    return 1;
}

static int close_enough(float a, float b, float tol)
{
    if (a != a || b != b)
        return a != a && b != b;
    if (isinf(a) || isinf(b))
        return a == b || (fabsf(a) > 1e30f && fabsf(b) > 1e30f && (a > 0) == (b > 0));
    return fabsf(a - b) <= tol * (1.0f + fmaxf(fabsf(a), fabsf(b)));
}

int main(void)
{
    static uint32_t w[136][4];
    int prog, programs = 1000, bad_progs = 0, k;
    float cst[192][4];

    if (!xbox_vk_backend() || !gpu_setup()) {
        printf("FAIL: no Vulkan\n");
        return 1;
    }
    layout_attrs();
    for (prog = 0; prog < programs; prog++) {
        uint32_t n = random_program(w), v, i;
        VkDrawUniforms *U = (VkDrawUniforms *)umap;
        int bad = 0;

        fill_vertices();
        for (i = 0; i < 192; i++)
            for (k = 0; k < 4; k++)
                cst[i][k] = frand(-2.0f, 2.0f);
        /* CPU reference. */
        for (i = 0; i < 136; i++)
            nv2a_vsh_set_instruction(i, w[i < n ? i : n - 1]);
        nv2a_vsh_set_start_slot(1);           /* invalidate the decode, */
        nv2a_vsh_set_start_slot(0);           /* then run from slot 0 */
        for (i = 0; i < 192; i++)
            nv2a_vsh_set_constant(i, cst[i]);
        /* GPU inputs. */
        memset(U, 0, sizeof *U);
        for (k = 0; k < 16; k++) {
            U->attr[k][0] = g_attr[k].off;
            U->attr[k][1] = g_stride;
            U->attr[k][2] = g_attr[k].type | g_attr[k].size << 4;
            memcpy(U->imm[k], g_imm[k], 16);
        }
        memcpy(cmap, cst, sizeof cst);
        memcpy(bmap, g_blob, sizeof g_blob);
        memset(omap, 0, NV * NOUT * 16);
        if (!gpu_run((const uint32_t (*)[4])w, n)) {
            printf("FAIL program %d: GPU run failed\n", prog);
            bad_progs++;
            continue;
        }
        for (v = 0; v < NV; v++) {
            float in[16][4];
            Nv2aVshOutput o;
            const float *ref[NOUT];
            const float *got = (const float *)omap + (size_t)v * NOUT * 4;
            for (k = 0; k < 16; k++)
                fetch((uint32_t)k, v, in[k]);
            if (!nv2a_vsh_run((const float (*)[4])in, &o)) {
                bad++;
                continue;
            }
            ref[0] = o.pos; ref[1] = o.d0; ref[2] = o.d1; ref[3] = o.fog;
            ref[4] = o.tex[0]; ref[5] = o.tex[1]; ref[6] = o.tex[2]; ref[7] = o.tex[3];
            for (i = 0; i < NOUT; i++)
                for (k = 0; k < 4; k++)
                    if (!close_enough(ref[i][k], got[i * 4 + k], g_imprecise ? 1e-3f : 1e-4f)) {
                        if (!bad && bad_progs < 5)
                            printf("FAIL program %d (%u ins): vertex %u output %u.%c = %g, want %g\n",
                                   prog, n, v, i, "xyzw"[k], got[i * 4 + k], ref[i][k]);
                        bad++;
                    }
        }
        bad_progs += bad != 0;
    }
    printf("%s vertex programs: %d of %d differ\n", bad_progs ? "FAIL" : "ok  ",
           bad_progs, programs);
    printf(bad_progs ? "FAILED\n" : "PASS\n");
    return bad_progs ? 1 : 0;
}
