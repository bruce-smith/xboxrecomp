/**
 * GLSL to SPIR-V at run time (glslang, C interface), and shader modules
 * cached by source.
 *
 * The NV2A's vertex programs and register combiners are translated to GLSL
 * per program; the same source always yields the same module, so modules are
 * kept in a hash table for the life of the process.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <glslang/Include/glslang_c_interface.h>
#include <glslang/Public/resource_limits_c.h>
#include "vk_internal.h"

static int s_glslang_up;

static uint64_t fnv1a64(const char *s)
{
    uint64_t h = 1469598103934665603ull;
    for (; *s; s++)
        h = (h ^ (uint8_t)*s) * 1099511628211ull;
    return h;
}

static int glsl_compile(int fragment, const char *src, uint32_t **words, size_t *nwords)
{
    glslang_input_t in;
    glslang_shader_t *sh;
    glslang_program_t *prog;
    glslang_stage_t stage = fragment ? GLSLANG_STAGE_FRAGMENT : GLSLANG_STAGE_VERTEX;
    int ok = 0;

    if (!s_glslang_up) {
        glslang_initialize_process();
        s_glslang_up = 1;
    }
    memset(&in, 0, sizeof in);
    in.language = GLSLANG_SOURCE_GLSL;
    in.stage = stage;
    in.client = GLSLANG_CLIENT_VULKAN;
    in.client_version = GLSLANG_TARGET_VULKAN_1_3;
    in.target_language = GLSLANG_TARGET_SPV;
    in.target_language_version = GLSLANG_TARGET_SPV_1_6;
    in.code = src;
    in.default_version = 450;
    in.default_profile = GLSLANG_NO_PROFILE;
    in.messages = GLSLANG_MSG_DEFAULT_BIT;
    in.resource = glslang_default_resource();

    sh = glslang_shader_create(&in);
    if (!glslang_shader_preprocess(sh, &in) || !glslang_shader_parse(sh, &in)) {
        fprintf(stderr, "  [VK] GLSL %s shader failed:\n%s\n%s\n--- source ---\n%s\n",
                fragment ? "fragment" : "vertex", glslang_shader_get_info_log(sh),
                glslang_shader_get_info_debug_log(sh), src);
        glslang_shader_delete(sh);
        return 0;
    }
    prog = glslang_program_create();
    glslang_program_add_shader(prog, sh);
    if (!glslang_program_link(prog, GLSLANG_MSG_SPV_RULES_BIT | GLSLANG_MSG_VULKAN_RULES_BIT)) {
        fprintf(stderr, "  [VK] GLSL link failed:\n%s\n", glslang_program_get_info_log(prog));
    } else {
        glslang_program_SPIRV_generate(prog, stage);
        *nwords = glslang_program_SPIRV_get_size(prog);
        *words = (uint32_t *)malloc(*nwords * sizeof(uint32_t));
        if (*words) {
            glslang_program_SPIRV_get(prog, *words);
            ok = 1;
        }
    }
    glslang_program_delete(prog);
    glslang_shader_delete(sh);
    return ok;
}

static unsigned s_compiled, s_cached;
static double s_glsl_ms;

/* The on-disk cache first (vk_cache.c), glslang otherwise. */
int vk_glsl_to_spirv(int fragment, const char *src, uint32_t **words, size_t *nwords)
{
    double t0;
    int ok;
    if (vk_cache_spirv_get(fragment, src, words, nwords)) {
        s_cached++;
        return 1;
    }
    t0 = vk_cache_now_ms();
    ok = glsl_compile(fragment, src, words, nwords);
    s_glsl_ms += vk_cache_now_ms() - t0;
    s_compiled++;
    if (ok)
        vk_cache_spirv_put(fragment, src, *words, *nwords);
    return ok;
}

/* Counted without a lock: the executor compiles nearly all of them, and
 * these are statistics. */
void vk_shader_stats(unsigned *compiled, unsigned *cached, double *ms)
{
    *compiled = s_compiled;
    *cached = s_cached;
    *ms = s_glsl_ms;
    s_compiled = s_cached = 0;
    s_glsl_ms = 0;
}

/* Modules by source hash; collisions are checked against the source. */
#define VK_SHADER_SLOTS 4096
static struct { uint64_t h; char *src; VkShaderModule m; } s_mods[VK_SHADER_SLOTS];
static unsigned s_nmods;

VkShaderModule vk_shader_module(int fragment, const char *src)
{
    uint64_t h = fnv1a64(src) ^ (uint64_t)fragment;
    unsigned i = (unsigned)(h % VK_SHADER_SLOTS), probes;
    VkShaderModuleCreateInfo ci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    uint32_t *words = NULL;
    size_t n = 0;
    VkShaderModule m = VK_NULL_HANDLE;
    VkResult r_;

    for (probes = 0; probes < VK_SHADER_SLOTS; probes++, i = (i + 1) % VK_SHADER_SLOTS) {
        if (!s_mods[i].src)
            break;
        if (s_mods[i].h == h && !strcmp(s_mods[i].src, src))
            return s_mods[i].m;
    }
    /* A prelude after #version:
     * - NaN and infinity kept, as the CPU path keeps them (RCP of 0 is +inf
     *   there, and what follows from it): without SignedZeroInfNanPreserve
     *   a driver may fold x * 0 to 0 and the like.
     * - CQ, the interpolation of the vertex colours and fog: perspective-
     *   correct (smooth), as xemu does and as the software span loops do
     *   since 1b37fdc -- which is also the only way the software clipper's
     *   clip-space interpolation at the near plane agrees with Vulkan's
     *   clipper. RECOMP_VK_SMOOTH_COLOR=0: screen-linear (noperspective),
     *   the old behaviour, for comparing with older captures. */
    if (!strncmp(src, "#version 450\n", 13)) {
        static const char nan_mode[] =
            "#extension GL_EXT_spirv_intrinsics : require\n"
            "spirv_execution_mode(capabilities = [4466], 4461, 32);\n";
        static int smooth = -1;
        char cq[64];
        size_t l = strlen(src);
        char *full = (char *)malloc(l + sizeof nan_mode + sizeof cq);
        int ok;
        if (!full)
            return VK_NULL_HANDLE;
        if (smooth < 0) {
            const char *e = getenv("RECOMP_VK_SMOOTH_COLOR");
            smooth = !(e && *e == '0');
        }
        snprintf(cq, sizeof cq, "#define CQ %s\n", smooth ? "smooth" : "noperspective");
        snprintf(full, l + sizeof nan_mode + sizeof cq, "#version 450\n%s%s%s",
                 g_vk.nan_preserve ? nan_mode : "", cq, src + 13);
        ok = vk_glsl_to_spirv(fragment, full, &words, &n);
        free(full);
        if (!ok)
            return VK_NULL_HANDLE;
    } else if (!vk_glsl_to_spirv(fragment, src, &words, &n)) {
        return VK_NULL_HANDLE;
    }
    ci.codeSize = n * sizeof(uint32_t);
    ci.pCode = words;
    if (!VK_OK(vkCreateShaderModule(g_vk.device, &ci, NULL, &m)))
        m = VK_NULL_HANDLE;
    free(words);
    if (m && probes < VK_SHADER_SLOTS && s_nmods < VK_SHADER_SLOTS * 3 / 4) {
        s_mods[i].h = h;
        s_mods[i].src = (char *)malloc(strlen(src) + 1);
        if (s_mods[i].src)
            strcpy(s_mods[i].src, src);
        s_mods[i].m = m;
        s_nmods++;
    }
    return m;
}
