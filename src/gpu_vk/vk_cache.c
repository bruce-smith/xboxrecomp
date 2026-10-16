/**
 * The on-disk shader and pipeline cache.
 *
 * A pipeline seen for the first time costs a GLSL-to-SPIR-V compile per
 * stage (glslang) and the driver's own compile; a new stage of the game
 * brings dozens of them at once, so its first seconds stuttered. Both are
 * kept between runs:
 * - vk_spirv.bin: SPIR-V by GLSL source (two 64-bit hashes of the full
 *   source, stage included), consulted by vk_glsl_to_spirv;
 * - vk_pipelines.bin: the VkPipelineCache data, given back to the driver at
 *   start when it was written for the same device and driver.
 *
 * In RECOMP_VK_CACHE_DIR, else the platform's cache folder
 * (%LOCALAPPDATA%\xboxrecomp\vkcache, $XDG_CACHE_HOME/xboxrecomp/vkcache).
 * RECOMP_VK_CACHE=0: none. Files are written whole to a temporary name and
 * renamed over the old one, so a run that dies (or a power loss) leaves the
 * previous file; every record carries a checksum and reading stops at the
 * first bad one. Several processes may share the folder: the last to write
 * wins, which only costs the other one's new entries.
 *
 * Nothing here is guest-visible: a cached SPIR-V blob is the one glslang
 * would produce from the same source.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "vk_internal.h"

#define CACHE_MAGIC   0x4B565258u   /* "XRVK" */
#define CACHE_VERSION 1u            /* bump when the GLSL front end changes */
#define SPV_SLOTS     8192

typedef struct {
    uint64_t h1, h2;
    uint32_t n;                     /* words */
    uint32_t *w;
} Spv;

static struct {
    int      state;                 /* 0 not yet, 1 on, -1 off */
    char     dir[512];
    void    *mu;
    Spv      spv[SPV_SLOTS];
    unsigned nspv, spv_dirty;
    void    *pdata;                 /* pipeline cache data read at start */
    size_t   npdata;
    double   last_save;
} C;

double vk_cache_now_ms(void)
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static uint32_t sum32(const void *p, size_t n)
{
    const uint8_t *b = (const uint8_t *)p;
    uint32_t h = 2166136261u;
    while (n--)
        h = (h ^ *b++) * 16777619u;
    return h;
}

static void path(char *out, size_t n, const char *name)
{
    snprintf(out, n, "%s/%s", C.dir, name);
}

static Spv *slot(uint64_t h1, uint64_t h2, int *found)
{
    unsigned i = (unsigned)(h1 % SPV_SLOTS), k;
    for (k = 0; k < SPV_SLOTS; k++, i = (i + 1) % SPV_SLOTS) {
        if (!C.spv[i].w) {
            *found = 0;
            return &C.spv[i];
        }
        if (C.spv[i].h1 == h1 && C.spv[i].h2 == h2) {
            *found = 1;
            return &C.spv[i];
        }
    }
    *found = 0;
    return NULL;
}

static void *read_file(const char *p, size_t *n)
{
    FILE *f = fopen(p, "rb");
    long l;
    void *d = NULL;
    *n = 0;
    if (!f)
        return NULL;
    if (fseek(f, 0, SEEK_END) == 0 && (l = ftell(f)) > 0 && fseek(f, 0, SEEK_SET) == 0) {
        d = malloc((size_t)l);
        if (d && fread(d, 1, (size_t)l, f) == (size_t)l) {
            *n = (size_t)l;
        } else {
            free(d);
            d = NULL;
        }
    }
    fclose(f);
    return d;
}

/* Header: magic, version, kind, then the device it was made on. */
typedef struct {
    uint32_t magic, version, kind, vendor, device, driver;
    uint8_t  uuid[VK_UUID_SIZE];
} Head;

static void head_make(Head *h, uint32_t kind)
{
    memset(h, 0, sizeof *h);
    h->magic = CACHE_MAGIC;
    h->version = CACHE_VERSION;
    h->kind = kind;
    h->vendor = g_vk.props.vendorID;
    h->device = g_vk.props.deviceID;
    h->driver = g_vk.props.driverVersion;
    memcpy(h->uuid, g_vk.props.pipelineCacheUUID, VK_UUID_SIZE);
}

static void load_spirv(void)
{
    char p[600];
    size_t n, at;
    uint8_t *d;
    Head want, *got;

    path(p, sizeof p, "vk_spirv.bin");
    d = (uint8_t *)read_file(p, &n);
    if (!d)
        return;
    head_make(&want, 1);
    got = (Head *)d;
    /* SPIR-V does not depend on the device, but the GLSL prelude does
     * (NaN preservation), so the device is part of the key all the same. */
    if (n < sizeof want || memcmp(got, &want, sizeof want)) {
        free(d);
        return;
    }
    for (at = sizeof want; at + 24 <= n;) {
        uint64_t h1, h2;
        uint32_t nw, sum;
        Spv *s;
        int found;
        memcpy(&h1, d + at, 8);
        memcpy(&h2, d + at + 8, 8);
        memcpy(&nw, d + at + 16, 4);
        memcpy(&sum, d + at + 20, 4);
        if (!nw || nw > (n - at - 24) / 4 || sum32(d + at + 24, (size_t)nw * 4) != sum)
            break;                  /* torn or damaged: keep what came before */
        s = slot(h1, h2, &found);
        if (s && !found && C.nspv < SPV_SLOTS * 3 / 4) {
            s->w = (uint32_t *)malloc((size_t)nw * 4);
            if (s->w) {
                memcpy(s->w, d + at + 24, (size_t)nw * 4);
                s->h1 = h1;
                s->h2 = h2;
                s->n = nw;
                C.nspv++;
            }
        }
        at += 24 + (size_t)nw * 4;
    }
    free(d);
}

static void load_pipelines(void)
{
    char p[600];
    size_t n;
    uint8_t *d;
    Head want;
    uint64_t sz;
    uint32_t sum;

    path(p, sizeof p, "vk_pipelines.bin");
    d = (uint8_t *)read_file(p, &n);
    if (!d)
        return;
    head_make(&want, 2);
    if (n >= sizeof want + 12 && !memcmp(d, &want, sizeof want)) {
        memcpy(&sz, d + sizeof want, 8);
        memcpy(&sum, d + sizeof want + 8, 4);
        if (sz && sz == n - sizeof want - 12 && sum32(d + sizeof want + 12, (size_t)sz) == sum) {
            C.pdata = malloc((size_t)sz);
            if (C.pdata) {
                memcpy(C.pdata, d + sizeof want + 12, (size_t)sz);
                C.npdata = (size_t)sz;
            }
        }
    }
    free(d);
}

static int on(void)
{
    if (C.state == 0) {
        const char *e = getenv("RECOMP_VK_CACHE");
        C.state = -1;
        if (!(e && *e == '0') && vk_cache_dir(C.dir, sizeof C.dir) && (C.mu = vk_mutex_new())) {
            C.state = 1;
            load_spirv();
            load_pipelines();
            C.last_save = vk_cache_now_ms();
            fprintf(stderr, "  [VK] cache %s: %u shaders, %zu bytes of pipelines\n",
                    C.dir, C.nspv, C.npdata);
        }
    }
    return C.state > 0;
}

void vk_cache_init(void)
{
    (void)on();
}

const void *vk_cache_pipeline_data(size_t *n)
{
    *n = on() ? C.npdata : 0;
    return *n ? C.pdata : NULL;
}

static uint64_t hash64(const char *s, uint64_t h, uint64_t mul)
{
    for (; *s; s++)
        h = (h ^ (uint8_t)*s) * mul;
    return h ^ (h >> 31);
}

int vk_cache_spirv_get(int fragment, const char *src, uint32_t **words, size_t *nwords)
{
    uint64_t h1, h2;
    Spv *s;
    int found;
    if (!on())
        return 0;
    h1 = hash64(src, 1469598103934665603ull, 1099511628211ull) ^ (uint64_t)fragment;
    h2 = hash64(src, 0x9E3779B97F4A7C15ull, 0xC2B2AE3D27D4EB4Full);
    vk_mutex_lock(C.mu);
    s = slot(h1, h2, &found);
    if (found) {
        *words = (uint32_t *)malloc((size_t)s->n * 4);
        if (*words) {
            memcpy(*words, s->w, (size_t)s->n * 4);
            *nwords = s->n;
        }
    }
    vk_mutex_unlock(C.mu);
    return found && *words;
}

void vk_cache_spirv_put(int fragment, const char *src, const uint32_t *words, size_t nwords)
{
    uint64_t h1, h2;
    Spv *s;
    int found;
    if (!on() || !nwords || nwords > 0x3FFFFFFF)
        return;
    h1 = hash64(src, 1469598103934665603ull, 1099511628211ull) ^ (uint64_t)fragment;
    h2 = hash64(src, 0x9E3779B97F4A7C15ull, 0xC2B2AE3D27D4EB4Full);
    vk_mutex_lock(C.mu);
    s = slot(h1, h2, &found);
    if (s && !found && C.nspv < SPV_SLOTS * 3 / 4) {
        s->w = (uint32_t *)malloc(nwords * 4);
        if (s->w) {
            memcpy(s->w, words, nwords * 4);
            s->h1 = h1;
            s->h2 = h2;
            s->n = (uint32_t)nwords;
            C.nspv++;
            C.spv_dirty++;
        }
    }
    vk_mutex_unlock(C.mu);
}

/* Write `parts` to dir/name through a temporary file and a rename. */
static int write_whole(const char *name, const void *const *parts, const size_t *sizes, int nparts)
{
    char tmp[640], dst[600];
    FILE *f;
    int i, ok = 1;
    path(dst, sizeof dst, name);
    snprintf(tmp, sizeof tmp, "%s.%u.tmp", dst, vk_process_id());
    f = fopen(tmp, "wb");
    if (!f)
        return 0;
    for (i = 0; i < nparts && ok; i++)
        ok = !sizes[i] || fwrite(parts[i], 1, sizes[i], f) == sizes[i];
    ok = (fclose(f) == 0) && ok;
    if (!ok || !vk_file_replace(tmp, dst)) {
        remove(tmp);
        return 0;
    }
    return 1;
}

static void save_spirv(void)
{
    size_t total = sizeof(Head), at;
    uint8_t *d;
    unsigned i;
    vk_mutex_lock(C.mu);
    for (i = 0; i < SPV_SLOTS; i++)
        if (C.spv[i].w)
            total += 24 + (size_t)C.spv[i].n * 4;
    d = (uint8_t *)malloc(total);
    if (d) {
        head_make((Head *)d, 1);
        at = sizeof(Head);
        for (i = 0; i < SPV_SLOTS; i++) {
            const Spv *s = &C.spv[i];
            uint32_t sum;
            if (!s->w)
                continue;
            sum = sum32(s->w, (size_t)s->n * 4);
            memcpy(d + at, &s->h1, 8);
            memcpy(d + at + 8, &s->h2, 8);
            memcpy(d + at + 16, &s->n, 4);
            memcpy(d + at + 20, &sum, 4);
            memcpy(d + at + 24, s->w, (size_t)s->n * 4);
            at += 24 + (size_t)s->n * 4;
        }
        C.spv_dirty = 0;
    }
    vk_mutex_unlock(C.mu);
    if (d) {
        const void *parts[1] = { d };
        size_t sizes[1] = { total };
        write_whole("vk_spirv.bin", parts, sizes, 1);
        free(d);
    }
}

static void save_pipelines(VkPipelineCache pc)
{
    size_t n = 0;
    void *d;
    Head h;
    uint64_t sz;
    uint32_t sum;
    VkResult r_;
    if (vkGetPipelineCacheData(g_vk.device, pc, &n, NULL) != VK_SUCCESS || !n)
        return;
    d = malloc(n);
    if (!d)
        return;
    if (VK_OK(vkGetPipelineCacheData(g_vk.device, pc, &n, d)) && n) {
        const void *parts[4];
        size_t sizes[4];
        head_make(&h, 2);
        sz = n;
        sum = sum32(d, n);
        parts[0] = &h;  sizes[0] = sizeof h;
        parts[1] = &sz; sizes[1] = 8;
        parts[2] = &sum; sizes[2] = 4;
        parts[3] = d;   sizes[3] = n;
        write_whole("vk_pipelines.bin", parts, sizes, 4);
    }
    free(d);
}

void vk_cache_save(VkPipelineCache pc, unsigned new_pipelines, int force)
{
    static unsigned pending;
    double now;
    if (!on() || g_vk.lost)
        return;
    pending += new_pipelines;
    if (!pending && !C.spv_dirty)
        return;
    now = vk_cache_now_ms();
    /* At most every 10 s, and not while a stage is still bringing new
     * pipelines in: a burst ends before it is written. */
    if (!force && now - C.last_save < 10000.0)
        return;
    if (!force && new_pipelines)
        return;
    C.last_save = now;
    if (C.spv_dirty)
        save_spirv();
    if (pending && pc)
        save_pipelines(pc);
    pending = 0;
}
