/**
 * A GPU backend for the pushbuffer executor (nv2a_pb_exec.c).
 *
 * The executor decodes the title's NV2A command stream and keeps everything
 * guest-visible: the method state, D3D's fences (semaphore releases), the GPU
 * position in PGRAPH 0x400B10, NOTIFY interrupts, the flip counter and the
 * vblank. What it does with a draw, a clear, a visibility report or a flip is
 * either its own software rasteriser (the default, and the reference) or,
 * with RECOMP_GPU=vulkan, a backend that implements this interface
 * (src/gpu_vk).
 *
 * Everything here is plain data: guest addresses already resolved through
 * dma_resolve (so the 0x8xxxxxxx contiguous alias and the RAM tag are gone),
 * and no host pointers into guest memory except where named. A backend runs
 * on the executor thread, at the point in the stream the method arrives, and
 * must take whatever it needs from guest memory before it returns: the title
 * reuses vertex buffers and textures as soon as a later fence has passed,
 * and fences pass when the executor has read the commands.
 */
#ifndef XBOXRECOMP_NV2A_BACKEND_H
#define XBOXRECOMP_NV2A_BACKEND_H

#include <stdint.h>
#include "nv2a_vsh_interp.h"
#include "nv2a_combiner.h"

/* One texture stage as the executor decoded it (its Texture). */
typedef struct {
    uint32_t offset;                    /* guest address of texel (0,0)  */
    uint32_t width, height;
    uint32_t pitch;                     /* linear formats: bytes per row */
    uint32_t color;                     /* NV097 colour format code      */
    uint32_t addr_u, addr_v;            /* SET_TEXTURE_ADDRESS modes     */
    uint32_t palette;                   /* guest address of the CLUT     */
    uint32_t levels;
    uint32_t filter;                    /* SET_TEXTURE_FILTER            */
    int      cube, valid;
    int      normalized;                /* coordinates in [0,1] (swizzled,
                                           DXT), else in texels (linear) */
    uint32_t face_stride;               /* bytes between cube faces      */
    float    bump_mat[4], bump_scale, bump_offset;
} Nv2aTexState;

/* The state a draw or a clear is carried out with. */
typedef struct {
    /* Colour surface. */
    uint32_t color_addr;                /* resolved; 0 = none            */
    uint32_t surface_format;            /* SET_SURFACE_FORMAT            */
    uint32_t pitch, bpp;                /* bpp: 2 or 4                   */
    uint32_t swz_w, swz_h;              /* swizzled surface's size, 0 = pitch */
    uint32_t clip_x, clip_y, clip_w, clip_h;
    /* Zeta surface (depth + stencil). */
    uint32_t zeta_addr;                 /* resolved; 0 = none            */
    uint32_t zeta_format;               /* 1 Z16, 2 Z24S8                */
    uint32_t zstencil_clear, clear_color;
    /* Fragment operations, NV2A/GL enums as programmed. */
    uint32_t color_mask;                /* A<<24 R<<16 G<<8 B            */
    uint32_t alpha_test, alpha_func, alpha_ref;
    uint32_t blend_enable, blend_sfactor, blend_dfactor, blend_equation;
    uint32_t blend_color;
    uint32_t depth_test, depth_func, depth_mask;
    uint32_t st_test, st_func, st_ref, st_fmask, st_wmask;
    uint32_t st_fail, st_zfail, st_zpass;
    uint32_t cull_enable, cull_face, front_face;
    /* Viewport as the D3D epilogue applies it: c[58] scale, c[59] offset. */
    float    vp_scale[3], vp_offset[3];
    uint32_t fog_enable, fog_mode, fog_color;    /* fog colour: R in bits 0-7 */
    float    fog_param[2];
    /* Register combiners (rc_seen: programmed at all). */
    Nv2aCombiner rc;
    int      rc_seen;
    uint32_t clip_plane_mode, other_stage_input;
    uint32_t zpass_enable;              /* this draw counts for a report */
    Nv2aTexState tex[4];
} Nv2aDrawState;

/* One vertex attribute array as programmed (SET_VERTEX_DATA_ARRAY_*). */
typedef struct {
    uint32_t offset;            /* resolved guest address, or a byte offset
                                   into `inline_data` when that is set */
    uint32_t type, size, stride;
} Nv2aAttrDesc;

/* A draw's geometry: the primitive (SET_BEGIN_END's NV_PRIM_*) and either
 *  - v: n vertices already through the vertex program, in submission
 *    order (the executor ran it: a program that writes constants, or a
 *    backend without its own vertex programs); or
 *  - v == NULL: the raw draw, for a backend that runs the program itself:
 *    the program's words, the constants, the attribute arrays with each
 *    attribute's inline value (what an attribute with no array reads), and
 *    the n indices into the arrays. */
typedef struct {
    uint32_t prim;
    const Nv2aVshOutput *v;
    uint32_t n;
    const uint32_t (*prog)[4];
    uint32_t prog_n;
    const float (*constants)[4];        /* NV2A_VSH_CONSTANTS of them */
    Nv2aAttrDesc attr[16];
    const float (*imm)[4];              /* 16 inline values */
    const uint16_t *idx;
    const uint8_t *inline_data;         /* INLINE_ARRAY / immediate payload */
    uint32_t inline_bytes;
} Nv2aDrawGeom;

typedef struct Nv2aBackend {
    const char *name;
    /* Takes raw draws (Nv2aDrawGeom.v == NULL) and runs vertex programs
     * itself. RECOMP_VK_CPU_VSH=1 makes the executor transform anyway. */
    int raw_vertices;
    /* CLEAR_SURFACE: `what` is its parameter (0xF0 colour, 1 depth, 2
     * stencil), over the state's clip rectangle. */
    void (*clear)(const Nv2aDrawState *s, uint32_t what);
    void (*draw)(const Nv2aDrawState *s, const Nv2aDrawGeom *g);
    /* Visibility tests: CLEAR_REPORT_VALUE starts counting from zero, and
     * GET_REPORT asks for the pixels that passed the depth test, in draws
     * with zpass_enable, since then. The backend writes the 16-byte report
     * at guest address `at` -- `stamp`, the count (100000 with `force`,
     * RECOMP_ZPASS_FORCE), then 0 for "done", last -- when the GPU has the
     * answer, possibly from another thread: the title polls the last word. */
    void (*report_clear)(void);
    void (*report)(uint32_t at, uint64_t stamp, int force);
    /* FLIP_STALL: the frame in the colour surface at `addr` is finished.
     * Afterwards guest memory must hold it (the framebuffer window and the
     * frame dumps read it there). */
    void (*flip)(uint32_t addr, uint32_t pitch, uint32_t bpp,
                 uint32_t width, uint32_t height);
    /* RECOMP_VK_COMPARE: the same, but guest memory already holds the
     * software path's frame; diff against it instead of writing it. */
    void (*compare)(uint32_t addr, uint32_t pitch, uint32_t bpp,
                    uint32_t width, uint32_t height, uint32_t flip);
} Nv2aBackend;

/* Offered by the executor to backends: one texel of a stage in its own
 * format, decoded exactly as the software path samples it (swizzle, DXT,
 * palettes, YUY2...), as 0xAARRGGBB. u, v are texel coordinates, already
 * inside the texture; face_offset selects a cube face. 0: format not
 * decodable. */
int nv2a_exec_texel(const Nv2aTexState *t, uint32_t face_offset,
                    uint32_t u, uint32_t v, uint32_t *argb);

#endif
