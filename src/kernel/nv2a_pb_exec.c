/**
 * Execute the parts of the title's pushbuffer that produce visible pixels.
 *
 * The title builds NV2A commands in guest RAM and advances DMA_PUT; without
 * something consuming them the framebuffer stays whatever it was, which is how
 * a fully booted title renders a black screen. This walks the same command
 * stream nv2a_pb_scan.c surveys and carries out the subset that decides what is
 * on screen: which surface is being drawn into, and clearing it.
 *
 * It also rasterises geometry, but only the part that can be drawn honestly:
 * batches whose attribute 0 is already in screen space, flat-shaded, straight
 * into the same guest framebuffer the clear writes. Titles draw their UI, HUD
 * and 2D overlays that way, so it is the first geometry to appear. Batches that
 * need a vertex program executed are counted and skipped rather than drawn
 * somewhere wrong -- see raster_batch(). Texturing, depth and vertex programs
 * are still a renderer, not a command decoder; the upgrade path is the D3D11
 * translator in src/nv2a/nv2a_pgraph_d3d11.c.
 *
 * Everything this does not handle is counted and ranked by
 * nv2a_pb_exec_report(), so what remains is a list rather than a guess.
 *
 * Enabled with RECOMP_PB_EXEC. RECOMP_RASTER_TEST draws one known triangle
 * after every clear, which separates "the pixel path is broken" from "the title
 * has not given us any vertices". RECOMP_FB_DUMP=<prefix> writes the surface to
 * <prefix>NNN.bmp, so the result can be looked at without a display.
 */
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "kernel.h"   /* XBOX_CONTIG_BASE / XBOX_CONTIG_SIZE */
#include "xbox_memory_layout.h"   /* xbox_Nv2aFrameCounterFlip */
/* The swizzle decoder the D3D8 layer already uses -- one implementation of
 * Morton order, not a second one that can disagree with it. */
#include "../d3d/d3d8_swizzle.h"
#include "nv2a_vsh_interp.h"
#include "nv2a_combiner.h"
#include "nv2a_backend.h"
#if defined(XBOXRECOMP_HAVE_VULKAN)
extern const Nv2aBackend *xbox_vk_backend(void);   /* src/gpu_vk */
#endif
#if defined(_WIN32)
#include <windows.h>
#define NV_TLS __declspec(thread)
#else
#define NV_TLS __thread
#endif

extern ptrdiff_t xbox_GetMemoryOffset(void);
extern void xbox_FramebufferWindowSet(uint32_t fb_va, uint32_t pitch);
extern void xbox_FramebufferWindowStart(void);
extern uint32_t g_xbox_image_lo, g_xbox_image_hi;

/* Would writing this surface land on the title's own image?
 *
 * NV097_SET_SURFACE_COLOR_OFFSET is an offset within the colour DMA object,
 * not a guest virtual address, and this executor has always used it as one.
 * That is harmless while the two happen to agree and catastrophic when they do
 * not: the Xbox Dashboard names surface 0x00088000 at 1280x960x4, so clearing
 * it wrote 4.9 MB of opaque black from 0x00088000 to 0x00538000 -- straight
 * over its own code, its D3D context at 0x000BBFC0 and the register-block
 * pointer at 0x000BE2C4. The symptom was a title that submitted one perfect
 * frame and then spun forever in a pushbuffer-full loop, three layers away,
 * with every D3D global reading 0xFF000000: the clear colour.
 *
 * So refuse, and say so. Getting the address right needs the DMA object base
 * this ignores (NV097_SET_CONTEXT_DMA_COLOR); until that exists, writing
 * nothing is strictly better than writing over the guest, and a title that
 * cannot draw is easier to debug than one that has been overwritten.
 */
static int surface_hits_image(uint32_t base, uint32_t bytes)
{
    if (!g_xbox_image_hi || !bytes)
        return 0;
    return base < g_xbox_image_hi && base + bytes > g_xbox_image_lo;
}

/* Where a DMA-object offset actually lives.
 *
 * NV097_SET_SURFACE_COLOR_OFFSET is an offset inside the colour DMA object,
 * and for a framebuffer that object covers physical memory -- so the offset
 * is a physical address, not a guest VA. Those are the same number in this
 * runtime, which is why treating it as a VA works until it does not: on
 * hardware the image is mapped at VA 0x00010000 from arbitrary physical
 * pages, so a framebuffer at physical 0x84000 does not overlap it. Here it
 * would.
 *
 * The title tells us which it is by where it allocated. Half-Life 2's
 * framebuffer comes from MmAllocateContiguousMemory, which this runtime
 * serves from the window at XBOX_CONTIG_BASE, so physical P is visible at
 * XBOX_CONTIG_BASE + P -- clear of the image, and the same bytes the title's
 * own writes and the framebuffer window reach.
 *
 * So: use the offset as a VA when that is credible, and fall back to the
 * physical mirror exactly when it is not. Titles whose surfaces already sit
 * in ordinary RAM (Wreckless renders to the tiled alias of physical
 * 0x01954000) keep the first path and are unaffected.
 */
static uint32_t dma_resolve(uint32_t offset)
{
    extern int xbox_ContiguousOwnsPhys(uint32_t phys);

    /* MmGetPhysicalAddress's answer for image or heap memory: the VA,
     * tagged (XBOX_PHYS_RAM_TAG). */
    if (offset & 0x40000000u)
        return offset & ~0x40000000u;

    /* Did this runtime hand the offset out as contiguous memory? Then the
     * bytes live in the window, and that is not a guess: the arena is a bump
     * allocator from XBOX_CONTIG_BASE, so everything below its high-water
     * mark is memory some MmAllocateContiguousMemory call returned. The
     * title's own writes go through the window, so the executor's must too.
     *
     * Checking this BEFORE the image test is the whole point. The image test
     * only catches an offset that would land on the title's code, and whether
     * it does is an accident of where the image happens to end: Half-Life 2's
     * colour surface is physical 0x00A6C000, which clears the image by 700 KB.
     * So it looked like an ordinary VA, and the executor cleared 1.2 MB of
     * black straight through the guest heap -- which faulted the title three
     * frames later on a pointer that had been overwritten, while the real
     * framebuffer at 0x80A6C000 stayed untouched and the screen stayed black. */
    if (xbox_ContiguousOwnsPhys(offset))
        return XBOX_CONTIG_BASE + offset;
    if (!surface_hits_image(offset, 1))
        return offset;
    if ((uint64_t)offset < XBOX_CONTIG_SIZE)
        return XBOX_CONTIG_BASE + offset;
    return offset;                         /* nothing better to offer */
}

static int surface_write_refused(uint32_t base, uint32_t bytes, const char *what)
{
    static int said;

    if (!surface_hits_image(base, bytes))
        return 0;
    if (!said) {
        said = 1;
        fprintf(stderr,
                "  [GPU] REFUSING to %s surface 0x%08X..0x%08X: that overlaps "
                "the loaded image (0x%08X..0x%08X).\n"
                "  [GPU]   SET_SURFACE_COLOR_OFFSET is a DMA-object offset, not "
                "a guest VA, and this executor treats it as one. Writing here "
                "would destroy the title's own code and globals.\n",
                what, base, base + bytes, g_xbox_image_lo, g_xbox_image_hi);
        fflush(stderr);
    }
    return 1;
}

/* NV097 methods this executor acts on. */
/* Blending. The pair this title programs, read from its own pushbuffer
 * rather than guessed: BLEND_ENABLE written 1168 times and left on,
 * SFACTOR 0x0302 (SRC_ALPHA) and DFACTOR 0x0303 (ONE_MINUS_SRC_ALPHA).
 * ALPHA_TEST_ENABLE is written 390 times and left at zero, so this is
 * blending and not an alpha test. */
#define NV097_SET_BLEND_ENABLE            0x0304
#define NV097_SET_BLEND_FUNC_SFACTOR      0x0344
#define NV097_SET_BLEND_FUNC_DFACTOR      0x0348
#define NV_BLEND_SRC_ALPHA                0x0302
#define NV_BLEND_ONE_MINUS_SRC_ALPHA      0x0303
#define NV097_SET_BLEND_COLOR             0x034C
#define NV097_SET_BLEND_EQUATION          0x0350

#define NV097_SET_SURFACE_CLIP_HORIZONTAL 0x0200
#define NV097_SET_SURFACE_CLIP_VERTICAL   0x0204
#define NV097_SET_SURFACE_FORMAT          0x0208
#define NV097_SET_SURFACE_PITCH           0x020C
#define NV097_SET_SURFACE_COLOR_OFFSET    0x0210
#define NV097_SET_COLOR_CLEAR_VALUE       0x1D90
#define NV097_CLEAR_SURFACE               0x1D94
#define NV097_SET_VERTEX_DATA_ARRAY_OFFSET 0x1720   /* +i*4, 16 attributes */
#define NV097_SET_VERTEX_DATA_ARRAY_FORMAT 0x1760   /* +i*4 */
#define NV097_SET_BEGIN_END               0x17FC
#define NV097_SET_TEXTURE_OFFSET          0x1B00   /* +i*0x40 */
#define NV097_SET_TEXTURE_FORMAT          0x1B04
#define NV097_SET_TEXTURE_PALETTE         0x1B20   /* offset | size<<2 | dma */
#define NV097_SET_TEXTURE_ADDRESS         0x1B08
#define NV097_SET_TEXTURE_CONTROL1        0x1B10
#define NV097_SET_TEXTURE_IMAGE_RECT      0x1B1C
/* The buffer flip. A title double-buffers by telling the GPU which buffer
 * the CRTC reads and which it draws into, advancing the write index and
 * then stalling until the flip has happened. Ignoring these means the
 * stall never clears: Half-Life 2's loader submits its initialisation,
 * asks for a flip, and waits for it in a loop that makes no kernel calls
 * and burns no dispatch, which reads as a hang with no cause.
 *
 * ponytail: the flip completes the moment it is asked for, because there is
 * no scanout to be in the middle of. That makes every frame land instantly
 * and a title that paces itself on the flip runs as fast as it can draw.
 * Pacing wants the vblank clock in the kernel, not a sleep in here. */
#define NV097_SET_FLIP_READ               0x0120
#define NV097_SET_FLIP_WRITE              0x0124
#define NV097_SET_FLIP_MODULO             0x0128
#define NV097_FLIP_INCREMENT_WRITE        0x012C
#define NV097_FLIP_STALL                  0x0130
#define NV097_ARRAY_ELEMENT16             0x1800
/* Draw a run of vertices straight out of the arrays, with no index list:
 * bits 0..23 are the first vertex, bits 24..31 the count minus one. It may
 * appear several times inside one BEGIN_END to draw a longer run. */
#define NV097_DRAW_ARRAYS                 0x1810
#define NV097_INLINE_ARRAY                0x1818
/* Immediate-mode vertices. SET_VERTEX3F/4F carry the position, and writing
 * its last component completes a vertex using whatever the SET_VERTEX_DATA*
 * registers currently hold for the other attributes. This is how Half-Life
 * 2's Xbox loader and the game's own 2D drawing submit every quad -- neither
 * uses INLINE_ARRAY -- so without these the executor saw SET_BEGIN_END pairs
 * with nothing attached and reported `draws 0` while a million and a half
 * textured quads a minute went past it. */
#define NV097_SET_VERTEX3F                0x1500   /* +0..0x08, 3 floats */
#define NV097_SET_VERTEX4F                0x1518   /* +0..0x0C, 4 floats */
#define NV097_SET_VERTEX_DATA2F_M         0x1880   /* + attr*8,  2 floats */
#define NV097_SET_VERTEX_DATA4F_M         0x1A00   /* + attr*16, 4 floats */
#define NV097_SET_VERTEX_DATA4UB          0x1940   /* + attr*4,  4 x u8 */
#define NV097_SET_VERTEX_DATA2S           0x1900   /* + attr*4,  2 x s16 */
#define NV097_SET_VERTEX_DATA4S_M         0x1980   /* + attr*8,  4 x s16 */

/* One immediate vertex, as this file packs it for the shared draw path:
 * position float4, diffuse D3DCOLOR, texcoord0 float2. */
/* An immediate-mode vertex: all 16 attributes as float4. */
#define IMM_VERTEX_DWORDS (NV_VERTEX_ATTRS * 4)

#define NV097_CLEAR_COLOR_MASK            0xF0   /* R,G,B,A bits */

/* One vertex attribute stream, as the title describes it. Attribute 0 is
 * position; the rest are colours, texture coordinates and so on. */
typedef struct {
    uint32_t offset;      /* guest address of element 0 */
    uint32_t type;        /* NV097 data type nibble */
    uint32_t size;        /* components per element */
    uint32_t stride;      /* bytes between elements */
} VertexAttr;

#define NV_VERTEX_ATTRS 16
/* Indices one draw can carry. Was 4096, and the rest of a longer draw was
 * dropped without a word: Blinx 2's sea surface is one 4592-index draw of
 * quads (rows of the camera-facing grid out to 89 degrees, then a row at
 * the horizon), so its last rows were never drawn and the open sea showed
 * the dark base pass beneath. A draw past this cap is logged once. */
#define NV_MAX_INDICES  16384
#define NV_MAX_INLINE   65536           /* dwords of INLINE_ARRAY per batch */

/* Texture stage 0, decoded from what the title programmed.
 *
 * Only stage 0: it is the only one the dashboard configures, and a stage
 * nothing writes to is a stage nothing can be sampled from. The rest arrive
 * as unhandled methods and are counted as such, which is how the next title
 * that needs them will say so. */
typedef struct {
    uint32_t offset;                    /* guest address of texel (0,0)  */
    uint32_t width, height;             /* from IMAGE_RECT               */
    uint32_t pitch;                     /* bytes per row, from CONTROL1  */
    uint32_t color;                     /* NV097 colour-format code      */
    uint32_t addr_u, addr_v;            /* wrap mode per axis            */
    uint32_t palette;                   /* guest address of the CLUT, P8 */
    uint32_t levels;                    /* mip levels, from FORMAT       */
    uint32_t filter;                    /* SET_TEXTURE_FILTER            */
    int      cube;                      /* FORMAT: six faces, not one    */
    int      valid;
    /* BUMPENVMAP: the 2x2 matrix (00 01 10 11) and, for _LUM, the luminance
     * scale and offset (SET_TEXTURE_SET_BUMP_ENV_MAT/_SCALE/_OFFSET). */
    float    bump_mat[4], bump_scale, bump_offset;
    /* Swizzled, non-DXT: texel index of x at swz[x], of y at swz[width + y].
     * Set on the copies a draw carries (pix_capture), not in s_gpu. */
    const uint32_t *swz;
} Texture;

static struct {
    VertexAttr attr[NV_VERTEX_ATTRS];
    uint32_t   prim;                    /* SET_BEGIN_END parameter, 0 = ended */
    uint16_t   idx[NV_MAX_INDICES];
    uint32_t   idx_count;
    /* INLINE_ARRAY payload: vertices written straight into the pushbuffer
     * instead of into a buffer the title points at. Same vertex format, a
     * different place to read them from. */
    uint32_t   inline_buf[NV_MAX_INLINE];
    uint32_t   inline_count;
    /* Current values of the immediate-mode attributes, and how many complete
     * vertices they have produced in this batch. */
    /* Immediate mode: every attribute's current value, as the SET_VERTEX
     * methods leave it. Writing attribute 0 (position) emits a vertex. */
    float      imm_attr[NV_VERTEX_ATTRS][4];
    uint16_t   imm_used;                /* attributes written since BEGIN */
    uint32_t   imm_count;
    int        inline_active;
    uint32_t   draws, verts, nonzero_draws;
    float      min_x, max_x, min_y, max_y;
    uint32_t color_offset, color_base, pitch, format;
    /* The surface the last batch actually drew into. A double-buffered title
     * has already pointed color_offset at the next buffer and cleared it by
     * the time the flip arrives, so dumping the current one dumps the frame
     * that has not been drawn yet -- which is how a correctly rendered
     * sequence came out as 12 black BMPs. */
    uint32_t drawn_offset;
    /* ...and the shape of that surface. A title also draws into small
     * render targets (Burnout 3: 128x128 shadow/reflection maps, pitch 512),
     * often after the main scene, so "last surface drawn" alone picked one of
     * those and read it with the framebuffer's pitch: horizontal noise. The
     * biggest surface drawn since the last flip is the one being presented. */
    uint32_t drawn_pitch, drawn_x, drawn_y, drawn_w, drawn_h, drawn_bpp;
    int drawn_stale;      /* set at a flip: the next draw starts a new frame */
    uint64_t pixels;
    uint32_t pixel_max;   /* brightest value any pixel write carried */
    uint32_t clip_x, clip_w, clip_y, clip_h;
    uint32_t clear_color;
    uint32_t clears, unhandled_total;
    uint32_t flip_read, flip_write, flip_modulo, flips;
    uint32_t tris_drawn, tris_skipped_offscreen, batches_untransformed;
    /* Why a batch came out flat. "Untextured" has two causes that look
     * identical on screen and want opposite fixes: the batch carried no
     * texture coordinates, or it did and the stage was not usable. */
    uint32_t batches_textured, batches_no_uv, batches_no_tex;
    /* The same split for vertex-program batches, which are nearly all of a 3D
     * scene's. A program always outputs texcoords, so the only question is
     * whether any stage had a texture bound. These were never counted, which
     * is why the report said "0 textured" over frames full of textures. */
    uint32_t program_textured, program_no_tex;
    /* Raster batches (rq_kick) and their sizes: ~6 a frame in the hub. */
    uint32_t rq_pooled, rq_inline;     /* inline: no pool (RASTER_THREADS=1) */
    uint64_t rq_tris, rq_px;
    uint64_t xf_parallel_verts;          /* set up on the pool */
    uint32_t blend_enable, blend_sfactor, blend_dfactor;
    uint32_t blend_equation, blend_color;
    uint32_t color_mask;        /* SET_COLOR_MASK: A<<24 R<<16 G<<8 B */
    /* Visibility tests (D3D's Begin/EndVisibilityTest): pixels that pass the
     * depth test while counting is on, reported by GET_REPORT. */
    uint32_t zpass_enable, zpass_count, reports;
    uint32_t blend_pairs[16];   /* sfactor<<16 | dfactor seen, for the report */
    int blend_npairs;
    Texture  texs[4];                   /* one per texture stage */
    /* Register combiners and the per-pixel state around them. rc_seen: the
     * title has programmed the combiners, so they decide every pixel's
     * colour; until then the old "stage 0 texel * diffuse" stands in. */
    Nv2aCombiner rc;
    int      rc_seen;
    uint32_t clip_plane_mode;           /* SET_SHADER_CLIP_PLANE_MODE */
    uint32_t other_stage_input;         /* SET_SHADER_OTHER_STAGE_INPUT */
    uint32_t report_epoch;              /* CLEAR_REPORT_VALUEs so far */
    uint32_t alpha_test, alpha_func, alpha_ref;
    uint32_t fog_enable, fog_mode, fog_color;
    float    fog_param[2];
    /* Vertex programs (nv2a_vsh_interp.c) and the depth buffer, which
     * together are what a 3D scene needs and a 2D one never used. */
    uint32_t xf_mode;                   /* TRANSFORM_EXECUTION_MODE: 2 = program */
    uint32_t batches_program, verts_program, tris_behind;
    /* Where program-path triangles go, so "nothing drew" has a reason. */
    uint32_t xf_degenerate, xf_offscreen, xf_drawn;
    uint64_t xf_depth_fail, xf_pixels;
    float xf_min[3], xf_max[3];
    uint32_t xf_fmt[16];  /* attribute formats seen: type | size<<4 | slot<<8 */
    int xf_nfmt, xf_seeded;
    uint32_t depth_test, depth_func, depth_mask;
    /* Stencil (SET_STENCIL_*): test enable, function/reference/compare mask,
     * write mask and the three operations, GL enums as the NV2A takes. */
    uint32_t st_test, st_func, st_ref, st_fmask, st_wmask;
    uint32_t st_fail, st_zfail, st_zpass;
    /* Face culling: CULL_FACE_ENABLE, CULL_FACE (0x404 front, 0x405 back,
     * 0x408 both), FRONT_FACE (0x900 CW, 0x901 CCW). */
    uint32_t cull_enable, cull_face, front_face;
    uint32_t tris_culled;
    uint32_t zeta_offset, zstencil_clear;
    uint32_t sem_offset, sem_releases;   /* semaphore (D3D fences) */
    uint64_t csum;                       /* shaded colour sum (FRAME_SUMMARY=2) */
} s_gpu;

/* Deferred rasterisation (below): drain everything queued. Counted by call
 * site, because a drain is where the executor and the raster workers stop
 * overlapping, and which ones a frame pays for is the question. */
static void rq_sync_at(int line);
#define rq_sync() rq_sync_at(__LINE__)

/* Unhandled methods, ranked. The interesting output is not that something was
 * skipped but which things dominate, because that is the order to implement
 * them in. */
#define PB_EXEC_MAX_UNHANDLED 2048
typedef struct { uint32_t method, count, last_param; } PbUnhandled;
static PbUnhandled s_unhandled[PB_EXEC_MAX_UNHANDLED];
static int s_unhandled_count;

/* Every texture-stage register, as the title last set it.
 * Texturing is not implemented yet; knowing which formats and sizes a title
 * actually programs is what decides which ones are worth implementing. */
#define NV_TEX_FIRST 0x1B00
#define NV_TEX_LAST  0x1BFC
static uint32_t s_tex_reg[(NV_TEX_LAST - NV_TEX_FIRST) / 4 + 1];
static uint8_t  s_tex_set[(NV_TEX_LAST - NV_TEX_FIRST) / 4 + 1];

/* Formats whose dimensions come from the format word and whose coordinates
 * arrive normalised, rather than from a pitch and SET_TEXTURE_IMAGE_RECT with
 * coordinates in texels. Swizzled and block-compressed are both in this group,
 * and every place that used to test only for swizzled needs the pair. */
static int tex_size_from_format(uint32_t fmt)
{
    return d3d8_format_is_swizzled(fmt) || d3d8_format_dxt_block_bytes(fmt);
}

static void tex_update_valid(Texture *t)
{
    t->valid = t->offset && t->width && t->height
            && (tex_size_from_format(t->color) || t->pitch);
}

static void record_tex_reg(uint32_t method, uint32_t param)
{
    s_tex_reg[(method - NV_TEX_FIRST) / 4] = param;
    s_tex_set[(method - NV_TEX_FIRST) / 4] = 1;
    /* A pitch is a linear texture's property. A swizzled one has no rows and
     * so no pitch, and requiring one here refused every swizzled texture --
     * which is nearly all of them, since swizzled is the Xbox default. That
     * left the title's own textures unsampled and every textured quad drawn in
     * flat vertex colour. */
    tex_update_valid(&s_gpu.texs[0]);
}

/* Every distinct texture a batch was drawn with, and how many batches used it.
 *
 * The per-draw verbose print shows the first few draws of the first frame,
 * which is enough to see that texturing works at all and not enough to answer
 * "is a font page ever bound". This is the same shape as the unhandled-method
 * table below it: a small set, ranked, printed with the rest of the report. */
#define PB_EXEC_MAX_TEXTURES 64
typedef struct {
    uint32_t offset, color, width, height, batches;
} PbTexUse;
static PbTexUse s_tex_use[PB_EXEC_MAX_TEXTURES];
static int s_tex_use_count;

/* Defined below, next to the sampler it goes through. */
static void dump_texture_bmp(uint32_t seq);

/* Repeat dumps are numbered from well past the first-use sequence, so a
 * listing sorts them after the textures they came from and no first-use file
 * is ever overwritten by one. */
#define TEX_DUMP_SEQ_BASE 1000u
#define TEX_DUMP_SEQ_MAX  40u

static void note_texture_use(void)
{
    int i;

    if (!s_gpu.texs[0].valid)
        return;
    for (i = 0; i < s_tex_use_count; i++) {
        if (s_tex_use[i].offset == s_gpu.texs[0].offset
         && s_tex_use[i].color  == s_gpu.texs[0].color) {
            s_tex_use[i].batches++;
            /* Dump a surface that is redrawn, every Nth time it is bound.
             *
             * First use alone cannot tell a decode error that is wrong in
             * every frame from one that accumulates across them. A block
             * transform that is wrong is wrong on its own, in the keyframe
             * as much as anywhere; motion compensation that is wrong starts
             * from a clean keyframe and smears further with each predicted
             * frame after it. In a single frame the two look identical, and
             * in a sequence they look nothing alike -- so the sequence is
             * what has to be captured.
             *
             * It belongs on this side of the return: a video surface keeps
             * one address for the whole film, so after the first frame it is
             * only ever found here, and the first-use dump below never fires
             * for it again. RECOMP_TEX_DUMP_EVERY=<n> sets the interval, and
             * RECOMP_TEX_DUMP still names the files. */
            {
                static int every = -1;
                static unsigned binds, seq;
                if (every < 0) {
                    const char *e = getenv("RECOMP_TEX_DUMP_EVERY");
                    every = e ? atoi(e) : 0;
                }
                if (every > 0 && ++binds % (unsigned)every == 0
                    && seq < TEX_DUMP_SEQ_MAX)
                    dump_texture_bmp(TEX_DUMP_SEQ_BASE + seq++);
            }
            return;
        }
    }
    if (s_tex_use_count < PB_EXEC_MAX_TEXTURES) {
        s_tex_use[s_tex_use_count].offset  = s_gpu.texs[0].offset;
        s_tex_use[s_tex_use_count].color   = s_gpu.texs[0].color;
        s_tex_use[s_tex_use_count].width   = s_gpu.texs[0].width;
        s_tex_use[s_tex_use_count].height  = s_gpu.texs[0].height;
        s_tex_use[s_tex_use_count].batches = 1;
        s_tex_use_count++;
        dump_texture_bmp((uint32_t)s_tex_use_count - 1);
    }
}

static void note_unhandled(uint32_t method, uint32_t param)
{
    int i;

    s_gpu.unhandled_total++;
    for (i = 0; i < s_unhandled_count; i++) {
        if (s_unhandled[i].method == method) {
            s_unhandled[i].count++;
            s_unhandled[i].last_param = param;
            return;
        }
    }
    if (s_unhandled_count < PB_EXEC_MAX_UNHANDLED) {
        s_unhandled[s_unhandled_count].method = method;
        s_unhandled[s_unhandled_count].count = 1;
        s_unhandled[s_unhandled_count].last_param = param;
        s_unhandled_count++;
    }
}

/* Read attribute `a` of vertex `index` as floats. Only the float and the
 * normalised-byte types appear in practice; anything else returns 0 so a
 * caller sees a degenerate vertex rather than reading past the array. */
static int fetch_attr(const VertexAttr *a, uint32_t index, float out[4])
{
    const uint8_t *mem = (const uint8_t *)xbox_GetMemoryOffset();
    const uint8_t *p;
    uint32_t i;

    out[0] = out[1] = out[2] = 0.0f;
    out[3] = 1.0f;
    if (!a->size || !a->stride)
        return 0;
    if (s_gpu.inline_active) {
        /* The batch arrived as INLINE_ARRAY, so `offset` is a byte offset into
         * the buffered payload rather than a guest address -- and 0 is a legal
         * one there, which is why the offset test is on the other side of this
         * branch. */
        size_t at = (size_t)a->offset + (size_t)index * a->stride;
        if (at + 4 > (size_t)s_gpu.inline_count * 4)
            return 0;
        p = (const uint8_t *)s_gpu.inline_buf + at;
    } else {
        if (!a->offset)
            return 0;
        p = mem + a->offset + (size_t)index * a->stride;
    }

    switch (a->type) {
    case 0:                                  /* D3DCOLOR */
        /* A DWORD 0xAARRGGBB, so little-endian bytes are B,G,R,A -- not the
         * component order of every other format here. Returned as R,G,B,A so
         * callers need not know which format the title chose. */
        out[0] = (float)p[2] / 255.0f;
        out[1] = (float)p[1] / 255.0f;
        out[2] = (float)p[0] / 255.0f;
        out[3] = (float)p[3] / 255.0f;
        return 1;
    case 2:                                  /* float */
        for (i = 0; i < a->size && i < 4; i++)
            out[i] = ((const float *)p)[i];
        return 1;
    case 4:                                  /* unsigned byte, normalised */
        for (i = 0; i < a->size && i < 4; i++)
            out[i] = (float)p[i] / 255.0f;
        return 1;
    /* The formats 3D geometry uses and screen-space quads never did. */
    case 1:                                  /* signed short, normalised */
        for (i = 0; i < a->size && i < 4; i++)
            out[i] = (float)((const int16_t *)p)[i] / 32767.0f;
        return 1;
    case 5:                                  /* signed short, as is */
        for (i = 0; i < a->size && i < 4; i++)
            out[i] = (float)((const int16_t *)p)[i];
        return 1;
    case 6: {                                /* packed 11:11:10 normal */
        uint32_t v = *(const uint32_t *)p;
        out[0] = (float)((int32_t)(v << 21) >> 21) / 1023.0f;
        out[1] = (float)((int32_t)(v << 10) >> 21) / 1023.0f;
        out[2] = (float)((int32_t)v >> 22) / 511.0f;
        return 1;
    }
    default:
        return 0;
    }
}

/* Vertex-program input `a` of vertex `index`. An attribute with no array
 * (size 0, which is how D3D disables a slot) reads its inline register --
 * whatever SET_VERTEX_DATA* last left there -- not (0,0,0,1): that is what
 * the NV2A does (xemu pgraph uses the attribute's inline_value). D3D puts a
 * constant diffuse, a material colour or a tint there for geometry whose
 * vertices carry none, and reading zeros would draw those batches black. */
static void fetch_input(uint32_t a, uint32_t index, float out[4])
{
    if (!s_gpu.attr[a].size) {
        memcpy(out, s_gpu.imm_attr[a], 4 * sizeof(float));
        return;
    }
    fetch_attr(&s_gpu.attr[a], index, out);
}

static uint32_t surface_bpp(void);

static void note_drawn(void)
{
    if (!s_gpu.drawn_stale && s_gpu.drawn_offset
        && s_gpu.clip_w * s_gpu.clip_h < s_gpu.drawn_w * s_gpu.drawn_h)
        return;
    s_gpu.drawn_stale = 0;
    s_gpu.drawn_offset = s_gpu.color_offset;
    s_gpu.drawn_pitch = s_gpu.pitch;
    s_gpu.drawn_bpp = surface_bpp();
    s_gpu.drawn_x = s_gpu.clip_x; s_gpu.drawn_y = s_gpu.clip_y;
    s_gpu.drawn_w = s_gpu.clip_w; s_gpu.drawn_h = s_gpu.clip_h;
}

static uint32_t surface_bpp(void)
{
    /* The surface format says it outright (NV097_SET_SURFACE_FORMAT_COLOR).
     * Deriving it from pitch / clip width -- the only way before -- is right
     * only while the clip spans the whole surface: Burnout 3 narrows the clip
     * to a 250-pixel window to draw its option values, 2560 / 250 came out as
     * 10 bytes a pixel, and every such text quad was refused. */
    switch (s_gpu.format & 0xF) {
    case 0x1: case 0x2: case 0x3: case 0xA:
        return 2;
    case 0x4: case 0x5: case 0x6: case 0x7: case 0x8:
        return 4;
    case 0x9:
        return 1;
    default:
        break;
    }
    /* The pitch and the clip width together give the pixel size, which is more
     * reliable than decoding the format field: the format's colour code is
     * only meaningful alongside a type the title also sets, while the pitch is
     * always exactly how many bytes a row occupies. */
    if (!s_gpu.clip_w)
        return 0;
    return s_gpu.pitch / s_gpu.clip_w;
}


/* Write the current surface out as a 24-bit BMP.
 *
 * A framebuffer window needs someone watching it. A file does not, which makes
 * this the only way to check what a title actually rendered on a machine you
 * are not sitting at -- and the only way to put a picture in a bug report.
 *
 * ponytail: bottom-up 24bpp BMP, no palette, no compression. That is the one
 * format every viewer reads and it is 30 lines; PNG would need a dependency.
 */
/* Set while RECOMP_FB_DUMP_FLIPS is dumping every flip: the report and
 * after-draw dumps stand down then, or they would put duplicate frames into
 * the sequence. */
static int s_flip_dumping;
static int s_flip_naming;   /* RECOMP_FB_DUMP_FLIPS is set: see dump names */

static void dump_surface_bmp(void)
{
    rq_sync();
    const char *prefix = getenv("RECOMP_FB_DUMP");
    const uint8_t *mem = (const uint8_t *)xbox_GetMemoryOffset();
    int drawn = s_gpu.drawn_offset != 0;
    uint32_t offset = drawn ? s_gpu.drawn_offset : s_gpu.color_offset;
    uint32_t pitch = drawn ? s_gpu.drawn_pitch : s_gpu.pitch;
    uint32_t cx = drawn ? s_gpu.drawn_x : s_gpu.clip_x;
    uint32_t cy = drawn ? s_gpu.drawn_y : s_gpu.clip_y;
    static int seq;
    char path[512];
    uint32_t w = drawn ? s_gpu.drawn_w : s_gpu.clip_w;
    uint32_t h = drawn ? s_gpu.drawn_h : s_gpu.clip_h, y, x;
    uint32_t bpp = drawn ? s_gpu.drawn_bpp : surface_bpp();
    uint32_t row_bytes, pad, filesz;
    uint8_t hdr[54];
    FILE *f;

    if (!prefix || !w || !h || (bpp != 2 && bpp != 4) || !offset)
        return;

    row_bytes = w * 3;
    pad = (4 - (row_bytes & 3)) & 3;
    filesz = 54 + (row_bytes + pad) * h;

    /* Flip dumps are named by flip number, which the frame summary
     * (RECOMP_FRAME_SUMMARY) prints too, so a bad frame finds its draws. */
    /* Other dumps (the periodic report's, the first after-draw ones) are
     * taken mid-frame. Numbered like the flips they overwrote flip dumps of
     * the same number with half-drawn frames -- no HUD, no reflection yet --
     * which looked exactly like a rendering glitch. So under flip dumping
     * they get an 'r' and their own sequence. */
    if (s_flip_dumping)
        snprintf(path, sizeof path, "%s%05u.bmp", prefix, s_gpu.flips);
    else if (s_flip_naming)
        snprintf(path, sizeof path, "%sr%05d.bmp", prefix, seq);
    else
        snprintf(path, sizeof path, "%s%05d.bmp", prefix, seq);
    seq++;
    f = fopen(path, "wb");
    if (!f)
        return;

    memset(hdr, 0, sizeof hdr);
    hdr[0] = 'B'; hdr[1] = 'M';
    memcpy(hdr + 2, &filesz, 4);
    hdr[10] = 54;
    hdr[14] = 40;
    memcpy(hdr + 18, &w, 4);
    memcpy(hdr + 22, &h, 4);
    hdr[26] = 1;
    hdr[28] = 24;
    fwrite(hdr, 1, sizeof hdr, f);

    /* BMP rows run bottom-up. */
    for (y = h; y-- > 0; ) {
        const uint8_t *row = mem + dma_resolve(offset)
                           + (size_t)(cy + y) * pitch;
        for (x = 0; x < w; x++) {
            uint8_t bgr[3];
            if (bpp == 4) {
                uint32_t v = ((const uint32_t *)row)[cx + x];
                bgr[0] = (uint8_t)(v);
                bgr[1] = (uint8_t)(v >> 8);
                bgr[2] = (uint8_t)(v >> 16);
            } else {
                uint16_t v = ((const uint16_t *)row)[cx + x];
                bgr[0] = (uint8_t)(( v        & 0x1F) << 3);
                bgr[1] = (uint8_t)(((v >>  5) & 0x3F) << 2);
                bgr[2] = (uint8_t)(((v >> 11) & 0x1F) << 3);
            }
            fwrite(bgr, 1, 3, f);
        }
        if (pad) {
            static const uint8_t zero[3] = {0, 0, 0};
            fwrite(zero, 1, pad, f);
        }
    }
    fclose(f);
    if (seq == 1)
        fprintf(stderr, "  [GPU] framebuffer dump: %s (%ux%u from 0x%08X %ubpp)\n",
                path, w, h, s_gpu.color_offset, bpp);
}

/* Defined below, next to the rest of the rasteriser; the clear path uses it
 * for RECOMP_RASTER_TEST. */
static void raster_triangle(const float a[2], const float b[2],
                            const float c[2], uint32_t argb,
                            const float uv[3][2]);

static uint8_t *s_surface;          /* host address of surface row 0 */

/* Swizzled colour surfaces. SURFACE_FORMAT's type field (bits 8-11) is 1 for
 * a pitch surface and 2 for a swizzled one, whose size is log2 in bits 16-23
 * (width) and 24-31 (height). A title renders into a swizzled surface when it
 * means to sample the result as an ordinary swizzled texture -- Blinx 2's hub
 * floor reflection is two 512x512 A8R8G8B8 render targets. Written row by row
 * with the pitch they came out as a linear image, and sampling that as
 * swizzled drew the floor as stair-stepped bands.
 * The tables hold each coordinate's byte offset, rebuilt when the format
 * changes, so a pixel costs two loads and an OR. */
#define NV_SWZ_MAX 4096
#define NV_SWZ_TABLES 4
typedef struct {
    uint32_t format;                    /* SURFACE_FORMAT the tables are for */
    uint32_t w, h;
    uint32_t x[NV_SWZ_MAX], y[NV_SWZ_MAX];
} SwzTable;
/* A few, by format: with rasterisation deferred, queued triangles may still
 * be writing through one table when the title moves on to a surface of
 * another size. */
static SwzTable s_swz_tab[NV_SWZ_TABLES];
static int s_swz_used, s_swz_next;

/* The current colour surface's swizzle tables, or NULL for a pitch surface. */
static const SwzTable *surface_swizzled(void)
{
    uint32_t f = s_gpu.format, lw, lh, mx, my, i, bpp;
    SwzTable *t;
    int k;

    lw = (f >> 16) & 0xFF;
    lh = (f >> 24) & 0xFF;
    if (((f >> 8) & 0xF) != 2 || lw > 12 || lh > 12)
        return NULL;
    for (k = 0; k < s_swz_used; k++)
        if (s_swz_tab[k].format == f)
            return &s_swz_tab[k];
    if (s_swz_used < NV_SWZ_TABLES) {
        k = s_swz_used++;
    } else {
        rq_sync();                    /* queued triangles may use the evictee */
        k = s_swz_next++ % NV_SWZ_TABLES;
    }
    t = &s_swz_tab[k];
    switch (f & 0xF) {                            /* as surface_bpp() */
    case 0x1: case 0x2: case 0x3: case 0xA: bpp = 2; break;
    case 0x9: bpp = 1; break;
    default: bpp = 4; break;
    }
    t->format = f;
    t->w = 1u << lw;
    t->h = 1u << lh;
    xbox_swizzle_masks(t->w, t->h, &mx, &my);
    for (i = 0; i < t->w; i++)
        t->x[i] = swizzle_deposit(i, mx) * bpp;
    for (i = 0; i < t->h; i++)
        t->y[i] = swizzle_deposit(i, my) * bpp;
    return t;
}

/* Texel-index tables for swizzled textures, one per size, built on the
 * executor and never freed (13 x 13 sizes at most). sample_tex used to
 * recompute the masks and deposit every coordinate bit by bit per texel --
 * four times per bilinear tap, a tenth of the pixel path in Blinx 2's hub. */
static const uint32_t *tex_swizzle_table(const Texture *t)
{
    static uint32_t *tabs[13][13];
    uint32_t lw, lh, mx, my, i, *tab;

    if (!t->valid || !d3d8_format_is_swizzled(t->color)
        || d3d8_format_dxt_block_bytes(t->color))
        return NULL;
    for (lw = 0; lw < 13 && (1u << lw) != t->width; lw++) ;
    for (lh = 0; lh < 13 && (1u << lh) != t->height; lh++) ;
    if (lw == 13 || lh == 13)
        return NULL;
    if (tabs[lw][lh])
        return tabs[lw][lh];
    tab = (uint32_t *)malloc(sizeof(uint32_t) * (t->width + t->height));
    if (!tab)
        return NULL;
    xbox_swizzle_masks(t->width, t->height, &mx, &my);
    for (i = 0; i < t->width; i++)
        tab[i] = swizzle_deposit(i, mx);
    for (i = 0; i < t->height; i++)
        tab[t->width + i] = swizzle_deposit(i, my);
    tabs[lw][lh] = tab;
    return tab;
}

/* Everything a pixel needs from the GPU state, copied once per draw call.
 *
 * Rasterisation is deferred (see "Deferred rasterisation") and runs on worker
 * threads while the executor walks on and changes the live state, so the
 * pixel path must not read s_gpu. It reads P, the state of the triangle the
 * calling thread is drawing: xf_rows sets it per triangle, and drawing done
 * directly on the executor sets it to s_live. */
typedef struct {
    uint8_t *surface;                   /* host address of the colour surface */
    const SwzTable *swz;                /* NULL: a pitch surface */
    uint32_t pitch;
    uint32_t clip_x, clip_y, clip_w, clip_h;
    uint32_t color_mask, alpha_test, alpha_func, alpha_ref;
    uint32_t blend_enable, blend_sfactor, blend_dfactor, blend_equation;
    uint32_t blend_color;
    uint32_t depth_test, depth_func, depth_mask, zpass_enable;
    uint32_t st_test, st_func, st_ref, st_fmask, st_wmask;
    uint32_t st_fail, st_zfail, st_zpass;
    uint32_t clip_plane_mode;
    uint32_t other_stage_input;         /* which stage a stage 2/3 reads */
    uint32_t report_epoch;              /* which visibility test counts it */
    Nv2aCombinerPrep rcp;               /* constants unpacked per draw */
    Texture texs[4];
    uint32_t self_sample;               /* a stage samples the colour surface */
} PixState;

static NV_TLS const PixState *P;
static PixState s_live;

static uint32_t tex_face_stride(const Texture *t);

static void pix_capture(PixState *ps)
{
    memset(ps, 0, sizeof *ps);            /* padding too: states are memcmp'd */
    ps->surface = s_surface;
    ps->swz = surface_swizzled();
    ps->pitch = s_gpu.pitch;
    ps->clip_x = s_gpu.clip_x; ps->clip_y = s_gpu.clip_y;
    ps->clip_w = s_gpu.clip_w; ps->clip_h = s_gpu.clip_h;
    ps->color_mask = s_gpu.color_mask;
    ps->alpha_test = s_gpu.alpha_test;
    ps->alpha_func = s_gpu.alpha_func;
    ps->alpha_ref = s_gpu.alpha_ref;
    ps->blend_enable = s_gpu.blend_enable;
    ps->blend_sfactor = s_gpu.blend_sfactor;
    ps->blend_dfactor = s_gpu.blend_dfactor;
    ps->blend_equation = s_gpu.blend_equation;
    ps->blend_color = s_gpu.blend_color;
    ps->depth_test = s_gpu.depth_test;
    ps->depth_func = s_gpu.depth_func;
    ps->depth_mask = s_gpu.depth_mask;
    {
        static int off = -1;               /* RECOMP_NO_STENCIL=1: A/B */
        if (off < 0)
            off = getenv("RECOMP_NO_STENCIL") != NULL;
        ps->st_test = s_gpu.st_test && !off;
    }
    ps->st_func = s_gpu.st_func;
    ps->st_ref = s_gpu.st_ref & 0xFF;
    ps->st_fmask = s_gpu.st_fmask & 0xFF;
    ps->st_wmask = s_gpu.st_wmask & 0xFF;
    ps->st_fail = s_gpu.st_fail;
    ps->st_zfail = s_gpu.st_zfail;
    ps->st_zpass = s_gpu.st_zpass;
    ps->zpass_enable = s_gpu.zpass_enable;
    ps->clip_plane_mode = s_gpu.clip_plane_mode;
    ps->other_stage_input = s_gpu.other_stage_input;
    ps->report_epoch = s_gpu.report_epoch;
    nv2a_rc_prepare(&s_gpu.rc, &ps->rcp);
    memcpy(ps->texs, s_gpu.texs, sizeof ps->texs);
    {
        int st;
        for (st = 0; st < 4; st++)
            ps->texs[st].swz = tex_swizzle_table(&ps->texs[st]);
    }
    /* A draw that samples the surface it draws into reads its own fresh
     * pixels, so its result depends on the order pixels are written in
     * (undefined on the console too). The four-lane path writes in a
     * different order than the scalar one, so such draws stay scalar. */
    if (s_gpu.color_offset && s_gpu.pitch) {
        uint32_t lo = dma_resolve(s_gpu.color_offset), hi;
        int st;
        hi = lo + (ps->swz ? ps->swz->w * ps->swz->h * surface_bpp()
                           : (s_gpu.clip_y + s_gpu.clip_h) * s_gpu.pitch);
        for (st = 0; st < 4; st++) {
            const Texture *t = &s_gpu.texs[st];
            uint32_t tlo, thi;
            if (!t->valid)
                continue;
            if (s_gpu.rc_seen ? !((s_gpu.rc.stage_program >> (st * 5)) & 0x1F)
                              : st != 0)
                continue;
            tlo = t->offset;
            thi = tlo + tex_face_stride(t) * (t->cube ? 6u : 1u);
            if (tlo < hi && lo < thi)
                ps->self_sample = 1;
        }
    }
}

/* Drawing straight from the executor thread (screen-space batches, texture
 * dumps): whatever is queued goes first, then the live state applies. */
static void pix_direct(void)
{
    rq_sync();
    pix_capture(&s_live);
    P = &s_live;
}

/* Host address of surface pixel (x, y), swizzled or pitch. NULL when a
 * swizzled surface's clip reaches past its own size. */
static uint8_t *surface_pixel(int x, int y, uint32_t bpp)
{
    const SwzTable *t = P->swz;
    if (t) {
        if ((uint32_t)x >= t->w || (uint32_t)y >= t->h)
            return NULL;
        return P->surface + t->x[x] + t->y[y];
    }
    return P->surface + (size_t)y * P->pitch + (size_t)x * bpp;
}

static int surface_begin_batch(const uint8_t *mem)
{
    uint32_t base = dma_resolve(s_gpu.color_offset);

    if (surface_hits_image(base, (s_gpu.clip_y + s_gpu.clip_h) * s_gpu.pitch))
        return 0;
    s_surface = (uint8_t *)mem + base;
    return 1;
}

static int rq_async(void);
static int rq_clear_job(uint8_t *mem, uint32_t bpp);

static void clear_surface(uint32_t param)
{
    uint8_t *mem = (uint8_t *)xbox_GetMemoryOffset();
    uint32_t bpp = surface_bpp();
    uint32_t y, x;
    const SwzTable *swz;

    if (!rq_async())
        rq_sync();                        /* queued pixels land first */
    if (!(param & NV097_CLEAR_COLOR_MASK))
        return;                            /* depth/stencil only */
    if (!s_gpu.color_offset || !s_gpu.pitch || !s_gpu.clip_h || bpp == 0)
        return;
    {
        uint32_t base = dma_resolve(s_gpu.color_offset);
        if (surface_write_refused(base,
                                  (s_gpu.clip_y + s_gpu.clip_h) * s_gpu.pitch,
                                  "clear"))
            return;
        s_gpu.color_base = base;
    }

    if (rq_async() && rq_clear_job(mem, bpp)) {
        y = s_gpu.clip_h;                 /* queued: skip both pixel loops */
        swz = NULL;
    } else if ((swz = surface_swizzled()) != NULL) {
        /* Pixel by pixel through the swizzle tables. A clip covering the whole
         * surface could be one fill, but clears are not where the time goes. */
        uint16_t v16 = (uint16_t)(((s_gpu.clear_color >> 8) & 0xF800)
                                | ((s_gpu.clear_color >> 5) & 0x07E0)
                                | ((s_gpu.clear_color >> 3) & 0x001F));
        for (y = s_gpu.clip_y; y < s_gpu.clip_y + s_gpu.clip_h && y < swz->h; y++)
            for (x = s_gpu.clip_x; x < s_gpu.clip_x + s_gpu.clip_w && x < swz->w; x++) {
                uint8_t *p = mem + s_gpu.color_base + swz->x[x] + swz->y[y];
                if (bpp == 4)
                    *(uint32_t *)p = s_gpu.clear_color;
                else if (bpp == 2)
                    *(uint16_t *)p = v16;
            }
        y = s_gpu.clip_h;                         /* skip the pitch loop */
    } else
        y = 0;
    for (; y < s_gpu.clip_h; y++) {
        uint8_t *row = mem + s_gpu.color_base
                     + (size_t)(s_gpu.clip_y + y) * s_gpu.pitch;
        if (bpp == 4) {
            uint32_t *p = (uint32_t *)row + s_gpu.clip_x;
            for (x = 0; x < s_gpu.clip_w; x++)
                p[x] = s_gpu.clear_color;
        } else if (bpp == 2) {
            /* The clear value is always given as A8R8G8B8; a 16-bit surface
             * takes the same colour reduced to 5:6:5. */
            uint16_t v = (uint16_t)(((s_gpu.clear_color >> 8) & 0xF800)
                                  | ((s_gpu.clear_color >> 5) & 0x07E0)
                                  | ((s_gpu.clear_color >> 3) & 0x001F));
            uint16_t *p = (uint16_t *)row + s_gpu.clip_x;
            for (x = 0; x < s_gpu.clip_w; x++)
                p[x] = v;
        }
    }
    s_gpu.clears++;
    /* Progress markers, interleaved with everything else in the log. The
     * summary says drawing stopped; only a marker next to the surrounding
     * activity says what the title was doing when it stopped. */
    if ((s_gpu.clears % 100) == 0)
        fprintf(stderr, "  [GPU] clear #%u\n", s_gpu.clears);
    /* Distinct clear colours actually used. "Cleared to black" and "the clear
     * never ran" look identical in the framebuffer, and only one of them is a
     * bug -- so record what was asked for, not just how often. */
    {
        static uint32_t seen[8];
        static int n;
        int i;
        for (i = 0; i < n; i++)
            if (seen[i] == s_gpu.clear_color) break;
        if (i == n && n < 8) {
            seen[n++] = s_gpu.clear_color;
            fprintf(stderr, "  [GPU] clear colour 0x%08X -> surface 0x%08X"
                            " (%ubpp)\n",
                    s_gpu.clear_color, s_gpu.color_offset, surface_bpp());
        }
    }

    /* Prove the pixel path end to end, independent of whether the title has
     * given us any geometry yet.
     *
     * "Nothing on screen" has three very different causes -- the surface
     * address or pitch is wrong, the rasteriser is broken, or the title's
     * vertex buffers are empty -- and they are indistinguishable from a black
     * window. RECOMP_RASTER_TEST draws one known triangle into the surface
     * just cleared, so a visible triangle rules out the first two and leaves
     * only the third. On Wreckless it is the third: attribute 0 decodes
     * correctly (float, size 2, stride 16) and the buffer it points at stays
     * zero.
     *
     * ponytail: bring-up aid, not a feature. It costs one branch per clear. */
    static int raster_test = -1;
    if (raster_test < 0)
        raster_test = getenv("RECOMP_RASTER_TEST") != NULL;
    if (raster_test) {
        static int announced;
        /* Every clear, not once: the title clears each frame and double-buffers,
         * so a triangle drawn a single time is erased before anyone sees it. */
        if (s_gpu.clip_w && s_gpu.clip_h) {
            float a[2], b[2], c[2];
            a[0] = s_gpu.clip_w * 0.5f; a[1] = s_gpu.clip_h * 0.15f;
            b[0] = s_gpu.clip_w * 0.85f; b[1] = s_gpu.clip_h * 0.85f;
            c[0] = s_gpu.clip_w * 0.15f; c[1] = s_gpu.clip_h * 0.85f;
            raster_triangle(a, b, c, 0xFFFF00FFu, NULL);  /* magenta: never a clear colour */
            if (announced++ == 0)
            fprintf(stderr, "  [GPU] raster self-test: triangle (%.0f,%.0f)"
                            " (%.0f,%.0f) (%.0f,%.0f) into 0x%08X %ubpp\n",
                    a[0], a[1], b[0], b[1], c[0], c[1],
                    s_gpu.color_offset, surface_bpp());
        }
    }

    /* Show the surface actually being drawn into. A title that double-buffers
     * renders into the back buffer, so following AvSetDisplayMode's address
     * would show the one nothing is writing. */
    /* The window has to read where the pixels actually are, which is the
     * resolved address rather than the DMA-object offset. */
    /* Only until the title flips. Following the draw surface on every scan
     * shows the buffer being written right now, half a frame at a time; past
     * the first flip the window is repointed at the finished one instead. */
    if (s_gpu.flips == 0)
        xbox_FramebufferWindowSet(dma_resolve(s_gpu.color_offset), s_gpu.pitch);

    /* And open the window, rather than waiting for AvSetDisplayMode to do it.
     *
     * That was the only caller, so a title which draws before setting a display
     * mode -- or never sets one at all -- got no window however much it
     * rendered. The Xbox Dashboard clears a 1280x960 surface at 0x00088000 on
     * its first frame and had not called AvSetDisplayMode by then, so
     * RECOMP_FB_WINDOW=1 was set, the executor knew the address and the pitch,
     * and nothing appeared.
     *
     * Here is the better trigger anyway: this runs when a surface address is
     * known to be real, because a clear just used it. Idempotent and gated on
     * RECOMP_FB_WINDOW, so the cost is one interlocked compare per clear. */
    xbox_FramebufferWindowStart();
}


/* ── Rasteriser ──────────────────────────────────────────────────────────
 *
 * Fills triangles straight into the guest framebuffer, the same memory
 * clear_surface() writes and the framebuffer window already shows. That is the
 * whole reason it is done on the CPU rather than through D3D: nothing new has
 * to be plumbed for the result to be visible.
 *
 * ponytail: flat-shaded, no depth buffer, no texturing, no perspective
 * correction, and only batches whose attribute 0 is already in screen space.
 * A title running a vertex program hands over object-space positions that mean
 * nothing without executing the program, so those batches are counted and
 * skipped rather than drawn somewhere wrong. Upgrade path is the D3D11
 * translator in src/nv2a/nv2a_pgraph_d3d11.c once vertex programs are
 * translated; this exists to get the first geometry on screen for every title,
 * which in practice is UI, HUD and 2D overlays -- all pre-transformed.
 */

/* One texel, in the title's own format.
 *
 * The codes are the NV097 colour field, which is the Xbox D3DFMT_ enum --
 * src/d3d/d3d8_xbox.h is the table, and it is the table to check against
 * rather than recollection: 0x1E is LIN_X8R8G8B8 and not, as this first read
 * it, a byte-reversed BGRA. Getting that one wrong turned an opaque black
 * render target into a screen of pure blue, which is the kind of wrong that
 * looks like content.
 *
 * Only the linear (LIN_) formats are read. A swizzled texture stores its
 * texels in Morton order rather than in rows, so reading one as if it had a
 * pitch does not give a slightly wrong colour, it gives a different image --
 * and inventing that image is exactly what this is not for. An unsupported
 * format samples nothing and the caller keeps the vertex colour, which is
 * visibly wrong rather than quietly wrong.
 *
 * ponytail: nearest texel, no filtering, whatever SET_TEXTURE_FILTER asked
 * for. Bilinear when a title's output actually depends on it.
 */
/* Off the edge of the texture, the way the title asked for.
 *
 * Refusing to sample instead is not neutral: it hands the caller back the
 * vertex colour, so a pass whose coordinates reach the last texel by half a
 * texel gets a bright line down the edge of the screen. The dashboard's
 * resolve does exactly that -- its last column and last row, 1119 pixels of
 * white on a black frame, from a rounding step at the boundary.
 */
static uint32_t wrap_coord(uint32_t c, uint32_t size, uint32_t mode)
{
    /* Coordinates are signed: a bilinear tap one texel left of or above
     * texel 0 is -1, which must wrap to the far edge or clamp to 0 -- as an
     * unsigned value it clamped to the far edge instead. */
    int32_t sc = (int32_t)c, n = (int32_t)size;
    if (!size)
        return 0;
    if (mode == 1) {                       /* wrap */
        /* Texture sizes are powers of two, where the floored modulo is the
         * low bits, negative coordinates included (two's complement): the
         * same result without the two divisions per coordinate -- per tap,
         * up to sixteen a bilinear pixel. */
        if (!(size & (size - 1)))
            return c & (size - 1);
        return (uint32_t)(((sc % n) + n) % n);
    }
    return sc < 0 ? 0u : (sc >= n ? size - 1 : c);   /* clamp, and the rest */
}

static uint32_t expand(uint32_t v, uint32_t bits)
{
    return d3d8_expand_channel(v, bits);
}

/* The linear format that decodes the same texels as a swizzled one.
 *
 * Swizzling changes where a texel lives, not what it says: A8R8G8B8 (0x06) and
 * LIN_A8R8G8B8 (0x12) are the same four bytes in the same order. So the whole
 * difference is the address calculation, and one of those lets every format
 * below serve both. Pairs read off the table in d3d8_xbox.h rather than
 * recalled -- the comment above this one is about getting exactly that wrong. */
static uint32_t linear_twin(uint32_t fmt)
{
    switch (fmt) {
    case 0x00: return 0x13;                /* L8        -> LIN_L8        */
    case 0x02: return 0x10;                /* A1R5G5B5  -> LIN_A1R5G5B5  */
    case 0x03: return 0x1C;                /* X1R5G5B5  -> LIN_X1R5G5B5  */
    case 0x04: return 0x1D;                /* A4R4G4B4  -> LIN_A4R4G4B4  */
    case 0x05: return 0x11;                /* R5G6B5    -> LIN_R5G6B5    */
    case 0x06: return 0x12;                /* A8R8G8B8  -> LIN_A8R8G8B8  */
    case 0x07: return 0x1E;                /* X8R8G8B8  -> LIN_X8R8G8B8  */
    case 0x19: return 0x1F;                /* A8        -> LIN_A8        */
    case 0x28: return 0x17;                /* G8B8      -> LIN_G8B8      */
    default:   return fmt;                 /* already linear, or unhandled */
    }
}

/* A DXT block's 16 texels, decoded.
 *
 * Decoded blocks, direct-mapped by address. Every texel of a DXT texture
 * decodes its whole 4x4 block, and bilinear reads four neighbours that
 * nearly always share one -- this was a tenth of the executor's time. Keyed
 * on the block's bytes as well as its address, so a texture rewritten in
 * place (a render target reused, a streamed mip) never returns stale texels.
 * Per thread. */
static const uint32_t *dxt_block_px(const uint8_t *blk, uint32_t fmt, uint32_t bb)
{
    static NV_TLS struct { uintptr_t key; uint32_t fmt; uint64_t raw[2];
                    uint32_t px[16]; } cache[4096];
    uintptr_t key = (uintptr_t)blk;
    size_t slot = ((key / bb) ^ (key >> 16)) & 4095;
    uint64_t raw[2] = {0, 0};
    memcpy(raw, blk, bb);
    if (cache[slot].key != key || cache[slot].fmt != fmt
        || cache[slot].raw[0] != raw[0] || cache[slot].raw[1] != raw[1]) {
        /* The whole block in one pass: decoding it texel by texel
         * re-derived the palette sixteen times (bit-identical). */
        d3d8_dxt_decode_block(blk, fmt, cache[slot].px);
        cache[slot].key = key;
        cache[slot].fmt = fmt;
        cache[slot].raw[0] = raw[0];
        cache[slot].raw[1] = raw[1];
    }
    return cache[slot].px;
}

/* Sample texel (u, v) of stage texture `t`; `face` offsets a cube map. */
static int sample_tex(const Texture *t, uint32_t face_offset,
                      uint32_t u, uint32_t v, uint32_t *argb)
{
    /* The format's class, remembered per thread: the predicates are calls
     * into the d3d8 library, and every texel of a draw asks them about the
     * same format. */
    static NV_TLS uint32_t k_fmt = 0xFFFFFFFFu, k_dxt, k_swz;
    extern ptrdiff_t g_xbox_mem_offset;
    const uint8_t *mem = (const uint8_t *)g_xbox_mem_offset;
    const uint8_t *p;
    uint32_t fmt, base = t->offset + face_offset;

    if (!t->valid)
        return 0;
    u = wrap_coord(u, t->width,  t->addr_u);
    v = wrap_coord(v, t->height, t->addr_v);

    fmt = t->color;
    if (fmt != k_fmt) {
        k_dxt = d3d8_format_dxt_block_bytes(fmt);
        k_swz = d3d8_format_is_swizzled(fmt) != 0;
        k_fmt = fmt;
    }
    if (k_dxt) {
        uint32_t bw = (t->width + 3) >> 2;
        const uint8_t *blk = mem + base + ((size_t)(v >> 2) * bw + (u >> 2)) * k_dxt;
        *argb = dxt_block_px(blk, fmt, k_dxt)[(v & 3) * 4 + (u & 3)];
        return 1;
    }
    if (k_swz) {
        /* Morton order: a texel's index is interleaved from x and y instead of
         * v*pitch + u, so index from the base of the image. The switch below
         * casts to each format's own width, which makes that index a texel
         * index for every one of them. */
        fmt = linear_twin(fmt);
        p = mem + base;
        u = t->swz ? t->swz[u] + t->swz[t->width + v]
                   : swizzle_offset(u, v, t->width, t->height);
    } else {
        p = mem + base + (size_t)v * t->pitch;
    }

    switch (fmt) {

    /* 32-bit, alpha-red-green-blue in the dword. */
    case 0x12:                                      /* LIN_A8R8G8B8 */
        *argb = ((const uint32_t *)p)[u];
        return 1;
    case 0x1E:                                      /* LIN_X8R8G8B8 */
        *argb = ((const uint32_t *)p)[u] | 0xFF000000u;
        return 1;

    /* 8-bit palette index, swizzled. Burnout 3 draws its logo and frontend
     * header art this way; without a case every such quad came out as the
     * unsupported-format fill, a white box where the logo should be. */
    case 0x0B:                                      /* SZ_I8_A8R8G8B8 */
        if (!t->palette)
            return 0;
        *argb = ((const uint32_t *)(mem + t->palette))[p[u]];
        return 1;

    /* 32-bit, other channel orders. The name gives the byte order from the
     * top of the dword down, so each is a permutation of the same four. */
    case 0x3F: {                                    /* LIN_A8B8G8R8 */
        uint32_t t = ((const uint32_t *)p)[u];
        *argb = (t & 0xFF00FF00u) | ((t & 0xFF) << 16) | ((t >> 16) & 0xFF);
        return 1;
    }
    case 0x40: {                                    /* LIN_B8G8R8A8 */
        uint32_t t = ((const uint32_t *)p)[u];
        *argb = ((t & 0xFFu) << 24)                 /* A, from the bottom */
              | (((t >>  8) & 0xFFu) << 16)         /* R */
              | (((t >> 16) & 0xFFu) <<  8)         /* G */
              |  ((t >> 24) & 0xFFu);               /* B, from the top */
        return 1;
    }
    case 0x41: {                                    /* LIN_R8G8B8A8 */
        uint32_t t = ((const uint32_t *)p)[u];
        *argb = ((t & 0xFFu) << 24) | (t >> 8);
        return 1;
    }

    /* 16-bit. */
    case 0x10: {                                    /* LIN_A1R5G5B5 */
        uint32_t t = ((const uint16_t *)p)[u];
        *argb = ((t & 0x8000u) ? 0xFF000000u : 0u)
              | (expand((t >> 10) & 0x1F, 5) << 16)
              | (expand((t >>  5) & 0x1F, 5) <<  8)
              |  expand( t        & 0x1F, 5);
        return 1;
    }
    case 0x1C: {                                    /* LIN_X1R5G5B5 */
        uint32_t t = ((const uint16_t *)p)[u];
        *argb = 0xFF000000u
              | (expand((t >> 10) & 0x1F, 5) << 16)
              | (expand((t >>  5) & 0x1F, 5) <<  8)
              |  expand( t        & 0x1F, 5);
        return 1;
    }
    case 0x11: {                                    /* LIN_R5G6B5 */
        uint32_t t = ((const uint16_t *)p)[u];
        *argb = 0xFF000000u
              | (expand((t >> 11) & 0x1F, 5) << 16)
              | (expand((t >>  5) & 0x3F, 6) <<  8)
              |  expand( t        & 0x1F, 5);
        return 1;
    }
    case 0x1D: {                                    /* LIN_A4R4G4B4 */
        uint32_t t = ((const uint16_t *)p)[u];
        *argb = (expand((t >> 12) & 0x0F, 4) << 24)
              | (expand((t >>  8) & 0x0F, 4) << 16)
              | (expand((t >>  4) & 0x0F, 4) <<  8)
              |  expand( t        & 0x0F, 4);
        return 1;
    }

    /* G8B8: G in the high byte, B in the low one, as the name orders
     * them; red 0, alpha opaque. Xbox D3D's V8U8 is this format (V high,
     * U low), so it is what bump maps arrive as -- signed or not is the
     * texture filter's per-channel bits (rc_texel), not the format. */
    case 0x17: {                                    /* LIN_G8B8 */
        uint32_t t = ((const uint16_t *)p)[u];
        *argb = 0xFF000000u | ((t >> 8) << 8) | (t & 0xFFu);
        return 1;
    }

    /* 8-bit. */
    case 0x13: {                                    /* LIN_L8 */
        uint32_t t = p[u];
        *argb = 0xFF000000u | (t << 16) | (t << 8) | t;
        return 1;
    }
    case 0x1F:                                      /* LIN_A8 */
        *argb = ((uint32_t)p[u] << 24) | 0x00FFFFFFu;
        return 1;

    /* 4:2:2 packed YUV, two texels per four bytes.
     *
     * This is how a title hands over a decoded video frame, and without it
     * the frame falls through to `default` -- which returns 0, so the caller
     * paints the quad's vertex colour and the movie is a flat rectangle.
     *
     * The chroma pair is shared between an even texel and the one after it,
     * so the group is found by masking the bottom bit of the index. BT.601,
     * the same coefficients the D3D8 upload path converts with, so the two
     * paths agree rather than each having its own idea of the colour. */
    case 0x24:                                      /* LC_CR8YB8CB8YA8, YUY2 */
    case 0x25: {                                    /* LC_YB8CR8YA8CB8, UYVY */
        uint32_t yoff = (fmt == 0x24) ? 0u : 1u;
        const uint8_t *g = p + (size_t)(u & ~1u) * 2;
        int c  = (int)g[(u & 1u) ? 2 + yoff : yoff] - 16;
        int cu = (int)g[1 - yoff] - 128;
        int cv = (int)g[3 - yoff] - 128;
        int r = (298 * c + 409 * cv + 128) >> 8;
        int gg = (298 * c - 100 * cu - 208 * cv + 128) >> 8;
        int b = (298 * c + 516 * cu + 128) >> 8;
        if (r < 0) r = 0;
        if (r > 255) r = 255;
        if (gg < 0) gg = 0;
        if (gg > 255) gg = 255;
        if (b < 0) b = 0;
        if (b > 255) b = 255;
        *argb = 0xFF000000u | ((uint32_t)r << 16) | ((uint32_t)gg << 8)
              | (uint32_t)b;
        return 1;
    }

    default:
        return 0;
    }
}

static int sample_texture(uint32_t u, uint32_t v, uint32_t *argb)
{
    return sample_tex(&P->texs[0], 0, u, v, argb);
}

/* Write a bound texture out as a BMP, through the sampler rather than around it.
 *
 * "Which texture is this" is not answerable from an address and a format, and
 * it is the question behind most of the ones that matter -- is that a font
 * page or an icon atlas, did the swizzle decode, is the alpha inverted. Going
 * through sample_texture means the file shows exactly what the rasteriser
 * sees, so a decode bug appears here rather than only as a wrong-looking
 * triangle.
 *
 * ponytail: RGB only, alpha dropped. A glyph page is alpha and would come out
 * black, so alpha is composited onto mid-grey to stay legible; that is a
 * viewing choice, not a decode. One file per distinct texture, first use only.
 */
static void dump_texture_bmp(uint32_t seq)
{
    pix_direct();
    const char *prefix = getenv("RECOMP_TEX_DUMP");
    uint32_t w = s_gpu.texs[0].width, h = s_gpu.texs[0].height, x, y;
    uint32_t row_bytes, pad, filesz;
    uint8_t hdr[54];
    char path[512];
    FILE *f;

    if (!prefix || !w || !h || w > 4096 || h > 4096)
        return;
    row_bytes = w * 3;
    pad = (4 - (row_bytes & 3)) & 3;
    filesz = 54 + (row_bytes + pad) * h;

    snprintf(path, sizeof path, "%s%02u_%08X_fmt%02X.bmp",
             prefix, seq, s_gpu.texs[0].offset, s_gpu.texs[0].color);
    f = fopen(path, "wb");
    if (!f)
        return;
    memset(hdr, 0, sizeof hdr);
    hdr[0] = 'B'; hdr[1] = 'M';
    memcpy(hdr + 2, &filesz, 4);
    hdr[10] = 54; hdr[14] = 40;
    memcpy(hdr + 18, &w, 4);
    memcpy(hdr + 22, &h, 4);
    hdr[26] = 1; hdr[28] = 24;
    fwrite(hdr, 1, sizeof hdr, f);

    for (y = 0; y < h; y++) {
        for (x = 0; x < w; x++) {
            uint32_t argb = 0, a;
            uint8_t px[3];
            if (!sample_texture(x, h - 1 - y, &argb))
                argb = 0;
            a = (argb >> 24) & 0xFFu;
            /* over mid-grey, so an alpha-only page is visible either way */
            px[0] = (uint8_t)(((argb & 0xFFu) * a + 128u * (255u - a)) / 255u);
            px[1] = (uint8_t)((((argb >> 8) & 0xFFu) * a + 128u * (255u - a)) / 255u);
            px[2] = (uint8_t)((((argb >> 16) & 0xFFu) * a + 128u * (255u - a)) / 255u);
            fwrite(px, 1, 3, f);
        }
        if (pad) {
            static const uint8_t zero[3] = {0, 0, 0};
            fwrite(zero, 1, pad, f);
        }
    }
    fclose(f);
    fprintf(stderr, "  [TEXDUMP] %s (%ux%u fmt 0x%02X)\n",
            path, w, h, s_gpu.texs[0].color);
    fflush(stderr);
}

/* The surface, resolved once per batch.
 *
 * dma_resolve consults the contiguous arena's high-water mark and
 * surface_hits_image walks the image range; both were being done per pixel
 * -- dma_resolve twice -- which cost more than the rasterisation they
 * guarded. Neither answer can change inside a batch, because the colour
 * offset arrives as a method and a method cannot arrive mid-triangle.
 *
 * This is not a micro-optimisation for its own sake: the loader's video
 * paces on frames actually presented, so the rasteriser's throughput is the
 * playback rate. */

static uint32_t pack_color(const float c[4]);

/* One GL blend factor, per channel (r g b a). */
static void blend_factor(uint32_t f, const float s[4], const float d[4],
                         float out[4])
{
    int k;
    for (k = 0; k < 4; k++) {
        float c = (float)((P->blend_color >> (k == 3 ? 24 : 16 - 8 * k))
                          & 0xFF) / 255.0f;
        switch (f) {
        case 0x0000: out[k] = 0.0f;            break;   /* ZERO                */
        case 0x0001: out[k] = 1.0f;            break;   /* ONE                 */
        case 0x0300: out[k] = s[k];            break;   /* SRC_COLOR           */
        case 0x0301: out[k] = 1.0f - s[k];     break;
        case 0x0302: out[k] = s[3];            break;   /* SRC_ALPHA           */
        case 0x0303: out[k] = 1.0f - s[3];     break;
        case 0x0304: out[k] = d[3];            break;   /* DST_ALPHA           */
        case 0x0305: out[k] = 1.0f - d[3];     break;
        case 0x0306: out[k] = d[k];            break;   /* DST_COLOR           */
        case 0x0307: out[k] = 1.0f - d[k];     break;
        case 0x0308: out[k] = k == 3 ? 1.0f : fminf(s[3], 1.0f - d[3]); break;
        case 0x8001: out[k] = c;               break;   /* CONSTANT_COLOR      */
        case 0x8002: out[k] = 1.0f - c;        break;
        case 0x8003: out[k] = (float)(P->blend_color >> 24) / 255.0f; break;
        case 0x8004: out[k] = 1.0f - (float)(P->blend_color >> 24) / 255.0f; break;
        default:     out[k] = f ? 1.0f : 0.0f; break;
        }
    }
}

/* Pixels written and the brightest, per thread. They were s_gpu fields that
 * every raster worker bumped for every pixel, and that one contended cache
 * line held 16 workers to the throughput of about eight. Folded into s_gpu
 * by stats_fold() on the executor. */
static NV_TLS uint64_t t_pixels;
static NV_TLS uint32_t t_pixel_max;

static void stats_take(uint64_t *n, uint32_t *mx)
{
    *n = t_pixels;
    *mx = t_pixel_max;
    t_pixels = 0;
    t_pixel_max = 0;
}

static void stats_fold(uint64_t n, uint32_t mx)
{
    s_gpu.pixels += n;
    if ((mx & 0x00FFFFFFu) > (s_gpu.pixel_max & 0x00FFFFFFu))
        s_gpu.pixel_max = mx;
}

/* RECOMP_PIXEL_WATCH=x,y[;x,y...] (diagnostic, up to 8): log every write to
 * those pixels of a 640-wide surface -- incoming colour, result, blend, stage
 * program and the stage textures -- to find which draw paints a
 * wrong-looking spot. */
static int s_pw_n, s_pw_xs[8], s_pw_ys[8];
/* RECOMP_TEX_WATCH=<hex address>: the same combiner-input line ([PXW-RC])
 * for every 997th pixel of draws whose stage 0 samples that texture, and a
 * [PXW-Z] line (depth and stored depth) for every 997th that fails the depth
 * test (scalar path, RECOMP_SCALAR_PIXELS=1), whatever the camera does. */
static uint32_t s_tw_addr;
static volatile LONG s_tw_count;

static void pixel_watch(int x, int y, uint32_t src, uint32_t out)
{
    int st;
    fprintf(stderr, "[PXW] flip %u (%d,%d) surf %p src %08X -> %08X | blend %u %X/%X eq %X"
            " | prog %05X fin %08X %08X | alpha %u %X %X |", s_gpu.flips, x, y,
            (void *)P->surface, src, out, P->blend_enable, P->blend_sfactor,
            P->blend_dfactor, P->blend_equation, P->rcp.rc.stage_program,
            P->rcp.rc.final0, P->rcp.rc.final1, P->alpha_test, P->alpha_func,
            P->alpha_ref);
    for (st = 0; st < 4; st++)
        if (P->texs[st].valid)
            fprintf(stderr, " t%d %08X %ux%u f%02X", st, P->texs[st].offset,
                    P->texs[st].width, P->texs[st].height, P->texs[st].color);
    fputc(10, stderr);
}

static void put_pixel(uint8_t *mem, uint32_t bpp, int x, int y, uint32_t argb)
{
    uint8_t *px;
    uint32_t src = argb;

    (void)mem;
    if (x < (int)P->clip_x || x >= (int)(P->clip_x + P->clip_w))
        return;
    if (y < (int)P->clip_y || y >= (int)(P->clip_y + P->clip_h))
        return;
    t_pixels++;
    if ((argb & 0x00FFFFFFu) > (t_pixel_max & 0x00FFFFFFu))
        t_pixel_max = argb;
    px = surface_pixel(x, y, bpp);
    if (!px)
        return;

    /* Blending, with the GL factor set and equations the NV2A takes.
     *
     * Only SRC_ALPHA / ONE_MINUS_SRC_ALPHA used to be honoured and every other
     * pair was written opaque. Menus never noticed; a race does: Burnout 3
     * finishes its 3D frame with screen-space passes that modulate or add
     * over the scene, and written opaque they paint the whole view over. */
    if (P->blend_enable
        && !(P->blend_sfactor == 1 && P->blend_dfactor == 0)) {
        uint32_t dst = 0xFF000000u;
        float sc[4], dc[4], sf[4], df[4], out[4];
        int k;
        if (bpp == 4) {
            dst = *(const uint32_t *)px;
        } else if (bpp == 2) {
            uint32_t t = *(const uint16_t *)px;
            dst = 0xFF000000u | ((t & 0xF800u) << 8) | ((t & 0x07E0u) << 5)
                | ((t & 0x001Fu) << 3);
        }
        for (k = 0; k < 4; k++) {           /* r g b a, 0..1 */
            int sh = k == 3 ? 24 : 16 - 8 * k;
            sc[k] = (float)((argb >> sh) & 0xFF) / 255.0f;
            dc[k] = (float)((dst  >> sh) & 0xFF) / 255.0f;
        }
        blend_factor(P->blend_sfactor, sc, dc, sf);
        blend_factor(P->blend_dfactor, sc, dc, df);
        for (k = 0; k < 4; k++) {
            float a = sc[k] * sf[k], b = dc[k] * df[k];
            switch (P->blend_equation) {
            case 0x800A: out[k] = a - b; break;               /* SUBTRACT     */
            case 0x800B: out[k] = b - a; break;               /* REV_SUBTRACT */
            case 0x8007: out[k] = fminf(sc[k], dc[k]); break; /* MIN          */
            case 0x8008: out[k] = fmaxf(sc[k], dc[k]); break; /* MAX          */
            default:     out[k] = a + b; break;               /* ADD          */
            }
        }
        argb = pack_color(out);
    }

    if (P->color_mask != 0x01010101u) {
        uint32_t keep = 0, old;
        if (!(P->color_mask & 0x01000000u)) keep |= 0xFF000000u;
        if (!(P->color_mask & 0x00010000u)) keep |= 0x00FF0000u;
        if (!(P->color_mask & 0x00000100u)) keep |= 0x0000FF00u;
        if (!(P->color_mask & 0x00000001u)) keep |= 0x000000FFu;
        if (keep == 0xFFFFFFFFu)
            return;
        if (bpp == 4) {
            old = *(const uint32_t *)px;
        } else {
            uint32_t t = *(const uint16_t *)px;
            old = ((t & 0xF800u) << 8) | ((t & 0x07E0u) << 5) | ((t & 0x001Fu) << 3);
        }
        argb = (argb & ~keep) | (old & keep);
    }
    if (bpp == 4) {
        *(uint32_t *)px = argb;
    } else if (bpp == 2) {
        *(uint16_t *)px = (uint16_t)(((argb >> 8) & 0xF800)
                                        | ((argb >> 5) & 0x07E0)
                                        | ((argb >> 3) & 0x001F));
    }
    if (s_pw_n && P->clip_w == 640) {
        int i;
        for (i = 0; i < s_pw_n; i++)
            if (x == s_pw_xs[i] && y == s_pw_ys[i])
                pixel_watch(x, y, src, argb);
    }
}

/* Half-space fill. Barycentric edge functions rather than scanline slopes:
 * the same test decides both windings, so a title that emits clockwise
 * triangles does not silently render nothing. */
static void raster_triangle(const float a[2], const float b[2],
                            const float c[2], uint32_t argb,
                            const float uv[3][2])
{
    uint8_t *mem = (uint8_t *)xbox_GetMemoryOffset();
    uint32_t bpp = surface_bpp();
    float area;
    int minx, maxx, miny, maxy, x, y;
    int textured = uv && s_gpu.texs[0].valid;

    if (bpp != 4 && bpp != 2)
        return;

    area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    if (area == 0.0f)
        return;                            /* degenerate */

    /* Where this batch writes. The same check the per-pixel path made, made
     * once: a surface address landing on the title's own image is no safer
     * one pixel at a time than 4.9 MB at once. */
    if (!surface_begin_batch(mem))
        return;
    pix_direct();

    minx = (int)floorf(fminf(a[0], fminf(b[0], c[0])));
    maxx = (int)ceilf (fmaxf(a[0], fmaxf(b[0], c[0])));
    miny = (int)floorf(fminf(a[1], fminf(b[1], c[1])));
    maxy = (int)ceilf (fmaxf(a[1], fmaxf(b[1], c[1])));

    if (minx < (int)s_gpu.clip_x) minx = (int)s_gpu.clip_x;
    if (miny < (int)s_gpu.clip_y) miny = (int)s_gpu.clip_y;
    if (maxx > (int)(s_gpu.clip_x + s_gpu.clip_w)) maxx = (int)(s_gpu.clip_x + s_gpu.clip_w);
    if (maxy > (int)(s_gpu.clip_y + s_gpu.clip_h)) maxy = (int)(s_gpu.clip_y + s_gpu.clip_h);
    if (minx >= maxx || miny >= maxy) {
        s_gpu.tris_skipped_offscreen++;
        return;
    }

    for (y = miny; y < maxy; y++) {
        for (x = minx; x < maxx; x++) {
            float px = (float)x + 0.5f, py = (float)y + 0.5f;
            float w0 = (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0]);
            float w1 = (c[0] - b[0]) * (py - b[1]) - (c[1] - b[1]) * (px - b[0]);
            float w2 = (a[0] - c[0]) * (py - c[1]) - (a[1] - c[1]) * (px - c[0]);
            if (!((w0 >= 0 && w1 >= 0 && w2 >= 0)
               || (w0 <= 0 && w1 <= 0 && w2 <= 0)))
                continue;
            if (textured) {
                /* Barycentric, straight from the edge functions already
                 * computed: w1 is the area opposite a, w2 opposite b, w0
                 * opposite c, and the three sum to the whole triangle.
                 *
                 * No perspective divide. These are screen-space vertices with
                 * no w to divide by -- which is exactly the case a full-screen
                 * pass is, and the only case that reaches here. */
                uint32_t texel;
                float su = (w1 * uv[0][0] + w2 * uv[1][0] + w0 * uv[2][0]) / area;
                float sv = (w1 * uv[0][1] + w2 * uv[1][1] + w0 * uv[2][1]) / area;
                if (su < 0.0f) su = 0.0f;
                if (sv < 0.0f) sv = 0.0f;
                if (sample_texture((uint32_t)su, (uint32_t)sv, &texel)) {
                    put_pixel(mem, bpp, x, y, texel);
                    continue;
                }
            }
            put_pixel(mem, bpp, x, y, argb);
        }
    }
    s_gpu.tris_drawn++;
    note_drawn();
}

/* Attribute 3 is diffuse colour in every NV2A layout that sets one. Absent it,
 * white -- a visible wrong colour beats an invisible correct one during
 * bring-up. */
/* Which attribute carries the colour.
 *
 * Slot 3 is diffuse by convention and titles that follow it are read straight
 * from there. Half-Life 2 does not: its vertex is position, colour, texcoord
 * at stride 24, and the colour arrives in slot 5. So fall back to the format
 * rather than the slot number -- D3DCOLOR is the one attribute type that is
 * only ever a colour, which makes it a stronger signal than the convention. */
static const VertexAttr *color_attr(void)
{
    uint32_t a;

    if (s_gpu.attr[3].offset && s_gpu.attr[3].stride)
        return &s_gpu.attr[3];
    for (a = 0; a < NV_VERTEX_ATTRS; a++)
        if (s_gpu.attr[a].type == 0 && s_gpu.attr[a].size == 4
                && s_gpu.attr[a].offset && s_gpu.attr[a].stride)
            return &s_gpu.attr[a];
    return &s_gpu.attr[3];
}

/* Attribute 9 is texture coordinate 0 in the NV2A vertex layout, the same way
 * 0 is position and 3 is diffuse -- for a title that follows the convention.
 *
 * Half-Life 2 does not, in either place. Its menu and HUD vertex is position,
 * colour, texcoord at stride 24, with the colour in slot 5 and the texcoords
 * in slot 7, so reading slot 9 found nothing and every batch drew untextured.
 * That is invisible rather than wrong-looking: the menu paints a full-screen
 * quad and then draws its text over it, and with no sampling both come out
 * white, so the screen is blank white and nothing suggests the text was ever
 * drawn.
 *
 * Falling back to the format works because the three attributes of such a
 * vertex are distinguishable: position is float3, colour is D3DCOLOR, and a
 * float2 is a texture coordinate and nothing else.
 *
 * ponytail: takes the first float2 it finds, so a title with two texcoord sets
 * gets stage 0's -- which is what this single-texture rasteriser samples
 * anyway. Multi-texture wants the D3D11 translator, not another heuristic. */
static const VertexAttr *texcoord_attr(void)
{
    uint32_t a;

    if (s_gpu.attr[9].offset && s_gpu.attr[9].stride)
        return &s_gpu.attr[9];
    for (a = 0; a < NV_VERTEX_ATTRS; a++)
        if (s_gpu.attr[a].type == 2 && s_gpu.attr[a].size == 2
                && s_gpu.attr[a].offset && s_gpu.attr[a].stride)
            return &s_gpu.attr[a];
    return &s_gpu.attr[9];
}

/* Texel coordinates, whichever convention the title used.
 *
 * The two are not interchangeable and the format decides which is in force: a
 * swizzled texture is addressed in [0,1], a linear one in texels. Both are
 * scaled to texels here so that everything downstream -- the barycentric
 * interpolation and the sampler -- works in one unit.
 *
 * This mattered the moment swizzled formats became samplable. Normalised
 * coordinates truncated to a texel index land on texel 0 for any coordinate
 * below 1.0, so a whole quad sampled a single texel and came out flat: the
 * background painted one near-black colour, which looks like a texture that
 * decoded wrong rather than one that was never indexed. */
static int fetch_texcoord(uint32_t index, float out[2])
{
    float t[4];

    if (!fetch_attr(texcoord_attr(), index, t))
        return 0;
    out[0] = t[0];
    out[1] = t[1];
    if (tex_size_from_format(s_gpu.texs[0].color)) {
        out[0] *= (float)s_gpu.texs[0].width;
        out[1] *= (float)s_gpu.texs[0].height;
    }
    return 1;
}

static uint32_t vertex_color(uint32_t index)
{
    float c[4];

    if (!fetch_attr(color_attr(), index, c))
        return 0xFFFFFFFFu;
    return ((uint32_t)(c[3] * 255.0f) << 24)
         | ((uint32_t)(c[0] * 255.0f) << 16)
         | ((uint32_t)(c[1] * 255.0f) <<  8)
         |  (uint32_t)(c[2] * 255.0f);
}

/* An untransformed batch drawn as if it were screen space smears a few pixels
 * into the corner, so the batch has to be classified before it is rasterised.
 *
 * This used to demand that every vertex land inside the surface, which is a
 * different question and the wrong one: geometry that extends past the
 * viewport is ordinary, and clipping it is raster_triangle's job (it clamps
 * its span to the clip rect). The dashboard is exactly the case that exposed
 * it -- a full-screen pass drawn as one oversized triangle, vertices at
 * (-0.5,-0.5), (2*w,-0.5), (-0.5,2*h), all correct and all rejected.
 *
 * What actually separates the two is scale. Object-space positions are model
 * units, a handful either side of the origin; screen-space ones are measured
 * in pixels of a surface hundreds of pixels wide. So: the batch has to be able
 * to touch the surface at all, and it has to be bigger than object space.
 *
 * ponytail: a genuinely tiny screen-space sprite reads as object space and is
 * skipped. It is counted as skipped rather than silently dropped, and the
 * unambiguous answer needs the vertex-program state, which is not tracked yet.
 */
#define OBJECT_SPACE_SPAN 8.0f

static int batch_is_screen_space(void)
{
    float p[4], lo_x, hi_x, lo_y, hi_y;
    uint32_t i;

    if (!s_gpu.clip_w || !s_gpu.clip_h || !s_gpu.idx_count)
        return 0;
    if (!fetch_attr(&s_gpu.attr[0], s_gpu.idx[0], p))
        return 0;
    lo_x = hi_x = p[0];
    lo_y = hi_y = p[1];
    for (i = 1; i < s_gpu.idx_count; i++) {
        if (!fetch_attr(&s_gpu.attr[0], s_gpu.idx[i], p))
            return 0;
        if (p[0] < lo_x) lo_x = p[0];
        if (p[0] > hi_x) hi_x = p[0];
        if (p[1] < lo_y) lo_y = p[1];
        if (p[1] > hi_y) hi_y = p[1];
    }

    /* Entirely off the surface: nothing to draw under either reading. */
    if (hi_x < (float)s_gpu.clip_x
     || lo_x > (float)(s_gpu.clip_x + s_gpu.clip_w)
     || hi_y < (float)s_gpu.clip_y
     || lo_y > (float)(s_gpu.clip_y + s_gpu.clip_h))
        return 0;

    /* Small enough to be model units rather than pixels. */
    if (hi_x - lo_x < OBJECT_SPACE_SPAN && hi_y - lo_y < OBJECT_SPACE_SPAN)
        return 0;

    return 1;
}

/* NV097 primitive types.
 *
 * These are the operand of SET_BEGIN_END, where 0 is END and the list starts
 * at 1. They were each one too low, so every title's geometry was decomposed
 * as the primitive below the one it asked for -- a strip as a fan, a fan as
 * quads, and TRIANGLES, the one case whose vertex count must be a multiple
 * of three, as a strip.
 *
 * The vertex order says which numbering is right without taking a table on
 * trust: a strip arrives in Z order and a fan in cyclic order, and they only
 * line up with the primitive under this one. */
#define NV_PRIM_POINTS         1
#define NV_PRIM_LINES          2
#define NV_PRIM_LINE_LOOP      3
#define NV_PRIM_LINE_STRIP     4
#define NV_PRIM_TRIANGLES      5
#define NV_PRIM_TRIANGLE_STRIP 6
#define NV_PRIM_TRIANGLE_FAN   7
#define NV_PRIM_QUADS          8
#define NV_PRIM_QUAD_STRIP     9
#define NV_PRIM_POLYGON        10

/* How many post-draw captures to keep: enough to see whether the geometry
 * is stable from frame to frame, few enough not to fill a directory. */
#define FB_DUMP_AFTER_DRAW 8
static int s_drawn_dumps;

static void dump_surface_bmp(void);

/* One triangle by vertex index: gather position and, if the batch has one,
 * texture coordinate 0. A vertex whose position cannot be read is not drawn;
 * a batch whose texcoords cannot be read is drawn untextured rather than not
 * at all, so a missing coordinate stream costs the colour and not the shape.
 */
static void raster_indexed(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t argb)
{
    float p[3][4], uv[3][2];
    int textured;

    if (!fetch_attr(&s_gpu.attr[0], i0, p[0])
     || !fetch_attr(&s_gpu.attr[0], i1, p[1])
     || !fetch_attr(&s_gpu.attr[0], i2, p[2]))
        return;

    textured = fetch_texcoord(i0, uv[0])
            && fetch_texcoord(i1, uv[1])
            && fetch_texcoord(i2, uv[2]);

    raster_triangle(p[0], p[1], p[2], argb,
                    textured ? (const float (*)[2])uv : NULL);
}

/* ---- 3D: vertex programs and depth -------------------------------------
 *
 * A batch drawn while TRANSFORM_EXECUTION_MODE says "program" carries
 * model-space positions. Each vertex is run through the title's own vertex
 * program (nv2a_vsh_interp.c), whose oPos output on Xbox D3D is already
 * screen space -- the runtime appends the viewport transform and the divide
 * by w -- with the clip-space w left in oPos.w. So the rasteriser gets pixel
 * coordinates and a depth, and w for perspective-correct texturing.
 *
 * ponytail: no clipping. A triangle with any vertex behind the eye (w <= 0)
 * is dropped rather than clipped, which loses the slivers that cross the near
 * plane; road right under the camera is where that shows. Clip in
 * homogeneous space if it matters. */

/* Depth buffers, host-side, one per zeta surface the title uses. The real one
 * lives in guest memory in a tiled format nothing here decodes, and nothing
 * the title does reads it back, so a float buffer per zeta offset is enough.
 * Its 8-bit stencil follows it in the same block (zbuf_stencil). */
#define NV_ZBUF_W 1024
#define NV_ZBUF_H 1024
#define zbuf_stencil(z) ((uint8_t *)((z) + (size_t)NV_ZBUF_W * NV_ZBUF_H))
static struct { uint32_t offset; float *z; } s_zbufs[4];
static int s_ftrace;            /* frame trace: 0 idle, 2 tracing, 3 done */
static int s_zbuf_next;

static float zclear_value(void)
{
    /* Zeta format in SURFACE_FORMAT bits 4-7: 1 is Z16, 2 is Z24S8. */
    uint32_t zf = (s_gpu.format >> 4) & 0xF;
    return zf == 1 ? (float)(s_gpu.zstencil_clear & 0xFFFF)
                   : (float)(s_gpu.zstencil_clear >> 8);
}

static float *zbuf_current(int create)
{
    int i;
    size_t k;

    for (i = 0; i < 4; i++)
        if (s_zbufs[i].z && s_zbufs[i].offset == s_gpu.zeta_offset)
            return s_zbufs[i].z;
    if (!create)
        return NULL;
    rq_sync();                     /* the slot reused may still be in use */
    i = s_zbuf_next++ & 3;
    if (!s_zbufs[i].z)
        s_zbufs[i].z = (float *)malloc((sizeof(float) + 1) * NV_ZBUF_W * NV_ZBUF_H);
    if (!s_zbufs[i].z)
        return NULL;
    s_zbufs[i].offset = s_gpu.zeta_offset;
    for (k = 0; k < (size_t)NV_ZBUF_W * NV_ZBUF_H; k++)
        s_zbufs[i].z[k] = 3.4e38f;
    memset(zbuf_stencil(s_zbufs[i].z), 0, (size_t)NV_ZBUF_W * NV_ZBUF_H);
    return s_zbufs[i].z;
}

static int rq_async(void);
static int rq_zclear_job(float *z, float v, uint32_t what, uint8_t sv);

/* CLEAR_SURFACE's depth (bit 0) and stencil (bit 1) parts. The stencil
 * value is ZSTENCIL_CLEAR's low byte (Z24S8). */
static void zbuf_clear(uint32_t what)
{
    float *z, v;
    uint8_t sv = (uint8_t)(s_gpu.zstencil_clear & 0xFF);
    size_t k;
    if (rq_async()) {
        z = zbuf_current(1);
        if (z && rq_zclear_job(z, zclear_value(), what, sv))
            return;
    }
    rq_sync();
    z = zbuf_current(1); v = zclear_value();
    if (!z)
        return;
    if (what & 1)
        for (k = 0; k < (size_t)NV_ZBUF_W * NV_ZBUF_H; k++)
            z[k] = v;
    if (what & 2)
        memset(zbuf_stencil(z), sv, (size_t)NV_ZBUF_W * NV_ZBUF_H);
}

/* The stencil test and operations (GL enums, as the NV2A). */
static int stencil_pass(uint8_t st)
{
    uint32_t r = P->st_ref & P->st_fmask, v = st & P->st_fmask;
    switch (P->st_func) {
    case 0x200: return 0;
    case 0x201: return r <  v;
    case 0x202: return r == v;
    case 0x203: return r <= v;
    case 0x204: return r >  v;
    case 0x205: return r != v;
    case 0x206: return r >= v;
    default:    return 1;
    }
}

static void stencil_op(uint8_t *sp, uint32_t op)
{
    uint32_t v = *sp, n;
    switch (op) {
    case 0x0000: n = 0; break;                          /* ZERO      */
    case 0x1E01: n = P->st_ref; break;                  /* REPLACE   */
    case 0x1E02: n = v < 255 ? v + 1 : 255; break;      /* INCR      */
    case 0x1E03: n = v > 0 ? v - 1 : 0; break;          /* DECR      */
    case 0x150A: n = ~v & 0xFF; break;                  /* INVERT    */
    case 0x8507: n = (v + 1) & 0xFF; break;             /* INCR_WRAP */
    case 0x8508: n = (v - 1) & 0xFF; break;             /* DECR_WRAP */
    default:     return;                                /* KEEP      */
    }
    *sp = (uint8_t)((v & ~P->st_wmask) | (n & P->st_wmask));
}

static int depth_pass(float z, float stored)
{
    switch (P->depth_func) {                   /* GL enums, as the NV2A */
    case 0x200: return 0;
    case 0x201: return z <  stored;
    case 0x202: return z == stored;
    case 0x203: return z <= stored;
    case 0x204: return z >  stored;
    case 0x205: return z != stored;
    case 0x206: return z >= stored;
    default:    return 1;
    }
}

static uint32_t pack_color(const float c[4])
{
    int i;
    uint32_t b[4];
    for (i = 0; i < 4; i++) {
        float f = c[i] < 0.0f ? 0.0f : (c[i] > 1.0f ? 1.0f : c[i]);
        b[i] = (uint32_t)(f * 255.0f + 0.5f);
    }
    return (b[3] << 24) | (b[0] << 16) | (b[1] << 8) | b[2];
}

static uint32_t modulate(uint32_t t, const float c[4])
{
    float f[4];
    f[0] = (float)((t >> 16) & 0xFF) / 255.0f * c[0];
    f[1] = (float)((t >> 8) & 0xFF) / 255.0f * c[1];
    f[2] = (float)(t & 0xFF) / 255.0f * c[2];
    f[3] = (float)(t >> 24) / 255.0f * c[3];
    return pack_color(f);
}

/* Bytes one texel of a (non-DXT) format takes, for cube-face strides. */
static uint32_t tex_texel_bytes(uint32_t fmt)
{
    switch (fmt) {
    case 0x00: case 0x01: case 0x0B: case 0x13: case 0x19: case 0x1F:
        return 1;
    case 0x02: case 0x03: case 0x04: case 0x05: case 0x10: case 0x11:
    case 0x1C: case 0x1D:
        return 2;
    default:
        return 4;
    }
}

/* Bytes between cube faces: one face with all its mip levels, rounded up to
 * the NV2A's 128-byte face alignment (xemu texture.c). */
static uint32_t tex_face_stride(const Texture *t)
{
    uint32_t w = t->width, h = t->height, lv, total = 0;
    uint32_t block = d3d8_format_dxt_block_bytes(t->color);
    uint32_t levels = t->levels ? t->levels : 1;

    if (!tex_size_from_format(t->color))
        return (t->pitch * t->height + 127u) & ~127u;
    for (lv = 0; lv < levels; lv++) {
        uint32_t lw = (w >> lv) ? (w >> lv) : 1, lh = (h >> lv) ? (h >> lv) : 1;
        total += block ? ((lw + 3) / 4) * ((lh + 3) / 4) * block
                       : lw * lh * tex_texel_bytes(t->color);
    }
    return (total + 127u) & ~127u;
}

/* One texel at normalised (swizzled/DXT) or texel (linear) coordinates, as
 * floats. An unusable stage reads white, so a missing texture multiplies
 * through instead of blacking the pixel out. */
/* The texture filter's signed bits (28 A, 29 R, 30 G, 31 B): those channels
 * hold two's-complement bytes, -1..1, not 0..1. Converted per texel before
 * any filtering, so a bilinear blend of -1/255 and +1/255 is about 0 and
 * not about one half. D3D sets them for bump maps (V8U8 is G8B8 with G and
 * B signed). */
/* The 2x2 texels (iu, iv) .. (iu+1, iv+1) of a bilinear tap: bit k of the
 * result says out[k] is set, k = (dy << 1) | dx, exactly as four
 * sample_tex calls would give. The common cases skip three of the four
 * format checks, half the wraps and three DXT cache lookups: 32-bit ARGB
 * swizzled through the texture's table, and DXT when the four share a
 * block. Everything else is four sample_tex calls on the wrapped
 * coordinates (wrapping twice changes nothing). */
static int sample_tex_quad(const Texture *t, uint32_t face_offset,
                           uint32_t iu, uint32_t iv, uint32_t out[4])
{
    extern ptrdiff_t g_xbox_mem_offset;
    const uint8_t *mem = (const uint8_t *)g_xbox_mem_offset;
    uint32_t u[2], v[2], fmt = t->color, bb;
    int k, m = 0;

    if (!t->valid)
        return 0;
    u[0] = wrap_coord(iu, t->width, t->addr_u);
    u[1] = wrap_coord(iu + 1, t->width, t->addr_u);
    v[0] = wrap_coord(iv, t->height, t->addr_v);
    v[1] = wrap_coord(iv + 1, t->height, t->addr_v);
    if ((fmt == 0x06 || fmt == 0x07) && t->swz) {   /* A8R8G8B8 / X8R8G8B8 */
        const uint32_t *p = (const uint32_t *)(mem + t->offset + face_offset);
        const uint32_t *sx = t->swz, *sy = t->swz + t->width;
        uint32_t a = fmt == 0x07 ? 0xFF000000u : 0u;
        for (k = 0; k < 4; k++)
            out[k] = p[sx[u[k & 1]] + sy[v[k >> 1]]] | a;
        return 15;
    }
    if (fmt == 0x12 || fmt == 0x1E) {               /* LIN_A8R8G8B8 / X8R8G8B8 */
        const uint8_t *b0 = mem + t->offset + face_offset;
        uint32_t a = fmt == 0x1E ? 0xFF000000u : 0u;
        for (k = 0; k < 4; k++)
            out[k] = ((const uint32_t *)(b0 + (size_t)v[k >> 1] * t->pitch))[u[k & 1]] | a;
        return 15;
    }
    bb = d3d8_format_dxt_block_bytes(fmt);
    if (bb && (u[0] >> 2) == (u[1] >> 2) && (v[0] >> 2) == (v[1] >> 2)) {
        uint32_t bw = (t->width + 3) >> 2;
        const uint8_t *blk = mem + t->offset + face_offset
                           + ((size_t)(v[0] >> 2) * bw + (u[0] >> 2)) * bb;
        const uint32_t *px = dxt_block_px(blk, fmt, bb);
        for (k = 0; k < 4; k++)
            out[k] = px[(v[k >> 1] & 3) * 4 + (u[k & 1] & 3)];
        return 15;
    }
    for (k = 0; k < 4; k++)
        if (sample_tex(t, face_offset, u[k & 1], v[k >> 1], &out[k]))
            m |= 1 << k;
    return m;
}

static void texel_unpack(const Texture *t, uint32_t texel, float o[4])
{
    nv2a_rc_unpack(texel, o);
    if (t->filter & 0xF0000000u) {
        static const int ch[4] = {3, 0, 1, 2};     /* bit 28 + i: A R G B */
        static const int sh[4] = {24, 16, 8, 0};   /* their byte in ARGB */
        int i;
        for (i = 0; i < 4; i++)
            if (t->filter >> (28 + i) & 1) {
                float f = (float)(int8_t)(texel >> sh[i]) / 127.0f;
                o[ch[i]] = f < -1.0f ? -1.0f : f;
            }
    }
}

static void rc_texel(const Texture *t, uint32_t face, float u, float v,
                     float out[4])
{
    uint32_t texel, mag = (t->filter >> 24) & 0xF, min = (t->filter >> 16) & 0xFF;
    if (tex_size_from_format(t->color)) {
        u *= (float)t->width;
        v *= (float)t->height;
    }
    /* TENT (bilinear) when the title asks for it, magnifying or minifying:
     * a 64x32 sky gradient stretched over the screen is blocks with nearest
     * texels and a gradient with four. MAG 2 is tent; MIN 2 is tent at LOD 0
     * and 4/6 the tent mip modes.
     * ponytail: level 0 only, no mip selection -- minified textures shimmer.
     * Add LOD from the screen-space derivative when that shows. */
    if (mag == 2 || min == 2 || min == 4 || min == 6) {
        float fu = u - 0.5f, fv = v - 0.5f, wu, wv, c[4][4];
        int32_t iu = (int32_t)floorf(fu), iv = (int32_t)floorf(fv), k, j;
        wu = fu - (float)iu;
        wv = fv - (float)iv;
        {
            uint32_t tx[4];
            int got = sample_tex_quad(t, face, (uint32_t)iu, (uint32_t)iv, tx);
            for (k = 0; k < 4; k++) {
                if (got >> k & 1)
                    texel_unpack(t, tx[k], c[k]);
                else
                    c[k][0] = c[k][1] = c[k][2] = c[k][3] = 1.0f;
            }
        }
        {
            /* The four channels at once, the same products and sums in the
             * same order as per channel -- bit-identical without FMA. A
             * glow pass in stage 01-03 runs four of these per pixel over
             * the whole screen, four times a frame. */
            __m128 a = _mm_set1_ps(1.0f - wu), b = _mm_set1_ps(wu);
            __m128 cv = _mm_set1_ps(1.0f - wv), dv = _mm_set1_ps(wv);
            __m128 top = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(c[0]), a),
                                    _mm_mul_ps(_mm_loadu_ps(c[1]), b));
            __m128 bot = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(c[2]), a),
                                    _mm_mul_ps(_mm_loadu_ps(c[3]), b));
            _mm_storeu_ps(out, _mm_add_ps(_mm_mul_ps(top, cv), _mm_mul_ps(bot, dv)));
        }
        (void)j;
        return;
    }
    if (sample_tex(t, face, (uint32_t)(int32_t)floorf(u),
                   (uint32_t)(int32_t)floorf(v), &texel))
        texel_unpack(t, texel, out);
    else
        out[0] = out[1] = out[2] = out[3] = 1.0f;
}

/* What texture stage `st` contributes, by its SHADER_STAGE_PROGRAM mode
 * (xemu psh.c). Returns 0 when a clip-plane stage kills the pixel. */
static int rc_stage_fetch(int st, const float c[4], const float prev[4][4],
                          float out[4])
{
    const Texture *t = &P->texs[st];
    uint32_t mode = (P->rcp.rc.stage_program >> (st * 5)) & 0x1F, j;
    float q = c[3] != 0.0f ? c[3] : 1.0f;

    switch (mode) {
    case 0:                                       /* NONE */
        out[0] = out[1] = out[2] = 0.0f; out[3] = 1.0f;
        return 1;
    case 1: case 2:                               /* PROJECT2D / 3D */
        rc_texel(t, 0, c[0] / q, c[1] / q, out);
        return 1;
    case 3: {                                     /* CUBEMAP */
        float x = c[0], y = c[1], z = c[2];
        float ax = fabsf(x), ay = fabsf(y), az = fabsf(z), ma, sc, tc;
        uint32_t face;
        if (!t->cube) {
            rc_texel(t, 0, c[0], c[1], out);
            return 1;
        }
        if (ax >= ay && ax >= az) {
            face = x > 0 ? 0 : 1; ma = ax; sc = x > 0 ? -z : z; tc = -y;
        } else if (ay >= az) {
            face = y > 0 ? 2 : 3; ma = ay; sc = x; tc = y > 0 ? z : -z;
        } else {
            face = z > 0 ? 4 : 5; ma = az; sc = z > 0 ? x : -x; tc = -y;
        }
        if (ma == 0.0f)
            ma = 1.0f;
        rc_texel(t, face * tex_face_stride(t), (sc / ma + 1.0f) * 0.5f,
                 (tc / ma + 1.0f) * 0.5f, out);
        return 1;
    }
    case 4:                                       /* PASSTHRU */
        for (j = 0; j < 4; j++)
            out[j] = c[j] < 0.0f ? 0.0f : (c[j] > 1.0f ? 1.0f : c[j]);
        return 1;
    case 5:                                       /* CLIPPLANE */
        out[0] = out[1] = out[2] = out[3] = 0.0f;
        for (j = 0; j < 4; j++) {
            int ge = (P->clip_plane_mode >> (st * 4 + j)) & 1;
            if (ge ? c[j] >= 0.0f : c[j] < 0.0f)
                return 0;
        }
        return 1;
    case 6: case 7: {                             /* BUMPENVMAP (_LUM) */
        /* The input stage's texel is an offset (ds, dt) = (b, g), signed --
         * as the filter's signed bits made it, else from 0..1 -- turned by
         * this stage's bump matrix and added to this stage's coordinates
         * (xemu psh.c). Stage 1 reads stage 0; 2 and 3 the stage
         * SHADER_OTHER_STAGE_INPUT names. Blinx 2's water: a G8B8 normal map
         * on stage 0 perturbing two reflection/colour maps on 1 and 2. These
         * read 0 before, so the water was drawn from the combiner's other
         * terms alone. */
        int in = st == 1 ? 0 : (int)(P->other_stage_input >> (st == 2 ? 16 : 20)) & 3;
        const Texture *ti = &P->texs[in];
        float ds = prev[in][2], dt = prev[in][1], du, dv, u, v;
        if (!(ti->filter >> 31 & 1)) ds = ds * 2.0f - 1.0f;
        if (!(ti->filter >> 30 & 1)) dt = dt * 2.0f - 1.0f;
        /* Added in the coordinates' own units, as xemu does (texScale *
         * (pT + dsdt)): texels for a linear texture, whose coordinates are
         * texels. Scaling the offset by the texture size for those turned a
         * few texels of ripple into hundreds. */
        du = t->bump_mat[0] * ds + t->bump_mat[2] * dt;
        dv = t->bump_mat[1] * ds + t->bump_mat[3] * dt;
        u = c[0] + du;
        v = c[1] + dv;
        rc_texel(t, 0, u, v, out);
        if (mode == 7) {
            float lum = prev[in][0] * t->bump_scale + t->bump_offset;
            lum = lum < 0.0f ? 0.0f : (lum > 1.0f ? 1.0f : lum);
            out[0] *= lum; out[1] *= lum; out[2] *= lum;
        }
        return 1;
    }
    default:
        /* ponytail: the dot-product and dependent modes read zero -- add
         * them when a title's look depends on one. */
        out[0] = out[1] = out[2] = out[3] = 0.0f;
        return 1;
    }
}

/* The fixed-function fog unit, per vertex, from the program's oFog.x
 * (xemu vsh.c). Fog disabled reads as factor 1: no fog. */
/* What triangle set-up reads of the GPU state, captured once per draw so
 * set-up can run on any thread (see the raster pool's geometry jobs). */
typedef struct {
    uint8_t *mem;
    uint32_t bpp;
    int ok;                             /* the surface can be drawn into */
    int textured, use_rc;
    float su, sv;                       /* stage 0 size, for T.uv */
    float *zb;
    int clip_x, clip_y, clip_w, clip_h;
    int cull_enable, front_face, cull_face, no_cull;
    float k[3], o[3];                   /* viewport: c[58] scale, c[59] offset */
    uint32_t fog_enable, fog_mode;
    float fog_param[2], fogc[4];
} GeoCtx;

/* Set-up's tallies, folded into s_gpu by whoever owns it. */
typedef struct {
    uint32_t behind, degenerate, culled, offscreen, drawn;
    float xmin[3], xmax[3];
    int seeded;
} GeoStats;

static float geo_fog(const GeoCtx *g, float d)
{
    float f, px = g->fog_param[0], py = g->fog_param[1];
    if (!g->fog_enable)
        return 1.0f;
    switch (g->fog_mode) {
    case 0x800: case 0x802:                       /* EXP, EXP_ABS */
        f = px + exp2f(d * py * 16.0f) - 1.5f;
        break;
    case 0x801: case 0x803:                       /* EXP2, EXP2_ABS */
        f = px + exp2f(-d * d * py * py * 32.0f) - 1.5f;
        break;
    default:                                      /* LINEAR, LINEAR_ABS */
        f = px + d * py - 1.0f;
        break;
    }
    if (g->fog_mode == 0x802 || g->fog_mode == 0x803
        || g->fog_mode == 0x804)
        f = fabsf(f);
    if (f != f)
        f = 1.0f;
    return f < 0.0f ? 0.0f : (f > 1.0f ? 1.0f : f);
}

static int alpha_test_pass(float a)
{
    int v = (int)(a * 255.0f + 0.5f), r = (int)(P->alpha_ref & 0xFF);
    switch (P->alpha_func) {
    case 0x200: return 0;                         /* NEVER    */
    case 0x201: return v <  r;                    /* LESS     */
    case 0x202: return v == r;                    /* EQUAL    */
    case 0x203: return v <= r;                    /* LEQUAL   */
    case 0x204: return v >  r;                    /* GREATER  */
    case 0x205: return v != r;                    /* NOTEQUAL */
    case 0x206: return v >= r;                    /* GEQUAL   */
    default:    return 1;                         /* ALWAYS   */
    }
}

/* Everything a triangle's pixels need, computed once per triangle, so the
 * pixel loop can run over any subset of rows on any thread. */
typedef struct {
    uint8_t *mem;
    uint32_t bpp;
    const Nv2aVshOutput *va, *vb, *vc;
    const float *a, *b, *c;
    float iw[3], uv[3][2], stc[3][4][4], vfog[3], fogc[4], inv_area;
    /* Edge functions w0 (a->b), w1 (b->c), w2 (c->a) as planes
     * ea*dx + (eb*dy + ec), dx/dy from the pixel (minx, miny); see
     * geo_edges. */
    float ea[3], eb[3], ec[3];
    float *zb;
    int minx, maxx, miny, maxy, use_rc, textured;
    const PixState *ps;                 /* the draw's state, for P */
    /* Not a triangle but an ordered job over its rectangle (rq_push_job):
     * 1 colour clear, 2 depth clear. */
    int job;
    uint32_t jcolor, jpitch, jbase;
    float jz;
    const SwzTable *jswz;
} XfTri;

/* Z-pass counts per visibility test (report epoch), so a test can be
 * answered when its own batches are done, without draining the queue. */
#define NV_ZEPOCHS 64

typedef struct {
    uint64_t depth_fail, pixels, written;
    uint32_t zpass, written_max;
    uint64_t csum;                      /* r+g+b shaded, for the draw trace */
    uint32_t zpe[NV_ZEPOCHS];           /* z-pass per report epoch */
} XfCount;

/* Rows y0, y0+step, ... < y1 of triangle T, one pixel at a time: the
 * reference for xf_rows_span4. */
static void xf_rows_span_scalar(const XfTri *T, int y0, int y1, int step,
                                XfCount *cnt)
{
    const float *a = T->a, *b = T->b, *c = T->c;
    const Nv2aVshOutput *va = T->va, *vb = T->vb, *vc = T->vc;
    int x, y, k;

    P = T->ps;
    for (y = y0; y < y1; y += step) {
        float dy = (float)(y - T->miny);
        float r0 = T->eb[0] * dy + T->ec[0];
        float r1 = T->eb[1] * dy + T->ec[1];
        float r2 = T->eb[2] * dy + T->ec[2];
        for (x = T->minx; x < T->maxx; x++) {
            float dx = (float)(x - T->minx);
            float w0 = T->ea[0] * dx + r0;
            float w1 = T->ea[1] * dx + r1;
            float w2 = T->ea[2] * dx + r2;
            float l0, l1, l2, z, col[4], pw;
            uint32_t argb;
            if (!((w0 >= 0 && w1 >= 0 && w2 >= 0)
               || (w0 <= 0 && w1 <= 0 && w2 <= 0)))
                continue;
            /* w1 is opposite a, w2 opposite b, w0 opposite c. */
            l0 = w1 * T->inv_area; l1 = w2 * T->inv_area; l2 = w0 * T->inv_area;
            z = l0 * a[2] + l1 * b[2] + l2 * c[2];
            {
                float *zp = T->zb ? &T->zb[(size_t)y * NV_ZBUF_W + x] : NULL;
                int shade = (P->color_mask & 0x01010101u) || P->alpha_test;
                /* Stencil test, depth test, then shade, then alpha test, and
                 * only then the depth write: an alpha-tested texel that is
                 * cut away must not leave its depth behind (foliage, fences).
                 * The stencil operations apply to what survives the alpha
                 * test (GL's order: alpha, stencil, depth), so a rejected
                 * pixel whose operation changes the stencil is still shaded
                 * when an alpha test could cut it (`rej`, operation `pend`). */
                uint8_t *sp = P->st_test && T->zb
                            ? zbuf_stencil(T->zb) + (size_t)y * NV_ZBUF_W + x : NULL;
                uint32_t pend = 0;
                int rej = 0;
                if (sp && !stencil_pass(*sp)) {
                    if (P->st_fail == 0x1E00 || !P->alpha_test) {
                        stencil_op(sp, P->st_fail);
                        continue;
                    }
                    pend = P->st_fail;
                    rej = 1;
                } else if (zp && P->depth_test && !depth_pass(z, *zp)) {
                    if (sp && P->st_zfail != 0x1E00 && P->alpha_test) {
                        pend = P->st_zfail;
                        rej = 1;
                    } else {
                        if (sp)
                            stencil_op(sp, P->st_zfail);
                        cnt->depth_fail++;
                        if (s_tw_addr && P->texs[0].offset == s_tw_addr
                            && InterlockedIncrement(&s_tw_count) % 997 == 0)
                            fprintf(stderr, "[PXW-Z] (%d,%d) z %.1f stored %.1f%c",
                                    x, y, z, *zp, 10);
                        continue;
                    }
                }
                argb = 0;
                if (shade) {
                    for (k = 0; k < 4; k++)
                        col[k] = l0 * va->d0[k] + l1 * vb->d0[k] + l2 * vc->d0[k];
                    pw = l0 * T->iw[0] + l1 * T->iw[1] + l2 * T->iw[2];
                    if (T->use_rc) {
                        float d1[4], fog[4], t[4][4], out[4], tc[4];
                        int st, j, keep = 1;
                        for (k = 0; k < 4; k++)
                            d1[k] = l0 * va->d1[k] + l1 * vb->d1[k] + l2 * vc->d1[k];
                        fog[0] = T->fogc[0]; fog[1] = T->fogc[1]; fog[2] = T->fogc[2];
                        fog[3] = l0 * T->vfog[0] + l1 * T->vfog[1] + l2 * T->vfog[2];
                        for (st = 0; st < 4 && keep; st++) {
                            if (!((P->rcp.rc.stage_program >> (st * 5)) & 0x1F)) {
                                t[st][0] = t[st][1] = t[st][2] = 0.0f;
                                t[st][3] = 1.0f;
                                continue;
                            }
                            for (j = 0; j < 4; j++)
                                tc[j] = (l0 * T->stc[0][st][j] + l1 * T->stc[1][st][j]
                                       + l2 * T->stc[2][st][j]) / pw;
                            keep = rc_stage_fetch(st, tc, (const float (*)[4])t, t[st]);
                        }
                        if (!keep)
                            continue;
                        nv2a_rc_eval_prep(&P->rcp, col, d1, fog,
                                     (const float (*)[4])t, out);
                        if ((s_pw_n || s_tw_addr) && P->clip_w == 640) {
                            int i, hit = s_tw_addr && P->texs[0].offset == s_tw_addr
                                         && InterlockedIncrement(&s_tw_count) % 997 == 0;
                            for (i = 0; i < (hit ? 1 : s_pw_n); i++)
                                if (hit || (x == s_pw_xs[i] && y == s_pw_ys[i]))
                                    fprintf(stderr, "[PXW-RC] (%d,%d) v0 %.3f %.3f %.3f %.3f"
                                            " v1 %.3f %.3f %.3f | fog %.3f %.3f %.3f f %.3f"
                                            " | t0 %.3f %.3f %.3f %.3f t1 %.3f %.3f %.3f"
                                            " t2 %.3f %.3f %.3f t3 %.3f %.3f %.3f %.3f"
                                            " | out %.3f %.3f %.3f %.3f | filt %08X %08X%c", x, y,
                                            col[0], col[1], col[2], col[3], d1[0], d1[1],
                                            d1[2], fog[0], fog[1], fog[2], fog[3],
                                            t[0][0], t[0][1], t[0][2], t[0][3],
                                            t[1][0], t[1][1], t[1][2], t[2][0], t[2][1],
                                            t[2][2], t[3][0], t[3][1], t[3][2], t[3][3],
                                            out[0], out[1], out[2], out[3],
                                            P->texs[0].filter, P->texs[3].filter, 10);
                        }
                        if (P->alpha_test && !alpha_test_pass(out[3]))
                            continue;
                        argb = pack_color(out);
                    } else {
                        argb = pack_color(col);
                        if (T->textured) {
                            uint32_t texel;
                            float tu, tv;
                            tu = (l0 * T->uv[0][0] + l1 * T->uv[1][0] + l2 * T->uv[2][0]) / pw;
                            tv = (l0 * T->uv[0][1] + l1 * T->uv[1][1] + l2 * T->uv[2][1]) / pw;
                            if (sample_texture((uint32_t)(int32_t)floorf(tu),
                                               (uint32_t)(int32_t)floorf(tv), &texel))
                                argb = modulate(texel, col);
                        }
                        if (P->alpha_test
                            && !alpha_test_pass((float)(argb >> 24) / 255.0f))
                            continue;
                    }
                }
                if (rej) {                     /* rejected; alpha test passed */
                    stencil_op(sp, pend);
                    continue;
                }
                if (sp)
                    stencil_op(sp, P->st_zpass);
                if (zp && P->depth_mask)
                    *zp = z;
                if (P->zpass_enable)
                    { cnt->zpass++; cnt->zpe[P->report_epoch & (NV_ZEPOCHS - 1)]++; }
                if (!(P->color_mask & 0x01010101u))
                    continue;              /* colour writes off: depth only */
                cnt->pixels++;
                cnt->csum += ((argb >> 16) & 0xFF) + ((argb >> 8) & 0xFF) + (argb & 0xFF);
            }
            put_pixel(T->mem, T->bpp, x, y, argb);
        }
    }
}

/* Four pixels of a row at once, for draws that go through the register
 * combiners (T->use_rc): barycentrics, depth, colours, texture coordinates
 * and the combiners in SSE lanes, with the same operations in the same order
 * as xf_rows_span, so every pixel comes out the same. Texture fetches, the
 * alpha test, packing and the write (blending, colour mask) run per lane
 * through the scalar code. Draws without combiners stay scalar.
 * RECOMP_SCALAR_PIXELS=1 turns this off; RECOMP_PIXEL_CHECK=1 runs both on
 * every span and reports the first differences. */
static int depth4_mask(uint32_t func, __m128 z, __m128 st)
{
    __m128 m;
    switch (func) {
    case 0x200: m = _mm_setzero_ps(); break;
    case 0x201: m = _mm_cmplt_ps(z, st); break;
    case 0x202: m = _mm_cmpeq_ps(z, st); break;
    case 0x203: m = _mm_cmple_ps(z, st); break;
    case 0x204: m = _mm_cmpgt_ps(z, st); break;
    case 0x205: m = _mm_cmpneq_ps(z, st); break;
    case 0x206: m = _mm_cmpge_ps(z, st); break;
    default:    return 0xF;
    }
    return _mm_movemask_ps(m);
}

static __m128 bary3(__m128 l0, __m128 l1, __m128 l2, float a, float b, float c)
{
    return _mm_add_ps(_mm_add_ps(_mm_mul_ps(l0, _mm_set1_ps(a)),
                                 _mm_mul_ps(l1, _mm_set1_ps(b))),
                      _mm_mul_ps(l2, _mm_set1_ps(c)));
}

static void xf_rows_span4(const XfTri *T, int y0, int y1, XfCount *cnt)
{
    const float *a = T->a, *b = T->b, *c = T->c;
    const Nv2aVshOutput *va = T->va, *vb = T->vb, *vc = T->vc;
    const __m128 zero = _mm_setzero_ps();
    const __m128 inv = _mm_set1_ps(T->inv_area);
    const __m128 ea0 = _mm_set1_ps(T->ea[0]), ea1 = _mm_set1_ps(T->ea[1]);
    const __m128 ea2 = _mm_set1_ps(T->ea[2]);
    int x, y, k, l;

    P = T->ps;
    for (y = y0; y < y1; y++) {
        float dy = (float)(y - T->miny);
        __m128 e0 = _mm_set1_ps(T->eb[0] * dy + T->ec[0]);
        __m128 e1 = _mm_set1_ps(T->eb[1] * dy + T->ec[1]);
        __m128 e2 = _mm_set1_ps(T->eb[2] * dy + T->ec[2]);
        for (x = T->minx; x < T->maxx; x += 4) {
            __m128 dx = _mm_cvtepi32_ps(_mm_setr_epi32(x - T->minx, x + 1 - T->minx,
                                                       x + 2 - T->minx, x + 3 - T->minx));
            __m128 w0 = _mm_add_ps(_mm_mul_ps(ea0, dx), e0);
            __m128 w1 = _mm_add_ps(_mm_mul_ps(ea1, dx), e1);
            __m128 w2 = _mm_add_ps(_mm_mul_ps(ea2, dx), e2);
            __m128 l0, l1, l2, z;
            float zl[4], zs[4];
            uint32_t argb[4] = {0, 0, 0, 0};
            int mask, n = T->maxx - x < 4 ? T->maxx - x : 4;
            int shade = (P->color_mask & 0x01010101u) || P->alpha_test;
            uint8_t *sr = P->st_test && T->zb
                        ? zbuf_stencil(T->zb) + (size_t)y * NV_ZBUF_W + x : NULL;
            uint32_t pend[4] = {0, 0, 0, 0};
            int rejm = 0;                       /* stencil op pending the alpha test */

            mask = (_mm_movemask_ps(_mm_and_ps(_mm_and_ps(_mm_cmpge_ps(w0, zero),
                                                          _mm_cmpge_ps(w1, zero)),
                                               _mm_cmpge_ps(w2, zero)))
                  | _mm_movemask_ps(_mm_and_ps(_mm_and_ps(_mm_cmple_ps(w0, zero),
                                                          _mm_cmple_ps(w1, zero)),
                                               _mm_cmple_ps(w2, zero))))
                 & ((1 << n) - 1);
            if (!mask)
                continue;
            /* w1 is opposite a, w2 opposite b, w0 opposite c. */
            l0 = _mm_mul_ps(w1, inv); l1 = _mm_mul_ps(w2, inv); l2 = _mm_mul_ps(w0, inv);
            z = bary3(l0, l1, l2, a[2], b[2], c[2]);
            _mm_storeu_ps(zl, z);
            /* Stencil, as xf_rows_span_scalar, per lane: each lane is its
             * own pixel, so the order within a pixel is all that matters. */
            if (sr) {
                for (l = 0; l < 4; l++) {
                    if (!(mask >> l & 1) || stencil_pass(sr[l]))
                        continue;
                    if (P->st_fail == 0x1E00 || !P->alpha_test) {
                        stencil_op(&sr[l], P->st_fail);
                        mask &= ~(1 << l);
                    } else {
                        pend[l] = P->st_fail;
                        rejm |= 1 << l;
                    }
                }
                if (!mask)
                    continue;
            }
            if (T->zb && P->depth_test && (mask & ~rejm)) {
                int pass;
                for (l = 0; l < 4; l++)
                    zs[l] = (mask >> l & 1) ? T->zb[(size_t)y * NV_ZBUF_W + x + l] : 0.0f;
                pass = depth4_mask(P->depth_func, z, _mm_loadu_ps(zs));
                for (l = 0; l < 4; l++) {
                    if (!(mask >> l & 1) || (rejm >> l & 1) || (pass >> l & 1))
                        continue;
                    if (sr && P->st_zfail != 0x1E00 && P->alpha_test) {
                        pend[l] = P->st_zfail;
                        rejm |= 1 << l;
                    } else {
                        if (sr)
                            stencil_op(&sr[l], P->st_zfail);
                        cnt->depth_fail++;
                        mask &= ~(1 << l);
                    }
                }
                if (!mask)
                    continue;
            }
            if (shade) {
                __m128 col[4], d1[4], fog[4], t[4][4], out[4], pw;
                float ol[4][4], lt[4][4][4];      /* lane, stage, channel */
                int st;
                for (k = 0; k < 4; k++) {
                    col[k] = bary3(l0, l1, l2, va->d0[k], vb->d0[k], vc->d0[k]);
                    d1[k] = bary3(l0, l1, l2, va->d1[k], vb->d1[k], vc->d1[k]);
                }
                pw = bary3(l0, l1, l2, T->iw[0], T->iw[1], T->iw[2]);
                fog[0] = _mm_set1_ps(T->fogc[0]);
                fog[1] = _mm_set1_ps(T->fogc[1]);
                fog[2] = _mm_set1_ps(T->fogc[2]);
                fog[3] = bary3(l0, l1, l2, T->vfog[0], T->vfog[1], T->vfog[2]);
                for (st = 0; st < 4; st++) {
                    __m128 tc[4];
                    float tcl[4][4], tl[4][4];
                    if (!((P->rcp.rc.stage_program >> (st * 5)) & 0x1F)) {
                        t[st][0] = t[st][1] = t[st][2] = zero;
                        t[st][3] = _mm_set1_ps(1.0f);
                        for (l = 0; l < 4; l++) {
                            lt[l][st][0] = lt[l][st][1] = lt[l][st][2] = 0.0f;
                            lt[l][st][3] = 1.0f;
                        }
                        continue;
                    }
                    for (k = 0; k < 4; k++) {
                        tc[k] = _mm_div_ps(bary3(l0, l1, l2, T->stc[0][st][k],
                                                 T->stc[1][st][k], T->stc[2][st][k]), pw);
                        _mm_storeu_ps(tcl[k], tc[k]);
                    }
                    for (l = 0; l < 4; l++) {
                        float cl[4], o[4] = {0, 0, 0, 0};
                        if (mask >> l & 1) {
                            for (k = 0; k < 4; k++)
                                cl[k] = tcl[k][l];
                            if (!rc_stage_fetch(st, cl, (const float (*)[4])lt[l], o))
                                mask &= ~(1 << l);     /* clip plane */
                        }
                        for (k = 0; k < 4; k++)
                            tl[k][l] = lt[l][st][k] = o[k];
                    }
                    for (k = 0; k < 4; k++)
                        t[st][k] = _mm_loadu_ps(tl[k]);
                }
                if (!mask)
                    continue;
                nv2a_rc_eval_prep4(&P->rcp, col, d1, fog, (const __m128 (*)[4])t, out);
                for (k = 0; k < 4; k++)
                    _mm_storeu_ps(ol[k], out[k]);
                for (l = 0; l < 4; l++) {
                    float o[4];
                    if (!(mask >> l & 1))
                        continue;
                    for (k = 0; k < 4; k++)
                        o[k] = ol[k][l];
                    if (P->alpha_test && !alpha_test_pass(o[3])) {
                        mask &= ~(1 << l);
                        continue;
                    }
                    argb[l] = pack_color(o);
                }
            }
            for (l = 0; l < 4; l++) {
                if (!(mask >> l & 1))
                    continue;
                if (rejm >> l & 1) {           /* rejected; alpha test passed */
                    stencil_op(&sr[l], pend[l]);
                    continue;
                }
                if (sr)
                    stencil_op(&sr[l], P->st_zpass);
                if (T->zb && P->depth_mask)
                    T->zb[(size_t)y * NV_ZBUF_W + x + l] = zl[l];
                if (P->zpass_enable)
                    { cnt->zpass++; cnt->zpe[P->report_epoch & (NV_ZEPOCHS - 1)]++; }
                if (!(P->color_mask & 0x01010101u))
                    continue;
                cnt->pixels++;
                cnt->csum += ((argb[l] >> 16) & 0xFF) + ((argb[l] >> 8) & 0xFF) + (argb[l] & 0xFF);
                put_pixel(T->mem, T->bpp, x + l, y, argb[l]);
            }
        }
    }
}

static void xf_rows_span_scalar(const XfTri *T, int y0, int y1, int step, XfCount *cnt);

/* RECOMP_PIXEL_CHECK=1: the four-lane path against the scalar one on the
 * same span. The rows' pixels and depths are saved; the span is drawn four at
 * a time, captured, restored, drawn by the scalar code, captured, and the two
 * compared. On a difference the scalar draw is repeated from the same start:
 * if that differs from the first scalar draw too, the input changed
 * underneath (a video frame decoded into the texture meanwhile) and the span
 * counts as unstable, not as a mismatch. The scalar result stays. Slow -- a
 * differential test, not a mode to play in. */
static void xf_rows_check(const XfTri *T, int y0, int y1, XfCount *cnt)
{
    static volatile LONG reported, spans, unstable;
    int w = T->maxx - T->minx, rows = y1 - y0, y, x, i, bad = -1;
    size_t npx = (size_t)(w > 0 ? w : 0) * (size_t)(rows > 0 ? rows : 0);
    uint32_t *orig = NULL, *r4 = NULL, *r1 = NULL;
    float *zorig = NULL, *z4 = NULL, *z1 = NULL;
    XfCount c4 = {0, 0, 0, 0, 0, 0}, c1 = {0, 0, 0, 0, 0, 0};

    if (!npx) {
        xf_rows_span_scalar(T, y0, y1, 1, cnt);
        return;
    }
    orig = (uint32_t *)malloc(npx * 4);
    r4 = (uint32_t *)malloc(npx * 4);
    r1 = (uint32_t *)malloc(npx * 4);
    if (T->zb) {                        /* depths, then the stencil bytes */
        zorig = (float *)malloc(npx * 5);
        z4 = (float *)malloc(npx * 5);
        z1 = (float *)malloc(npx * 5);
    }
    if (!orig || !r4 || !r1 || (T->zb && (!zorig || !z4 || !z1))) {
        free(orig); free(r4); free(r1); free(zorig); free(z4); free(z1);
        xf_rows_span_scalar(T, y0, y1, 1, cnt);
        return;
    }
    P = T->ps;
#define CK_GET(dst, zdst) do { \
        for (y = y0, i = 0; y < y1; y++) \
            for (x = T->minx; x < T->maxx; x++, i++) { \
                uint8_t *q = surface_pixel(x, y, T->bpp); \
                dst[i] = !q ? 0 : T->bpp == 4 ? *(uint32_t *)q : *(uint16_t *)q; \
            } \
        if (T->zb) \
            for (y = y0, i = 0; y < y1; y++, i += w) { \
                memcpy((zdst) + i, T->zb + (size_t)y * NV_ZBUF_W + T->minx, (size_t)w * 4); \
                memcpy((uint8_t *)((zdst) + npx) + i, \
                       zbuf_stencil(T->zb) + (size_t)y * NV_ZBUF_W + T->minx, (size_t)w); \
            } \
    } while (0)
#define CK_PUT(src, zsrc) do { \
        for (y = y0, i = 0; y < y1; y++) \
            for (x = T->minx; x < T->maxx; x++, i++) { \
                uint8_t *q = surface_pixel(x, y, T->bpp); \
                if (!q) continue; \
                if (T->bpp == 4) *(uint32_t *)q = src[i]; \
                else *(uint16_t *)q = (uint16_t)src[i]; \
            } \
        if (T->zb) \
            for (y = y0, i = 0; y < y1; y++, i += w) { \
                memcpy(T->zb + (size_t)y * NV_ZBUF_W + T->minx, (zsrc) + i, (size_t)w * 4); \
                memcpy(zbuf_stencil(T->zb) + (size_t)y * NV_ZBUF_W + T->minx, \
                       (const uint8_t *)((zsrc) + npx) + i, (size_t)w); \
            } \
    } while (0)
#define CK_DIFF(a, za, b, zb_) \
    (memcmp((a), (b), npx * 4) || (T->zb && memcmp((za), (zb_), npx * 5)))

    CK_GET(orig, zorig);
    xf_rows_span4(T, y0, y1, &c4);
    CK_GET(r4, z4);
    CK_PUT(orig, zorig);
    P = T->ps;
    xf_rows_span_scalar(T, y0, y1, 1, &c1);
    CK_GET(r1, z1);
    if (CK_DIFF(r1, z1, r4, z4) || c1.pixels != c4.pixels
        || c1.depth_fail != c4.depth_fail || c1.zpass != c4.zpass
        || c1.csum != c4.csum) {
        /* Both again from the same start. A real difference repeats: each
         * path agrees with itself and they still disagree. */
        uint32_t *r4b = (uint32_t *)malloc(npx * 4), *r1b = (uint32_t *)malloc(npx * 4);
        float *z4b = T->zb ? (float *)malloc(npx * 5) : NULL;
        float *z1b = T->zb ? (float *)malloc(npx * 5) : NULL;
        XfCount c4b = {0, 0, 0, 0, 0, 0}, c1b = {0, 0, 0, 0, 0, 0};
        if (r4b && r1b && (!T->zb || (z4b && z1b))) {
            CK_PUT(orig, zorig);
            xf_rows_span4(T, y0, y1, &c4b);
            CK_GET(r4b, z4b);
            CK_PUT(orig, zorig);
            P = T->ps;
            xf_rows_span_scalar(T, y0, y1, 1, &c1b);
            CK_GET(r1b, z1b);
            if (CK_DIFF(r1, z1, r1b, z1b) || CK_DIFF(r4, z4, r4b, z4b)
                || c1b.csum != c1.csum || c4b.csum != c4.csum) {
                InterlockedIncrement(&unstable);
            } else {
                for (i = 0; i < (int)npx && bad < 0; i++)
                    if (r1[i] != r4[i])
                        bad = i;
                if (InterlockedIncrement(&reported) <= 8)
                    fprintf(stderr, "[PIXCHECK] MISMATCH span y %d..%d x %d..%d,"
                            " first pixel %d (scalar %08X four-lane %08X);"
                            " pixels %llu/%llu zfail %llu/%llu csum %llu/%llu%c",
                            y0, y1, T->minx, T->maxx, bad,
                            bad >= 0 ? r1[bad] : 0, bad >= 0 ? r4[bad] : 0,
                            (unsigned long long)c1.pixels, (unsigned long long)c4.pixels,
                            (unsigned long long)c1.depth_fail,
                            (unsigned long long)c4.depth_fail,
                            (unsigned long long)c1.csum, (unsigned long long)c4.csum, 10);
            }
            c1 = c1b;
        }
        free(r4b); free(r1b); free(z4b); free(z1b);
    }
    if (!(InterlockedIncrement(&spans) & 0xFFFFF))
        fprintf(stderr, "[PIXCHECK] %ld spans compared, %ld mismatched, %ld"
                " unstable (input changed)%c", (long)spans, (long)reported,
                (long)unstable, 10);
#undef CK_GET
#undef CK_PUT
#undef CK_DIFF
    cnt->depth_fail += c1.depth_fail;
    cnt->pixels += c1.pixels;
    cnt->zpass += c1.zpass;
    cnt->csum += c1.csum;
    /* The visibility-report counts too: left out, every checked draw read as
     * hidden, and Blinx 2, which tests its player against a report, stopped
     * drawing him under RECOMP_PIXEL_CHECK. */
    for (i = 0; i < NV_ZEPOCHS; i++)
        cnt->zpe[i] += c1.zpe[i];
    free(orig); free(r4); free(r1); free(zorig); free(z4); free(z1);
}

/* Rows y0, y0+step, ... < y1 of triangle T. */
/* RECOMP_PIXEL_CHECK=<file> (not "1"): check only while the file exists,
 * looked at once a flip -- the check is far too slow to reach a scene with,
 * so a run gets there at speed and turns it on there (pixel_check_flip). */
static const char *s_pxchk_file;
static volatile LONG s_pxchk_on;

static void pixel_check_flip(void)
{
    FILE *f;
    if (!s_pxchk_file)
        return;
    f = fopen(s_pxchk_file, "rb");
    if (f)
        fclose(f);
    InterlockedExchange(&s_pxchk_on, f != NULL);
}

static void xf_rows_span(const XfTri *T, int y0, int y1, int step, XfCount *cnt)
{
    static volatile LONG mode = -1;     /* 0 scalar, 1 four-lane, 2 check, 3 check on a flag */
    if (mode < 0) {
        const char *pc = getenv("RECOMP_PIXEL_CHECK");
        LONG m = getenv("RECOMP_SCALAR_PIXELS") ? 0
               : pc ? (strcmp(pc, "1") ? 3 : 2) : 1;
        if (InterlockedCompareExchange(&mode, m, -1) == -1)
            fprintf(stderr, "[GPU] pixel shading: %s%c", m == 0 ? "scalar"
                    : m == 2 ? "four-lane, checked against scalar"
                    : m == 3 ? "four-lane, checked while the RECOMP_PIXEL_CHECK file exists"
                    : "four-lane", 10);
    }
    if (mode == 3) {
        if (s_pxchk_on) {
            if (step == 1 && T->use_rc && !T->ps->self_sample) {
                xf_rows_check(T, y0, y1, cnt);
                return;
            }
            xf_rows_span_scalar(T, y0, y1, step, cnt);
            return;
        }
    }
    /* A draw sampling its own target goes four-lane too, unless checking
     * (RECOMP_SELF_SAMPLE_SCALAR=1 keeps it scalar). Four lanes read before
     * they write, the scalar loop writes each pixel before the next reads,
     * so the two differ only where a pixel samples one written earlier in
     * its own group of four -- a read-after-write within a draw that the
     * hardware leaves undefined, and where it too returns the old texel.
     * Blinx 2's two full-screen colour passes over its main surface are
     * such draws, ~600K pixels a frame: 29 vs 27 fps median in stage
     * 01-01 gameplay, in-run A/B. */
    static int self_scalar = -1;
    if (self_scalar < 0)
        self_scalar = getenv("RECOMP_SELF_SAMPLE_SCALAR") != NULL;
    if (mode && step == 1 && T->use_rc
        && (!T->ps->self_sample || ((mode == 1 || mode == 3) && !self_scalar))) {
        if (mode == 2)
            xf_rows_check(T, y0, y1, cnt);
        else
            xf_rows_span4(T, y0, y1, cnt);
        return;
    }
    xf_rows_span_scalar(T, y0, y1, step, cnt);
}

static void xf_rows(const XfTri *T, int y0, int step, XfCount *cnt)
{
    xf_rows_span(T, y0, T->maxy, step, cnt);
}

/* Deferred rasterisation.
 *
 * Triangles used to be rasterised per draw call: set up, then the pool woken
 * for the call's rows, then waited for. A Blinx 2 hub frame is ~400 draw
 * calls, most a few hundred to a few thousand pixels, so the pool sat idle
 * while the executor shaded them itself -- it was ~70% inside its own xf_rows
 * and the title spent 85-92% of its time waiting for pushbuffer space.
 *
 * Now set-up triangles go into a batch together with a snapshot of the state
 * they need (PixState, one per draw call, shared while it does not change).
 * A full batch is handed to the workers and the executor goes on filling the
 * next one, so the vertex programs and the pushbuffer walk overlap the
 * shading. Every band of rows goes through the batches, and through each
 * batch's triangles, in submission order, so depth testing and blending stay
 * exactly sequential (see the pool below).
 *
 * Anything that touches the results from the executor thread drains the
 * queue first (rq_sync): clears, depth-buffer reuse, visibility reports,
 * flips, dumps, direct screen-space drawing -- and a draw that samples a
 * surface the queue is still writing (render to texture: Blinx's floor
 * reflection, its bloom downsample).
 * RECOMP_RASTER_THREADS=<n> sets the workers (1: everything on the executor),
 * RECOMP_RASTER_BATCH=0 drains after every draw call.
 * Every count a worker keeps is its own until the batch ends (XfCount on
 * its stack, put_pixel's in TLS): shared counters bumped per pixel made the
 * workers fight over cache lines. */
#define NV_RASTER_MAX_THREADS 32      /* default: cores - 4 */
#define NV_RQ_TRIS   4096
#define NV_RQ_STATES 256
#define NV_RQ_SURFS  8
#define NV_RQ_TEXS   16

/* One draw's geometry, deferred to the raster pool: its vertex inputs and
 * constants are copied into the batch, and a worker transforms, assembles,
 * clips and sets it up into the `cap` triangle slots reserved for it at
 * tri0 -- in order, so the batch draws exactly what the executor would have
 * queued. Slots it does not fill are left empty (no rows). */
static void geo_fold(const GeoStats *st);
static void geo_capture(GeoCtx *g);
static uint32_t geo_count(uint32_t prim, uint32_t n);
typedef void (*GeoSink)(void *sink, XfTri *T);
static void geo_assemble(const GeoCtx *g, GeoStats *st, GeoSink put, void *sink,
                         uint32_t prim, const Nv2aVshOutput *x, uint32_t n);
#define NV_RQ_GEO    1024               /* draws per batch */
#define NV_RQ_GVERTS 32768              /* their vertices */
typedef struct {
    const Nv2aVshProg *prog;
    uint32_t in0, n, prim, tri0, cap, cst;
    GeoCtx g;
    const PixState *ps;
    int ymin, ymax;                     /* rows drawn, once done */
    long px;
} GeoJob;

typedef struct {
    int n, nst, nsurf;                  /* nsurf < 0: too many to track */
    long px;                            /* bounding-box pixels queued */
    int ymin, ymax;                     /* rows the batch touches */
    XfTri tri[NV_RQ_TRIS];
    Nv2aVshOutput v[NV_RQ_TRIS][3];
    PixState st[NV_RQ_STATES];
    uint32_t surf_lo[NV_RQ_SURFS], surf_hi[NV_RQ_SURFS];
    uint32_t surf_lay[NV_RQ_SURFS];     /* how it was written: rq_layout */
    /* What the batch's draws sample, so a later draw that renders into one of
     * them starts a new batch (see rq_tex_overlaps). ntex < 0: too many. */
    int ntex;
    uint32_t tex_lo[NV_RQ_TEXS], tex_hi[NV_RQ_TEXS];
    GeoJob *geo;                        /* NV_RQ_GEO */
    int ngeo;
    uint32_t nin;
    float (*in)[NV2A_VSH_INPUTS][4];    /* NV_RQ_GVERTS */
    float (*cst)[NV2A_VSH_CONSTANTS][4];/* NV_RQ_GEO */
    /* Next job to claim, tagged with the batch's number in the high half
     * (geo_claim_take). A plain counter let a worker that saw the slot just
     * before it was retired claim a job of the batch filled into it next:
     * geo_left then reached 0 a job early, and the bands started on
     * triangles still being set up. */
    volatile LONG64 geo_claim;
    volatile LONG geo_left;
    int barrier;                       /* waits for every earlier batch */
} RasterBatch;

/* Batches: one filling, the rest drawing. A frame alternates stretches the
 * executor sets up (many small draws) with stretches the raster workers
 * chew on, so the ring is the buffer between them: 16 against 4 gave
 * stage 01-01 26-27 vs 25 fps median in two in-run A/Bs (async sync,
 * all-band barriers). ~12 MB a slot. */
#define NV_RQ_DEPTH  16

static RasterBatch *s_rq[NV_RQ_DEPTH];
static int s_rq_fill;                   /* the batch being filled */
static const PixState *s_rq_ps;         /* this draw's state, in s_rq[fill] */
static int s_rq_each = -1;              /* RECOMP_RASTER_BATCH=0 */

/* Work is dealt out in bands of NV_BAND rows by absolute y, and each band
 * goes through every triangle of the batch in submission order on one
 * thread, so no pixel is ever written out of order.
 *
 * Bands are claimed, not assigned. With rows dealt statically (y mod n) the
 * slowest worker set the pace -- sixteen workers next to the title's threads
 * on sixteen cores means some share a core -- and in Blinx 2's hub worker 0
 * sat 60% idle while the executor waited on the batch. A claimed band also
 * lets the executor work instead of waiting (rq_wait).
 * Four rows, not one: a 32-bit swizzled surface keeps a 4x4 block in each
 * 64-byte line, and rows dealt one by one put four threads on every line of
 * the reflection targets. */
#define NV_BAND 4

/* A batch in order on the calling thread: no pool. */
/* Rows [y0, y1) of a clear job, in the band that owns them. The colour
 * clear writes the same values clear_surface does; the depth clear fills
 * the job's rectangle of its depth buffer. */
static void xf_job_rows(const XfTri *T, int y0, int y1)
{
    int y, x, lo = y0 > T->miny ? y0 : T->miny, hi = y1 < T->maxy ? y1 : T->maxy;
    if (T->job == 2) {
        for (y = lo; y < hi; y++) {
            if (T->jpitch & 1)
                for (x = T->minx; x < T->maxx; x++)
                    T->zb[(size_t)y * NV_ZBUF_W + x] = T->jz;
            if (T->jpitch & 2)
                memset(zbuf_stencil(T->zb) + (size_t)y * NV_ZBUF_W + T->minx,
                       (int)T->jcolor, (size_t)(T->maxx - T->minx));
        }
        return;
    }
    {
        uint16_t v16 = (uint16_t)(((T->jcolor >> 8) & 0xF800)
                                | ((T->jcolor >> 5) & 0x07E0)
                                | ((T->jcolor >> 3) & 0x001F));
        const SwzTable *swz = T->jswz;
        for (y = lo; y < hi; y++) {
            if (swz) {
                if ((uint32_t)y >= swz->h)
                    break;
                for (x = T->minx; x < T->maxx && (uint32_t)x < swz->w; x++) {
                    uint8_t *p = T->mem + T->jbase + swz->x[x] + swz->y[y];
                    if (T->bpp == 4) *(uint32_t *)p = T->jcolor;
                    else if (T->bpp == 2) *(uint16_t *)p = v16;
                }
            } else {
                uint8_t *row = T->mem + T->jbase + (size_t)y * T->jpitch;
                for (x = T->minx; x < T->maxx; x++) {
                    if (T->bpp == 4) ((uint32_t *)row)[x] = T->jcolor;
                    else if (T->bpp == 2) ((uint16_t *)row)[x] = v16;
                }
            }
        }
    }
}

static void xf_rows_queue(const XfTri *q, int qn, XfCount *cnt)
{
    int i;
    for (i = 0; i < qn; i++) {
        if (q[i].job)
            xf_job_rows(&q[i], q[i].miny, q[i].maxy);
        else
            xf_rows(&q[i], q[i].miny, 1, cnt);
    }
}

/* One band through every triangle of a batch. */
static void xf_band(const XfTri *q, int qn, int band, XfCount *cnt)
{
    int y0 = band * NV_BAND, y1 = y0 + NV_BAND, i;
    for (i = 0; i < qn; i++) {
        const XfTri *T = &q[i];
        int lo = y0 > T->miny ? y0 : T->miny;
        int hi = y1 < T->maxy ? y1 : T->maxy;
        if (lo < hi) {
            if (T->job)
                xf_job_rows(T, lo, hi);
            else
                xf_rows_span(T, lo, hi, 1, cnt);
        }
    }
}

static void rq_count(const XfCount *c)
{
    s_gpu.xf_depth_fail += c->depth_fail;
    s_gpu.xf_pixels += c->pixels;
    s_gpu.zpass_count += c->zpass;
    s_gpu.csum += c->csum;
    stats_fold(c->written, c->written_max);
}

#if defined(_WIN32)
/* The pool works on a pipeline of batches rather than one at a time.
 *
 * Each band of NV_BAND rows takes the submitted batches strictly in order --
 * band_next[b] is the next batch it owes -- so depth testing and blending
 * stay exactly sequential for every pixel, but different bands may be on
 * different batches. A worker claims a free band and runs it through every
 * batch submitted so far, then looks for another. One batch at a time (all
 * workers start it, all finish it, then the next) left a worker idle 44% of
 * the time in Blinx 2's hub: a batch touching a few rows has only a few
 * bands, and every batch ended in a tail the whole pool waited out, with the
 * executor waiting behind it -- and the title waiting on the executor.
 *
 * The executor fills one batch while up to NV_RQ_DEPTH - 1 are in flight;
 * it waits only when the ring is full or something drains (rq_wait). */
#define NV_MAX_BANDS 1024                   /* 4096 rows */

static struct {
    int n;                                  /* worker threads */
    HANDLE thread[NV_RASTER_MAX_THREADS];
    SRWLOCK lock;
    CONDITION_VARIABLE work_cv, done_cv;
    volatile LONG head;                     /* batches submitted */
    LONG tail;                              /* batches retired (executor) */
    volatile LONG band_next[NV_MAX_BANDS];  /* next batch each band owes */
    volatile LONG band_busy[NV_MAX_BANDS];
    int band0[NV_RQ_DEPTH], band1[NV_RQ_DEPTH]; /* bands a batch touches */
    volatile LONG left[NV_RQ_DEPTH];        /* its bands not yet drawn */
    volatile LONG done[NV_RQ_DEPTH];
    volatile LONG slot_seq[NV_RQ_DEPTH];    /* which batch a slot holds */
    volatile LONG geo_ready[NV_RQ_DEPTH];   /* its geometry is all set up */
    volatile LONG barrier[NV_RQ_DEPTH];     /* waits for every earlier batch */
    volatile LONG done_prefix;              /* batches 0..n-1 all drawn */
    Nv2aVshOutput *xf_scratch[NV_RASTER_MAX_THREADS + 1];
    GeoStats gst[NV_RASTER_MAX_THREADS + 1];
    int nbands;                             /* bands any batch has used */
    XfCount cnt[NV_RASTER_MAX_THREADS + 1]; /* + 1: the executor helping */
} s_pool;

/* Visibility reports waiting for the batches before them (RECOMP_ASYNC_SYNC):
 * written by whoever extends done_prefix past `seq`. */
#define NV_REPORTS_PENDING 32
static struct { uint32_t at, epoch; LONG seq; uint64_t stamp; }
    s_rep[NV_REPORTS_PENDING];
static int s_rep_head, s_rep_n;             /* under s_pool.lock */
static uint32_t s_zpe_total[NV_ZEPOCHS];    /* folded at drains */
static void rq_report_write(uint32_t at, uint64_t stamp, uint32_t epoch);

/* Under s_pool.lock: extend the run of batches drawn in full, and answer
 * the visibility tests that were waiting for them. */
static void rq_advance_prefix(void)
{
    LONG d = s_pool.done_prefix;
    while (d < s_pool.head && s_pool.slot_seq[d % NV_RQ_DEPTH] == d
           && s_pool.done[d % NV_RQ_DEPTH])
        d++;
    s_pool.done_prefix = d;
    while (s_rep_n && s_rep[s_rep_head].seq < d) {
        rq_report_write(s_rep[s_rep_head].at, s_rep[s_rep_head].stamp,
                        s_rep[s_rep_head].epoch);
        s_rep_head = (s_rep_head + 1) % NV_REPORTS_PENDING;
        s_rep_n--;
    }
}

static int band_has_work(int b)
{
    return s_pool.band_next[b] < s_pool.head && !s_pool.band_busy[b];
}

/* Where the raster pool's time goes: workers asleep because every band
 * with work waits (on a batch's geometry, or a barrier) or because there is
 * none, the executor waiting for a ring slot or a drain, and the batches it
 * kicked. Always counted -- a QueryPerformanceCounter around each sleep,
 * which is noise next to the sleep -- for the once-a-second [PERF] line
 * (nv2a_pb_exec_perf). RECOMP_RQ_STATS=1 also prints the per-frame detail
 * every 60 flips. */
static struct {
    volatile LONG64 blk_geo_us, blk_bar_us, idle_us, wait_us;
    volatile LONG64 kicks, barriers, tris;
} s_rqst;
static int s_rqst_print = -1;
static __declspec(thread) int t_blk_why;    /* 1 geometry, 2 barrier */

static int rqst_on(void)
{
    return 1;
}

static LONG64 rqst_us(void)
{
    static LARGE_INTEGER f;
    LARGE_INTEGER t;
    if (!f.QuadPart)
        QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (LONG64)(t.QuadPart * 1000000 / f.QuadPart);
}

/* Claim one band that owes a batch and draw it through every batch
 * submitted so far. Returns 0 if there was nothing to claim. */
static int rq_work_one(int k, int *start)
{
    int nb = s_pool.nbands, i, blocked = 0, any_blocked = 0;

    for (i = 0; i < nb; i++) {
        int b = (*start + i) % nb;
        LONG seq;
        XfCount *c = &s_pool.cnt[k];

        LONG start_seq;
        if (!band_has_work(b)
            || InterlockedCompareExchange(&s_pool.band_busy[b], 1, 0))
            continue;
        start_seq = s_pool.band_next[b];
        while ((seq = s_pool.band_next[b]) < s_pool.head) {
            int slot = seq % NV_RQ_DEPTH, in;
            /* A band the batch does not touch can fall behind until the
             * slot holds a newer batch. Read the range between two looks
             * at slot_seq: if the slot moved on, batch seq was retired --
             * which needs every band it touches -- so this one it does not. */
            if (s_pool.slot_seq[slot] == seq && !s_pool.geo_ready[slot]) {
                /* Its triangles are still being set up, so which bands it
                 * touches is not known yet. */
                blocked = 1;
                t_blk_why = 1;
                break;
            }
            in = s_pool.slot_seq[slot] == seq
                 && b >= s_pool.band0[slot] && b < s_pool.band1[slot];
            MemoryBarrier();
            if (s_pool.slot_seq[slot] == seq && s_pool.barrier[slot]
                && s_pool.done_prefix < seq) {
                /* Render to texture, either way round: this batch samples
                 * what an earlier one draws, or draws what an earlier one
                 * samples, anywhere on the surface -- so it waits for all
                 * of them, not only for this band. Every band stops here,
                 * also those the batch does not touch: then everything
                 * after a barrier is ordered after everything before it,
                 * and a later draw sampling the same target needs no
                 * barrier of its own (s_rq_bar_seq). Holding back only the
                 * batch's own bands took ~17 barriers a frame in stage 01-01
                 * -- one per batch that sampled a render target still
                 * drawing -- and the workers slept 25-55% of the frame on
                 * them. */
                blocked = 1;
                t_blk_why = 2;
                break;
            }
            if (in && s_pool.slot_seq[slot] == seq) {
                const RasterBatch *rb = s_rq[slot];
                uint64_t wn;
                uint32_t wmx;
                xf_band(rb->tri, rb->n, b, c);
                stats_take(&wn, &wmx);          /* resets: accumulate */
                c->written += wn;
                if ((wmx & 0x00FFFFFFu) > (c->written_max & 0x00FFFFFFu))
                    c->written_max = wmx;
                InterlockedExchange(&s_pool.band_next[b], seq + 1);
                if (InterlockedDecrement(&s_pool.left[slot]) == 0) {
                    AcquireSRWLockExclusive(&s_pool.lock);
                    s_pool.done[slot] = 1;
                    rq_advance_prefix();
                    WakeAllConditionVariable(&s_pool.done_cv);
                    ReleaseSRWLockExclusive(&s_pool.lock);
                }
            } else {
                InterlockedExchange(&s_pool.band_next[b], seq + 1);
            }
        }
        InterlockedExchange(&s_pool.band_busy[b], 0);
        *start = b + 1;
        if (blocked) {
            /* Stopped at a batch it may not start yet: whatever it drew
             * before that counts as work, otherwise try the other bands. */
            blocked = 0;
            any_blocked = 1;
            if (s_pool.band_next[b] == start_seq)
                continue;
        }
        return 1;
    }
    return any_blocked ? 2 : 0;
}

static void geo_slot_sink(void *sink, XfTri *T);

typedef struct {
    RasterBatch *b;
    uint32_t next, end;
    const PixState *ps;
    int ymin, ymax;
    long px;
} GeoSlots;

/* A batch's last geometry job is done: its rows are known, so hand it to
 * the bands, as rq_kick does for a batch with none. */
static void rq_geo_finish(RasterBatch *b, int slot)
{
    int i, ymin = b->ymin, ymax = b->ymax, b0, b1;
    for (i = 0; i < b->ngeo; i++) {
        if (b->geo[i].ymin < ymin) ymin = b->geo[i].ymin;
        if (b->geo[i].ymax > ymax) ymax = b->geo[i].ymax;
    }
    b0 = ymin / NV_BAND;
    b1 = (ymax + NV_BAND - 1) / NV_BAND;
    if (b1 > s_pool.nbands) b1 = s_pool.nbands;
    if (b0 > b1) b0 = b1;
    AcquireSRWLockExclusive(&s_pool.lock);
    s_pool.band0[slot] = b0;
    s_pool.band1[slot] = b1;
    s_pool.left[slot] = b1 - b0;
    s_pool.done[slot] = b1 == b0;
    MemoryBarrier();
    s_pool.geo_ready[slot] = 1;
    rq_advance_prefix();
    WakeAllConditionVariable(&s_pool.work_cv);
    WakeAllConditionVariable(&s_pool.done_cv);
    ReleaseSRWLockExclusive(&s_pool.lock);
}

/* Claim the next geometry job of batch `seq`, or -1 if it has none left or
 * the slot no longer holds that batch. */
static LONG geo_claim_take(RasterBatch *rb, LONG seq)
{
    for (;;) {
        LONG64 v = rb->geo_claim;
        LONG j = (LONG)(v & 0xFFFFFFFF);
        if ((LONG)(v >> 32) != seq || j >= rb->ngeo)
            return -1;
        if (InterlockedCompareExchange64(&rb->geo_claim, v + 1, v) == v)
            return j;
    }
}

static int geo_has_jobs(const RasterBatch *rb)
{
    return (LONG)(rb->geo_claim & 0xFFFFFFFF) < rb->ngeo;
}

/* Claim and run one geometry job of the oldest batch that has one left. */
static int rq_geo_work(int k)
{
    LONG seq;
    for (seq = s_pool.tail; seq < s_pool.head; seq++) {
        int slot = seq % NV_RQ_DEPTH;
        RasterBatch *rb = s_rq[slot];
        LONG j;
        GeoJob *J;
        GeoSlots S;
        Nv2aVshOutput *xf;
        uint32_t i;

        if (s_pool.slot_seq[slot] != seq || s_pool.geo_ready[slot]
            || (j = geo_claim_take(rb, seq)) < 0)
            continue;
        J = &rb->geo[j];
        if (!s_pool.xf_scratch[k])
            s_pool.xf_scratch[k] = (Nv2aVshOutput *)malloc(
                sizeof(Nv2aVshOutput) * NV_MAX_INDICES);
        xf = s_pool.xf_scratch[k];
        S.b = rb; S.next = J->tri0; S.end = J->tri0 + J->cap; S.ps = J->ps;
        S.ymin = 1 << 30; S.ymax = 0; S.px = 0;
        if (xf && nv2a_vsh_transform(J->prog, (const float (*)[4])rb->cst[J->cst],
                                     (const float (*)[NV2A_VSH_INPUTS][4])
                                         (rb->in + J->in0), J->n, xf))
            geo_assemble(&J->g, &s_pool.gst[k], geo_slot_sink, &S, J->prim,
                         xf, J->n);
        for (i = S.next; i < S.end; i++) {     /* unused: no rows */
            rb->tri[i].miny = rb->tri[i].maxy = 0;
            rb->tri[i].minx = rb->tri[i].maxx = 0;
        }
        J->ymin = S.ymin; J->ymax = S.ymax; J->px = S.px;
        if (InterlockedDecrement(&rb->geo_left) == 0)
            rq_geo_finish(rb, slot);
        return 1;
    }
    return 0;
}

static int rq_any_work(void)
{
    int b;
    LONG seq;
    for (seq = s_pool.tail; seq < s_pool.head; seq++) {
        int slot = seq % NV_RQ_DEPTH;
        if (s_pool.slot_seq[slot] == seq && !s_pool.geo_ready[slot]
            && geo_has_jobs(s_rq[slot]))
            return 1;
    }
    for (b = 0; b < s_pool.nbands; b++)
        if (band_has_work(b))
            return 1;
    return 0;
}

static DWORD WINAPI raster_worker(LPVOID arg)
{
    int k = (int)(intptr_t)arg, start = k * 7;
    for (;;) {
        if (rq_geo_work(k))                 /* geometry first: bands wait on it */
            continue;
        switch (rq_work_one(k, &start)) {
        case 1: continue;
        case 2:
            /* Every band with work is waiting on a batch (its geometry, or
             * a barrier): sleep until a batch finishes rather than spin --
             * spinning workers starved the executor of its core. */
        {
            LONG64 t0 = rqst_on() ? rqst_us() : 0;
            AcquireSRWLockExclusive(&s_pool.lock);
            SleepConditionVariableSRW(&s_pool.done_cv, &s_pool.lock, 1, 0);
            ReleaseSRWLockExclusive(&s_pool.lock);
            if (t0)
                InterlockedAdd64(t_blk_why == 2 ? &s_rqst.blk_bar_us
                                                : &s_rqst.blk_geo_us,
                                 rqst_us() - t0);
            continue;
        }
        }
        {
        LONG64 t0 = rqst_on() ? rqst_us() : 0;
        AcquireSRWLockExclusive(&s_pool.lock);
        while (!rq_any_work())
            SleepConditionVariableSRW(&s_pool.work_cv, &s_pool.lock,
                                      INFINITE, 0);
        ReleaseSRWLockExclusive(&s_pool.lock);
        if (t0)
            InterlockedAdd64(&s_rqst.idle_us, rqst_us() - t0);
        }
    }
    return 0;
}

/* Worker threads; 1 means none, everything on the executor. The default
 * leaves a few cores for the title and the host. */
static int raster_pool_size(void)
{
    static int init;
    if (!init) {
        const char *e = getenv("RECOMP_RASTER_THREADS");
        SYSTEM_INFO si;
        int n, k;
        init = 1;
        GetSystemInfo(&si);
        n = e ? atoi(e) : (int)si.dwNumberOfProcessors - 4;
        if (n < 1) n = 1;
        if (n > NV_RASTER_MAX_THREADS) n = NV_RASTER_MAX_THREADS;
        s_pool.n = n;
        InitializeSRWLock(&s_pool.lock);
        InitializeConditionVariable(&s_pool.work_cv);
        InitializeConditionVariable(&s_pool.done_cv);
        /* Every band from the start: with geometry set up on the pool a
         * batch's rows are known only when it is done, so bands cannot be
         * added as batches are submitted. */
        s_pool.nbands = NV_ZBUF_H / NV_BAND;
        if (n > 1)
            for (k = 0; k < n; k++)
                s_pool.thread[k] = CreateThread(NULL, 0, raster_worker,
                                                (LPVOID)(intptr_t)k, 0, NULL);
    }
    return s_pool.n;
}

/* Raster worker k's thread handle, or NULL before the pool starts (on the
 * first draw) or without one. For a sampling profiler: with the executor
 * waiting on the workers, theirs is the time worth timing. */
void *nv2a_pb_exec_raster_thread(int k)
{
    return k >= 0 && k < s_pool.n ? s_pool.thread[k] : NULL;
}

/* Retire finished batches in order, so their slots can be refilled. */
static void rq_reap(void)
{
    s_pool.tail = s_pool.done_prefix;
}

/* Wait until at most `keep` batches are in flight, drawing bands alongside
 * the workers rather than sleeping. */
static void rq_wait_until(LONG keep)
{
    int start = 0;
    LONG64 t0 = 0;
    for (;;) {
        rq_reap();
        if (s_pool.head - s_pool.tail <= keep) {
            if (t0)
                InterlockedAdd64(&s_rqst.wait_us, rqst_us() - t0);
            return;
        }
        if (!t0 && rqst_on())
            t0 = rqst_us();
        if (rq_geo_work(s_pool.n) || rq_work_one(s_pool.n, &start) == 1)
            continue;
        AcquireSRWLockExclusive(&s_pool.lock);
        if (!s_pool.done[s_pool.tail % NV_RQ_DEPTH] && !rq_any_work())
            SleepConditionVariableSRW(&s_pool.done_cv, &s_pool.lock, 1, 0);
        ReleaseSRWLockExclusive(&s_pool.lock);
    }
}

/* Everything queued drawn, and every count folded in. */
static void rq_wait(void)
{
    int k;
    /* Fold the counts even when nothing is in flight: a batch retired by an
     * earlier reap (rq_kick, rq_queue_overlaps) left its counts in cnt[], and
     * returning early here added them at some later drain instead. That
     * reached GET_REPORT -- visibility counts short by whole batches, or
     * carried past CLEAR_REPORT_VALUE into the next test -- and the per-draw
     * trace's pixel counts. */
    if (s_pool.head != s_pool.tail)
        rq_wait_until(0);
    for (k = 0; k <= s_pool.n; k++) {
        XfCount z = {0, 0, 0, 0, 0};
        int e;
        rq_count(&s_pool.cnt[k]);
        for (e = 0; e < NV_ZEPOCHS; e++)
            s_zpe_total[e] += s_pool.cnt[k].zpe[e];
        s_pool.cnt[k] = z;
        geo_fold(&s_pool.gst[k]);
        memset(&s_pool.gst[k], 0, sizeof s_pool.gst[k]);
    }
}

static int rq_in_flight(void)
{
    return s_pool.head != s_pool.tail;
}

/* For the [PERF] line: since the last call, the share of the workers' time
 * spent asleep with nothing to do (idle) and blocked on barriers or
 * geometry (blocked), and of the wall time the executor spent waiting for
 * the pool (ring slots, drains). Shares in [0,1]; one caller. */
void nv2a_pb_exec_perf(double *wk_idle, double *wk_blocked, double *exec_wait)
{
    static LONG64 t0, idle0, blk0, wait0;
    LONG64 t = rqst_us(), idle = s_rqst.idle_us;
    LONG64 blk = s_rqst.blk_geo_us + s_rqst.blk_bar_us, wait = s_rqst.wait_us;
    double span = (double)(t - t0), wk = span * (s_pool.n > 1 ? s_pool.n : 1);
    *wk_idle = *wk_blocked = *exec_wait = 0;
    if (t0 && span > 0) {
        *wk_idle = (idle - idle0) / wk;
        *wk_blocked = (blk - blk0) / wk;
        *exec_wait = (wait - wait0) / span;
    }
    t0 = t; idle0 = idle; blk0 = blk; wait0 = wait;
}

/* For the stall watchdog: the pool's bookkeeping, read racily -- a stuck
 * pool does not move, so a torn read is not the question. */
void nv2a_pb_exec_dump_state(void)
{
    int i, owed = 0, busy = 0, shown = 0;
    fprintf(stderr, "  [RQ] head %ld tail %ld done_prefix %ld fill %d"
            " reports pending %d (oldest seq %ld) epoch %u\n",
            (long)s_pool.head, (long)s_pool.tail, (long)s_pool.done_prefix,
            s_rq_fill, s_rep_n, s_rep_n ? (long)s_rep[s_rep_head].seq : -1L,
            s_gpu.report_epoch);
    for (i = 0; i < NV_RQ_DEPTH; i++) {
        const RasterBatch *b = s_rq[i];
        fprintf(stderr, "  [RQ] slot %d: seq %ld done %ld left %ld geo_ready %ld"
                " barrier %ld bands %d..%d ngeo %d geo_left %ld geo_next %ld"
                " tris %u\n", i, (long)s_pool.slot_seq[i], (long)s_pool.done[i],
                (long)s_pool.left[i], (long)s_pool.geo_ready[i],
                (long)s_pool.barrier[i], s_pool.band0[i], s_pool.band1[i],
                b ? b->ngeo : -1, b ? (long)b->geo_left : -1L,
                b ? (long)(b->geo_claim & 0xFFFFFFFF) : -1L, b ? b->n : 0u);
    }
    for (i = 0; i < s_pool.nbands; i++) {
        if (s_pool.band_busy[i])
            busy++;
        if (s_pool.band_next[i] < s_pool.head) {
            owed++;
            if (shown++ < 8)
                fprintf(stderr, "  [RQ] band %d owes from seq %ld%s\n", i,
                        (long)s_pool.band_next[i],
                        s_pool.band_busy[i] ? " (busy)" : "");
        }
    }
    fprintf(stderr, "  [RQ] %d of %d bands owe work, %d busy\n",
            owed, s_pool.nbands, busy);
}
#else
void nv2a_pb_exec_perf(double *wk_idle, double *wk_blocked, double *exec_wait)
{
    *wk_idle = *wk_blocked = *exec_wait = 0;
}
void nv2a_pb_exec_dump_state(void) { }
void *nv2a_pb_exec_raster_thread(int k) { (void)k; return NULL; }
static int raster_pool_size(void) { return 1; }
static void rq_wait(void) { }
static int rq_in_flight(void) { return 0; }
#endif

/* For the [PERF] line: draw calls and pixels written so far (pixels are
 * folded in at drains, so per second they lag by at most a frame). */
void nv2a_pb_exec_counts(uint32_t *draws, uint64_t *pixels)
{
    *draws = s_gpu.draws;
    *pixels = s_gpu.xf_pixels;
}

/* Hand the batch being filled to the workers (or draw it here, with no
 * pool) and start filling the next one. */
static void rq_kick(void)
{
    RasterBatch *b = s_rq[s_rq_fill];

    if (!b)
        return;
    if (b->n) {
        s_gpu.rq_tris += (uint64_t)b->n;
        s_gpu.rq_px += (uint64_t)b->px;
#if defined(_WIN32)
        if (raster_pool_size() > 1) {
            int slot = s_rq_fill;
            /* The slot's new number before its new range: a band that fell
             * behind on the slot's previous batch rereads slot_seq after the
             * range (rq_work_one), so it must never see the new range with
             * the old number. The fence keeps the compiler from sinking this
             * store below the plain ones that follow. */
            s_pool.slot_seq[slot] = s_pool.head;
            MemoryBarrier();
            s_pool.barrier[slot] = b->barrier;
            if (b->ngeo) {
                /* Rows unknown until its geometry is set up: rq_geo_finish
                 * hands it to the bands. */
                s_pool.done[slot] = 0;
                s_pool.geo_ready[slot] = 0;
                b->geo_claim = (LONG64)s_pool.head << 32;
                b->geo_left = b->ngeo;
            } else {
                int b0 = b->ymin / NV_BAND;
                int b1 = (b->ymax + NV_BAND - 1) / NV_BAND;
                if (b1 > s_pool.nbands) b1 = s_pool.nbands;
                if (b0 > b1) b0 = b1;
                s_pool.band0[slot] = b0;
                s_pool.band1[slot] = b1;
                s_pool.left[slot] = b1 - b0;
                s_pool.done[slot] = b1 == b0;
                s_pool.geo_ready[slot] = 1;
            }
            if (rqst_on()) {
                InterlockedIncrement64(&s_rqst.kicks);
                InterlockedAdd64(&s_rqst.tris, b->n);
                if (b->barrier)
                    InterlockedIncrement64(&s_rqst.barriers);
            }
            AcquireSRWLockExclusive(&s_pool.lock);
            InterlockedIncrement(&s_pool.head);
            rq_advance_prefix();              /* a batch with no rows is done */
            WakeAllConditionVariable(&s_pool.work_cv);
            ReleaseSRWLockExclusive(&s_pool.lock);
            s_gpu.rq_pooled++;
            /* The next slot must be retired before it is refilled. */
            rq_wait_until(NV_RQ_DEPTH - 1);
            s_rq_fill = s_pool.head % NV_RQ_DEPTH;
        } else
#endif
        {
            XfCount c = {0, 0, 0, 0, 0, 0};
            xf_rows_queue(b->tri, b->n, &c);
            rq_count(&c);
            s_gpu.rq_inline++;
        }
    }
    /* An empty batch is not handed over, and the one filled next is the
     * same batch under the same number. A barrier set on it is still owed
     * to whatever goes in next: rq_barrier recorded that number as
     * s_rq_bar_seq, and later render-to-texture checks skip every batch
     * before it on the strength of it. Cleared here, it went in without a
     * barrier while the skip stayed. That happened whenever a kick (GET
     * reaching PUT, a notify) came between the barrier and the draw's first
     * triangle, or the draw was culled entirely. The draws after it then
     * sampled render targets still being drawn: one-frame dark flashes in
     * stage 01-01's water (P7). */
    {
        int keep = b == s_rq[s_rq_fill] ? b->barrier : 0;
        b = s_rq[s_rq_fill];
        b->barrier = keep;
    }
    b->n = b->nst = b->nsurf = b->ntex = 0;
    b->ngeo = 0;
    b->nin = 0;
    b->px = 0;
    b->ymin = 1 << 30;
    b->ymax = 0;
    s_rq_ps = NULL;
}

/* Drains by call site: how many found work pending, and how long the
 * executor waited in them (ms) -- the drains sit on the kick-ack path. */
static struct { int line; uint32_t n; double ms; } s_sync_sites[24];

static void rq_sync_at(int line)
{
    uint64_t n;
    uint32_t mx;
    int i = 24;
#if defined(_WIN32)
    LARGE_INTEGER t0, t1, f;
#endif
    if (rq_in_flight() || (s_rq[s_rq_fill] && s_rq[s_rq_fill]->n)) {
        for (i = 0; i < 24 && s_sync_sites[i].line && s_sync_sites[i].line != line; i++)
            ;
        if (i < 24) {
            s_sync_sites[i].line = line;
            s_sync_sites[i].n++;
        }
    }
#if defined(_WIN32)
    if (i < 24)
        QueryPerformanceCounter(&t0);
#endif
    rq_kick();
    rq_wait();
#if defined(_WIN32)
    if (i < 24) {
        QueryPerformanceCounter(&t1);
        QueryPerformanceFrequency(&f);
        s_sync_sites[i].ms += 1e3 * (double)(t1.QuadPart - t0.QuadPart)
                            / (double)f.QuadPart;
    }
#endif
    stats_take(&n, &mx);              /* the executor's own direct drawing */
    stats_fold(n, mx);
}

/* BACK_END_WRITE_SEMAPHORE_RELEASE: D3D's fence.
 *
 * By default it is released as soon as the executor reaches it: every
 * command before it has been read and every vertex fetched, which is what
 * the title's pushbuffer-space fences (most of Blinx 2's ~9 a frame) need.
 * Queued pixels may still be sampling textures; the cases where the title
 * looks at what was drawn (flips, clears, visibility reports, render to
 * texture) drain the queue already.
 *
 * RECOMP_FENCE_STRICT=1 releases only once the pixels are written as well,
 * by draining the queue first. The title then waits on the rasteriser at
 * every fence: Blinx 2's hub ran at 11.6 fps that way against 17.0. */
static void rq_fence(uint32_t off, uint32_t val)
{
    static int strict = -1;
    if (strict < 0) {
        const char *e = getenv("RECOMP_FENCE_STRICT");
        strict = e && *e == '1';
    }
    if (strict)
        rq_sync();
    xbox_Nv2aSemaphoreRelease(off, val);
}

/* The physical address of a guest one, for render-to-texture tracking.
 * Blinx 2 draws its water reflections into surfaces at 0x01364000 and
 * 0x012E4000 and samples them as textures at 0x81364000 and 0x812E4000:
 * the same memory through the contiguous window. Compared raw, the
 * ranges never overlapped, so the water sampled a reflection still being
 * drawn without a barrier: dark one-frame flashes in stage 01-01's sea
 * under async sync (P7), hidden before by the drains at the clears. Every
 * range noted or asked about goes through here. */
static uint32_t rq_phys(uint32_t va)
{
    return va & 0x0FFFFFFFu;
}

/* How a surface write lays its texels out: the pitch, or swizzled at the
 * surface's size. Bands split work by pitch rows; a swizzled texel's bytes
 * lie in other bands' rows, so two writes to the same memory in different
 * layouts are only in order if the first is finished everywhere. D3D
 * clears a swizzled target as pitch and then draws it swizzled (P7,
 * 4810b1d: Blinx 2's reflection targets); this is the general case of it:
 * any write after a queued write of other layout to overlapping memory. */
static uint32_t rq_layout(const SwzTable *swz, uint32_t pitch)
{
    return swz ? 0x80000000u | (swz->w << 16) | swz->h : pitch;
}

static int rq_surf_overlaps(const RasterBatch *b, uint32_t lo, uint32_t hi)
{
    int i;
    if (b->nsurf < 0)
        return 1;
    for (i = 0; i < b->nsurf; i++)
        if (lo < b->surf_hi[i] && b->surf_lo[i] < hi)
            return 1;
    return 0;
}

/* Does batch b write memory overlapping [lo, hi) in another layout? */
static int rq_surf_relayout(const RasterBatch *b, uint32_t lo, uint32_t hi,
                            uint32_t lay)
{
    int i;
    if (b->nsurf < 0)
        return 1;
    for (i = 0; i < b->nsurf; i++)
        if (lo < b->surf_hi[i] && b->surf_lo[i] < hi && b->surf_lay[i] != lay)
            return 1;
    return 0;
}

static void rq_note_surface(RasterBatch *b, uint32_t lo, uint32_t hi,
                            uint32_t lay)
{
    int i;
    hi = rq_phys(lo) + (hi - lo);
    lo = rq_phys(lo);
    if (b->nsurf < 0)
        return;
    for (i = 0; i < b->nsurf; i++)
        if (b->surf_lo[i] == lo && b->surf_hi[i] == hi && b->surf_lay[i] == lay)
            return;
    if (b->nsurf == NV_RQ_SURFS) {
        b->nsurf = -1;                    /* every textured draw syncs */
        return;
    }
    b->surf_lo[b->nsurf] = lo;
    b->surf_hi[b->nsurf] = hi;
    b->surf_lay[b->nsurf] = lay;
    b->nsurf++;
}

/* The latest barrier batch (async): every band waits there for every
 * batch before it, so batches before it are already ordered ahead of
 * anything queued from then on and need no further barrier. */
#if defined(_WIN32)
static LONG s_rq_bar_seq;

static LONG rq_unordered_from(void)
{
    return s_rq_bar_seq > s_pool.tail ? s_rq_bar_seq : s_pool.tail;
}
#endif

/* Does anything queued -- filling or still being drawn -- write [lo, hi)? */
static int rq_queue_overlaps(uint32_t lo, uint32_t hi)
{
    hi = rq_phys(lo) + (hi - lo);
    lo = rq_phys(lo);
    if (s_rq[s_rq_fill] && rq_surf_overlaps(s_rq[s_rq_fill], lo, hi))
        return 2;
#if defined(_WIN32)
    {
        LONG seq;
        rq_reap();
        for (seq = rq_unordered_from(); seq < s_pool.head; seq++)
            if (!s_pool.done[seq % NV_RQ_DEPTH]
                && rq_surf_overlaps(s_rq[seq % NV_RQ_DEPTH], lo, hi))
                return 1;
    }
#endif
    return 0;
}

/* Does anything queued write memory overlapping [lo, hi) in another
 * layout than `lay` (rq_layout)? Kinds as rq_queue_overlaps. */
static int rq_queue_relayout(uint32_t lo, uint32_t hi, uint32_t lay)
{
    hi = rq_phys(lo) + (hi - lo);
    lo = rq_phys(lo);
    if (s_rq[s_rq_fill] && rq_surf_relayout(s_rq[s_rq_fill], lo, hi, lay))
        return 2;
#if defined(_WIN32)
    {
        LONG seq;
        rq_reap();
        for (seq = rq_unordered_from(); seq < s_pool.head; seq++)
            if (!s_pool.done[seq % NV_RQ_DEPTH]
                && rq_surf_relayout(s_rq[seq % NV_RQ_DEPTH], lo, hi, lay))
                return 1;
    }
#endif
    return 0;
}

/* Render to texture: a draw that samples memory the queue is still writing
 * has to wait for it. Only stages the draw can sample count: those the
 * shader stage program enables, or stage 0 before the combiners are
 * programmed. Counting every bound stage made stale bindings -- a reflection
 * target left on stage 1 -- drain the queue ~40 times a frame in Blinx 2's
 * hub. */
/* The memory texture stage `st` can sample in the current draw, or 0. */
static int draw_samples(int st, uint32_t *lo, uint32_t *hi)
{
    const Texture *t = &s_gpu.texs[st];
    if (!t->valid)
        return 0;
    if (s_gpu.rc_seen ? !((s_gpu.rc.stage_program >> (st * 5)) & 0x1F)
                      : st != 0)
        return 0;
    *lo = rq_phys(t->offset);
    *hi = *lo + tex_face_stride(t) * (t->cube ? 6u : 1u);
    return 1;
}

/* Asynchronous sync points, the default (RECOMP_ASYNC_SYNC=0 restores the
 * drains). Blinx 2's first mission ran 28 against 25 fps median with them,
 * alternating within one run; 15 soak runs (hub idle, stage 01-01) came
 * through without a stall once 1d4f335/38db67f were in.
 *
 * The per-frame sync points no longer drain the queue on the executor.
 * Each one emptied the pipeline -- the executor waited,
 * the title waited on the executor, and the workers then idled while the
 * executor walked on -- and removing any single one only moved the wait to
 * the next (P8's per-site timings). Render-to-texture hazards become
 * barrier batches, clears become ordered jobs, visibility reports are
 * written when their batches are done; the flip is the one drain left. */
static int rq_ready(void);
static int rq_queue_samples(uint32_t lo, uint32_t hi);
static int s_async = -1;

static int rq_async(void)
{
    if (s_async < 0) {
        const char *e = getenv("RECOMP_ASYNC_SYNC");
        s_async = !(e && *e == '0') && raster_pool_size() > 1 && rq_ready();
    }
    return s_async;
}

/* Order what comes next after everything queued, for a render-to-texture
 * hazard of kind `k` (rq_queue_overlaps / rq_queue_samples: 2, the batch
 * being filled is involved and is closed; 1, only batches already drawing).
 * Returns whether there was one. */
static int rq_barrier(int k)
{
    if (!k)
        return 0;
    if (!rq_async()) {
        rq_sync();
        return 1;
    }
    if (k == 2)
        rq_kick();
    s_rq[s_rq_fill]->barrier = 1;
#if defined(_WIN32)
    s_rq_bar_seq = s_pool.head;           /* the number it is kicked with */
#endif
    return 1;
}

/* A clear job into the batch being filled, in order with the triangles. */
static void rq_push_job(const XfTri *T)
{
    RasterBatch *b = s_rq[s_rq_fill];
    if (b->n == NV_RQ_TRIS) {
        rq_kick();
        b = s_rq[s_rq_fill];
    }
    b->tri[b->n++] = *T;
    if (T->miny < b->ymin) b->ymin = T->miny;
    if (T->maxy > b->ymax) b->ymax = T->maxy;
}

static void rq_report_write(uint32_t at, uint64_t stamp, uint32_t epoch)
{
    static int force = -1;
    uint8_t *mem = (uint8_t *)xbox_GetMemoryOffset();
    uint32_t zc = s_zpe_total[epoch & (NV_ZEPOCHS - 1)], zero = 0;
    int k;
    for (k = 0; k <= s_pool.n; k++)
        zc += s_pool.cnt[k].zpe[epoch & (NV_ZEPOCHS - 1)];
    if (force < 0)
        force = getenv("RECOMP_ZPASS_FORCE") != NULL;
    if (force)
        zc = 100000;
    memcpy(mem + at, &stamp, 8);
    memcpy(mem + at + 8, &zc, 4);
    MemoryBarrier();
    memcpy(mem + at + 12, &zero, 4);       /* done, last */
}

/* Colour clear of the clip rectangle as an ordered job (RECOMP_ASYNC_SYNC):
 * each band clears its own rows, after the draws queued before it and
 * before those after. */
static int rq_clear_job(uint8_t *mem, uint32_t bpp)
{
    XfTri J;
    const SwzTable *swz = surface_swizzled();
    uint32_t lo = s_gpu.color_base, hi;

    hi = lo + (swz ? swz->w * swz->h * bpp
                   : (s_gpu.clip_y + s_gpu.clip_h) * s_gpu.pitch);
    (void)rq_barrier(rq_queue_samples(lo, hi));    /* a queued draw samples it */
    (void)rq_barrier(rq_queue_relayout(lo, hi,       /* or wrote it otherwise */
                                       rq_layout(swz, s_gpu.pitch)));
    memset(&J, 0, sizeof J);
    J.job = 1;
    J.mem = mem;
    J.bpp = bpp;
    J.jbase = s_gpu.color_base;
    J.jpitch = s_gpu.pitch;
    J.jcolor = s_gpu.clear_color;
    J.jswz = swz;
    J.minx = (int)s_gpu.clip_x; J.maxx = (int)(s_gpu.clip_x + s_gpu.clip_w);
    J.miny = (int)s_gpu.clip_y; J.maxy = (int)(s_gpu.clip_y + s_gpu.clip_h);
    rq_push_job(&J);
    rq_note_surface(s_rq[s_rq_fill], lo, hi,      /* a later sample orders after */
                    rq_layout(swz, s_gpu.pitch));
    /* The job clears its rows band by band, which keeps it in order with
     * draws that use the same layout. D3D clears a swizzled render target
     * with the surface set to pitch, then draws into it swizzled (Blinx 2's
     * reflection targets: format 0x08080128 for the clear, 0x08080228 for
     * the draws). A swizzled texel's bytes lie in the pitch rows of other
     * bands, so one band could still be clearing rows another had already
     * drawn into: one-frame dark blocks in stage 01-01's sea. The surface
     * noted above with its layout catches that: a draw into it in the other
     * layout waits for the clear (rq_queue_relayout in rq_draw_state). Every
     * colour clear used to be followed by a barrier for it, about one a
     * frame; RECOMP_CLEAR_BARRIER=1 restores that. */
    {
        static int clear_bar = -1;
        if (clear_bar < 0)
            clear_bar = getenv("RECOMP_CLEAR_BARRIER") != NULL;
        if (clear_bar)
            (void)rq_barrier(2);
    }
    return 1;
}

/* Depth clear of the whole buffer as an ordered job. */
static int rq_zclear_job(float *z, float v, uint32_t what, uint8_t sv)
{
    XfTri J;
    memset(&J, 0, sizeof J);
    J.job = 2;
    J.zb = z;
    J.jz = v;
    J.jpitch = what;                     /* bit 0 depth, bit 1 stencil */
    J.jcolor = sv;
    J.minx = 0; J.maxx = NV_ZBUF_W;
    J.miny = 0; J.maxy = NV_ZBUF_H;
    rq_push_job(&J);
    return 1;
}

static void rq_check_textures(void)
{
    int st;
    for (st = 0; st < 4; st++) {
        uint32_t lo, hi;
        if (!draw_samples(st, &lo, &hi))
            continue;
        if (rq_barrier(rq_queue_overlaps(lo, hi)))
            return;
    }
}

/* The other half of render to texture: a draw that renders into memory a
 * queued draw still samples. Bands run in parallel, each through every
 * triangle of every batch in order, and different bands may be on different
 * batches, so one band could be writing the new image while another, still on
 * the earlier draw, samples rows of it -- part old texture, part new. Such a
 * draw drains the queue first. Only sampled ranges that are render targets
 * (a surface drawn into lately) are tracked: an ordinary texture is never
 * written by a draw, and tracking every one would overflow the list and
 * drain all the time. Checked before the draw's own textures are noted, so a
 * draw sampling its own target (undefined on the console too) does not
 * drain. */
#define NV_RT_SEEN 16
static struct { uint32_t lo, hi; } s_rt_seen[NV_RT_SEEN];
static int s_rt_next;

static void rt_seen_note(uint32_t lo, uint32_t hi)
{
    int i;
    hi = rq_phys(lo) + (hi - lo);
    lo = rq_phys(lo);
    for (i = 0; i < NV_RT_SEEN; i++)
        if (s_rt_seen[i].lo == lo && s_rt_seen[i].hi == hi)
            return;
    s_rt_seen[s_rt_next].lo = lo;
    s_rt_seen[s_rt_next].hi = hi;
    s_rt_next = (s_rt_next + 1) % NV_RT_SEEN;
}

static int rt_seen_overlaps(uint32_t lo, uint32_t hi)
{
    int i;
    for (i = 0; i < NV_RT_SEEN; i++)
        if (lo < s_rt_seen[i].hi && s_rt_seen[i].lo < hi)
            return 1;
    return 0;
}

static int rq_tex_overlaps(const RasterBatch *b, uint32_t lo, uint32_t hi)
{
    int i;
    if (!b->n)
        return 0;
    if (b->ntex < 0)
        return 1;
    for (i = 0; i < b->ntex; i++)
        if (lo < b->tex_hi[i] && b->tex_lo[i] < hi)
            return 1;
    return 0;
}

/* Does anything queued -- filling or still being drawn -- sample [lo, hi)? */
static int rq_queue_samples(uint32_t lo, uint32_t hi)
{
    hi = rq_phys(lo) + (hi - lo);
    lo = rq_phys(lo);
    if (s_rq[s_rq_fill] && rq_tex_overlaps(s_rq[s_rq_fill], lo, hi))
        return 2;
#if defined(_WIN32)
    {
        LONG seq;
        rq_reap();
        for (seq = rq_unordered_from(); seq < s_pool.head; seq++)
            if (!s_pool.done[seq % NV_RQ_DEPTH]
                && rq_tex_overlaps(s_rq[seq % NV_RQ_DEPTH], lo, hi))
                return 1;
    }
#endif
    return 0;
}

static void rq_note_textures(RasterBatch *b)
{
    int st, i;
    for (st = 0; st < 4 && b->ntex >= 0; st++) {
        uint32_t lo, hi;
        if (!draw_samples(st, &lo, &hi) || !rt_seen_overlaps(lo, hi))
            continue;
        for (i = 0; i < b->ntex; i++)
            if (b->tex_lo[i] == lo && b->tex_hi[i] == hi)
                break;
        if (i < b->ntex)
            continue;
        if (b->ntex == NV_RQ_TEXS) {
            b->ntex = -1;
            break;
        }
        b->tex_lo[b->ntex] = lo;
        b->tex_hi[b->ntex] = hi;
        b->ntex++;
    }
}

static int rq_ready(void)
{
    static int failed;
    int i;
    if (s_rq[0])
        return 1;
    if (failed)
        return 0;
    for (i = 0; i < NV_RQ_DEPTH; i++) {
        s_rq[i] = (RasterBatch *)calloc(1, sizeof(RasterBatch));
        if (!s_rq[i]) {
            while (i--) {
                free(s_rq[i]);
                s_rq[i] = NULL;
            }
            failed = 1;
            fprintf(stderr, "  [GPU] no memory for the raster queue;"
                            " drawing on the executor\n");
            return 0;
        }
        s_rq[i]->ymin = 1 << 30;
        /* Geometry job space; without it every draw is set up here. */
        s_rq[i]->geo = (GeoJob *)malloc(sizeof(GeoJob) * NV_RQ_GEO);
        s_rq[i]->in = (float (*)[NV2A_VSH_INPUTS][4])malloc(
            sizeof(float) * NV2A_VSH_INPUTS * 4 * NV_RQ_GVERTS);
        s_rq[i]->cst = (float (*)[NV2A_VSH_CONSTANTS][4])malloc(
            sizeof(float) * NV2A_VSH_CONSTANTS * 4 * NV_RQ_GEO);
        if (!s_rq[i]->geo || !s_rq[i]->in || !s_rq[i]->cst) {
            free(s_rq[i]->geo); free(s_rq[i]->in); free(s_rq[i]->cst);
            s_rq[i]->geo = NULL; s_rq[i]->in = NULL; s_rq[i]->cst = NULL;
        }
    }
    return 1;
}

/* A draw's state into batch b, once, at its first triangle (or when its
 * geometry is queued): the PixState snapshot, and the surface and texture
 * ranges the render-to-texture checks look at. A draw that renders into
 * something a queued draw samples drains first, so b may change. */
static RasterBatch *rq_draw_state(RasterBatch *b, uint32_t bpp)
{
    const SwzTable *swz = surface_swizzled();
    uint32_t lo = dma_resolve(s_gpu.color_offset), bytes;
    PixState *ps;

    bytes = swz ? swz->w * swz->h * bpp
                : (s_gpu.clip_y + s_gpu.clip_h) * s_gpu.pitch;
    if (rq_barrier(rq_queue_samples(lo, lo + bytes)))
        b = s_rq[s_rq_fill];
    if (rq_barrier(rq_queue_relayout(lo, lo + bytes, rq_layout(swz, s_gpu.pitch))))
        b = s_rq[s_rq_fill];
    rt_seen_note(lo, lo + bytes);
    ps = &b->st[b->nst];
    pix_capture(ps);
    if (b->nst && memcmp(ps, &b->st[b->nst - 1], sizeof *ps) == 0)
        ps = &b->st[b->nst - 1];
    else
        b->nst++;
    s_rq_ps = ps;
    rq_note_surface(b, lo, lo + bytes, rq_layout(swz, s_gpu.pitch));
    rq_note_textures(b);
    return b;
}

/* Triangle T into slot i of batch b, with its vertices copied alongside. */
static void rq_put_tri(RasterBatch *b, int i, XfTri *T, const PixState *ps)
{
    b->v[i][0] = *T->va;
    b->v[i][1] = *T->vb;
    b->v[i][2] = *T->vc;
    T->va = &b->v[i][0]; T->a = b->v[i][0].pos;
    T->vb = &b->v[i][1]; T->b = b->v[i][1].pos;
    T->vc = &b->v[i][2]; T->c = b->v[i][2].pos;
    T->ps = ps;
    b->tri[i] = *T;
}

/* Queue a set-up triangle with this draw call's state. */
static void xfq_push(XfTri *T)
{
    RasterBatch *b;
    int i;

    if (!rq_ready()) {
        XfCount c = {0, 0, 0, 0, 0, 0};
        pix_capture(&s_live);
        T->ps = &s_live;
        xf_rows(T, T->miny, 1, &c);
        rq_count(&c);
        return;
    }
    /* Before taking pointers into the batch: a swizzle-table eviction syncs,
     * and a sync swaps the batches. */
    if (!s_rq_ps)
        (void)surface_swizzled();
    b = s_rq[s_rq_fill];
    if (b->n == NV_RQ_TRIS || (!s_rq_ps && b->nst == NV_RQ_STATES)) {
        rq_kick();
        b = s_rq[s_rq_fill];
    }
    if (!s_rq_ps)
        b = rq_draw_state(b, T->bpp);
    i = b->n++;
    rq_put_tri(b, i, T, s_rq_ps);
    b->px += (long)(T->maxx - T->minx) * (T->maxy - T->miny);
    if (T->miny < b->ymin) b->ymin = T->miny;
    if (T->maxy > b->ymax) b->ymax = T->maxy;
}

/* A geometry job's triangle into the next of its reserved slots. */
static void geo_slot_sink(void *sink, XfTri *T)
{
    GeoSlots *S = (GeoSlots *)sink;
    if (S->next >= S->end)
        return;                       /* cannot happen: 2 slots a triangle */
    rq_put_tri(S->b, (int)S->next++, T, S->ps);
    S->px += (long)(T->maxx - T->minx) * (T->maxy - T->miny);
    if (T->miny < S->ymin) S->ymin = T->miny;
    if (T->maxy > S->ymax) S->ymax = T->maxy;
}

/* Queue a draw's geometry for the pool (see GeoJob): fetch its inputs and
 * snapshot its constants and set-up state now, reserve its triangle slots,
 * and let a worker do the rest. Returns 0 if the draw has to be set up
 * here instead. */
static int geo_queue(const Nv2aVshProg *prog, uint32_t n)
{
    GeoCtx g;
    RasterBatch *b;
    GeoJob *J;
    uint32_t tris = geo_count(s_gpu.prim, n), cap = 2 * tris, i, a;

    if (cap > NV_RQ_TRIS || n > NV_RQ_GVERTS)
        return 0;
    geo_capture(&g);
    if (!tris || !g.ok)
        return 1;                     /* draws nothing either way */
    (void)surface_swizzled();
    b = s_rq[s_rq_fill];
    if (b->n + cap > NV_RQ_TRIS || b->nst == NV_RQ_STATES
        || b->ngeo == NV_RQ_GEO || b->nin + n > NV_RQ_GVERTS) {
        rq_kick();
        b = s_rq[s_rq_fill];
    }
    b = rq_draw_state(b, g.bpp);
    {
        /* Only the inputs the program reads: a typical one reads three to
         * five of the sixteen, and fetching all of them for every vertex
         * was the executor's largest single cost per draw (fetch_attr).
         * The rest are never read (nv2a_vsh_prog_inputs). */
        uint32_t used = nv2a_vsh_prog_inputs(prog);
        for (i = 0; i < n; i++)
            for (a = 0; a < NV2A_VSH_INPUTS; a++)
                if (used >> a & 1)
                    fetch_input(a, s_gpu.idx[i], b->in[b->nin + i][a]);
    }
    memcpy(b->cst[b->ngeo], nv2a_vsh_constants(), sizeof b->cst[0]);
    J = &b->geo[b->ngeo];
    J->prog = prog;
    J->in0 = b->nin;
    J->n = n;
    J->prim = s_gpu.prim;
    J->tri0 = (uint32_t)b->n;
    J->cap = cap;
    J->cst = (uint32_t)b->ngeo;
    J->g = g;
    J->ps = s_rq_ps;
    J->ymin = 1 << 30;
    J->ymax = 0;
    J->px = 0;
    b->ngeo++;
    b->nin += n;
    b->n += (int)cap;
    note_drawn();
    s_gpu.xf_parallel_verts += n;
    return 1;
}

/* A draw call starts: its state is captured afresh by its first triangle. */
static void xfq_begin(void)
{
    if (s_rq_each < 0) {
        const char *e = getenv("RECOMP_RASTER_BATCH");
        s_rq_each = e && *e == '0';
    }
    rq_check_textures();
    s_rq_ps = NULL;
}

/* A frame trace counts each draw's pixels, so it drains per draw. */
static void xfq_end(void)
{
    if (s_rq_each || s_ftrace == 2)
        rq_sync();
}

/* Back-face culling, on the screen-space signed area.
 *
 * Winding is judged against the viewport's own y direction. D3D's usual
 * viewport maps NDC y up to screen y down (c[58].y < 0); a clockwise
 * triangle on screen then has area > 0 and FRONT_FACE 0x900 (CW) means
 * exactly that. A viewport with c[58].y > 0 -- Blinx 2's render-to-texture
 * passes into its 256x256 targets -- mirrors the image vertically and with it
 * every triangle's on-screen winding, so the test inverts. Judged on screen
 * alone, those passes lost their visible faces and kept the hidden ones, and
 * the main pass, which samples them, tinted the whole attract stage magenta.
 * The title also switches CULL_FACE between FRONT and BACK per object (both
 * with FRONT_FACE CW), which is why neither fixed sense was right everywhere.
 * RECOMP_NO_CULL=1 turns culling off for A/B comparisons. */
static int geo_cull(const GeoCtx *g, GeoStats *st, float area)
{
    int cw = area > 0.0f, front;

    if (!g->cull_enable || g->no_cull)
        return 0;
    if (g->k[1] > 0.0f)
        cw = !cw;                               /* y-up viewport */
    front = g->front_face == 0x901 ? !cw : cw;
    switch (g->cull_face) {
    case 0x404: if (!front) return 0; break;      /* FRONT */
    case 0x408: break;                            /* FRONT_AND_BACK */
    default:    if (front) return 0; break;       /* BACK */
    }
    st->culled++;
    return 1;
}

/* Where a set-up triangle goes (GeoSink): the raster queue on the executor
 * (xfq_push), or a geometry job's slots in its batch on the pool. */

/* The three edge functions as planes, set up in double and handed to the
 * pixel loops relative to the triangle's first pixel.
 *
 * They were evaluated per pixel in float as (b0 - a0) * (py - a1) -
 * (b1 - a1) * (px - a0). Near-plane clipping (NV_CLIP_W) puts vertices
 * millions of pixels off screen -- Blinx 2's ground near the camera reaches
 * x = -1.9e7 -- and there each product is ~1e14, so float rounding is wrong
 * by many pixels: pixels along an edge two triangles share were inside
 * neither. That was the stair-stepped black lines and the scattered short
 * dashes across the ground in the first mission.
 *
 * Here each edge is A*px + B*py + C with C = a0*b1 - a1*b0, computed in
 * double; the edge's other triangle computes exactly the negation. The
 * plane is then evaluated once, at the centre of pixel (minx, miny), and the
 * pixel loops only add A*dx + B*dy for on-screen dx, dy. Rounding now is
 * relative to a few hundred pixels' worth of edge, ~1e-4 pixel. The area
 * (for the barycentrics) is done in double for the same reason. */
static void geo_edges(XfTri *T, const float *a, const float *b, const float *c)
{
    const float *p[3], *q[3];
    double ox = (double)T->minx + 0.5, oy = (double)T->miny + 0.5, area;
    int i;

    p[0] = a; q[0] = b;                          /* w0: a -> b */
    p[1] = b; q[1] = c;                          /* w1: b -> c */
    p[2] = c; q[2] = a;                          /* w2: c -> a */
    for (i = 0; i < 3; i++) {
        double A = (double)p[i][1] - (double)q[i][1];
        double B = (double)q[i][0] - (double)p[i][0];
        double C = (double)p[i][0] * q[i][1] - (double)p[i][1] * q[i][0];
        T->ea[i] = (float)A;
        T->eb[i] = (float)B;
        T->ec[i] = (float)(A * ox + B * oy + C);
    }
    area = ((double)b[0] - a[0]) * ((double)c[1] - a[1])
         - ((double)b[1] - a[1]) * ((double)c[0] - a[0]);
    T->inv_area = (float)(1.0 / area);
}

static void geo_triangle(const GeoCtx *g, GeoStats *st, GeoSink put, void *sink,
                         const Nv2aVshOutput *va, const Nv2aVshOutput *vb,
                         const Nv2aVshOutput *vc)
{
    XfTri T;
    const Nv2aVshOutput *v[3];
    const float *a, *b, *c;
    float area, su = g->su, sv = g->sv;
    int k;

    memset(&T, 0, sizeof T);
    T.mem = g->mem;
    T.bpp = g->bpp;
    T.textured = g->textured;
    T.use_rc = g->use_rc;
    T.va = va; T.vb = vb; T.vc = vc;

    v[0] = va; v[1] = vb; v[2] = vc;
    a = va->pos; b = vb->pos; c = vc->pos;
    T.a = a; T.b = b; T.c = c;
    if (!g->ok)
        return;
    if (a[3] <= 0.0f || b[3] <= 0.0f || c[3] <= 0.0f) {
        st->behind++;
        return;
    }
    area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    if (area == 0.0f || area != area) {
        st->degenerate++;
        return;
    }
    if (geo_cull(g, st, area))
        return;
    for (k = 0; k < 3; k++) {
        if (!st->seeded)
            st->xmin[k] = st->xmax[k] = a[k];
        if (a[k] < st->xmin[k]) st->xmin[k] = a[k];
        if (a[k] > st->xmax[k]) st->xmax[k] = a[k];
    }
    st->seeded = 1;
    T.zb = g->zb;
    for (k = 0; k < 3; k++) {
        T.iw[k] = 1.0f / v[k]->pos[3];
        T.uv[k][0] = v[k]->tex[0][0] * su * T.iw[k];
        T.uv[k][1] = v[k]->tex[0][1] * sv * T.iw[k];
    }
    if (T.use_rc) {
        int sti, j;
        for (k = 0; k < 3; k++) {
            for (sti = 0; sti < 4; sti++)
                for (j = 0; j < 4; j++)
                    T.stc[k][sti][j] = v[k]->tex[sti][j] * T.iw[k];
            T.vfog[k] = geo_fog(g, v[k]->fog[0]);
        }
        memcpy(T.fogc, g->fogc, sizeof T.fogc);
    }

    T.minx = (int)floorf(fminf(a[0], fminf(b[0], c[0])));
    T.maxx = (int)ceilf (fmaxf(a[0], fmaxf(b[0], c[0])));
    T.miny = (int)floorf(fminf(a[1], fminf(b[1], c[1])));
    T.maxy = (int)ceilf (fmaxf(a[1], fmaxf(b[1], c[1])));
    if (T.minx < g->clip_x) T.minx = g->clip_x;
    if (T.miny < g->clip_y) T.miny = g->clip_y;
    if (T.maxx > g->clip_x + g->clip_w) T.maxx = g->clip_x + g->clip_w;
    if (T.maxy > g->clip_y + g->clip_h) T.maxy = g->clip_y + g->clip_h;
    if (T.maxx > NV_ZBUF_W) T.maxx = NV_ZBUF_W;
    if (T.maxy > NV_ZBUF_H) T.maxy = NV_ZBUF_H;
    if (T.minx >= T.maxx || T.miny >= T.maxy) {
        st->offscreen++;
        return;
    }
    geo_edges(&T, a, b, c);
    st->drawn++;
    put(sink, &T);
}

/* Near-plane clipping.
 *
 * A triangle with a vertex behind the eye (w <= 0) used to be dropped whole.
 * That is invisible for distant geometry and ruinous close up: in a chase
 * view the road under the camera is exactly the set of triangles that reach
 * behind it, so the bottom of the frame showed a hole -- a quarter of every
 * race frame's triangles went this way.
 *
 * Clipping has to happen in clip space, but the D3D epilogue has already
 * divided x, y and z by w and applied the viewport (c[58] scale, c[59]
 * offset) while leaving w itself alone. So each vertex is taken back to clip
 * space first: clip = (screen - offset) / scale * w. Everything else the
 * vertex carries -- colours, fog, texture coordinates -- is linear in clip
 * space and is interpolated directly. The polygon is clipped against
 * w = NV_CLIP_W (Sutherland-Hodgman, one plane), re-projected, and fanned.
 * ponytail: the near plane only; x/y/far fall to the rasteriser's bounds
 * check, which is correct, just slower for huge off-screen triangles. */
#define NV_CLIP_W 1e-3f

static void xf_to_clip(const Nv2aVshOutput *v, const float k[3],
                       const float o[3], float c[3])
{
    int i;
    for (i = 0; i < 3; i++)
        c[i] = k[i] != 0.0f ? (v->pos[i] - o[i]) / k[i] * v->pos[3] : 0.0f;
}

static void xf_lerp(const Nv2aVshOutput *a, const Nv2aVshOutput *b, float t,
                    Nv2aVshOutput *out)
{
    const float *pa = (const float *)a, *pb = (const float *)b;
    float *po = (float *)out;
    size_t i, n = sizeof(Nv2aVshOutput) / sizeof(float);
    for (i = 0; i < n; i++)
        po[i] = pa[i] + (pb[i] - pa[i]) * t;
}

/* A polygon vertex while clipping: the vertex, its clip-space position, and
 * whether it is an original vertex (drawn as it came, see below). */
typedef struct {
    Nv2aVshOutput v;
    float c[4];
    int keep;
} ClipVert;

/* Clip-space distance of v to plane `pl`: 0 the near plane (z >= 0, D3D's
 * clip space), 1 the w > NV_CLIP_W guard. */
static float clip_dist(const ClipVert *v, int pl)
{
    return pl == 0 ? v->c[2] : v->c[3] - NV_CLIP_W;
}

/* One Sutherland-Hodgman pass of polygon in[0..n) against plane pl. */
static int clip_poly(const ClipVert *in, int n, int pl, ClipVert *out)
{
    int i, m = 0;
    for (i = 0; i < n; i++) {
        const ClipVert *p = &in[i], *q = &in[(i + 1) % n];
        float dp = clip_dist(p, pl), dq = clip_dist(q, pl);
        if (dp >= 0.0f)
            out[m++] = *p;
        if ((dp >= 0.0f) != (dq >= 0.0f)) {
            float t = dp / (dp - dq);
            Nv2aVshOutput va = p->v, vb = q->v;
            int j;
            for (j = 0; j < 3; j++) { va.pos[j] = p->c[j]; vb.pos[j] = q->c[j]; }
            va.pos[3] = p->c[3]; vb.pos[3] = q->c[3];
            xf_lerp(&va, &vb, t, &out[m].v);
            for (j = 0; j < 4; j++)
                out[m].c[j] = p->c[j] + (q->c[j] - p->c[j]) * t;
            out[m].keep = 0;
            m++;
        }
    }
    return m;
}

/* Near-plane clipping, against the real near plane.
 *
 * This used to clip at w = NV_CLIP_W (1e-3) only. A triangle that reaches
 * behind the camera -- Blinx 2's sea, a plane around the whole island --
 * then got a vertex at w = 0.001: millions of pixels off screen, with a
 * screen depth around -2e10. The rest of the triangle's depth was
 * interpolated from that in float, and came out far too near over most of
 * the open sea. The bump-mapped water surface drawn over it then failed
 * its LESS depth test there, and that part of the sea stayed the dark base
 * pass instead of turquoise.
 *
 * D3D's clip space puts the near plane at z = 0 (0 <= z <= w), and that is
 * where the console clips. The vertex program's output is already in screen
 * space (divided by w, viewport applied, w kept), so each vertex is taken
 * back: clip = (screen - offset) / scale * w, which holds for any w != 0. The
 * polygon is clipped against z >= 0, then against w > NV_CLIP_W as a guard
 * for vertices at the eye, re-projected and fanned. An original vertex
 * inside both planes is drawn exactly as it came: a round trip moves it by
 * a rounding error, its neighbours drawn unclipped do not move, and the
 * shared edge would open into a crack. */
static void geo_clipped(const GeoCtx *g, GeoStats *st, GeoSink put, void *sink,
                        const Nv2aVshOutput *a, const Nv2aVshOutput *b,
                        const Nv2aVshOutput *c)
{
    const Nv2aVshOutput *in[3];
    ClipVert p0[3], p1[5], p2[6];
    float k[3], o[3];
    int n, i, j, inside = 1, outside0 = 1, outside1 = 1;

    for (i = 0; i < 3; i++) { k[i] = g->k[i]; o[i] = g->o[i]; }
    in[0] = a; in[1] = b; in[2] = c;
    for (i = 0; i < 3; i++) {
        float w = in[i]->pos[3];
        p0[i].v = *in[i];
        p0[i].keep = 1;
        xf_to_clip(in[i], k, o, p0[i].c);
        p0[i].c[3] = w;
        if (k[2] == 0.0f || w != w)
            p0[i].c[2] = w;               /* no depth scale: w alone decides */
        if (p0[i].c[2] != p0[i].c[2])
            p0[i].c[2] = -1.0f;           /* NaN: outside */
        if (!(clip_dist(&p0[i], 0) >= 0.0f && clip_dist(&p0[i], 1) >= 0.0f))
            inside = 0;
        if (clip_dist(&p0[i], 0) >= 0.0f) outside0 = 0;
        if (clip_dist(&p0[i], 1) >= 0.0f) outside1 = 0;
    }
    if (inside) {
        geo_triangle(g, st, put, sink, a, b, c);
        return;
    }
    if (outside0 || outside1) {
        st->behind++;                         /* wholly behind: nothing */
        return;
    }
    n = clip_poly(p0, 3, 0, p1);
    n = clip_poly(p1, n, 1, p2);
    /* Back to screen space: divide by the (now positive) w, viewport. */
    for (i = 0; i < n; i++)
        if (!p2[i].keep)
            for (j = 0; j < 3; j++)
                p2[i].v.pos[j] = p2[i].c[j] / p2[i].c[3] * k[j] + o[j];
    for (i = 1; i + 1 < n; i++)
        geo_triangle(g, st, put, sink, &p2[0].v, &p2[i].v, &p2[i + 1].v);
}

/* The live state, for a draw about to be set up. Run on the executor. */
static void geo_capture(GeoCtx *g)
{
    static int no_rc = -1, no_cull = -1;
    const float *sc = nv2a_vsh_constant(58), *of = nv2a_vsh_constant(59);
    int i;

    if (no_rc < 0)
        no_rc = getenv("RECOMP_NO_COMBINERS") != NULL;
    if (no_cull < 0)
        no_cull = getenv("RECOMP_NO_CULL") != NULL;
    memset(g, 0, sizeof *g);
    g->mem = (uint8_t *)xbox_GetMemoryOffset();
    g->bpp = surface_bpp();
    g->textured = s_gpu.texs[0].valid;
    g->use_rc = s_gpu.rc_seen && !no_rc;
    g->ok = (g->bpp == 4 || g->bpp == 2) && surface_begin_batch(g->mem);
    g->su = g->sv = 1.0f;
    if (tex_size_from_format(s_gpu.texs[0].color)) {
        g->su = (float)s_gpu.texs[0].width;
        g->sv = (float)s_gpu.texs[0].height;
    }
    if (g->ok && (s_gpu.depth_test || s_gpu.depth_mask || s_gpu.st_test))
        g->zb = zbuf_current(1);
    g->clip_x = (int)s_gpu.clip_x; g->clip_y = (int)s_gpu.clip_y;
    g->clip_w = (int)s_gpu.clip_w; g->clip_h = (int)s_gpu.clip_h;
    g->cull_enable = s_gpu.cull_enable != 0;
    g->front_face = (int)s_gpu.front_face;
    g->cull_face = (int)s_gpu.cull_face;
    g->no_cull = no_cull;
    for (i = 0; i < 3; i++) { g->k[i] = sc[i]; g->o[i] = of[i]; }
    g->fog_enable = s_gpu.fog_enable;
    g->fog_mode = s_gpu.fog_mode;
    g->fog_param[0] = s_gpu.fog_param[0];
    g->fog_param[1] = s_gpu.fog_param[1];
    /* FOG_COLOR is R in bits 0-7, the reverse of a D3DCOLOR. */
    nv2a_rc_unpack(s_gpu.fog_color, g->fogc);
    { float r = g->fogc[2]; g->fogc[2] = g->fogc[0]; g->fogc[0] = r; }
}

/* Set-up's tallies into s_gpu. Run on the executor. */
static void geo_fold(const GeoStats *st)
{
    int k;
    s_gpu.tris_behind += st->behind;
    s_gpu.xf_degenerate += st->degenerate;
    s_gpu.tris_culled += st->culled;
    s_gpu.tris_skipped_offscreen += st->offscreen;
    s_gpu.xf_offscreen += st->offscreen;
    s_gpu.xf_drawn += st->drawn;
    s_gpu.tris_drawn += st->drawn;
    if (st->seeded) {
        for (k = 0; k < 3; k++) {
            if (!s_gpu.xf_seeded)
                s_gpu.xf_min[k] = s_gpu.xf_max[k] = st->xmin[k];
            if (st->xmin[k] < s_gpu.xf_min[k]) s_gpu.xf_min[k] = st->xmin[k];
            if (st->xmax[k] > s_gpu.xf_max[k]) s_gpu.xf_max[k] = st->xmax[k];
        }
        s_gpu.xf_seeded = 1;
    }
}

/* The triangles a primitive of n vertices makes, before clipping. */
static uint32_t geo_count(uint32_t prim, uint32_t n)
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

/* A transformed primitive into triangles, clipped and set up, in order. */
static void geo_assemble(const GeoCtx *g, GeoStats *st, GeoSink put, void *sink,
                         uint32_t prim, const Nv2aVshOutput *x, uint32_t n)
{
    uint32_t i;
    switch (prim) {
    case NV_PRIM_TRIANGLES:
        for (i = 0; i + 2 < n; i += 3)
            geo_clipped(g, st, put, sink, &x[i], &x[i+1], &x[i+2]);
        break;
    case NV_PRIM_TRIANGLE_STRIP:
        /* Every other strip triangle is listed in the opposite order; swap
         * it back so all of them share the strip's winding (culling). */
        for (i = 0; i + 2 < n; i++)
            if (i & 1)
                geo_clipped(g, st, put, sink, &x[i+1], &x[i], &x[i+2]);
            else
                geo_clipped(g, st, put, sink, &x[i], &x[i+1], &x[i+2]);
        break;
    case NV_PRIM_TRIANGLE_FAN:
    case NV_PRIM_POLYGON:
        for (i = 1; i + 1 < n; i++)
            geo_clipped(g, st, put, sink, &x[0], &x[i], &x[i+1]);
        break;
    case NV_PRIM_QUADS:
        for (i = 0; i + 3 < n; i += 4) {
            geo_clipped(g, st, put, sink, &x[i], &x[i+1], &x[i+2]);
            geo_clipped(g, st, put, sink, &x[i], &x[i+2], &x[i+3]);
        }
        break;
    case NV_PRIM_QUAD_STRIP:
        for (i = 0; i + 3 < n; i += 2) {
            geo_clipped(g, st, put, sink, &x[i], &x[i+1], &x[i+3]);
            geo_clipped(g, st, put, sink, &x[i], &x[i+3], &x[i+2]);
        }
        break;
    default:
        break;
    }
}

static void xfq_sink(void *unused, XfTri *T)
{
    (void)unused;
    xfq_push(T);
}

static Nv2aVshOutput s_xf[NV_MAX_INDICES];

static int transform_vertex(uint32_t index, Nv2aVshOutput *out)
{
    float in[NV2A_VSH_INPUTS][4];
    uint32_t a;

    for (a = 0; a < NV2A_VSH_INPUTS; a++)
        fetch_attr(&s_gpu.attr[a], index, in[a]);   /* absent: 0,0,0,1 */
    return nv2a_vsh_run((const float (*)[4])in, out);
}

static void raster_batch_program(void)
{
    uint32_t i, n = s_gpu.idx_count;
    /* Geometry on the raster pool (GeoJob) unless something needs it here:
     * RECOMP_SERIAL_XF=1, a vertex trace or timing (they read this loop),
     * CXT_WRITE
     * (vertices then depend on each other), no pool, or a program that
     * could not be kept. */
    static int serial_xf = -1, vtrace = -1;
    const Nv2aVshProg *prog = NULL;
    int deferred;
    if (serial_xf < 0) {
        serial_xf = getenv("RECOMP_SERIAL_XF") != NULL;
        vtrace = getenv("RECOMP_VSH_TRACE") != NULL
                 || getenv("RECOMP_VSH_TIME") != NULL;    /* times this loop */
    }
    deferred = !serial_xf && !vtrace && raster_pool_size() > 1 && rq_ready()
               && s_rq[0]->geo && !nv2a_vsh_cxt_write()
               && (prog = nv2a_vsh_snapshot()) != NULL;

    /* RECOMP_VSH_TIME=1: fetch + vertex-program time per vertex, every 10 s,
     * for A/B runs of the interpreter. */
#if defined(_WIN32)
    static int vtime = -1;
    static uint64_t fticks, batches;
    LARGE_INTEGER vt0, vf0, vf1;
    if (vtime < 0)
        vtime = getenv("RECOMP_VSH_TIME") != NULL;
    if (vtime)
        QueryPerformanceCounter(&vt0);
#endif
    /* Four at a time (nv2a_vsh_run4, bit-identical to one at a time): the
     * vertex program was a third of the executor in Blinx 2's hub. A short
     * tail repeats its last vertex to fill the lanes. */
    for (i = 0; !deferred && i < n; i += 4) {
        static float in[4][NV2A_VSH_INPUTS][4];
        static Nv2aVshOutput tail[4];
        const float (*ip[4])[4];
        Nv2aVshOutput *op[4];
        uint32_t l, a;
#if defined(_WIN32)
        if (vtime)
            QueryPerformanceCounter(&vf0);
#endif
        for (l = 0; l < 4; l++) {
            uint32_t j = i + l < n ? i + l : n - 1;
            for (a = 0; a < NV2A_VSH_INPUTS; a++)
                fetch_input(a, s_gpu.idx[j], in[l][a]);
            ip[l] = (const float (*)[4])in[l];
            op[l] = i + l < n ? &s_xf[i + l] : &tail[l];
        }
#if defined(_WIN32)
        if (vtime) {
            QueryPerformanceCounter(&vf1);
            fticks += (uint64_t)(vf1.QuadPart - vf0.QuadPart);
        }
#endif
        if (!nv2a_vsh_run4(ip, op))
            return;                                    /* no program loaded */
    }
#if defined(_WIN32)
    if (vtime) {
        static uint64_t ticks, verts;
        static DWORD last;
        LARGE_INTEGER vt1, f;
        QueryPerformanceCounter(&vt1);
        ticks += (uint64_t)(vt1.QuadPart - vt0.QuadPart);
        verts += n;
        batches++;
        if (GetTickCount() - last > 10000 && verts) {
            QueryPerformanceFrequency(&f);
            extern unsigned long nv2a_vsh_decodes;
            fprintf(stderr, "[VSHTIME] %.1f ns/vertex over %llu vertices (%.0f ms),"
                    " fetch %.0f ms, %llu batches, %lu decodes%c",
                    1e9 * (double)ticks / (double)f.QuadPart / (double)verts,
                    (unsigned long long)verts, 1e3 * (double)ticks / (double)f.QuadPart,
                    1e3 * (double)fticks / (double)f.QuadPart,
                    (unsigned long long)batches, nv2a_vsh_decodes, 10);
            nv2a_vsh_decodes = 0;
            ticks = verts = fticks = batches = 0;
            last = GetTickCount();
        }
    }
#endif
    s_gpu.batches_program++;
    s_gpu.verts_program += n;
    if (s_gpu.blend_enable) {
        uint32_t pair = s_gpu.blend_sfactor << 16 | (s_gpu.blend_dfactor & 0xFFFF);
        int j;
        for (j = 0; j < s_gpu.blend_npairs && s_gpu.blend_pairs[j] != pair; j++)
            ;
        if (j == s_gpu.blend_npairs && j < 16)
            s_gpu.blend_pairs[s_gpu.blend_npairs++] = pair;
    }
    {
        /* RECOMP_VSH_TRACE=<n>: print n program batches -- attribute setup,
         * raw inputs, key constants, and what came out. */
        static int left = -1;
        if (left < 0) {
            const char *t = getenv("RECOMP_VSH_TRACE");
            left = t ? atoi(t) : 0;
        }
        /* Spread out: one every 20000 program batches, so a run that spends
         * its first half in menus still traces the 3D scene. */
        if (left > 0 && n >= 3 && s_gpu.batches_program % 20000 == 0) {
            float in[4];
            left--;
            fprintf(stderr, "[VTRACE] prim %u n %u idx %u %u %u\n", s_gpu.prim,
                    n, s_gpu.idx[0], s_gpu.idx[1], s_gpu.idx[2]);
            for (i = 0; i < NV_VERTEX_ATTRS; i++) {
                const VertexAttr *at = &s_gpu.attr[i];
                if (!at->size)
                    continue;
                fetch_attr(at, s_gpu.idx[0], in);
                fprintf(stderr, "[VTRACE]   v%u off %08X type %u size %u"
                        " stride %u = %g %g %g %g\n", i, at->offset, at->type,
                        at->size, at->stride, in[0], in[1], in[2], in[3]);
            }
            {
                static const uint32_t cs[] = {58, 59, 96, 97, 112, 113, 114, 115};
                for (i = 0; i < 8; i++) {
                    const float *c = nv2a_vsh_constant(cs[i]);
                    fprintf(stderr, "[VTRACE]   c[%u] %g %g %g %g%c", cs[i],
                            c[0], c[1], c[2], c[3], 10);
                }
            }
            for (i = 0; i < 3; i++)
                fprintf(stderr, "[VTRACE]   out%u pos %g %g %g %g\n", i,
                        s_xf[i].pos[0], s_xf[i].pos[1], s_xf[i].pos[2],
                        s_xf[i].pos[3]);
        }
    }
    for (i = 0; i < NV2A_VSH_INPUTS; i++) {
        const VertexAttr *at = &s_gpu.attr[i];
        uint32_t f = at->type | (at->size << 4) | (i << 8), j;
        if (!at->size || !at->stride)
            continue;
        for (j = 0; j < (uint32_t)s_gpu.xf_nfmt && s_gpu.xf_fmt[j] != f; j++)
            ;
        if (j == (uint32_t)s_gpu.xf_nfmt && s_gpu.xf_nfmt < 16)
            s_gpu.xf_fmt[s_gpu.xf_nfmt++] = f;
    }
    {
        int st, any = 0;
        for (st = 0; st < 4; st++)
            any |= s_gpu.texs[st].valid;
        if (any)
            s_gpu.program_textured++;
        else
            s_gpu.program_no_tex++;
    }
    if (s_gpu.texs[0].valid)
        note_texture_use();

    xfq_begin();
    if (!deferred || !geo_queue(prog, n)) {
        GeoCtx g;
        GeoStats st;
        if (deferred) {                 /* too big for a batch: transform here */
            for (i = 0; i < n; i += 4) {
                static float in[4][NV2A_VSH_INPUTS][4];
                static Nv2aVshOutput tail[4];
                const float (*ip[4])[4];
                Nv2aVshOutput *op[4];
                uint32_t l, a;
                for (l = 0; l < 4; l++) {
                    uint32_t j = i + l < n ? i + l : n - 1;
                    for (a = 0; a < NV2A_VSH_INPUTS; a++)
                        fetch_input(a, s_gpu.idx[j], in[l][a]);
                    ip[l] = (const float (*)[4])in[l];
                    op[l] = i + l < n ? &s_xf[i + l] : &tail[l];
                }
                if (!nv2a_vsh_run4(ip, op))
                    return;
            }
        }
        memset(&st, 0, sizeof st);
        geo_capture(&g);
        geo_assemble(&g, &st, xfq_sink, NULL, s_gpu.prim, s_xf, n);
        geo_fold(&st);
        if (st.drawn)
            note_drawn();
    }
    xfq_end();
}

/* ── GPU backend (nv2a_backend.h) ─────────────────────────────────────────
 *
 * RECOMP_GPU=vulkan hands draws, clears, visibility reports and flips to a
 * backend instead of the software rasteriser. Everything else here --
 * decoding, fences, the GPU position, NOTIFY, flips counted, the vblank --
 * stays as it is, so the guest sees the same GPU either way. Unset (or
 * "soft"), nothing below runs and the software path is the reference. */
static const Nv2aBackend *s_backend;
/* RECOMP_VK_COMPARE=1 (with RECOMP_GPU=vulkan): both renderers. The
 * software path stays in charge of everything the guest sees -- its pixels,
 * its visibility counts -- and the backend draws the same stream alongside;
 * at every flip the backend's frame is diffed against the software one. The
 * same frames by construction, which is what a parity check needs. */
static int s_backend_compare;

static void backend_select(void)
{
    const char *g = getenv("RECOMP_GPU");
    if (!g || !*g || !strcmp(g, "soft"))
        return;
#if defined(XBOXRECOMP_HAVE_VULKAN)
    if (!strcmp(g, "vulkan"))
        s_backend = xbox_vk_backend();
#endif
    s_backend_compare = s_backend && getenv("RECOMP_VK_COMPARE") != NULL;
    fprintf(stderr, "  [GPU] RECOMP_GPU=%s: %s%s\n", g,
            s_backend ? s_backend->name : "not available, rendering in software",
            s_backend_compare ? " (compared against software, which renders)" : "");
}

static void tex_to_backend(const Texture *t, Nv2aTexState *o)
{
    memset(o, 0, sizeof *o);
    o->offset = t->offset;
    o->width = t->width;
    o->height = t->height;
    o->pitch = t->pitch;
    o->color = t->color;
    o->addr_u = t->addr_u;
    o->addr_v = t->addr_v;
    o->palette = t->palette;
    o->levels = t->levels;
    o->filter = t->filter;
    o->cube = t->cube;
    o->valid = t->valid;
    o->normalized = tex_size_from_format(t->color);
    o->face_stride = t->valid ? tex_face_stride(t) : 0;
    memcpy(o->bump_mat, t->bump_mat, sizeof o->bump_mat);
    o->bump_scale = t->bump_scale;
    o->bump_offset = t->bump_offset;
}

int nv2a_exec_texel(const Nv2aTexState *t, uint32_t face_offset,
                    uint32_t u, uint32_t v, uint32_t *argb)
{
    Texture tx;
    memset(&tx, 0, sizeof tx);
    tx.offset = t->offset;
    tx.width = t->width;
    tx.height = t->height;
    tx.pitch = t->pitch;
    tx.color = t->color;
    tx.addr_u = t->addr_u;
    tx.addr_v = t->addr_v;
    tx.palette = t->palette;
    tx.levels = t->levels;
    tx.filter = t->filter;
    tx.cube = t->cube;
    tx.valid = t->valid;
    tx.swz = tex_swizzle_table(&tx);
    return sample_tex(&tx, face_offset, u, v, argb);
}

static void backend_capture(Nv2aDrawState *d)
{
    const SwzTable *swz = surface_swizzled();
    const float *sc = nv2a_vsh_constant(58), *of = nv2a_vsh_constant(59);
    int i;

    memset(d, 0, sizeof *d);
    d->color_addr = s_gpu.color_offset ? dma_resolve(s_gpu.color_offset) : 0;
    d->surface_format = s_gpu.format;
    d->pitch = s_gpu.pitch;
    d->bpp = surface_bpp();
    if (swz) {
        d->swz_w = swz->w;
        d->swz_h = swz->h;
    }
    d->clip_x = s_gpu.clip_x; d->clip_y = s_gpu.clip_y;
    d->clip_w = s_gpu.clip_w; d->clip_h = s_gpu.clip_h;
    /* The software path keys its depth buffers by the raw offset. */
    d->zeta_addr = s_gpu.zeta_offset;
    d->zeta_format = (s_gpu.format >> 4) & 0xF;
    d->zstencil_clear = s_gpu.zstencil_clear;
    d->clear_color = s_gpu.clear_color;
    d->color_mask = s_gpu.color_mask;
    d->alpha_test = s_gpu.alpha_test;
    d->alpha_func = s_gpu.alpha_func;
    d->alpha_ref = s_gpu.alpha_ref;
    d->blend_enable = s_gpu.blend_enable;
    d->blend_sfactor = s_gpu.blend_sfactor;
    d->blend_dfactor = s_gpu.blend_dfactor;
    d->blend_equation = s_gpu.blend_equation;
    d->blend_color = s_gpu.blend_color;
    d->depth_test = s_gpu.depth_test;
    d->depth_func = s_gpu.depth_func;
    d->depth_mask = s_gpu.depth_mask;
    d->st_test = s_gpu.st_test;
    d->st_func = s_gpu.st_func;
    d->st_ref = s_gpu.st_ref;
    d->st_fmask = s_gpu.st_fmask;
    d->st_wmask = s_gpu.st_wmask;
    d->st_fail = s_gpu.st_fail;
    d->st_zfail = s_gpu.st_zfail;
    d->st_zpass = s_gpu.st_zpass;
    d->cull_enable = s_gpu.cull_enable;
    d->cull_face = s_gpu.cull_face;
    d->front_face = s_gpu.front_face;
    for (i = 0; i < 3; i++) {
        d->vp_scale[i] = sc[i];
        d->vp_offset[i] = of[i];
    }
    d->fog_enable = s_gpu.fog_enable;
    d->fog_mode = s_gpu.fog_mode;
    d->fog_color = s_gpu.fog_color;
    d->fog_param[0] = s_gpu.fog_param[0];
    d->fog_param[1] = s_gpu.fog_param[1];
    d->rc = s_gpu.rc;
    d->rc_seen = s_gpu.rc_seen;
    d->clip_plane_mode = s_gpu.clip_plane_mode;
    d->other_stage_input = s_gpu.other_stage_input;
    d->zpass_enable = s_gpu.zpass_enable;
    for (i = 0; i < 4; i++)
        tex_to_backend(&s_gpu.texs[i], &d->tex[i]);
}

static void backend_clear(uint32_t param)
{
    Nv2aDrawState d;
    backend_capture(&d);
    s_backend->clear(&d, param);
    if (!(param & NV097_CLEAR_COLOR_MASK) || s_backend_compare)
        return;
    s_gpu.clears++;
    if (s_gpu.flips == 0)
        xbox_FramebufferWindowSet(dma_resolve(s_gpu.color_offset), s_gpu.pitch);
    xbox_FramebufferWindowStart();
}

/* A vertex-program batch: the program runs here, as in the software path's
 * serial mode (nv2a_vsh_run4), and the backend draws the outputs. */
static void backend_draw(void)
{
    uint32_t i, n = s_gpu.idx_count;
    Nv2aDrawState d;
    Nv2aDrawGeom g;

    if ((s_gpu.xf_mode & 3) != 2) {
        s_gpu.batches_untransformed++;     /* fixed function: not drawn yet */
        return;
    }
    /* The backend runs the program itself, unless it writes constants
     * (CXT_WRITE: each vertex sees the last one's writes, so it runs here,
     * in order) or RECOMP_VK_CPU_VSH=1 asks for the CPU anyway. */
    {
        static int cpu_vsh = -1;
        const uint32_t (*w)[4];
        uint32_t pn, k;
        int raw;
        if (cpu_vsh < 0)
            cpu_vsh = getenv("RECOMP_VK_CPU_VSH") != NULL;
        raw = s_backend->raw_vertices && !cpu_vsh && !nv2a_vsh_cxt_write()
           && (pn = nv2a_vsh_program_words(&w)) != 0;
        for (k = 0; raw && k < pn; k++)
            if (nv2a_vsh_ins_writes_constant(w[k]))
                raw = 0;                   /* writes c[] */
        if (raw) {
            memset(&g, 0, sizeof g);
            g.prim = s_gpu.prim;
            g.n = n;
            g.prog = w;
            g.prog_n = pn;
            g.constants = nv2a_vsh_constants();
            for (k = 0; k < NV_VERTEX_ATTRS; k++) {
                g.attr[k].offset = s_gpu.attr[k].offset;
                g.attr[k].type = s_gpu.attr[k].type;
                g.attr[k].size = s_gpu.attr[k].size;
                g.attr[k].stride = s_gpu.attr[k].stride;
            }
            g.imm = (const float (*)[4])s_gpu.imm_attr;
            g.idx = s_gpu.idx;
            if (s_gpu.inline_active) {
                g.inline_data = (const uint8_t *)s_gpu.inline_buf;
                g.inline_bytes = s_gpu.inline_count * 4;
            }
            s_gpu.batches_program++;
            s_gpu.verts_program += n;
            backend_capture(&d);
            s_backend->draw(&d, &g);
            note_drawn();
            return;
        }
    }
    for (i = 0; i < n; i += 4) {
        static float in[4][NV2A_VSH_INPUTS][4];
        static Nv2aVshOutput tail[4];
        const float (*ip[4])[4];
        Nv2aVshOutput *op[4];
        uint32_t l, a;
        for (l = 0; l < 4; l++) {
            uint32_t j = i + l < n ? i + l : n - 1;
            for (a = 0; a < NV2A_VSH_INPUTS; a++)
                fetch_input(a, s_gpu.idx[j], in[l][a]);
            ip[l] = (const float (*)[4])in[l];
            op[l] = i + l < n ? &s_xf[i + l] : &tail[l];
        }
        if (!nv2a_vsh_run4(ip, op))
            return;                        /* no program loaded */
    }
    s_gpu.batches_program++;
    s_gpu.verts_program += n;
    backend_capture(&d);
    memset(&g, 0, sizeof g);
    g.prim = s_gpu.prim;
    g.v = s_xf;
    g.n = n;
    s_backend->draw(&d, &g);
    note_drawn();
}

static void raster_batch(void)
{
    uint32_t i;
    uint32_t before = s_gpu.tris_drawn;

    if (s_gpu.idx_count < 3)
        return;
    if (s_backend) {
        backend_draw();
        if (!s_backend_compare)
            return;
    }
    static int no_vsh = -1;
    if (no_vsh < 0)
        no_vsh = getenv("RECOMP_NO_VSH") != NULL;
    if ((s_gpu.xf_mode & 3) == 2 && !no_vsh) {
        raster_batch_program();
        return;
    }
    if (!batch_is_screen_space()) {
        s_gpu.batches_untransformed++;
        return;
    }

    /* Count why, once per batch: the texture stage cannot change inside one. */
    {
        const VertexAttr *tc = texcoord_attr();

        if (!(tc->offset && tc->stride))
            s_gpu.batches_no_uv++;
        else if (!s_gpu.texs[0].valid)
            s_gpu.batches_no_tex++;
        else {
            s_gpu.batches_textured++;
            note_texture_use();
        }
    }

    switch (s_gpu.prim) {
    case NV_PRIM_TRIANGLES:
        for (i = 0; i + 2 < s_gpu.idx_count; i += 3)
            raster_indexed(s_gpu.idx[i], s_gpu.idx[i+1], s_gpu.idx[i+2],
                           vertex_color(s_gpu.idx[i]));
        break;
    case NV_PRIM_TRIANGLE_STRIP:
        for (i = 0; i + 2 < s_gpu.idx_count; i++)
            raster_indexed(s_gpu.idx[i], s_gpu.idx[i+1], s_gpu.idx[i+2],
                           vertex_color(s_gpu.idx[i]));
        break;
    case NV_PRIM_TRIANGLE_FAN:
    case NV_PRIM_POLYGON:
        for (i = 1; i + 1 < s_gpu.idx_count; i++)
            raster_indexed(s_gpu.idx[0], s_gpu.idx[i], s_gpu.idx[i+1],
                           vertex_color(s_gpu.idx[0]));
        break;
    case NV_PRIM_QUADS:
        /* Independent quads, four vertices each. A batch of eight is two
         * quads, not one six-triangle fan around the first vertex; with
         * exactly four the two agreed, which is why sharing the fan arm
         * looked right. */
        for (i = 0; i + 3 < s_gpu.idx_count; i += 4) {
            raster_indexed(s_gpu.idx[i], s_gpu.idx[i+1], s_gpu.idx[i+2],
                           vertex_color(s_gpu.idx[i]));
            raster_indexed(s_gpu.idx[i], s_gpu.idx[i+2], s_gpu.idx[i+3],
                           vertex_color(s_gpu.idx[i]));
        }
        break;
    case NV_PRIM_QUAD_STRIP:
        /* Each vertex pair past the first closes another quad against the
         * pair before it. */
        for (i = 0; i + 3 < s_gpu.idx_count; i += 2) {
            raster_indexed(s_gpu.idx[i], s_gpu.idx[i+1], s_gpu.idx[i+3],
                           vertex_color(s_gpu.idx[i]));
            raster_indexed(s_gpu.idx[i], s_gpu.idx[i+3], s_gpu.idx[i+2],
                           vertex_color(s_gpu.idx[i]));
        }
        break;
    default:
        break;                             /* points and lines: not yet */
    }

    if (s_gpu.tris_drawn && (s_gpu.tris_drawn % 500) == 0)
        fprintf(stderr, "  [GPU] %u triangles rasterised\n", s_gpu.tris_drawn);

    /* Capture the surface while the geometry is still on it.
     *
     * The periodic report dumps too, but a title clears every frame and draws
     * in only some of them, so a report almost always lands on a surface that
     * was wiped a moment ago -- which reads as "nothing was drawn" when the
     * triangles went down correctly just before it. A few frames that actually
     * contain geometry are worth more than any number of clears. */
    if (s_gpu.tris_drawn != before && s_drawn_dumps < FB_DUMP_AFTER_DRAW
        && !s_flip_dumping) {
        s_drawn_dumps++;
        dump_surface_bmp();
    }
}

/* What a batch actually contains. Before anything can be rasterised, the
 * question is what space attribute 0 arrives in: a title running a vertex
 * program hands over object-space positions that mean nothing without running
 * it, while pre-transformed screen-space coordinates can be drawn directly. */
/* RECOMP_FRAME_TRACE=<flag file>: once the file exists, log every batch of
 * the next whole frame (flip to flip) -- where it drew, with what texture,
 * blend and depth state, and how many pixels it actually wrote. "The frame
 * is black" has many causes and this is what tells them apart. */

static void frame_trace_flip(void)
{
    static const char *flag = (const char *)-1;
    if (flag == (const char *)-1)
        flag = getenv("RECOMP_FRAME_TRACE");
    if (!flag)
        return;
    if (s_ftrace == 3) {
        /* Deleting the flag file re-arms the trace for the next time it
         * appears, so one run can trace several moments. */
        FILE *f = fopen(flag, "rb");
        if (f)
            fclose(f);
        else
            s_ftrace = 0;
        return;
    }
    if (s_ftrace == 2) {
        s_ftrace = 3;
        fprintf(stderr, "[FTRACE] end of frame\n");
        return;
    }
    if (s_ftrace == 0) {
        FILE *f = fopen(flag, "rb");
        if (!f)
            return;
        fclose(f);
        s_ftrace = 2;
        fprintf(stderr, "[FTRACE] frame begins (flip %u)\n", s_gpu.flips);
    }
}

/* RECOMP_FRAME_SUMMARY=1: one line per presented frame -- which surfaces it
 * drew into, how many draws and triangles each, and the visibility-test
 * results it reported. Unlike the frame trace it does not drain per draw, so
 * it leaves the timing alone and can run for thousands of frames; a frame
 * that looks wrong in a flip dump (named by the same flip number) shows
 * whether the title submitted something different or the same work came out
 * differently. */
#define FSUM_SURFS 12
#define FSUM_REPS  24
static int s_fsum = -1;
static struct { uint32_t off, w, h, draws, tris; } s_fsum_s[FSUM_SURFS];
static int s_fsum_n;
static uint32_t s_fsum_rep[FSUM_REPS];
static int s_fsum_nrep;
#define FSUM_FOGS 6
static struct { uint32_t mode, color; float p0, p1; uint32_t draws;
                float omin, omax; } s_fsum_fog[FSUM_FOGS];
static int s_fsum_nfog;
#define FSUM_CLEARS 16
static struct { uint32_t off, param, color, x, y, w, h, at; } s_fsum_clr[FSUM_CLEARS];
static int s_fsum_nclr;
static uint32_t s_fsum_draws;           /* draws this frame, for "at" */

extern void xbox_WatchNote(uint32_t lo, uint32_t hi, uint32_t kind);
extern volatile uint32_t g_watch_flip;

static void fsum_draw(uint32_t tris)
{
    int i;
    {
        uint32_t lo = dma_resolve(s_gpu.color_offset);
        int st;
        xbox_WatchNote(lo, lo + (s_gpu.clip_y + s_gpu.clip_h) * s_gpu.pitch, 1);
        for (st = 0; st < 4; st++) {
            uint32_t tlo, thi;
            if (draw_samples(st, &tlo, &thi))
                xbox_WatchNote(tlo, thi, 2);
        }
    }
    for (i = 0; i < s_fsum_n; i++)
        if (s_fsum_s[i].off == s_gpu.color_offset && s_fsum_s[i].w == s_gpu.clip_w)
            break;
    if (i == s_fsum_n) {
        if (s_fsum_n == FSUM_SURFS)
            return;
        s_fsum_n++;
        s_fsum_s[i].off = s_gpu.color_offset;
        s_fsum_s[i].w = s_gpu.clip_w;
        s_fsum_s[i].h = s_gpu.clip_h;
        s_fsum_s[i].draws = s_fsum_s[i].tris = 0;
    }
    s_fsum_s[i].draws++;
    s_fsum_s[i].tris += tris;
    s_fsum_draws++;
    if (s_gpu.fog_enable && tris && (s_gpu.xf_mode & 3) == 2) {
        float of = s_xf[0].fog[0];
        for (i = 0; i < s_fsum_nfog; i++)
            if (s_fsum_fog[i].mode == s_gpu.fog_mode
             && s_fsum_fog[i].color == s_gpu.fog_color
             && s_fsum_fog[i].p0 == s_gpu.fog_param[0]
             && s_fsum_fog[i].p1 == s_gpu.fog_param[1])
                break;
        if (i == s_fsum_nfog) {
            if (s_fsum_nfog == FSUM_FOGS)
                return;
            s_fsum_nfog++;
            s_fsum_fog[i].mode = s_gpu.fog_mode;
            s_fsum_fog[i].color = s_gpu.fog_color;
            s_fsum_fog[i].p0 = s_gpu.fog_param[0];
            s_fsum_fog[i].p1 = s_gpu.fog_param[1];
            s_fsum_fog[i].draws = 0;
            s_fsum_fog[i].omin = s_fsum_fog[i].omax = of;
        }
        s_fsum_fog[i].draws++;
        if (of < s_fsum_fog[i].omin) s_fsum_fog[i].omin = of;
        if (of > s_fsum_fog[i].omax) s_fsum_fog[i].omax = of;
    }
}

static void fsum_flip(void)
{
    int i;
    if (s_fsum <= 0)
        return;
    {
        static uint64_t px0, zf0;
        static uint32_t dg0, off0, bh0, cu0;
        fprintf(stderr, "[FSUM] flip %u present %08X | px %llu zfail %llu degen %u"
                " off %u behind %u culled %u |", s_gpu.flips, s_gpu.drawn_offset,
                (unsigned long long)(s_gpu.xf_pixels - px0),
                (unsigned long long)(s_gpu.xf_depth_fail - zf0),
                s_gpu.xf_degenerate - dg0, s_gpu.xf_offscreen - off0,
                s_gpu.tris_behind - bh0, s_gpu.tris_culled - cu0);
        px0 = s_gpu.xf_pixels; zf0 = s_gpu.xf_depth_fail;
        dg0 = s_gpu.xf_degenerate; off0 = s_gpu.xf_offscreen;
        bh0 = s_gpu.tris_behind; cu0 = s_gpu.tris_culled;
    }
    for (i = 0; i < s_fsum_n; i++)
        fprintf(stderr, " %08X %ux%u d%u t%u;", s_fsum_s[i].off, s_fsum_s[i].w,
                s_fsum_s[i].h, s_fsum_s[i].draws, s_fsum_s[i].tris);
    fprintf(stderr, " | rep");
    for (i = 0; i < s_fsum_nrep; i++)
        fprintf(stderr, " %u", s_fsum_rep[i]);
    fprintf(stderr, " | clr");
    for (i = 0; i < s_fsum_nclr; i++)
        fprintf(stderr, " %08X %X %08X %u,%u+%ux%u @%u;", s_fsum_clr[i].off,
                s_fsum_clr[i].param, s_fsum_clr[i].color, s_fsum_clr[i].x,
                s_fsum_clr[i].y, s_fsum_clr[i].w, s_fsum_clr[i].h, s_fsum_clr[i].at);
    fprintf(stderr, " | fog");
    for (i = 0; i < s_fsum_nfog; i++)
        fprintf(stderr, " %X/%08X/%g/%g d%u o%g..%g;", s_fsum_fog[i].mode,
                s_fsum_fog[i].color, s_fsum_fog[i].p0, s_fsum_fog[i].p1,
                s_fsum_fog[i].draws, s_fsum_fog[i].omin, s_fsum_fog[i].omax);
    fputc(10, stderr);
    s_fsum_n = s_fsum_nrep = s_fsum_nfog = s_fsum_nclr = 0;
    s_fsum_draws = 0;
}

static void draw_primitive(void)
{
    float v[4];
    uint32_t i;

    if (!s_gpu.prim || !s_gpu.idx_count)
        return;
    s_gpu.draws++;
    /* A progress marker, rare enough to cost nothing: every 200 draws was
     * ~270 lines a second in Blinx 2's hub, and the executor then waited on
     * stderr's lock behind every other thread's logging. */
    if ((s_gpu.draws % 20000) == 0)
        fprintf(stderr, "  [GPU] draw #%u\n", s_gpu.draws);
    s_gpu.verts += s_gpu.idx_count;

    /* How many batches carry coordinates at all, and what range they span.
     * A pipeline that decodes perfectly and draws nothing is indistinguishable
     * from one that never ran, unless the vertices themselves are measured. */
    {
        float p[4];
        if (fetch_attr(&s_gpu.attr[0], s_gpu.idx[0], p)) {
            if (p[0] != 0.0f || p[1] != 0.0f || p[2] != 0.0f) {
                s_gpu.nonzero_draws++;
                if (p[0] < s_gpu.min_x) s_gpu.min_x = p[0];
                if (p[0] > s_gpu.max_x) s_gpu.max_x = p[0];
                if (p[1] < s_gpu.min_y) s_gpu.min_y = p[1];
                if (p[1] > s_gpu.max_y) s_gpu.max_y = p[1];
            }
        }
    }

    {
        uint32_t t0 = s_gpu.tris_drawn, c0 = s_gpu.tris_culled;
        uint32_t g0 = s_gpu.xf_degenerate, o0 = s_gpu.xf_offscreen;
        uint32_t b0 = s_gpu.tris_behind, z0 = s_gpu.zpass_count;
        uint64_t p0 = s_gpu.pixels, f0 = s_gpu.xf_depth_fail;
        raster_batch();
        if (s_fsum < 0)
        {
            const char *e = getenv("RECOMP_FRAME_SUMMARY");
            s_fsum = e ? (*e == '2' ? 2 : 1) : 0;
        }
        if (s_fsum)
            fsum_draw(s_gpu.tris_drawn - t0);
        if (s_fsum == 2 && s_gpu.tris_drawn != t0) {
            /* One line per draw: what it read and what it shaded. Needs
             * RECOMP_RASTER_BATCH=0 so the colour sum is this draw's. */
            static uint64_t c0, x0;
            int st;
            rq_sync();
            fprintf(stderr, "[FDRAW] %u %u %08X px %llu c %llu", s_gpu.flips,
                    s_gpu.draws, s_gpu.color_offset,
                    (unsigned long long)(s_gpu.xf_pixels - x0),
                    (unsigned long long)(s_gpu.csum - c0));
            fprintf(stderr, " bl %u %X/%X f0 %08X %08X %08X %08X fc %08X d0a %.3f zp %u",
                    s_gpu.blend_enable, s_gpu.blend_sfactor, s_gpu.blend_dfactor,
                    s_gpu.rc.factor0[0], s_gpu.rc.factor0[1], s_gpu.rc.factor0[2],
                    s_gpu.rc.factor0[3], s_gpu.rc.final_c0, s_xf[0].d0[3],
                    s_gpu.zpass_enable);
            c0 = s_gpu.csum;
            x0 = s_gpu.xf_pixels;
            for (st = 0; st < 4; st++) {
                uint32_t lo, hi;
                if (draw_samples(st, &lo, &hi)) {
                    const uint8_t *m = (const uint8_t *)xbox_GetMemoryOffset() + lo;
                    uint32_t h = 2166136261u, k, n = hi - lo < 4096 ? hi - lo : 4096;
                    for (k = 0; k < n; k += 4)
                        h = (h ^ *(const uint32_t *)(m + k)) * 16777619u;
                    fprintf(stderr, " t%d %08X f%02X h%08X", st, lo,
                            s_gpu.texs[st].color, h);
                    if (s_gpu.texs[st].color == 0x0B && s_gpu.texs[st].palette) {
                        const uint32_t *pal = (const uint32_t *)
                            ((const uint8_t *)xbox_GetMemoryOffset() + s_gpu.texs[st].palette);
                        h = 2166136261u;
                        for (k = 0; k < 256; k++)
                            h = (h ^ pal[k]) * 16777619u;
                        fprintf(stderr, " pal %08X h%08X", s_gpu.texs[st].palette, h);
                    }
                }
            }
            fputc(10, stderr);
        }
        if (s_ftrace == 2) {
            fprintf(stderr, "[FTRACE] alpha %u %X ref %X | zpass %u %u zfail %llu"
                    " | cmask %08X%c", s_gpu.alpha_test, s_gpu.alpha_func,
                    s_gpu.alpha_ref, s_gpu.zpass_enable,
                    s_gpu.zpass_count - z0,
                    (unsigned long long)(s_gpu.xf_depth_fail - f0),
                    s_gpu.color_mask, 10);
            fprintf(stderr, "[FTRACE] %s prim %u n %u surf %08X %ux%u+%u+%u"
                    " pitch %u | tex %s %08X %ux%u fmt %02X | blend %u %X/%X"
                    " eq %X | z test %u func %X mask %u | tris %u px %llu"
                    " | cull %u %X %X culled %u | degen %u off %u behind %u\n",
                    (s_gpu.xf_mode & 3) == 2 ? "PRG" : "FIX", s_gpu.prim,
                    s_gpu.idx_count, s_gpu.color_offset, s_gpu.clip_w,
                    s_gpu.clip_h, s_gpu.clip_x, s_gpu.clip_y, s_gpu.pitch,
                    s_gpu.texs[0].valid ? "on" : "off", s_gpu.texs[0].offset,
                    s_gpu.texs[0].width, s_gpu.texs[0].height, s_gpu.texs[0].color,
                    s_gpu.blend_enable, s_gpu.blend_sfactor,
                    s_gpu.blend_dfactor, s_gpu.blend_equation,
                    s_gpu.depth_test, s_gpu.depth_func, s_gpu.depth_mask,
                    s_gpu.tris_drawn - t0,
                    (unsigned long long)(s_gpu.pixels - p0),
                    s_gpu.cull_enable, s_gpu.cull_face, s_gpu.front_face,
                    s_gpu.tris_culled - c0, s_gpu.xf_degenerate - g0,
                    s_gpu.xf_offscreen - o0, s_gpu.tris_behind - b0);
            if ((s_gpu.xf_mode & 3) == 2) {
                extern uint32_t nv2a_vsh_cur_hash;
                fprintf(stderr, "[FTRACE]     vsh %08X%c", nv2a_vsh_cur_hash, 10);
            }
            if ((s_gpu.xf_mode & 3) == 2)
                fprintf(stderr, "[FTRACE]     v0 %g %g %g %g  v1 %g %g  v2 %g %g%c",
                        s_xf[0].pos[0], s_xf[0].pos[1], s_xf[0].pos[2],
                        s_xf[0].pos[3], s_xf[1].pos[0], s_xf[1].pos[1],
                        s_xf[2].pos[0], s_xf[2].pos[1], 10);
            if ((s_gpu.xf_mode & 3) == 2)
                fprintf(stderr, "[FTRACE]     d0 %.3g %.3g %.3g %.3g  d1 %.3g %.3g %.3g %.3g"
                        "  ofog %g | fog %u mode %X col %08X p %g %g%c",
                        s_xf[0].d0[0], s_xf[0].d0[1], s_xf[0].d0[2], s_xf[0].d0[3],
                        s_xf[0].d1[0], s_xf[0].d1[1], s_xf[0].d1[2], s_xf[0].d1[3],
                        s_xf[0].fog[0], s_gpu.fog_enable, s_gpu.fog_mode,
                        s_gpu.fog_color, s_gpu.fog_param[0], s_gpu.fog_param[1], 10);
            if (s_gpu.rc_seen) {
                int st;
                fprintf(stderr, "[FTRACE]     rc stages %u ctl %08X prog %05X fin %08X %08X",
                        s_gpu.rc.control & 0xFF, s_gpu.rc.control,
                        s_gpu.rc.stage_program, s_gpu.rc.final0, s_gpu.rc.final1);
                for (st = 0; st < 4; st++)
                    if ((s_gpu.rc.stage_program >> (st * 5)) & 0x1F)
                        fprintf(stderr, " | t%d %08X %ux%u fmt %02X%s", st,
                                s_gpu.texs[st].offset, s_gpu.texs[st].width,
                                s_gpu.texs[st].height, s_gpu.texs[st].color,
                                s_gpu.texs[st].valid ? "" : " (invalid)");
                fputc(10, stderr);
                for (st = 0; st < (int)(s_gpu.rc.control & 0xFF) && st < 8; st++)
                    fprintf(stderr, "[FTRACE]       s%d icw %08X %08X ocw %08X %08X c %08X %08X%c",
                            st, s_gpu.rc.color_icw[st], s_gpu.rc.alpha_icw[st],
                            s_gpu.rc.color_ocw[st], s_gpu.rc.alpha_ocw[st],
                            s_gpu.rc.factor0[st], s_gpu.rc.factor1[st], 10);
            }
            if ((s_gpu.xf_mode & 3) == 2) {
                uint32_t r;
                for (r = 112; r < 116; r++) {
                    const float *c = nv2a_vsh_constant(r);
                    fprintf(stderr, "[FTRACE]     c[%u] %g %g %g %g%c", r,
                            c[0], c[1], c[2], c[3], 10);
                }
            }
        }
    }

    static int verbose = -1;                /* getenv per draw was 3% */
    if (verbose < 0)
        verbose = getenv("RECOMP_PB_EXEC_VERBOSE") != NULL;
    if (verbose) {
        static int shown;
        if (shown++ < 6) {
            fprintf(stderr, "  [GPU] prim %u, %u indices, pos attr:"
                            " off 0x%08X type %u size %u stride %u\n",
                    s_gpu.prim, s_gpu.idx_count, s_gpu.attr[0].offset,
                    s_gpu.attr[0].type, s_gpu.attr[0].size, s_gpu.attr[0].stride);
            /* The texture stage, for either kind of batch. This used to print
             * only for inline batches, which meant a title drawing through
             * vertex arrays -- Half-Life 2's menu, for one -- showed no
             * texture state at all, and the reason a quad sampled flat was
             * invisible. */
            fprintf(stderr, "  [GPU]   tex: off 0x%08X %ux%u pitch %u"
                            " colour 0x%02X swizzled %d valid %d\n",
                    s_gpu.texs[0].offset, s_gpu.texs[0].width, s_gpu.texs[0].height,
                    s_gpu.texs[0].pitch, s_gpu.texs[0].color,
                    d3d8_format_is_swizzled(s_gpu.texs[0].color), s_gpu.texs[0].valid);
            {
                uint32_t k;
                for (k = 0; k < s_gpu.idx_count && k < 3; k++) {
                    float t[2];
                    if (fetch_texcoord(s_gpu.idx[k], t))
                        fprintf(stderr, "  [GPU]   uv[%u] = %.3f %.3f\n",
                                k, t[0], t[1]);
                }
            }
            /* An inline batch has no guest buffer to go and look at -- the
             * vertices are the payload -- so print the payload too. */
            if (s_gpu.inline_active) {
                uint32_t k;
                fprintf(stderr, "  [GPU]   inline %u dwords:", s_gpu.inline_count);
                for (k = 0; k < s_gpu.inline_count && k < 16; k++)
                    fprintf(stderr, " %08X", s_gpu.inline_buf[k]);
                fprintf(stderr, "\n");
                for (k = 0; k < s_gpu.idx_count && k < 4; k++) {
                    float t[2];
                    if (fetch_texcoord(s_gpu.idx[k], t))
                        fprintf(stderr, "  [GPU]   uv[%u] = %.3f %.3f\n",
                                k, t[0], t[1]);
                }
                fprintf(stderr, "  [GPU]   tex: off 0x%08X %ux%u pitch %u"
                                " colour 0x%02X valid %d\n",
                        s_gpu.texs[0].offset, s_gpu.texs[0].width, s_gpu.texs[0].height,
                        s_gpu.texs[0].pitch, s_gpu.texs[0].color, s_gpu.texs[0].valid);
            }
            {
                /* Every attribute the batch has, not just position. If the
                 * other streams carry data and position does not, the problem
                 * is one buffer rather than the whole vertex path. */
                const uint8_t *mem = (const uint8_t *)xbox_GetMemoryOffset();
                uint32_t a, k;
                for (a = 0; a < NV_VERTEX_ATTRS; a++) {
                    const VertexAttr *at = &s_gpu.attr[a];
                    uint32_t nz = 0;
                    if (!at->offset || !at->size)
                        continue;
                    for (k = 0; k < 64; k++)
                        if (mem[at->offset + k]) nz++;
                    fprintf(stderr, "  [GPU]   attr%-2u off 0x%08X type %u"
                                    " size %u stride %-3u  %u/64 bytes set\n",
                            a, at->offset, at->type, at->size, at->stride, nz);
                }
            }
            {
                /* Raw bytes at the array, in case the values read as zero:
                 * that looks the same whether the offset is wrong or the
                 * buffer genuinely has not been filled yet. */
                const uint8_t *mem = (const uint8_t *)xbox_GetMemoryOffset();
                uint32_t k;
                fprintf(stderr, "  [GPU]   bytes @0x%08X:", s_gpu.attr[0].offset);
                for (k = 0; k < 32; k++)
                    fprintf(stderr, " %02X", mem[s_gpu.attr[0].offset + k]);
                fprintf(stderr, "\n");
                fprintf(stderr, "  [GPU]   indices:");
                for (k = 0; k < s_gpu.idx_count && k < 8; k++)
                    fprintf(stderr, " %u", s_gpu.idx[k]);
                fprintf(stderr, "\n");
            }
            for (i = 0; i < s_gpu.idx_count && i < 3; i++) {
                if (fetch_attr(&s_gpu.attr[0], s_gpu.idx[i], v))
                    fprintf(stderr, "  [GPU]   v[%u] = %.3f %.3f %.3f %.3f\n",
                            s_gpu.idx[i], v[0], v[1], v[2], v[3]);
            }
        }
    }
}

/* Draw the vertices the title wrote straight into the pushbuffer.
 *
 * INLINE_ARRAY carries no offsets and no indices: the dwords between BEGIN and
 * END *are* the vertex buffer, packed in attribute order using the same
 * SET_VERTEX_DATA_ARRAY_FORMAT registers an ordinary array would use. So the
 * whole batch is describable as a vertex array whose base happens to be that
 * payload, which means synthesising the layout and handing it to the existing
 * path -- rather than a second copy of the topology and rasterisation code.
 *
 * The title's own attribute table is saved and put back: these offsets and
 * strides are ours, and it has not stopped using its.
 *
 * ponytail: each attribute is padded to a whole dword. That is exact for the
 * float and D3DCOLOR formats every inline batch actually uses; a packed
 * sub-dword attribute would need the unpadded layout.
 */
static void draw_inline_array(void)
{
    VertexAttr saved[NV_VERTEX_ATTRS];
    uint32_t off = 0, a, i, vsize, count;

    memcpy(saved, s_gpu.attr, sizeof saved);

    for (a = 0; a < NV_VERTEX_ATTRS; a++) {
        uint32_t bytes;
        if (!s_gpu.attr[a].size)
            continue;
        switch (s_gpu.attr[a].type) {
        case 0:  bytes = 4;                        break;  /* D3DCOLOR   */
        case 2:  bytes = 4 * s_gpu.attr[a].size;   break;  /* float      */
        case 4:  bytes = s_gpu.attr[a].size;       break;  /* ubyte norm */
        case 1: case 5:                                     /* shorts     */
                 bytes = 2 * s_gpu.attr[a].size;   break;
        case 6:  bytes = 4;                        break;  /* 11:11:10   */
        default: bytes = 4 * s_gpu.attr[a].size;   break;
        }
        s_gpu.attr[a].offset = off;
        off += (bytes + 3u) & ~3u;
    }
    vsize = off;
    if (!vsize)
        goto out;

    count = (s_gpu.inline_count * 4) / vsize;
    if (count < 3 || count > NV_MAX_INDICES)
        goto out;
    for (a = 0; a < NV_VERTEX_ATTRS; a++)
        if (s_gpu.attr[a].size)
            s_gpu.attr[a].stride = vsize;

    for (i = 0; i < count; i++)
        s_gpu.idx[i] = (uint16_t)i;
    s_gpu.idx_count = count;

    s_gpu.inline_active = 1;
    draw_primitive();
    s_gpu.inline_active = 0;

out:
    memcpy(s_gpu.attr, saved, sizeof saved);
    s_gpu.idx_count = 0;
}

/* Draw the vertices SET_VERTEX3F/4F completed.
 *
 * Same trick as draw_inline_array: rather than a second copy of the topology
 * and rasterisation code, describe what was accumulated as an ordinary vertex
 * array and hand it to the existing path. The layout is ours and fixed, so
 * the attribute table is written out here rather than derived from the
 * title's format registers.
 *
 * The title's own table is saved and put back -- it has not stopped using it.
 *
 * ponytail: position, diffuse and texcoord0 only. That is what a 2D quad
 * carries and what this rasteriser samples; a second texcoord set or a normal
 * would need the D3D11 translator, not more slots here.
 */
static void draw_immediate(void)
{
    VertexAttr saved[NV_VERTEX_ATTRS];
    uint32_t i;

    if (s_gpu.imm_count < 3)
        return;

    memcpy(saved, s_gpu.attr, sizeof saved);
    memset(s_gpu.attr, 0, sizeof s_gpu.attr);
    /* Offsets are byte offsets into inline_buf here, not guest addresses --
     * fetch_attr reads them that way while inline_active is set, which is
     * also why 0 is a legal offset for position. Every attribute is a float4;
     * the ones the batch never set carry their standing values, as on the
     * GPU. */
    for (i = 0; i < NV_VERTEX_ATTRS; i++) {
        s_gpu.attr[i].type = 2;
        s_gpu.attr[i].size = 4;
        s_gpu.attr[i].offset = i * 16;
        s_gpu.attr[i].stride = IMM_VERTEX_DWORDS * 4;
    }

    for (i = 0; i < s_gpu.imm_count && i < NV_MAX_INDICES; i++)
        s_gpu.idx[i] = (uint16_t)i;
    s_gpu.idx_count = i;

    /* fetch_attr bounds-checks against inline_count dwords. */
    s_gpu.inline_count = s_gpu.imm_count * IMM_VERTEX_DWORDS;
    s_gpu.inline_active = 1;
    draw_primitive();
    s_gpu.inline_active = 0;

    memcpy(s_gpu.attr, saved, sizeof saved);
    s_gpu.idx_count = 0;
    s_gpu.inline_count = 0;
}

/* A vertex is complete: append all 16 attributes in the layout
 * draw_immediate describes. */
static void imm_emit_vertex(void)
{
    uint32_t at = s_gpu.imm_count * IMM_VERTEX_DWORDS;

    if (!s_gpu.prim || at + IMM_VERTEX_DWORDS > NV_MAX_INLINE)
        return;
    memcpy(&s_gpu.inline_buf[at], s_gpu.imm_attr, sizeof s_gpu.imm_attr);
    s_gpu.imm_count++;
}

/* The immediate-mode writes. Returns 1 if `method` was one of them.
 *
 * Split out because it is a range test against five separate bases, and that
 * reads better than five more cases in an already long switch.
 */
/* The immediate-mode attribute methods (xemu pgraph.c SET_VERTEX_DATA*).
 * Each sets an attribute's current value; completing attribute 0 -- the
 * position -- emits a vertex carrying every attribute as it stands. This used
 * to keep position, diffuse and texcoord 0 only, which dropped any quad with
 * a second texture coordinate: Burnout 3 composites its whole 3D scene into
 * the frame with exactly such a quad, and the frame never got the scene. */
static int imm_vertex_method(uint32_t method, uint32_t param)
{
    union { uint32_t u; float f; } v;
    uint32_t attr, c;
    float *a;
    v.u = param;

    if (method >= NV097_SET_VERTEX4F && method < NV097_SET_VERTEX4F + 16) {
        c = (method - NV097_SET_VERTEX4F) / 4;
        s_gpu.imm_attr[0][c] = v.f;
        s_gpu.imm_used |= 1;
        if (c == 3)
            imm_emit_vertex();
        return 1;
    }
    if (method >= NV097_SET_VERTEX3F && method < NV097_SET_VERTEX3F + 12) {
        c = (method - NV097_SET_VERTEX3F) / 4;
        s_gpu.imm_attr[0][c] = v.f;
        s_gpu.imm_used |= 1;
        if (c == 2) {
            s_gpu.imm_attr[0][3] = 1.0f;
            imm_emit_vertex();
        }
        return 1;
    }
    if (method >= NV097_SET_VERTEX_DATA2F_M
            && method < NV097_SET_VERTEX_DATA2F_M + NV_VERTEX_ATTRS * 8) {
        attr = (method - NV097_SET_VERTEX_DATA2F_M) / 8;
        c = ((method - NV097_SET_VERTEX_DATA2F_M) % 8) / 4;
        a = s_gpu.imm_attr[attr];
        a[c] = v.f;
        if (c == 1) {
            a[2] = 0.0f; a[3] = 1.0f;
        }
        s_gpu.imm_used |= (uint16_t)(1u << attr);
        if (attr == 0 && c == 1)
            imm_emit_vertex();
        return 1;
    }
    if (method >= NV097_SET_VERTEX_DATA4F_M
            && method < NV097_SET_VERTEX_DATA4F_M + NV_VERTEX_ATTRS * 16) {
        attr = (method - NV097_SET_VERTEX_DATA4F_M) / 16;
        c = ((method - NV097_SET_VERTEX_DATA4F_M) % 16) / 4;
        s_gpu.imm_attr[attr][c] = v.f;
        s_gpu.imm_used |= (uint16_t)(1u << attr);
        if (attr == 0 && c == 3)
            imm_emit_vertex();
        return 1;
    }
    if (method >= NV097_SET_VERTEX_DATA2S
            && method < NV097_SET_VERTEX_DATA2S + NV_VERTEX_ATTRS * 4) {
        attr = (method - NV097_SET_VERTEX_DATA2S) / 4;
        a = s_gpu.imm_attr[attr];
        a[0] = (float)(int16_t)(param & 0xFFFF);
        a[1] = (float)(int16_t)(param >> 16);
        a[2] = 0.0f; a[3] = 1.0f;
        s_gpu.imm_used |= (uint16_t)(1u << attr);
        if (attr == 0)
            imm_emit_vertex();
        return 1;
    }
    if (method >= NV097_SET_VERTEX_DATA4UB
            && method < NV097_SET_VERTEX_DATA4UB + NV_VERTEX_ATTRS * 4) {
        /* Bytes in register order, x from the low byte (xemu). */
        attr = (method - NV097_SET_VERTEX_DATA4UB) / 4;
        a = s_gpu.imm_attr[attr];
        a[0] = (float)( param        & 0xFF) / 255.0f;
        a[1] = (float)((param >>  8) & 0xFF) / 255.0f;
        a[2] = (float)((param >> 16) & 0xFF) / 255.0f;
        a[3] = (float)( param >> 24        ) / 255.0f;
        s_gpu.imm_used |= (uint16_t)(1u << attr);
        if (attr == 0)
            imm_emit_vertex();
        return 1;
    }
    if (method >= NV097_SET_VERTEX_DATA4S_M
            && method < NV097_SET_VERTEX_DATA4S_M + NV_VERTEX_ATTRS * 8) {
        attr = (method - NV097_SET_VERTEX_DATA4S_M) / 8;
        c = ((method - NV097_SET_VERTEX_DATA4S_M) % 8) / 4;
        a = s_gpu.imm_attr[attr];
        a[c * 2]     = (float)(int16_t)(param & 0xFFFF);
        a[c * 2 + 1] = (float)(int16_t)(param >> 16);
        s_gpu.imm_used |= (uint16_t)(1u << attr);
        if (attr == 0 && c == 1)
            imm_emit_vertex();
        return 1;
    }
    return 0;
}
/* The texture stage registers, all four stages. Stage i's registers are
 * 0x1B00 + 0x40*i onwards; only stage 0 used to be decoded, which was
 * enough for menus drawn one texture at a time and not for a 3D frame that
 * composites its scene through a second stage. */
static void tex_stage_method(uint32_t method, uint32_t param)
{
    uint32_t stage = (method - NV_TEX_FIRST) / 0x40;
    uint32_t reg = NV_TEX_FIRST + ((method - NV_TEX_FIRST) & 0x3F);
    Texture *t = &s_gpu.texs[stage];

    switch (reg) {
    case NV097_SET_TEXTURE_OFFSET:
        /* A DMA-object offset like a surface's: physical, reached through
         * the contiguous window when it names contiguous memory. */
        t->offset = dma_resolve(param);
        break;
    case NV097_SET_TEXTURE_FORMAT:
        t->color = (param >> 8) & 0xFF;
        t->cube = (param >> 2) & 1;
        t->levels = (param >> 16) & 0xF;
        /* A swizzled texture carries its own dimensions here, as log2 in
         * BASE_SIZE_U/V. It has to: IMAGE_RECT describes a linear image, and
         * a title that only uses swizzled textures never sends one. */
        if (tex_size_from_format(t->color)) {
            t->width  = 1u << ((param >> 20) & 0xF);
            t->height = 1u << ((param >> 24) & 0xF);
        }
        break;
    case NV097_SET_TEXTURE_PALETTE:
        /* The low six bits carry the DMA context and the entry count. */
        t->palette = dma_resolve(param & ~0x3Fu);
        break;
    case NV097_SET_TEXTURE_ADDRESS:
        /* Four bits per axis. 1 is wrap, 3 clamp-to-edge; mirror and border
         * fall back to clamp, wrong at an edge rather than everywhere. */
        t->addr_u =  param        & 0xF;
        t->addr_v = (param >>  8) & 0xF;
        break;
    case NV097_SET_TEXTURE_CONTROL1:
        t->pitch = param >> 16;          /* linear formats only */
        break;
    case NV097_SET_TEXTURE_IMAGE_RECT:
        t->width  = param >> 16;
        t->height = param & 0xFFFF;
        break;
    case 0x1B14:                                  /* SET_TEXTURE_FILTER */
        t->filter = param;
        break;
    case 0x1B28: case 0x1B2C: case 0x1B30: case 0x1B34:  /* BUMP_ENV_MAT */
        memcpy(&t->bump_mat[(reg - 0x1B28) / 4], &param, 4);
        break;
    case 0x1B38: memcpy(&t->bump_scale, &param, 4); break;  /* BUMP_ENV_SCALE */
    case 0x1B3C: memcpy(&t->bump_offset, &param, 4); break; /* BUMP_ENV_OFFSET */
    default:
        break;
    }
    tex_update_valid(t);
    if (stage == 0)
        record_tex_reg(method, param);
}

/* A draw longer than NV_MAX_INDICES: the rest is dropped. Said once. */
static void idx_overflow(void)
{
    static int said;
    if (!said) {
        said = 1;
        fprintf(stderr, "  [GPU] a draw has more than %d indices; the rest is"
                        " not drawn%c", NV_MAX_INDICES, 10);
    }
}

void nv2a_pb_exec_method(uint32_t subch, uint32_t method, uint32_t param)
{
    static int inited;
    if (!inited) {
        inited = 1;
        s_gpu.color_mask = 0x01010101u;           /* all channels, as reset */
        {
            const char *w = getenv("RECOMP_TEX_WATCH");
            if (w)
                s_tw_addr = (uint32_t)strtoul(w, NULL, 16);
            w = getenv("RECOMP_PIXEL_WATCH");
            while (w && s_pw_n < 8
                   && sscanf(w, "%d,%d", &s_pw_xs[s_pw_n], &s_pw_ys[s_pw_n]) == 2) {
                s_pw_n++;
                w = strchr(w, ';');
                if (w)
                    w++;
            }
        }
        s_gpu.st_func = 0x207;                    /* stencil: ALWAYS, */
        s_gpu.st_fmask = s_gpu.st_wmask = 0xFF;   /* all bits, */
        s_gpu.st_fail = s_gpu.st_zfail = s_gpu.st_zpass = 0x1E00; /* KEEP */
        {
            int i;
            for (i = 0; i < NV_VERTEX_ATTRS; i++)
                s_gpu.imm_attr[i][3] = 1.0f;
        }
        s_gpu.min_x = s_gpu.min_y = 1e30f;
        s_gpu.max_x = s_gpu.max_y = -1e30f;
        backend_select();
    }
    /* Bring-up: the first parameters each surface method carries. A wrong
     * pitch or clip is indistinguishable from a method never arriving unless
     * the values are visible. Cached: this runs for every method, and an
     * uncached getenv here was a quarter of the executor's time. */
    static int verbose = -1;
    if (verbose < 0)
        verbose = getenv("RECOMP_PB_EXEC_VERBOSE") != NULL;
    if (verbose) {
        static int shown[8];
        int slot = -1;
        switch (method) {
        case NV097_SET_SURFACE_CLIP_HORIZONTAL: slot = 0; break;
        case NV097_SET_SURFACE_CLIP_VERTICAL:   slot = 1; break;
        case NV097_SET_SURFACE_FORMAT:          slot = 2; break;
        case NV097_SET_SURFACE_PITCH:           slot = 3; break;
        case NV097_SET_SURFACE_COLOR_OFFSET:    slot = 4; break;
        case NV097_SET_COLOR_CLEAR_VALUE:       slot = 5; break;
        case NV097_CLEAR_SURFACE:               slot = 6; break;
        default: break;
        }
        if (slot >= 0 && shown[slot]++ < 4)
            fprintf(stderr, "  [GPU] subch %u method 0x%04X param 0x%08X\n",
                    subch, method, param);
    }

    if (s_ftrace == 2) {
        /* RECOMP_FRAME_TRACE_METHODS: every method of the traced frame,
         * less the bulk (vertex data, program and constant uploads). */
        static int all = -1;
        if (all < 0)
            all = getenv("RECOMP_FRAME_TRACE_METHODS") != NULL;
        if (all && !(method >= 0x1828 && method < 0x1A00)
            && !(method >= 0x0B00 && method < 0x0C00))
            fprintf(stderr, "[FTRACE]   m %u:%04X = %08X%c", subch, method,
                    param, 10);
    }
    if (subch == 0 && method >= NV_TEX_FIRST && method <= NV_TEX_LAST) {
        tex_stage_method(method, param);
        return;
    }
    if (subch == 5 && method == 0x0310) {
        /* D3D's InsertFence puts this before every fence's semaphore
         * release: ((pushbuffer address << 3) | fence & 0x1F) << 2 | epoch.
         * The GPU latches it into PGRAPH 0x400B10, and that -- not DMA GET
         * -- is where D3D reads how far the GPU has consumed the pushbuffer
         * (Blinx 2's sub_002E7E00: (reg >> 5) & 0x7FFFFFC | 0x80000000).
         * Unexecuted, the register only ever held the fence bits the
         * semaphore release wrote, so D3D's idea of the GPU's position was
         * address 0. */
        xbox_Nv2aPgraphWrite(0x400B10u, param);
        return;
    }
    if (subch != 0) {                      /* 3D class lives on subchannel 0 */
        note_unhandled(method, param);
        return;
    }
    switch (method) {
    case NV097_SET_SURFACE_CLIP_HORIZONTAL:
        s_gpu.clip_x = param & 0xFFFF;
        s_gpu.clip_w = (param >> 16) & 0xFFFF;
        break;
    case NV097_SET_SURFACE_CLIP_VERTICAL:
        s_gpu.clip_y = param & 0xFFFF;
        s_gpu.clip_h = (param >> 16) & 0xFFFF;
        break;
    case NV097_SET_SURFACE_FORMAT:
        s_gpu.format = param;
        break;
    case NV097_SET_SURFACE_PITCH:
        s_gpu.pitch = param & 0xFFFF;      /* colour pitch; zeta is the top half */
        break;
    case NV097_SET_SURFACE_COLOR_OFFSET:
        if (s_ftrace == 2)
            fprintf(stderr, "[FTRACE] color offset -> %08X%c", param, 10);
        s_gpu.color_offset = param;
        break;
    case NV097_SET_COLOR_CLEAR_VALUE:
        s_gpu.clear_color = param;
        break;
    case NV097_SET_BLEND_ENABLE:
        s_gpu.blend_enable = param;
        break;
    case NV097_SET_BLEND_EQUATION:
        s_gpu.blend_equation = param;
        break;
    case 0x0358:                                  /* SET_COLOR_MASK */
        s_gpu.color_mask = param;
        break;

    /* Register combiners. */
    case 0x0288: s_gpu.rc.final0 = param; break;  /* SPECULAR_FOG_CW0 */
    case 0x028C: s_gpu.rc.final1 = param; break;  /* SPECULAR_FOG_CW1 */
    case 0x1E20: s_gpu.rc.final_c0 = param; break;/* SPECULAR_FOG_FACTOR */
    case 0x1E24: s_gpu.rc.final_c1 = param; break;
    case 0x1E60:                                  /* COMBINER_CONTROL */
        s_gpu.rc.control = param;
        s_gpu.rc_seen = 1;
        break;
    case 0x1E70: s_gpu.rc.stage_program = param; break;
    case 0x1E78: s_gpu.other_stage_input = param; break; /* OTHER_STAGE_INPUT */
    case 0x17F8: s_gpu.clip_plane_mode = param; break;

    /* Alpha test and fog. */
    case 0x0300: s_gpu.alpha_test = param; break;
    case 0x033C: s_gpu.alpha_func = param; break;
    case 0x0340: s_gpu.alpha_ref = param; break;
    case 0x02A4: s_gpu.fog_enable = param; break;
    case 0x02A8: s_gpu.fog_color = param; break;  /* R in bits 0-7 */
    case 0x029C: s_gpu.fog_mode = param; break;
    case 0x09C0: memcpy(&s_gpu.fog_param[0], &param, 4); break;
    case 0x09C4: memcpy(&s_gpu.fog_param[1], &param, 4); break;
    case 0x17C8:                                  /* CLEAR_REPORT_VALUE */
        if (s_backend && !s_backend_compare) {
            s_backend->report_clear();
            s_gpu.zpass_count = 0;
            break;
        }
        if (rq_async()) {
            /* A new test: draws from here count in a fresh epoch. */
            uint32_t e;
            int k;
            /* Far behind, or the epoch about to be reused still has a test
             * pending (CLEARs without GETs between): let the queue catch up
             * before its counters are zeroed. */
            if (s_rep_n >= NV_ZEPOCHS / 2 ||
                (s_rep_n && s_gpu.report_epoch + 1 - s_rep[s_rep_head].epoch
                                >= NV_ZEPOCHS))
                rq_sync();
            e = ++s_gpu.report_epoch & (NV_ZEPOCHS - 1);
            s_zpe_total[e] = 0;
            for (k = 0; k <= s_pool.n; k++)
                s_pool.cnt[k].zpe[e] = 0;
        } else
            rq_sync();
        s_gpu.zpass_count = 0;
        break;
    case 0x17CC:                                  /* SET_ZPASS_PIXEL_COUNT_ENABLE */
        s_gpu.zpass_enable = param;
        break;
    case 0x17D0: {                                /* GET_REPORT */
        /* The GPU's answer to a visibility test: 16 bytes at the report
         * DMA offset -- a timestamp, the pixel count, and 0 for "done".
         * Never written, a title polling it (D3D's GetVisibilityTestResult
         * starts the slot at 0xFFFFFFFF) sees every test incomplete. Burnout
         * 3 culls its world with these, so the race's main view drew only
         * the sky while the cube-map passes, which do not test, drew it all.
         * The report DMA object starts at physical 0 here, as D3D sets it. */
        uint8_t *mem = (uint8_t *)xbox_GetMemoryOffset();
        uint32_t at = dma_resolve(param & 0x00FFFFFFu);
        uint64_t stamp;
        static int rep_sync = -1;
        if (rep_sync < 0)
            rep_sync = getenv("RECOMP_REPORT_SYNC") != NULL;
        /* RECOMP_REPORT_SYNC=1: answer visibility tests at once, after a
         * drain, even under async sync points -- a diagnostic: a title that
         * polls a result once and takes "not ready" for "not visible" reads
         * differently when the answer comes late. */
        if (s_backend && !s_backend_compare) {
            static int force = -1;
            if (force < 0)
                force = getenv("RECOMP_ZPASS_FORCE") != NULL;
            stamp = (uint64_t)s_gpu.reports++ * 1000u;
            s_backend->report(at, stamp, force);   /* written when known */
            break;
        }
        if (rq_async() && !rep_sync) {
            /* Answered when the batches before it are done, by whichever
             * thread finishes them (rq_advance_prefix). The title polls the
             * slot until its last word reads 0, as it would for the GPU. */
            stamp = (uint64_t)s_gpu.reports++ * 1000u;
            rq_kick();
            AcquireSRWLockExclusive(&s_pool.lock);
            if (s_rep_n < NV_REPORTS_PENDING && s_pool.done_prefix < s_pool.head) {
                int i = (s_rep_head + s_rep_n) % NV_REPORTS_PENDING;
                s_rep[i].at = at;
                s_rep[i].stamp = stamp;
                s_rep[i].epoch = s_gpu.report_epoch;
                s_rep[i].seq = s_pool.head - 1;
                s_rep_n++;
                ReleaseSRWLockExclusive(&s_pool.lock);
            } else {
                ReleaseSRWLockExclusive(&s_pool.lock);
                rq_sync();
                rq_report_write(at, stamp, s_gpu.report_epoch);
            }
            break;
        }
        rq_sync();                         /* every queued pixel counted */
        stamp = (uint64_t)s_gpu.reports++ * 1000u;
        memcpy(mem + at, &stamp, 8);
        {
            /* RECOMP_ZPASS_FORCE=1 reports every visibility test as visible.
             * Diagnostic: in Blinx 2's attract stage it brings back the sky
             * and the cliffs' shading, so some test the title gates them on
             * comes out 0 here where the console counts pixels. Which one,
             * and why, is open; forcing also draws whatever really was
             * occluded (flares behind walls), so it is not the default. */
            static int force = -1;
            uint32_t zc = s_gpu.zpass_count;
            if (force < 0)
                force = getenv("RECOMP_ZPASS_FORCE") != NULL;
            if (force)
                zc = 100000;
            if (s_fsum > 0 && s_fsum_nrep < FSUM_REPS)
                s_fsum_rep[s_fsum_nrep++] = zc;
            memcpy(mem + at + 8, &zc, 4);
        }
        memset(mem + at + 12, 0, 4);
        break;
    }
    case NV097_SET_BLEND_COLOR:
        s_gpu.blend_color = param;
        break;
    case NV097_SET_BLEND_FUNC_SFACTOR:
        s_gpu.blend_sfactor = param;
        break;
    case NV097_SET_BLEND_FUNC_DFACTOR:
        s_gpu.blend_dfactor = param;
        break;
    case NV097_CLEAR_SURFACE:
        if (s_ftrace == 2)
            fprintf(stderr, "[FTRACE] clear %X surf %08X %ux%u colour %08X%c",
                    param, s_gpu.color_offset, s_gpu.clip_w, s_gpu.clip_h,
                    s_gpu.clear_color, 10);
        if (s_fsum > 0 && s_fsum_nclr < FSUM_CLEARS) {
            int c = s_fsum_nclr++;
            s_fsum_clr[c].off = s_gpu.color_offset;
            s_fsum_clr[c].param = param;
            s_fsum_clr[c].color = s_gpu.clear_color;
            s_fsum_clr[c].x = s_gpu.clip_x; s_fsum_clr[c].y = s_gpu.clip_y;
            s_fsum_clr[c].w = s_gpu.clip_w; s_fsum_clr[c].h = s_gpu.clip_h;
            s_fsum_clr[c].at = s_fsum_draws;
        }
        if (s_backend) {
            backend_clear(param);
            if (!s_backend_compare)
                break;
        }
        clear_surface(param);
        if (param & 0x03)                         /* Z, stencil */
            zbuf_clear(param & 0x03);
        break;

    case NV097_SET_BEGIN_END:
        if (param) {
            s_gpu.prim = param;
            s_gpu.idx_count = 0;
            s_gpu.inline_count = 0;
            s_gpu.imm_count = 0;
        } else {
            /* Three ways a batch can have arrived, and only one is in use at
             * a time: vertices completed by SET_VERTEX4F, a payload written
             * with INLINE_ARRAY, or indices into the title's own arrays. */
            if (s_gpu.imm_count)
                draw_immediate();
            else if (s_gpu.inline_count)
                draw_inline_array();
            else
                draw_primitive();
            s_gpu.prim = 0;
            s_gpu.inline_count = 0;
            s_gpu.imm_count = 0;
        }
        break;

    case NV097_INLINE_ARRAY:
        /* Vertex data, not a pointer to it. Buffered rather than decoded here
         * because the format is only fully known at END. */
        if (s_gpu.prim && s_gpu.inline_count < NV_MAX_INLINE)
            s_gpu.inline_buf[s_gpu.inline_count++] = param;
        break;

    case NV097_DRAW_ARRAYS: {
        /* The method this title actually draws with, and the reason the
         * executor reported zero draws while geometry was being submitted the
         * whole time: BEGIN_END arrived, END arrived, and in between came a
         * run description rather than the index list the draw path wanted, so
         * every batch ended with idx_count == 0 and was dropped in silence.
         *
         * Expanded into indices because that is what the rasteriser consumes,
         * and an implicit run is just the indices start..start+count-1. */
        uint32_t start = param & 0x00FFFFFFu;
        uint32_t count = ((param >> 24) & 0xFFu) + 1u;
        uint32_t i;

        if (!s_gpu.prim)
            break;
        for (i = 0; i < count && s_gpu.idx_count < NV_MAX_INDICES; i++)
            s_gpu.idx[s_gpu.idx_count++] = (uint16_t)(start + i);
        if (i < count)
            idx_overflow();
        break;
    }

    case NV097_ARRAY_ELEMENT16:
        /* Two 16-bit indices per parameter word. */
        if (s_gpu.prim && s_gpu.idx_count + 2 <= NV_MAX_INDICES) {
            s_gpu.idx[s_gpu.idx_count++] = (uint16_t)(param & 0xFFFF);
            s_gpu.idx[s_gpu.idx_count++] = (uint16_t)(param >> 16);
        } else if (s_gpu.prim) {
            idx_overflow();
        }
        break;

    case NV097_SET_FLIP_READ:
        if (s_ftrace == 2)
            fprintf(stderr, "[FTRACE] P_READ %u%c", param, 10);
        s_gpu.flip_read = param;
        return;

    case NV097_SET_FLIP_WRITE:
        if (s_ftrace == 2)
            fprintf(stderr, "[FTRACE] P_WRITE %u%c", param, 10);
        s_gpu.flip_write = param;
        return;

    case NV097_SET_FLIP_MODULO:
        s_gpu.flip_modulo = param;
        return;

    case NV097_FLIP_INCREMENT_WRITE:
        rq_sync();                       /* the frame is finished */
#if defined(_WIN32)
        if (s_rqst_print < 0)
            s_rqst_print = getenv("RECOMP_RQ_STATS") != NULL;
        if (s_rqst_print && s_gpu.flips % 60 == 59) {
            /* The time counters run on for nv2a_pb_exec_perf: deltas. */
            static LONG64 g0, b0, i0, w0;
            double f = 60.0;
            fprintf(stderr, "[RQSTATS] per frame: workers asleep %.1f ms blocked"
                    " on geometry, %.1f ms on barriers, %.1f ms idle (summed over"
                    " %d); executor waited %.1f ms; %.1f batches (%.1f barrier),"
                    " %.0f tris each%c",
                    (s_rqst.blk_geo_us - g0) / 1e3 / f,
                    (s_rqst.blk_bar_us - b0) / 1e3 / f,
                    (s_rqst.idle_us - i0) / 1e3 / f, s_pool.n,
                    (s_rqst.wait_us - w0) / 1e3 / f,
                    s_rqst.kicks / f, s_rqst.barriers / f,
                    s_rqst.kicks ? (double)s_rqst.tris / s_rqst.kicks : 0.0, 10);
            g0 = s_rqst.blk_geo_us; b0 = s_rqst.blk_bar_us;
            i0 = s_rqst.idle_us; w0 = s_rqst.wait_us;
            s_rqst.kicks = s_rqst.barriers = s_rqst.tris = 0;
        }
#endif
        s_gpu.flip_write = s_gpu.flip_modulo
                         ? (s_gpu.flip_write + 1) % s_gpu.flip_modulo
                         : s_gpu.flip_write + 1;
        s_gpu.flips++;
        frame_trace_flip();
        return;

    case NV097_FLIP_STALL:
        rq_sync();
        if (s_ftrace == 2)
            fprintf(stderr, "[FTRACE] FLIP_STALL presenting %08X (color now %08X)%c",
                    s_gpu.drawn_offset, s_gpu.color_offset, 10);
        /* The stall ends when the buffer being read is the one just finished.
         * There is no scanout here to wait for, so that is now. */
        s_gpu.flip_read = s_gpu.flip_write;
        /* And this is a completed swap, which is what a title's own swap
         * counter counts -- see xbox_Nv2aFrameCounterFlip. */
        xbox_Nv2aFrameCounterFlip();
        /* Hand the window a copy of the frame just finished.
         *
         * The buffer the title has finished is the one the last batch drew
         * into, which is what drawn_offset holds and why it exists: by the
         * flip, color_offset has already moved to the next buffer. Copying
         * here, rather than letting the window read guest memory on its own
         * clock, is what stops it showing a surface the rasteriser is still
         * writing. */
        {
            extern void xbox_FramebufferWindowFrameStats(uint32_t);
            static uint32_t draws_at_flip;
            xbox_FramebufferWindowFrameStats(s_gpu.draws - draws_at_flip);
            draws_at_flip = s_gpu.draws;
        }
        if (s_gpu.pitch) {
            extern void xbox_FramebufferWindowPresent(uint32_t, uint32_t);
            uint32_t done = s_gpu.drawn_offset ? s_gpu.drawn_offset
                                               : s_gpu.color_offset;
            uint32_t pitch = s_gpu.drawn_offset ? s_gpu.drawn_pitch
                                                : s_gpu.pitch;
            if (done && s_backend) {
                /* The frame is on the GPU: into guest memory, where the
                 * window and the dumps read it. */
                uint32_t bpp = s_gpu.drawn_offset ? s_gpu.drawn_bpp : surface_bpp();
                uint32_t w = s_gpu.drawn_offset ? s_gpu.drawn_x + s_gpu.drawn_w
                                                : s_gpu.clip_x + s_gpu.clip_w;
                uint32_t h = s_gpu.drawn_offset ? s_gpu.drawn_y + s_gpu.drawn_h
                                                : s_gpu.clip_y + s_gpu.clip_h;
                if (s_backend_compare)
                    s_backend->compare(dma_resolve(done), pitch, bpp, w, h, s_gpu.flips);
                else
                    s_backend->flip(dma_resolve(done), pitch, bpp, w, h);
            }
            if (done) {
                xbox_FramebufferWindowSet(dma_resolve(done), pitch);
                xbox_FramebufferWindowPresent(dma_resolve(done), pitch);
            }
        }
        {
            /* RECOMP_FB_DUMP_FLIPS: dump every presented frame, not one per
             * report -- a consecutive sequence, which is what a headless
             * recording (frames -> ffmpeg) needs. "1" dumps from boot; any
             * other value names a file, and frames are dumped while it
             * exists, so a capture can start mid-game without writing
             * gigabytes of menus first. */
            static const char *spec = (const char *)-1;
            if (spec == (const char *)-1)
                spec = getenv("RECOMP_FB_DUMP_FLIPS");
            if (spec) {
                int on = strcmp(spec, "1") == 0;
                s_flip_naming = 1;
                if (!on) {
                    FILE *f = fopen(spec, "rb");
                    if (f) { fclose(f); on = 1; }
                }
                s_flip_dumping = on;
                if (on)
                    dump_surface_bmp();
            }
        }
        {
            static int init;
            if (!init) {
                const char *pc = getenv("RECOMP_PIXEL_CHECK");
                init = 1;
                if (pc && strcmp(pc, "1"))
                    s_pxchk_file = pc;
            }
            pixel_check_flip();
        }
        fsum_flip();
        g_watch_flip = s_gpu.flips;
        s_gpu.drawn_stale = 1;

        if (getenv("RECOMP_PB_EXEC_VERBOSE")) {
            static unsigned n;
            if (n++ < 8) {
                fprintf(stderr, "  [GPU] flip %u: read=%u write=%u\n",
                        s_gpu.flips, s_gpu.flip_read, s_gpu.flip_write);
                fflush(stderr);
            }
        }
        return;


    /* Vertex programs and their constants: forwarded as they arrive. */
    case 0x1E94:                                  /* TRANSFORM_EXECUTION_MODE */
        s_gpu.xf_mode = param;
        break;
    case 0x1E98:                                  /* _PROGRAM_CXT_WRITE_EN */
        nv2a_vsh_set_cxt_write(param);
        break;
    case 0x1E9C:                                  /* _PROGRAM_LOAD */
        nv2a_vsh_set_load_slot(param);
        break;
    case 0x1EA0:                                  /* _PROGRAM_START */
        nv2a_vsh_set_start_slot(param);
        break;
    case 0x1EA4:                                  /* _CONSTANT_LOAD */
        nv2a_vsh_set_constant_load(param);
        break;

    /* Depth. */
    case 0x030C: s_gpu.depth_test = param; break; /* DEPTH_TEST_ENABLE */
    case 0x0354: s_gpu.depth_func = param; break; /* DEPTH_FUNC */
    case 0x035C: s_gpu.depth_mask = param; break; /* DEPTH_MASK */
    case 0x032C: s_gpu.st_test = param; break;    /* STENCIL_TEST_ENABLE */
    case 0x0360: s_gpu.st_wmask = param; break;   /* STENCIL_MASK */
    case 0x0364: s_gpu.st_func = param; break;    /* STENCIL_FUNC */
    case 0x0368: s_gpu.st_ref = param; break;     /* STENCIL_FUNC_REF */
    case 0x036C: s_gpu.st_fmask = param; break;   /* STENCIL_FUNC_MASK */
    case 0x0370: s_gpu.st_fail = param; break;    /* STENCIL_OP_FAIL */
    case 0x0374: s_gpu.st_zfail = param; break;   /* STENCIL_OP_ZFAIL */
    case 0x0378: s_gpu.st_zpass = param; break;   /* STENCIL_OP_ZPASS */
    case 0x0308: s_gpu.cull_enable = param; break;/* CULL_FACE_ENABLE */
    case 0x039C: s_gpu.cull_face = param; break;  /* CULL_FACE */
    case 0x03A0: s_gpu.front_face = param; break; /* FRONT_FACE */

    /* Nothing to do in an executor that runs every method in order, to
     * completion: NO_OPERATION, WAIT_FOR_IDLE, BREAK_VERTEX_BUFFER_CACHE.
     * Ranked as unhandled they were three of the top four entries in Blinx
     * 2's hub and pushed the methods that matter down the list. */
    case 0x0100:                                  /* NO_OPERATION */
        /* With a parameter it is a software notify: the GPU stops and
         * interrupts, and D3D's handler acts on the parameter. Blinx 2's
         * fence wait plants one (param 5) when the GPU is far behind and
         * sleeps until the interrupt wakes it. */
        if (param) {
            rq_kick();               /* hand over what precedes it */
            xbox_Nv2aNotify(method, param);
        }
        break;
    case 0x0110: case 0x1710:
        break;
    case 0x0214: s_gpu.zeta_offset = param; break;/* SURFACE_ZETA_OFFSET */
    case 0x1D8C: s_gpu.zstencil_clear = param; break; /* ZSTENCIL_CLEAR_VALUE */
    case 0x1D6C: s_gpu.sem_offset = param; break;     /* SET_SEMAPHORE_OFFSET */
    case 0x1D70:                     /* BACK_END_WRITE_SEMAPHORE_RELEASE */
        /* D3D's fence: the GPU writes the fence time once everything before
         * it is done, and the title waits on that value before reusing what
         * those commands read. Released here, in stream order, from the
         * thread that executes the stream, the title waits exactly as long
         * as the work takes -- not until a 1 ms poll notices. What counts
         * as done is rq_fence's call. */
        rq_fence(s_gpu.sem_offset, param);
        s_gpu.sem_releases++;
        break;

    default:
        if (method >= 0x0260 && method < 0x0280) {         /* ALPHA_ICW(i) */
            s_gpu.rc.alpha_icw[(method - 0x0260) / 4] = param;
            break;
        }
        if (method >= 0x0AC0 && method < 0x0AE0) {         /* COLOR_ICW(i) */
            s_gpu.rc.color_icw[(method - 0x0AC0) / 4] = param;
            break;
        }
        if (method >= 0x1E40 && method < 0x1E60) {         /* COLOR_OCW(i) */
            s_gpu.rc.color_ocw[(method - 0x1E40) / 4] = param;
            break;
        }
        if (method >= 0x0AA0 && method < 0x0AC0) {         /* ALPHA_OCW(i) */
            s_gpu.rc.alpha_ocw[(method - 0x0AA0) / 4] = param;
            break;
        }
        if (method >= 0x0A60 && method < 0x0A80) {         /* FACTOR0(i) */
            s_gpu.rc.factor0[(method - 0x0A60) / 4] = param;
            break;
        }
        if (method >= 0x0A80 && method < 0x0AA0) {         /* FACTOR1(i) */
            s_gpu.rc.factor1[(method - 0x0A80) / 4] = param;
            break;
        }
        /* SET_VIEWPORT_OFFSET / _SCALE. The GPU keeps these in the constant
         * file, at c[59] and c[58] -- exactly where the D3D epilogue reads
         * them -- so they are constants that happen to arrive by method.
         * Dropped, every vertex program put its whole batch at the origin:
         * no 3D anywhere, and a black car on the garage screen. */
        if (method >= 0x0A20 && method < 0x0A30) {
            nv2a_vsh_constant_component(59, (method - 0x0A20) / 4, param);
            break;
        }
        if (method >= 0x0AF0 && method < 0x0B00) {
            nv2a_vsh_constant_component(58, (method - 0x0AF0) / 4, param);
            break;
        }
        if (method >= 0x0B00 && method < 0x0B80) {    /* TRANSFORM_PROGRAM(i) */
            nv2a_vsh_program_word(param);
            break;
        }
        if (method >= 0x0B80 && method < 0x0C00) {    /* TRANSFORM_CONSTANT(i) */
            nv2a_vsh_constant_word(param);
            break;
        }
        if (method >= NV_TEX_FIRST && method <= NV_TEX_LAST)
            record_tex_reg(method, param);
        if (method >= NV097_SET_VERTEX_DATA_ARRAY_OFFSET
                && method < NV097_SET_VERTEX_DATA_ARRAY_OFFSET + NV_VERTEX_ATTRS * 4) {
            /* Resolved here, once, so every consumer -- the rasteriser's
             * attribute reads and the diagnostics alike -- sees the same
             * address. A vertex array offset is a DMA-object offset exactly
             * like a surface offset: physical, and addressable only through
             * the window when it names contiguous memory. */
            s_gpu.attr[(method - NV097_SET_VERTEX_DATA_ARRAY_OFFSET) / 4].offset =
                dma_resolve(param);
        } else if (method >= NV097_SET_VERTEX_DATA_ARRAY_FORMAT
                && method < NV097_SET_VERTEX_DATA_ARRAY_FORMAT + NV_VERTEX_ATTRS * 4) {
            VertexAttr *a = &s_gpu.attr[(method - NV097_SET_VERTEX_DATA_ARRAY_FORMAT) / 4];
            a->type   =  param        & 0x0F;
            a->size   = (param >> 4)  & 0x0F;
            a->stride = (param >> 8)  & 0xFF;
        } else if (!imm_vertex_method(method, param)) {
            note_unhandled(method, param);
        }
        break;
    }
}

/* Find where the title actually wrote its quad.
 *
 * The GPU is pointed at a buffer that stays zero, which says the data went
 * somewhere else -- and the only way to find somewhere else is to look for the
 * data. A screen-space quad for a 640x480 target contains 640.0f and 480.0f as
 * floats, which is a distinctive enough pair to search guest RAM for. Whatever
 * address that turns up is where the title's writes are landing, and the
 * difference from the programmed offset is the bug. */
static void find_quad_vertices(void)
{
    const uint32_t W = 0x44200000u;   /* 640.0f */
    const uint32_t H = 0x43F00000u;   /* 480.0f */
    const uint32_t *ram = (const uint32_t *)xbox_GetMemoryOffset();
    uint32_t i, hits = 0;

    fprintf(stderr, "[GPU] searching guest RAM for 640.0f/480.0f pairs...\n");
    for (i = 0x1000 / 4; i < (0x04000000u / 4) - 8 && hits < 12; i++) {
        if (ram[i] != W && ram[i] != H)
            continue;
        /* Both values within a few words of each other: a lone 640.0f is
         * common, the pair much less so. */
        {
            int has_w = 0, has_h = 0;
            uint32_t k;
            for (k = 0; k < 8; k++) {
                if (ram[i + k] == W) has_w = 1;
                if (ram[i + k] == H) has_h = 1;
            }
            if (!has_w || !has_h)
                continue;
        }
        hits++;
        fprintf(stderr, "  [GPU]   0x%08X:", i * 4);
        {
            uint32_t k;
            for (k = 0; k < 8; k++)
                fprintf(stderr, " %08X", ram[i + k]);
        }
        fprintf(stderr, "\n");
        i += 8;
    }
    if (!hits)
        fprintf(stderr, "  [GPU]   none found -- the quad is not in RAM in"
                        " that form\n");
    fflush(stderr);
}

/* Locate NaN-filled transform matrices in guest RAM.
 *
 * A matrix arriving as NaN says the maths went wrong somewhere upstream, and
 * the only way to find where is to find the matrix and watch who writes it.
 * Three consecutive real-indefinite values is a distinctive enough signature:
 * ordinary data does not contain runs of 0xFFC00000. */
static void find_nan_matrices(void)
{
    const uint32_t NAN_NEG = 0xFFC00000u;
    const uint32_t *ram = (const uint32_t *)xbox_GetMemoryOffset();
    uint32_t i, hits = 0;

    fprintf(stderr, "[GPU] searching guest RAM for NaN matrices...\n");
    for (i = 0x1000 / 4; i < (0x04000000u / 4) - 20 && hits < 10; i++) {
        if (ram[i] != NAN_NEG || ram[i + 1] != NAN_NEG || ram[i + 2] != NAN_NEG)
            continue;
        hits++;
        fprintf(stderr, "  [GPU]   0x%08X:", i * 4);
        {
            uint32_t k;
            for (k = 0; k < 16; k++)
                fprintf(stderr, " %08X", ram[i + k]);
        }
        fprintf(stderr, "\n");
        i += 16;
    }
    if (!hits)
        fprintf(stderr, "  [GPU]   none in RAM -- the NaNs are computed into"
                        " registers, not stored\n");
    fflush(stderr);
}

/* Print guest dwords named by RECOMP_PEEK, as hex and as float.
 *
 * Chasing a value backwards means reading it, and a value that is only wrong
 * for one frame in a thousand cannot be caught by stopping. Both
 * interpretations are printed because the question is usually "is this a
 * pointer or a number", and guessing wrong costs a run. */
static void peek_addresses(void)
{
    const char *spec = getenv("RECOMP_PEEK");
    const uint8_t *mem = (const uint8_t *)xbox_GetMemoryOffset();
    char buf[256];
    char *tok, *ctx = NULL;

    if (!spec)
        return;
    strncpy(buf, spec, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = 0;
    for (tok = strtok_s(buf, ",", &ctx); tok; tok = strtok_s(NULL, ",", &ctx)) {
        uint32_t va = (uint32_t)strtoul(tok, NULL, 0);
        uint32_t v;
        float f;
        if (va < 0x1000u || va >= 0x04000000u)
            continue;
        v = *(const uint32_t *)(mem + va);
        memcpy(&f, &v, 4);
        fprintf(stderr, "  [PEEK] 0x%08X = %08X  (%g)\n", va, v, f);
    }
    fflush(stderr);
}

/* Walk a pointer chain and print every step.
 *
 * RECOMP_PEEK_CHAIN="0x1315A8,8,0x10,0" starts at that address, and for each
 * offset dereferences the current pointer and adds it. Following a chain by
 * hand costs one run per level; this costs one run for the whole chain, and
 * prints where it goes wrong when a level is null.
 */
static void peek_chain(void)
{
    const char *spec = getenv("RECOMP_PEEK_CHAIN");
    const uint8_t *mem = (const uint8_t *)xbox_GetMemoryOffset();
    char buf[256], *tok, *ctx = NULL;
    uint32_t cur = 0;
    int step = 0;

    if (!spec)
        return;
    strncpy(buf, spec, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = 0;

    for (tok = strtok_s(buf, ",", &ctx); tok; tok = strtok_s(NULL, ",", &ctx)) {
        uint32_t off = (uint32_t)strtoul(tok, NULL, 0);
        if (step == 0) {
            cur = off;
            fprintf(stderr, "  [CHAIN] start 0x%08X\n", cur);
        } else {
            if (cur < 0x1000u || cur + 4 >= 0x04000000u) {
                fprintf(stderr, "  [CHAIN] step %d: 0x%08X is not a guest"
                                " pointer -- chain ends\n", step, cur);
                return;
            }
            cur = *(const uint32_t *)(mem + cur) + off;
            fprintf(stderr, "  [CHAIN] step %d: deref +0x%X -> 0x%08X\n",
                    step, off, cur);
        }
        step++;
    }
    if (cur >= 0x1000u && cur + 4 < 0x04000000u)
        fprintf(stderr, "  [CHAIN] final value at 0x%08X = 0x%08X\n",
                cur, *(const uint32_t *)(mem + cur));
    fflush(stderr);
}

/* Called by the NV2A thread before it moves GET up to PUT: hand the queued
 * triangles to the workers, without waiting for them.
 *
 * Waiting here (rq_sync) kept "GET passed" meaning "pixels done", but it is
 * also where the NV2A thread blocks before it gets back round to acking
 * D3D's next kick (the 0x100410 flush bit, which only that loop may clear),
 * and the title spends most of its frame spinning on that bit. Not waiting
 * was ~12.8 -> ~14.7 fps in Blinx 2's hub (alternating within one run). What GET releases is safe to
 * reuse: vertices are transformed and copied at set-up, render targets are
 * only written by the queue, and flips, clears and render-to-texture still
 * drain. The one gap is a texture the CPU rewrites right after GET passes
 * a draw still queued, which then samples the newer texels -- and the
 * GPU-idle shim already lets the title assume every fence has passed.
 * RECOMP_RASTER_GET_WAIT=1 restores the wait. */
void nv2a_pb_exec_drain(void)
{
    extern int xbox_SchedMode(void);
    static int wait = -1;
    /* Under a guest-CPU schedule (sched.c) the workers' writes to render
     * targets in guest memory must be done within the agent's turn, or they
     * land during someone else's: it waits. */
    if (wait < 0)
        wait = getenv("RECOMP_RASTER_GET_WAIT") != NULL || xbox_SchedMode();
    if (wait)
        rq_sync();
    else
        rq_kick();
}

void nv2a_pb_exec_report(void)
{
    rq_sync();
    peek_addresses();
    peek_chain();
    if (getenv("RECOMP_FIND_NAN")) {
        /* Every report, not once: the matrix is fine early on and only turns
         * to NaN later, so a single scan at startup finds nothing and says
         * nothing. */
        find_nan_matrices();
    }
    if (getenv("RECOMP_FIND_QUAD")) {
        static int done;
        if (!done) { done = 1; find_quad_vertices(); }
    }
    int i, j;

    {
        /* Frames per second of guest time, from flips between reports. */
        static uint32_t last_flips;
        static unsigned long last_ms;
        unsigned long now = (unsigned long)(clock() * 1000.0 / CLOCKS_PER_SEC);
        if (last_ms && now > last_ms)
            fprintf(stderr, "[GPU] %.2f fps (%u flips)%c",
                    (s_gpu.flips - last_flips) * 1000.0 / (now - last_ms),
                    s_gpu.flips, 10);
        last_flips = s_gpu.flips;
        last_ms = now;
    }
    fprintf(stderr, "[GPU] surface 0x%08X pitch %u clip %ux%u+%u+%u"
                    " clears %u | %u unhandled methods (%d distinct)"
                    " | %u fences released\n",
            s_gpu.color_offset, s_gpu.pitch, s_gpu.clip_w, s_gpu.clip_h,
            s_gpu.clip_x, s_gpu.clip_y, s_gpu.clears,
            s_gpu.unhandled_total, s_unhandled_count, s_gpu.sem_releases);
    fprintf(stderr, "[GPU] draws %u (%u with coordinates), %u indices;"
                    " x %.1f..%.1f  y %.1f..%.1f\n",
            s_gpu.draws, s_gpu.nonzero_draws, s_gpu.verts,
            s_gpu.min_x, s_gpu.max_x, s_gpu.min_y, s_gpu.max_y);
    /* One picture per report rather than per clear: a title clears hundreds of
     * times a second and nobody wants that many files. */
    if (!s_flip_dumping)
        dump_surface_bmp();

    /* Drawn and skipped separately: "nothing appeared" and "every batch needed
     * a vertex program we do not run" look identical on screen, and only one
     * of them means the rasteriser is broken. */
    fprintf(stderr, "[GPU] brightest pixel written 0x%08X\n", s_gpu.pixel_max);
    fprintf(stderr, "[GPU] %llu pixels written; draw surface 0x%08X"
                    " -> 0x%08X, clear surface 0x%08X -> 0x%08X\n",
            (unsigned long long)s_gpu.pixels, s_gpu.drawn_offset,
            dma_resolve(s_gpu.drawn_offset), s_gpu.color_offset,
            dma_resolve(s_gpu.color_offset));
    fprintf(stderr, "[GPU] rasterised %u triangles; %u batches skipped as not"
                    " screen-space, %u triangles fully off-surface\n",
            s_gpu.tris_drawn, s_gpu.batches_untransformed,
            s_gpu.tris_skipped_offscreen);
    fprintf(stderr, "[GPU] vertex programs: %u batches, %u vertices;"
                    " %u triangles dropped behind the eye, %u culled\n",
            s_gpu.batches_program, s_gpu.verts_program, s_gpu.tris_behind,
            s_gpu.tris_culled);
    fprintf(stderr, "[GPU]   of the rest: %u degenerate/NaN, %u off-surface,"
                    " %u drawn; pixels %llu written, %llu depth-failed;"
                    " x %.0f..%.0f y %.0f..%.0f z %g..%g\n",
            s_gpu.xf_degenerate, s_gpu.xf_offscreen, s_gpu.xf_drawn,
            (unsigned long long)s_gpu.xf_pixels,
            (unsigned long long)s_gpu.xf_depth_fail,
            s_gpu.xf_min[0], s_gpu.xf_max[0], s_gpu.xf_min[1], s_gpu.xf_max[1],
            s_gpu.xf_min[2], s_gpu.xf_max[2]);
    {
        fprintf(stderr, "[GPU]   blend pairs (src/dst):");
        for (j = 0; j < s_gpu.blend_npairs; j++)
            fprintf(stderr, " %X/%X", s_gpu.blend_pairs[j] >> 16,
                    s_gpu.blend_pairs[j] & 0xFFFF);
        fputc(10, stderr);
        fprintf(stderr, "[GPU]   program attribute formats (slot:type/size):");
        for (j = 0; j < s_gpu.xf_nfmt; j++)
            fprintf(stderr, " v%u:%u/%u", s_gpu.xf_fmt[j] >> 8,
                    s_gpu.xf_fmt[j] & 15, (s_gpu.xf_fmt[j] >> 4) & 15);
        fprintf(stderr, "\n");
    }

    /* And of the batches that did rasterise, how many sampled anything. A menu
     * that draws its background from one texture and its text from another
     * shows both as flat colour if either half is missing, so the split is
     * what says which half. */
    fprintf(stderr, "[GPU] screen-space batches: %u textured, %u with no"
                    " texcoords, %u with texcoords but no usable stage\n",
            s_gpu.batches_textured, s_gpu.batches_no_uv, s_gpu.batches_no_tex);
    fprintf(stderr, "[GPU] vertex-program batches: %u with a texture on some"
                    " stage, %u with none\n",
            s_gpu.program_textured, s_gpu.program_no_tex);
    {
        int i;
        fprintf(stderr, "[GPU] raster drains with work pending, by line:");
        for (i = 0; i < 24 && s_sync_sites[i].line; i++)
            fprintf(stderr, " %d:%u/%.0fms", s_sync_sites[i].line, s_sync_sites[i].n,
                    s_sync_sites[i].ms);
        fputc(10, stderr);
    }
    fprintf(stderr, "[GPU] raster batches: %u to the pool, %u inline;"
                    " %llu triangles, %llu bbox pixels\n",
            s_gpu.rq_pooled, s_gpu.rq_inline,
            (unsigned long long)s_gpu.rq_tris, (unsigned long long)s_gpu.rq_px);
    fprintf(stderr, "[GPU] vertices set up on the pool: %llu%c",
            (unsigned long long)s_gpu.xf_parallel_verts, 10);
    for (i = 0; i < s_tex_use_count; i++)
        fprintf(stderr, "  [TEXUSE] 0x%08X %ux%u fmt 0x%02X%s: %u batches\n",
                s_tex_use[i].offset, s_tex_use[i].width, s_tex_use[i].height,
                s_tex_use[i].color,
                d3d8_format_dxt_block_bytes(s_tex_use[i].color) ? " dxt"
                    : d3d8_format_is_swizzled(s_tex_use[i].color) ? " swz" : " lin",
                s_tex_use[i].batches);

    if (getenv("RECOMP_TEX_STATE")) {
        uint32_t k;
        for (k = 0; k < sizeof s_tex_set / sizeof s_tex_set[0]; k++)
            if (s_tex_set[k])
                fprintf(stderr, "  [TEX] 0x%04X = 0x%08X\n",
                        (unsigned)(NV_TEX_FIRST + k * 4), s_tex_reg[k]);
    }

    /* Top ten by frequency: selection sort over a small table, once every few
     * seconds, is not worth a better algorithm.
     *
     * RECOMP_PB_UNHANDLED_ALL lists every one instead. Ten is the right
     * default -- the tail is a long list of state registers nobody needs to
     * read -- but when a title stops and the question is which method it
     * stopped on, the answer is as likely to be the one seen twice as the
     * one seen a thousand times, and ten hides it. */
    {
        int shown = getenv("RECOMP_PB_UNHANDLED_ALL") ? s_unhandled_count : 10;
    for (i = 0; i < shown && i < s_unhandled_count; i++) {
        int best = i;
        for (j = i + 1; j < s_unhandled_count; j++)
            if (s_unhandled[j].count > s_unhandled[best].count)
                best = j;
        if (best != i) {
            PbUnhandled t = s_unhandled[i];
            s_unhandled[i] = s_unhandled[best];
            s_unhandled[best] = t;
        }
        {
            /* The value as well as the count. A method nobody decoded is
             * a guess until you see what it carried: screen coordinates,
             * a 0..1 texcoord and a packed colour are told apart at a
             * glance, and that is what says which vertex encoding a title
             * is using. */
            union { uint32_t u; float f; } v;
            v.u = s_unhandled[i].last_param;
            fprintf(stderr, "  [GPU]   0x%04X x%-8u last=0x%08X (%.4f)\n",
                    s_unhandled[i].method, s_unhandled[i].count,
                    v.u, v.f);
        }
    }
    }    fflush(stderr);
}
