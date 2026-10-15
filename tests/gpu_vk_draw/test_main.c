/* Headless test of the Vulkan backend through nv2a_backend.h: README.md. */
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "nv2a_backend.h"
#include "nv2a_combiner.h"
#include "xbox_vk.h"

extern const Nv2aBackend *xbox_vk_backend(void);

/* Guest memory: address A is g_mem + A. */
#define MEM_BYTES (8u << 20)
static uint8_t *g_mem;
ptrdiff_t xbox_GetMemoryOffset(void) { return (ptrdiff_t)g_mem; }
int xbox_SchedMode(void) { return 0; }

/* Texture contents: a 4x4 checker (green on even squares, blue on odd) at
 * TEXA; random 16x16 textures at RTEX + stage * 0x10000 (combiner test). */
#define RTEX 0x00400000u
static uint32_t g_rtex[4][16][16];

int nv2a_exec_texel(const Nv2aTexState *t, uint32_t face, uint32_t u, uint32_t v,
                    uint32_t *argb)
{
    (void)face;
    if (t->offset >= RTEX) {
        *argb = g_rtex[(t->offset - RTEX) / 0x10000][v & 15][u & 15];
        return 1;
    }
    *argb = ((u + v) & 1) ? 0xFF0000FFu : 0xFF00FF00u;
    return 1;
}

#define SURF   0x00100000u
#define TEXA   0x00200000u
#define W 64
#define H 64
#define ZMAX 16777215.0f

static const Nv2aBackend *be;
static int failures;

static void base_state(Nv2aDrawState *s)
{
    memset(s, 0, sizeof *s);
    s->color_addr = SURF;
    s->surface_format = 0x08 | (2 << 4);         /* A8R8G8B8, Z24S8 */
    s->pitch = W * 4;
    s->bpp = 4;
    s->clip_w = W;
    s->clip_h = H;
    s->zeta_addr = 0x00300000u;
    s->zeta_format = 2;
    s->zstencil_clear = 0xFFFFFF00u;             /* far, stencil 0 */
    s->color_mask = 0x01010101u;
    s->alpha_func = 0x207;
    s->blend_sfactor = 1;
    s->depth_func = 0x201;                       /* LESS */
    s->st_func = 0x207;
    s->st_fmask = s->st_wmask = 0xFF;
    s->st_fail = s->st_zfail = s->st_zpass = 0x1E00;
    s->cull_face = 0x405;
    s->front_face = 0x901;
    s->vp_scale[0] = W / 2.0f; s->vp_scale[1] = -H / 2.0f; s->vp_scale[2] = ZMAX;
    s->vp_offset[0] = W / 2.0f; s->vp_offset[1] = H / 2.0f;
}

/* A screen-space rectangle as a quad, colour c (r g b a), depth z in [0,1]. */
static void rect(const Nv2aDrawState *s, float x0, float y0, float x1, float y1,
                 float z, const float c[4], float u1)
{
    Nv2aVshOutput v[4];
    Nv2aDrawGeom g;
    int i;
    memset(v, 0, sizeof v);
    v[0].pos[0] = x0; v[0].pos[1] = y0;
    v[1].pos[0] = x1; v[1].pos[1] = y0;
    v[2].pos[0] = x1; v[2].pos[1] = y1;
    v[3].pos[0] = x0; v[3].pos[1] = y1;
    for (i = 0; i < 4; i++) {
        v[i].pos[2] = z * ZMAX;
        v[i].pos[3] = 1.0f;
        memcpy(v[i].d0, c, sizeof v[i].d0);
        v[i].tex[0][0] = (i == 1 || i == 2) ? u1 : 0.0f;
        v[i].tex[0][1] = (i >= 2) ? u1 : 0.0f;
        v[i].tex[0][3] = 1.0f;
    }
    g.prim = 8;                                  /* QUADS */
    g.v = v;
    g.n = 4;
    be->draw(s, &g);
}

static uint32_t px(int x, int y)
{
    return *(uint32_t *)(g_mem + SURF + (size_t)y * W * 4 + (size_t)x * 4);
}

/* GET_REPORT: answered asynchronously; poll the slot's last word as the
 * title does. */
#define REPORT 0x00380000u
static uint32_t report(void)
{
    volatile uint32_t *done = (volatile uint32_t *)(g_mem + REPORT + 12);
    uint32_t n, spins = 0;
    *done = 0xFFFFFFFFu;
    be->report(REPORT, 1000, 0);
    while (*done != 0 && spins++ < 200000000u)
        ;
    if (*done != 0) {
        printf("FAIL report never written\n");
        failures++;
    }
    memcpy(&n, g_mem + REPORT + 8, 4);
    return n;
}

static void flip(void)
{
    memset(g_mem + SURF, 0xCD, W * H * 4);       /* must be overwritten */
    be->flip(SURF, W * 4, 4, W, H);
}

static int near_px(uint32_t a, uint32_t b, int tol)
{
    int k;
    for (k = 0; k < 32; k += 8)
        if (abs((int)((a >> k) & 0xFF) - (int)((b >> k) & 0xFF)) > tol)
            return 0;
    return 1;
}

static void expect(const char *what, int x, int y, uint32_t want, int tol)
{
    uint32_t got = px(x, y);
    if (!near_px(got, want, tol)) {
        printf("FAIL %s: (%d,%d) = %08X, want %08X\n", what, x, y, got, want);
        failures++;
    } else {
        printf("ok   %s: (%d,%d) = %08X\n", what, x, y, got);
    }
}

/* ---- Combiners: random programs, GPU against nv2a_rc_eval --------------- */

static uint32_t rng = 12345;
static uint32_t rnd(void)
{
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return rng;
}

static uint32_t rnd_in(void)
{
    static const uint32_t regs[] = {0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13};
    return regs[rnd() % 12] | (rnd() & 1) << 4 | (rnd() & 7) << 5;
}

static uint32_t rnd_out_reg(void)
{
    static const uint32_t regs[] = {0, 4, 5, 8, 9, 10, 11, 12, 13};
    return regs[rnd() % 9];
}

static uint32_t rnd_ocw(int alpha)
{
    static const uint32_t maps[] = {0x00, 0x08, 0x10, 0x18, 0x20, 0x30};
    uint32_t f = maps[rnd() % 6];
    if (!alpha)
        f |= (rnd() & 7) | ((rnd() & 3) << 6);   /* dots, mux, blue-to-alpha */
    else
        f |= rnd() & 4;
    return (f << 12) | rnd_out_reg() << 8 | rnd_out_reg() << 4 | rnd_out_reg();
}

static int combiner_diff(int programs)
{
    int prog, bad_px = 0, bad_progs = 0, x, y, i;
    for (i = 0; i < 4; i++)
        for (y = 0; y < 16; y++)
            for (x = 0; x < 16; x++)
                g_rtex[i][y][x] = rnd();
    for (prog = 0; prog < programs; prog++) {
        Nv2aDrawState s;
        Nv2aVshOutput v[4];
        Nv2aDrawGeom g;
        float d0[4], d1[4], fog[4];
        int n = 1 + (int)(rnd() % 8), bad = 0;

        base_state(&s);
        s.clip_w = s.clip_h = 16;
        s.rc_seen = 1;
        s.rc.control = (uint32_t)n | (rnd() & 1) << 8 | (rnd() & 1) << 12 | (rnd() & 1) << 16;
        s.rc.stage_program = 1 | 1 << 5 | 1 << 10 | 1 << 15;
        for (i = 0; i < 8; i++) {
            s.rc.color_icw[i] = rnd_in() << 24 | rnd_in() << 16 | rnd_in() << 8 | rnd_in();
            s.rc.alpha_icw[i] = rnd_in() << 24 | rnd_in() << 16 | rnd_in() << 8 | rnd_in();
            s.rc.color_ocw[i] = rnd_ocw(0);
            s.rc.alpha_ocw[i] = rnd_ocw(1);
            s.rc.factor0[i] = rnd();
            s.rc.factor1[i] = rnd();
        }
        if (rnd() & 3) {
            uint32_t fin[4];
            static const uint32_t fregs[] = {0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15};
            for (i = 0; i < 4; i++)
                fin[i] = fregs[rnd() % 14] | (rnd() & 1) << 4 | (rnd() & 1) << 5;
            s.rc.final0 = fin[0] << 24 | fin[1] << 16 | fin[2] << 8 | fin[3];
            s.rc.final1 = (fregs[rnd() % 12] | (rnd() & 1) << 4) << 24
                        | (fregs[rnd() % 12] | (rnd() & 1) << 4) << 16
                        | (fregs[rnd() % 12] | (rnd() & 1) << 4 | (rnd() & 1) << 5) << 8
                        | (rnd() & 0xE0);
        }
        s.rc.final_c0 = rnd();
        s.rc.final_c1 = rnd();
        s.fog_color = rnd();
        for (i = 0; i < 4; i++) {
            s.tex[i].offset = RTEX + (uint32_t)i * 0x10000u;
            s.tex[i].width = s.tex[i].height = 16;
            s.tex[i].color = 0x06;
            s.tex[i].valid = 1;
            s.tex[i].normalized = 1;
            s.tex[i].face_stride = 1024;
        }
        nv2a_rc_unpack(rnd(), d0);
        nv2a_rc_unpack(rnd(), d1);
        fog[0] = (float)(s.fog_color & 0xFF) / 255.0f;
        fog[1] = (float)((s.fog_color >> 8) & 0xFF) / 255.0f;
        fog[2] = (float)((s.fog_color >> 16) & 0xFF) / 255.0f;
        fog[3] = 1.0f;                          /* fog off: factor 1 */

        memset(v, 0, sizeof v);
        for (i = 0; i < 4; i++) {
            float px = (i == 1 || i == 2) ? 16.0f : 0.0f, py = i >= 2 ? 16.0f : 0.0f;
            int k;
            v[i].pos[0] = px; v[i].pos[1] = py;
            v[i].pos[2] = 0.5f * ZMAX; v[i].pos[3] = 1.0f;
            memcpy(v[i].d0, d0, 16);
            memcpy(v[i].d1, d1, 16);
            for (k = 0; k < 4; k++) {
                v[i].tex[k][0] = px / 16.0f;
                v[i].tex[k][1] = py / 16.0f;
                v[i].tex[k][3] = 1.0f;
            }
        }
        s.clear_color = 0;
        be->clear(&s, 0xF3);
        g.prim = 8; g.v = v; g.n = 4;
        be->draw(&s, &g);
        memset(g_mem + SURF, 0xCD, W * 16 * 4);
        be->flip(SURF, W * 4, 4, 16, 16);
        for (y = 0; y < 16; y++)
            for (x = 0; x < 16; x++) {
                float t[4][4], out[4];
                uint32_t want, got = px(x, y);
                int k;
                for (k = 0; k < 4; k++)
                    nv2a_rc_unpack(g_rtex[k][y][x], t[k]);
                nv2a_rc_eval(&s.rc, d0, d1, fog, (const float (*)[4])t, out);
                want = 0;
                for (k = 0; k < 4; k++) {
                    uint32_t b = (uint32_t)(out[k] * 255.0f + 0.5f);
                    want |= b << (k == 3 ? 24 : 16 - 8 * k);
                }
                if (!near_px(got, want, 1)) {
                    if (!bad && bad_progs < 3)
                        printf("FAIL combiner program %d (%d stages, control %08X,"
                               " final %08X %08X): (%d,%d) = %08X want %08X\n",
                               prog, n, s.rc.control, s.rc.final0, s.rc.final1,
                               x, y, got, want);
                    bad++;
                }
            }
        bad_px += bad;
        bad_progs += bad != 0;
    }
    printf("%s combiners: %d of %d programs differ (%d pixels > 1/255)\n",
           bad_progs ? "FAIL" : "ok  ", bad_progs, programs, bad_px);
    return bad_progs;
}

int main(void)
{
    static const float red[4] = {1, 0, 0, 1}, green[4] = {0, 1, 0, 1};
    static const float blue[4] = {0, 0, 1, 1}, white_half[4] = {1, 1, 1, 0.5f};
    Nv2aDrawState s;
    uint32_t n;

    /* Every check reads the frame from guest memory right after its flip.
     * (Set before the backend reads it.) */
#ifdef _WIN32
    _putenv("RECOMP_VK_WRITEBACK=1");
#else
    setenv("RECOMP_VK_WRITEBACK", "1", 1);
#endif
    g_mem = (uint8_t *)calloc(1, MEM_BYTES);
    be = xbox_vk_backend();
    if (!g_mem || !be) {
        printf("FAIL: no backend\n");
        return 1;
    }

    /* Clear. */
    base_state(&s);
    s.clear_color = 0xFF202020u;
    be->clear(&s, 0xF3);
    flip();
    expect("clear", 5, 5, 0xFF202020u, 0);
    expect("clear", 63, 63, 0xFF202020u, 0);

    /* A flat quad on the left half; clip-rect scissor on the right. */
    base_state(&s);
    rect(&s, 0, 0, 32, 64, 0.5f, red, 0);
    s.clip_x = 48; s.clip_w = 8;
    rect(&s, 32, 0, 64, 64, 0.5f, green, 0);
    flip();
    expect("quad", 10, 30, 0xFFFF0000u, 0);
    expect("quad edge", 31, 30, 0xFFFF0000u, 0);
    expect("outside", 40, 30, 0xFF202020u, 0);
    expect("scissor in", 50, 30, 0xFF00FF00u, 0);
    expect("scissor out", 60, 30, 0xFF202020u, 0);

    /* Depth: near red, then green behind it, then blue nearer in a corner. */
    base_state(&s);
    s.clear_color = 0xFF000000u;
    be->clear(&s, 0xF3);
    s.depth_test = 1; s.depth_mask = 1;
    rect(&s, 0, 0, 64, 64, 0.3f, red, 0);
    rect(&s, 0, 0, 64, 64, 0.6f, green, 0);
    rect(&s, 0, 0, 16, 16, 0.1f, blue, 0);
    flip();
    expect("depth hidden", 40, 40, 0xFFFF0000u, 0);
    expect("depth nearer", 8, 8, 0xFF0000FFu, 0);

    /* Visibility: a quad behind counts nothing, an 8x8 one in front 64. */
    be->report_clear();
    s.zpass_enable = 1;
    s.depth_mask = 0;
    rect(&s, 0, 0, 64, 64, 0.9f, green, 0);
    n = report();
    printf("%s report behind = %u\n", n == 0 ? "ok  " : "FAIL", n);
    failures += n != 0;
    be->report_clear();
    rect(&s, 40, 40, 48, 48, 0.05f, green, 0);
    n = report();
    printf("%s report 8x8 = %u\n", n == 64 ? "ok  " : "FAIL", n);
    failures += n != 64;
    s.zpass_enable = 0;

    /* Blending: half-transparent white over black. */
    base_state(&s);
    s.clear_color = 0xFF000000u;
    be->clear(&s, 0xF3);
    s.blend_enable = 1;
    s.blend_sfactor = 0x302; s.blend_dfactor = 0x303;
    rect(&s, 0, 0, 64, 64, 0.5f, white_half, 0);
    flip();
    expect("blend", 20, 20, 0xBF808080u, 1);

    /* Alpha test: GREATER 0x80 rejects an alpha of 0.5. */
    base_state(&s);
    s.clear_color = 0xFF000000u;
    be->clear(&s, 0xF3);
    s.alpha_test = 1; s.alpha_func = 0x204; s.alpha_ref = 0x80;
    rect(&s, 0, 0, 64, 64, 0.5f, white_half, 0);
    s.alpha_ref = 0x7F;
    rect(&s, 0, 0, 8, 8, 0.5f, white_half, 0);
    flip();
    expect("alpha rejected", 20, 20, 0xFF000000u, 0);
    expect("alpha passed", 4, 4, 0x80FFFFFFu, 1);

    /* Texture: a 4x4 checker over the surface, nearest, normalised. */
    base_state(&s);
    s.tex[0].offset = TEXA;
    s.tex[0].width = s.tex[0].height = 4;
    s.tex[0].color = 0x06;
    s.tex[0].valid = 1;
    s.tex[0].normalized = 1;
    s.tex[0].face_stride = 64;
    {
        static const float white[4] = {1, 1, 1, 1};
        rect(&s, 0, 0, 64, 64, 0.5f, white, 1.0f);
    }
    flip();
    expect("texel 0,0", 8, 8, 0xFF00FF00u, 0);
    expect("texel 1,0", 24, 8, 0xFF0000FFu, 0);
    expect("texel 1,1", 24, 24, 0xFF00FF00u, 0);

    /* Render to texture, then the CPU reuses the memory: a target cleared to
     * red is sampled as the GPU left it; once the CPU has rewritten its
     * memory, the texture comes from guest memory (here the checker that
     * nv2a_exec_texel returns) instead of the old image. */
    {
        static const float white[4] = {1, 1, 1, 1};
        const uint32_t rt2 = 0x00280000u;
        base_state(&s);
        s.color_addr = rt2;
        s.clear_color = 0xFFFF0000u;
        be->clear(&s, 0xF0);
        base_state(&s);
        s.tex[0].offset = rt2;
        s.tex[0].width = s.tex[0].height = 4;
        s.tex[0].color = 0x06;
        s.tex[0].valid = 1;
        s.tex[0].normalized = 1;
        s.tex[0].face_stride = 64;
        rect(&s, 0, 0, 64, 64, 0.5f, white, 1.0f);
        flip();
        expect("render target sampled", 8, 8, 0xFFFF0000u, 0);
        memset(g_mem + rt2, 0x5A, W * H * 4);
        rect(&s, 0, 0, 64, 64, 0.5f, white, 1.0f);
        flip();
        expect("rewritten by the CPU", 8, 8, 0xFF00FF00u, 0);
    }

    /* Supersampling switched on while running (the settings API; it applies
     * at a flip): the same picture, and visibility still counts native
     * pixels at 3x. Then back to 1x. */
    {
        XboxVkSettings set;
        xbox_vk_settings_get(&set);
        set.render_scale = 3;
        xbox_vk_settings_set(&set);
        base_state(&s);
        s.clear_color = 0xFF000000u;
        be->clear(&s, 0xF3);
        flip();                                  /* 3x from here */
        base_state(&s);
        s.clear_color = 0xFF000000u;
        be->clear(&s, 0xF3);
        s.depth_test = 1; s.depth_mask = 1;
        rect(&s, 0, 0, 64, 64, 0.3f, red, 0);
        be->report_clear();
        s.zpass_enable = 1;
        s.depth_mask = 0;
        rect(&s, 0, 0, 64, 64, 0.9f, green, 0);  /* behind: 0 */
        n = report();
        printf("%s 3x report behind = %u\n", n == 0 ? "ok  " : "FAIL", n);
        failures += n != 0;
        be->report_clear();
        rect(&s, 40, 40, 48, 48, 0.05f, green, 0);
        n = report();
        printf("%s 3x report 8x8 = %u\n", n == 64 ? "ok  " : "FAIL", n);
        failures += n != 64;
        s.zpass_enable = 0;
        flip();
        expect("3x red", 20, 20, 0xFFFF0000u, 0);
        expect("3x green", 44, 44, 0xFF00FF00u, 0);
        set.render_scale = 1;
        xbox_vk_settings_set(&set);
        flip();                                  /* 1x again */
        xbox_vk_settings_get(&set);
        printf("%s back to scale %u\n", set.render_scale == 1 ? "ok  " : "FAIL",
               set.render_scale);
        failures += set.render_scale != 1;
    }

    /* Culling: back faces with front = CCW; a clockwise (on screen) quad
     * is a back face on a y-down viewport... and a front one when y is up. */
    base_state(&s);
    s.clear_color = 0xFF000000u;
    be->clear(&s, 0xF3);
    s.cull_enable = 1;
    s.vp_scale[1] = -H / 2.0f;
    rect(&s, 0, 0, 32, 64, 0.5f, red, 0);        /* x right, y down: area > 0 */
    s.vp_scale[1] = H / 2.0f;
    rect(&s, 32, 0, 64, 64, 0.5f, red, 0);
    flip();
    {
        /* Software: cw = area > 0; y-up flips; front = CCW ? !cw : cw;
         * BACK culls non-front. Left: cw, not flipped, front=!cw -> back,
         * culled. Right: flipped -> front, drawn. */
        expect("culled (back)", 10, 30, 0xFF000000u, 0);
        expect("drawn (front, y-up)", 50, 30, 0xFFFF0000u, 0);
    }

    failures += combiner_diff(500);
    printf(failures ? "FAILED (%d)\n" : "PASS\n", failures);
    return failures ? 1 : 0;
}
