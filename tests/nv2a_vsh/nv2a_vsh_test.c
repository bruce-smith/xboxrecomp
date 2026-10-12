/* The Xbox D3D screen-space epilogue, as every title's programs end:
 *
 *   mov o[0], v0                      position (stand-in for the DP4s)
 *   mul o[0].xyz, R12, c[58]  + rcc R1.x, R12.w
 *   mad o[0].xyz, R12, R1.x, c[59]
 *
 * The rcc is paired with a MAC op that writes no temp, and must still land
 * in R1; sent to the instruction's temp field instead, R1.x stayed 0 and
 * every vertex came out at c[59]. */
#include "nv2a_vsh_interp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { T = 1, V = 2, C = 3 };             /* source mux */
#define XYZW 0x1B                         /* swizzle x,y,z,w */
#define WWWW 0xFF
#define XXXX 0x00

static void src(uint32_t w[4], int which, int mux, int reg, int swz)
{
    if (which == 0) { w[1] |= swz; w[2] |= (mux << 26) | (reg << 28); }
    else if (which == 1) { w[2] |= (swz << 17) | (mux << 11) | (reg << 13); }
    else { w[2] |= (swz << 2) | (reg >> 2); w[3] |= (mux << 28) | ((reg & 3) << 30); }
}

/* nv2a_vsh_run4 must match nv2a_vsh_run bit for bit (NaN payloads aside):
 * random programs over every MAC and ILU op, with inputs and constants that
 * include zeros of both signs, infinities, NaN and extremes. */
static uint32_t rng = 12345u;
static uint32_t rnd(void) { rng = rng * 1664525u + 1013904223u; return rng >> 8; }

static float special(void)
{
    static const float k[] = { 0.0f, -0.0f, 1.0f, -1.0f, 0.5f, 1e-30f, -1e-30f,
                               1e30f, -1e30f, 3.0e38f, 1.0e-45f };
    uint32_t r = rnd() % 16;
    if (r < sizeof k / sizeof k[0]) return k[r];
    if (r == 11) return INFINITY;
    if (r == 12) return -INFINITY;
    if (r == 13) return NAN;
    return ((float)(int)(rnd() % 20001) - 10000.0f) / 100.0f;
}

static int same(float a, float b)
{
    uint32_t x, y;
    if (a != a && b != b) return 1;
    memcpy(&x, &a, 4); memcpy(&y, &b, 4);
    return x == y;
}

static int random_programs(void)
{
    int prog, bad = 0;
    for (prog = 0; prog < 20000 && bad < 5; prog++) {
        int n = 1 + (int)(rnd() % 24), i, l, k;
        float in[4][NV2A_VSH_INPUTS][4];
        const float (*ip[4])[4];
        Nv2aVshOutput o1[4], o4[4], *op[4];
        for (i = 0; i < NV2A_VSH_CONSTANTS; i++) {
            float c[4];
            for (k = 0; k < 4; k++) c[k] = special();
            nv2a_vsh_set_constant((uint32_t)i, c);
        }
        for (i = 0; i < n; i++) {
            uint32_t w[4] = {0, 0, 0, 0};
            int s;
            w[1] = ((rnd() % 8) << 25) | ((rnd() % 14) << 21)
                 | ((rnd() % NV2A_VSH_CONSTANTS) << 13) | ((rnd() % 16) << 9)
                 | ((rnd() & 1) << 8);
            for (s = 0; s < 3; s++)
                src(w, s, 1 + (int)(rnd() % 3), (int)(rnd() % 13), (int)(rnd() % 256));
            w[2] |= ((rnd() & 1) << 25) | ((rnd() & 1) << 10);
            w[3] |= ((rnd() % 16) << 24) | ((rnd() % 13) << 20) | ((rnd() % 16) << 16)
                  | ((rnd() % 16) << 12) | ((rnd() & 1) << 11)
                  | ((rnd() % 13) << 3) | ((rnd() & 1) << 2) | (rnd() & 2);
            if (i == n - 1) w[3] |= 1u;
            nv2a_vsh_set_instruction((uint32_t)i, w);
        }
        for (l = 0; l < 4; l++) {
            for (i = 0; i < NV2A_VSH_INPUTS; i++)
                for (k = 0; k < 4; k++) in[l][i][k] = special();
            ip[l] = (const float (*)[4])in[l];
            op[l] = &o4[l];
            nv2a_vsh_run(ip[l], &o1[l]);
        }
        nv2a_vsh_run4(ip, op);
        for (l = 0; l < 4; l++) {
            const float *a = (const float *)&o1[l], *b = (const float *)&o4[l];
            for (k = 0; k < (int)(sizeof(Nv2aVshOutput) / sizeof(float)); k++)
                if (!same(a[k], b[k])) {
                    printf("FAIL: program %d lane %d float %d: %g vs %g%c",
                           prog, l, k, a[k], b[k], 10);
                    bad++;
                    break;
                }
        }
    }
    return bad;
}

int main(void)
{
    uint32_t i0[4] = {0}, i1[4] = {0}, i2[4] = {0};
    const float c58[4] = {320, -240, 1000, 0}, c59[4] = {320, 240, 0, 0};
    float in[NV2A_VSH_INPUTS][4] = {{0}};
    Nv2aVshOutput o;

    /* mov o[0].xyzw, v0 */
    i0[1] = (1u << 21);                         /* MAC mov, input v0 */
    src(i0, 0, V, 0, XYZW);
    i0[3] |= (0xFu << 12) | (1u << 11);         /* o[0] mask xyzw */

    /* mul o[0].xyz, R12, c[58] ; rcc R1.x, R12.w (temp field says R7) */
    i1[1] = (3u << 25) | (2u << 21) | (58u << 13);
    src(i1, 0, T, 12, XYZW); src(i1, 1, C, 0, XYZW); src(i1, 2, T, 12, WWWW);
    i1[3] |= (7u << 20) | (8u << 16) | (0xEu << 12) | (1u << 11);

    /* mad o[0].xyz, R12, R1.x, c[59] ; final */
    i2[1] = (4u << 21) | (59u << 13);
    src(i2, 0, T, 12, XYZW); src(i2, 1, T, 1, XXXX); src(i2, 2, C, 0, XYZW);
    i2[3] |= (0xEu << 12) | (1u << 11) | 1u;

    nv2a_vsh_set_instruction(0, i0);
    nv2a_vsh_set_instruction(1, i1);
    nv2a_vsh_set_instruction(2, i2);
    nv2a_vsh_set_constant(58, c58);
    nv2a_vsh_set_constant(59, c59);

    in[0][0] = 1.0f; in[0][1] = 0.5f; in[0][2] = 0.25f; in[0][3] = 2.0f;
    if (!nv2a_vsh_run((const float (*)[4])in, &o)) {
        puts("FAIL: program did not run");
        return 1;
    }
    /* x = 1*320/2 + 320, y = 0.5*-240/2 + 240, z = 0.25*1000/2, w kept */
    if (fabsf(o.pos[0] - 480) > 1e-3f || fabsf(o.pos[1] - 180) > 1e-3f
        || fabsf(o.pos[2] - 125) > 1e-3f || o.pos[3] != 2.0f) {
        printf("FAIL: pos %g %g %g %g, want 480 180 125 2\n",
               o.pos[0], o.pos[1], o.pos[2], o.pos[3]);
        return 1;
    }
    if (random_programs())
        return 1;
    puts("ok");
    return 0;
}
