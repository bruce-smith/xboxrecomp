/**
 * NV2A register combiners. See nv2a_combiner.h.
 *
 * Register numbers (an input or output byte's low nibble):
 *   0 zero/discard  1 c0  2 c1  3 fog  4 v0  5 v1  8-11 t0-t3  12 r0  13 r1
 *   14 v1+r0 sum and 15 E*F product (final combiner only)
 * An input byte is reg | alpha<<4 | mapping<<5. In the RGB portion the alpha
 * bit replicates .a; in the alpha portion it picks .a over .b.
 */
#include "nv2a_combiner.h"

#include <string.h>

enum { R_ZERO = 0, R_C0 = 1, R_C1 = 2, R_FOG = 3, R_V0 = 4, R_V1 = 5,
       R_T0 = 8, R_R0 = 12, R_R1 = 13, R_SUM = 14, R_EF = 15 };

/* x / 255.0f for every byte, built on first use. The executor unpacks every
 * texel it samples, and four float divides per texel were ~14% of the NV2A
 * worker thread in Blinx 2's hub. Same values as the division, bit for bit.
 * Racing initialisers all write identical values, so no lock is needed. */
static float s_unorm8[256];
static volatile int s_unorm8_ready;

void nv2a_rc_unpack(uint32_t c, float o[4])
{
    if (!s_unorm8_ready) {
        int i;
        for (i = 0; i < 256; i++)
            s_unorm8[i] = (float)i / 255.0f;
        s_unorm8_ready = 1;
    }
    o[0] = s_unorm8[(c >> 16) & 0xFF];
    o[1] = s_unorm8[(c >>  8) & 0xFF];
    o[2] = s_unorm8[ c        & 0xFF];
    o[3] = s_unorm8[ c >> 24        ];
}

static float clampf(float x, float lo, float hi)
{
    return x < lo ? lo : (x > hi ? hi : x);
}

/* PS_INPUTMAPPING (psh.c get_input_var) applied to three components with
 * one decision, not one per component: this runs four times per stage per
 * pixel and its switch was a fifth of the rasteriser. */
static void map_in3(const float x[3], uint32_t byte, float v[3])
{
    int i;
    switch ((byte >> 5) & 7) {
    case 0:                                          /* unsigned identity */
        for (i = 0; i < 3; i++) v[i] = x[i] < 0.0f ? 0.0f : x[i];
        break;
    case 1:                                          /* unsigned invert   */
        for (i = 0; i < 3; i++) v[i] = 1.0f - clampf(x[i], 0.0f, 1.0f);
        break;
    case 2:                                          /* expand normal     */
        for (i = 0; i < 3; i++) v[i] = 2.0f * (x[i] < 0.0f ? 0.0f : x[i]) - 1.0f;
        break;
    case 3:                                          /* expand negate     */
        for (i = 0; i < 3; i++) v[i] = -2.0f * (x[i] < 0.0f ? 0.0f : x[i]) + 1.0f;
        break;
    case 4:                                          /* half-bias normal  */
        for (i = 0; i < 3; i++) v[i] = (x[i] < 0.0f ? 0.0f : x[i]) - 0.5f;
        break;
    case 5:                                          /* half-bias negate  */
        for (i = 0; i < 3; i++) v[i] = -(x[i] < 0.0f ? 0.0f : x[i]) + 0.5f;
        break;
    case 6:                                          /* signed identity   */
        for (i = 0; i < 3; i++) v[i] = x[i];
        break;
    default:                                         /* signed negate     */
        for (i = 0; i < 3; i++) v[i] = -x[i];
        break;
    }
}

/* The same for one value (alpha portion). */
static float map_in(float x, uint32_t byte)
{
    float p = x < 0.0f ? 0.0f : x;
    switch ((byte >> 5) & 7) {
    case 0:  return p;
    case 1:  return 1.0f - clampf(x, 0.0f, 1.0f);
    case 2:  return 2.0f * p - 1.0f;
    case 3:  return -2.0f * p + 1.0f;
    case 4:  return p - 0.5f;
    case 5:  return -p + 0.5f;
    case 6:  return x;
    default: return -x;
    }
}

/* PS_COMBINEROUTPUT scale/bias (psh.c get_output) as (x + bias) * scale:
 * 0x08 x-0.5, 0x10 2x, 0x18 2(x-0.5), 0x20 4x, 0x30 x/2. Exact for each. */
static void out_map(uint32_t mapping, float *bias, float *scale)
{
    switch (mapping) {
    case 0x08: *bias = -0.5f; *scale = 1.0f; break;
    case 0x10: *bias =  0.0f; *scale = 2.0f; break;
    case 0x18: *bias = -0.5f; *scale = 2.0f; break;
    case 0x20: *bias =  0.0f; *scale = 4.0f; break;
    case 0x30: *bias =  0.0f; *scale = 0.5f; break;
    default:   *bias =  0.0f; *scale = 1.0f; break;
    }
}

static void read_rgb(float R[16][4], uint32_t byte, float v[3])
{
    const float *r = R[byte & 0xF];
    if (byte & 0x10) {
        float a[3];
        a[0] = a[1] = a[2] = r[3];
        map_in3(a, byte, v);
    } else {
        map_in3(r, byte, v);
    }
}

static float read_alpha(float R[16][4], uint32_t byte)
{
    const float *r = R[byte & 0xF];
    return map_in((byte & 0x10) ? r[3] : r[2], byte);
}

void nv2a_rc_prepare(const Nv2aCombiner *rc, Nv2aCombinerPrep *p)
{
    uint32_t flags = rc->control >> 8, s;

    memset(p, 0, sizeof *p);
    p->rc = *rc;
    /* One c0/c1 for every stage, or one per stage (COMBINERCOUNT). */
    for (s = 0; s < 8; s++) {
        nv2a_rc_unpack(rc->factor0[(flags & 0x010) ? s : 0], p->c0[s]);
        nv2a_rc_unpack(rc->factor1[(flags & 0x100) ? s : 0], p->c1[s]);
    }
    nv2a_rc_unpack(rc->final_c0, p->fc0);
    nv2a_rc_unpack(rc->final_c1, p->fc1);
}

void nv2a_rc_eval(const Nv2aCombiner *rc, const float v0[4],
                  const float v1[4], const float fog[4],
                  const float t[4][4], float out[4])
{
    Nv2aCombinerPrep p;
    nv2a_rc_prepare(rc, &p);
    nv2a_rc_eval_prep(&p, v0, v1, fog, t, out);
}

void nv2a_rc_eval_prep(const Nv2aCombinerPrep *p, const float v0[4],
                       const float v1[4], const float fog[4],
                       const float t[4][4], float out[4])
{
    const Nv2aCombiner *rc = &p->rc;
    float R[16][4];
    uint32_t n = rc->control & 0xFF, flags = rc->control >> 8, s;
    int i;

    if (n > 8)
        n = 8;
    /* Only the registers a program can read before writing need a value:
     * zero, r0, r1 and the final-combiner specials. */
    for (i = 0; i < 4; i++) {
        R[0][i] = R[6][i] = R[7][i] = 0.0f;
        R[12][i] = R[13][i] = R[14][i] = R[15][i] = 0.0f;
    }
    for (i = 0; i < 4; i++) {
        R[R_FOG][i] = fog[i];
        R[R_V0][i] = v0[i];
        R[R_V1][i] = v1[i];
        R[R_T0 + 0][i] = t[0][i];
        R[R_T0 + 1][i] = t[1][i];
        R[R_T0 + 2][i] = t[2][i];
        R[R_T0 + 3][i] = t[3][i];
    }
    /* r0.a starts as texture 0's alpha, or 1 with stage 0 off (psh.c). */
    R[R_R0][3] = (rc->stage_program & 0x1F) ? t[0][3] : 1.0f;

    for (s = 0; s < n; s++) {
        uint32_t icw = rc->color_icw[s], ocw = rc->color_ocw[s];
        uint32_t aicw = rc->alpha_icw[s], aocw = rc->alpha_ocw[s];
        uint32_t fl = ocw >> 12, afl = aocw >> 12;
        float A[3], B[3], C[3], D[3], ab[3], cd[3], ms[3];
        float aA, aB, aC, aD, aab, acd, ams, bias, scale, dab, dcd;
        int mux_cd;

        memcpy(R[R_C0], p->c0[s], 16);
        memcpy(R[R_C1], p->c1[s], 16);
        mux_cd = (flags & 1) ? R[R_R0][3] >= 0.5f
                             : ((int)(R[R_R0][3] * 255.0f) & 1);

        /* Both portions read before either writes (psh.c emits all the
         * arithmetic, then the assignments). */
        read_rgb(R, icw >> 24, A); read_rgb(R, icw >> 16, B);
        read_rgb(R, icw >> 8, C);  read_rgb(R, icw, D);
        aA = read_alpha(R, aicw >> 24); aB = read_alpha(R, aicw >> 16);
        aC = read_alpha(R, aicw >> 8);  aD = read_alpha(R, aicw);

        out_map(fl & 0x38, &bias, &scale);
        dab = A[0] * B[0] + A[1] * B[1] + A[2] * B[2];
        dcd = C[0] * D[0] + C[1] * D[1] + C[2] * D[2];
        for (i = 0; i < 3; i++) {
            float pab = (fl & 2) ? dab : A[i] * B[i];
            float pcd = (fl & 1) ? dcd : C[i] * D[i];
            float sum = (fl & 4) ? (mux_cd ? pcd : pab) : pab + pcd;
            ab[i] = clampf((pab + bias) * scale, -1.0f, 1.0f);
            cd[i] = clampf((pcd + bias) * scale, -1.0f, 1.0f);
            ms[i] = clampf((sum + bias) * scale, -1.0f, 1.0f);
        }
        out_map(afl & 0x38, &bias, &scale);
        {
            float pab = aA * aB, pcd = aC * aD;
            float sum = (afl & 4) ? (mux_cd ? pcd : pab) : pab + pcd;
            aab = clampf((pab + bias) * scale, -1.0f, 1.0f);
            acd = clampf((pcd + bias) * scale, -1.0f, 1.0f);
            ams = clampf((sum + bias) * scale, -1.0f, 1.0f);
        }

        /* RGB destinations; blue-to-alpha also writes the destination's .a. */
        if ((ocw >> 4) & 0xF) {
            float *d = R[(ocw >> 4) & 0xF];
            d[0] = ab[0]; d[1] = ab[1]; d[2] = ab[2];
            if (fl & 0x80) d[3] = ab[2];
        }
        if (ocw & 0xF) {
            float *d = R[ocw & 0xF];
            d[0] = cd[0]; d[1] = cd[1]; d[2] = cd[2];
            if (fl & 0x40) d[3] = cd[2];
        }
        if ((ocw >> 8) & 0xF) {
            float *d = R[(ocw >> 8) & 0xF];
            d[0] = ms[0]; d[1] = ms[1]; d[2] = ms[2];
        }
        if ((aocw >> 4) & 0xF) R[(aocw >> 4) & 0xF][3] = aab;
        if (aocw & 0xF)        R[aocw & 0xF][3] = acd;
        if ((aocw >> 8) & 0xF) R[(aocw >> 8) & 0xF][3] = ams;
        /* Register 0 is zero whatever was "written" to it. */
        R[R_ZERO][0] = R[R_ZERO][1] = R[R_ZERO][2] = R[R_ZERO][3] = 0.0f;
    }

    if (rc->final0 || rc->final1) {
        uint32_t f0 = rc->final0, f1 = rc->final1, fflags = f1 & 0xFF;
        float A[3], B[3], C[3], D[3], E[3], F[3];

        memcpy(R[R_C0], p->fc0, 16);
        memcpy(R[R_C1], p->fc1, 16);
        /* V1R0 sum: optional complements (0x40 v1, 0x20 r0), clamp (0x80). */
        for (i = 0; i < 3; i++) {
            float a = (fflags & 0x40) ? 1.0f - R[R_V1][i] : R[R_V1][i];
            float b = (fflags & 0x20) ? 1.0f - R[R_R0][i] : R[R_R0][i];
            R[R_SUM][i] = (fflags & 0x80) ? clampf(a + b, 0.0f, 1.0f) : a + b;
        }
        R[R_SUM][3] = 0.0f;
        read_rgb(R, f1 >> 24, E);
        read_rgb(R, f1 >> 16, F);
        for (i = 0; i < 3; i++)
            R[R_EF][i] = E[i] * F[i];
        R[R_EF][3] = 0.0f;
        read_rgb(R, f0 >> 24, A); read_rgb(R, f0 >> 16, B);
        read_rgb(R, f0 >> 8, C);  read_rgb(R, f0, D);
        /* D + mix(C, B, A), alpha from G. */
        for (i = 0; i < 3; i++)
            out[i] = D[i] + C[i] * (1.0f - A[i]) + B[i] * A[i];
        out[3] = read_alpha(R, f1 >> 8);
    } else {
        for (i = 0; i < 4; i++)
            out[i] = R[R_R0][i];
    }
    for (i = 0; i < 4; i++)
        out[i] = clampf(out[i], 0.0f, 1.0f);
}
