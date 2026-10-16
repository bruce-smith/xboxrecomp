/*
 * recomp_math.c -- the x87 transcendentals of lifted code.
 *
 * The lifter turns fsin/fcos/fsincos/fptan/fpatan/fyl2x/fyl2xp1/f2xm1 into
 * calls to the functions below. By default they are the host C library's
 * sin/cos/tan/atan2/log2/exp2, exactly what the lifted code called before.
 * Those are not the same function everywhere: MSVC's CRT and glibc disagree
 * in the last bit for some arguments, and one bit is enough for two runs of
 * the same input to part. Blinx 2 builds a cosine table at boot (its IDCT)
 * whose off-by-one entries changed every later frame between Windows and
 * Linux, so an exact replay could not cross operating systems and a network
 * game between them would desync.
 *
 * RECOMP_PORTABLE_MATH=1 selects one implementation instead, written here in
 * plain IEEE double arithmetic -- additions, multiplications, divisions and
 * exact conversions in a fixed order, no library calls, no fused
 * multiply-add -- so every conforming compiler gives the same bits. The
 * algorithms are fdlibm's (Sun Microsystems, freely redistributable; see the
 * notice below): sin/cos/tan through a Cody-Waite reduction by pi/2 and
 * fdlibm's kernels, atan2 through fdlibm's atan, log2 through log, exp2
 * through exp. Accuracy is within a couple of ulps, about what the x87 gave;
 * determinism is the point.
 *
 * The choice is fixed before any guest code runs: the environment, a
 * schedule being replayed (which records it), or a network session
 * (recomp_set_portable_math). Off by default, so existing recordings replay
 * as they were made.
 *
 * ====================================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================================
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ---- the switch ------------------------------------------------------- */

static volatile int s_portable = -1;

int recomp_portable_math_enabled(void)
{
    int v = s_portable;
    if (v < 0) {
        const char *e = getenv("RECOMP_PORTABLE_MATH");
        v = (e && *e && *e != '0') ? 1 : 0;
        s_portable = v;
    }
    return v;
}

void recomp_set_portable_math(int on)
{
    s_portable = on ? 1 : 0;
}

/* ---- bits ------------------------------------------------------------- */

static uint32_t hi_of(double x)
{
    uint64_t u;
    memcpy(&u, &x, 8);
    return (uint32_t)(u >> 32);
}

static uint32_t lo_of(double x)
{
    uint64_t u;
    memcpy(&u, &x, 8);
    return (uint32_t)u;
}

static double from_words(uint32_t hi, uint32_t lo)
{
    uint64_t u = ((uint64_t)hi << 32) | lo;
    double d;
    memcpy(&d, &u, 8);
    return d;
}

static double with_hi(double x, uint32_t hi)
{
    return from_words(hi, lo_of(x));
}

static double pm_fabs(double x)
{
    return from_words(hi_of(x) & 0x7fffffffu, lo_of(x));
}

static double pm_nan(void)   { return from_words(0x7ff80000u, 0); }
static double pm_inf(void)   { return from_words(0x7ff00000u, 0); }

/* ---- sin, cos, tan ---------------------------------------------------- */

static const double
    S1 = -1.66666666666666324348e-01,
    S2 =  8.33333333332248946124e-03,
    S3 = -1.98412698298579493134e-04,
    S4 =  2.75573137070700676789e-06,
    S5 = -2.50507602534068634195e-08,
    S6 =  1.58969099521155010221e-10,
    C1 =  4.16666666666666019037e-02,
    C2 = -1.38888888888741095749e-03,
    C3 =  2.48015872894767294178e-05,
    C4 = -2.75573143513906633035e-07,
    C5 =  2.08757232129817482790e-09,
    C6 = -1.13596475577881948265e-11;

/* sin(x + y) for |x| <= pi/4, y the tail of x; iy == 0: y is zero. */
static double k_sin(double x, double y, int iy)
{
    double z = x * x, v = z * x;
    double r = S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)));
    if (iy == 0)
        return x + v * (S1 + z * r);
    return x - ((z * (0.5 * y - v * r) - y) - v * S1);
}

static double k_cos(double x, double y)
{
    uint32_t ix = hi_of(x) & 0x7fffffffu;
    double z = x * x, r, qx, hz, a;
    r = z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6)))));
    if (ix < 0x3fd33333u)                       /* |x| < 0.3 */
        return 1.0 - (0.5 * z - (z * r - x * y));
    if (ix > 0x3fe90000u)                       /* |x| > 0.78125 */
        qx = 0.28125;
    else
        qx = from_words(ix - 0x00200000u, 0);   /* x/4 */
    hz = 0.5 * z - qx;
    a = 1.0 - qx;
    return a - (hz - (z * r - x * y));
}

static const double
    invpio2 = 6.36619772367581382433e-01,
    pio2_1  = 1.57079632673412561417e+00,
    pio2_1t = 6.07710050650619224932e-11,
    pio2_2  = 6.07710050630396597660e-11,
    pio2_2t = 2.02226624879595063154e-21,
    pio2_3  = 2.02226624871116645580e-21,
    pio2_3t = 8.47842766036889956997e-32,
    two_pi  = 6.28318530717958623200e+00;

/* x = n*pi/2 + y[0] + y[1]; returns n. fdlibm's reduction for |x| up to
 * 2^20*pi/2 (three-part pi/2); larger arguments -- no game angle -- are
 * first folded by an exact fmod into one turn, deterministic but losing the
 * digits a full Payne-Hanek reduction would keep. */
static int rem_pio2(double x, double y[2])
{
    uint32_t hx = hi_of(x), ix = hx & 0x7fffffffu;
    double ax = pm_fabs(x), fn, r, w, t;
    int n, j, i;

    if (ix <= 0x3fe921fbu) {                    /* |x| <= pi/4 */
        y[0] = x;
        y[1] = 0;
        return 0;
    }
    if (ix >= 0x413921fbu) {                    /* |x| >= 2^20*pi/2 */
        ax = fmod(ax, two_pi);
        ix = hi_of(ax) & 0x7fffffffu;
    }
    n = (int)(ax * invpio2 + 0.5);
    fn = (double)n;
    r = ax - fn * pio2_1;
    w = fn * pio2_1t;
    j = (int)(ix >> 20);
    y[0] = r - w;
    i = j - (int)((hi_of(y[0]) >> 20) & 0x7ff);
    if (i > 16) {                               /* second iteration */
        t = r;
        w = fn * pio2_2;
        r = t - w;
        w = fn * pio2_2t - ((t - r) - w);
        y[0] = r - w;
        i = j - (int)((hi_of(y[0]) >> 20) & 0x7ff);
        if (i > 49) {                           /* third */
            t = r;
            w = fn * pio2_3;
            r = t - w;
            w = fn * pio2_3t - ((t - r) - w);
            y[0] = r - w;
        }
    }
    y[1] = (r - y[0]) - w;
    if ((int32_t)hx < 0) {
        y[0] = -y[0];
        y[1] = -y[1];
        return -n;
    }
    return n;
}

static double pm_sin(double x)
{
    uint32_t ix = hi_of(x) & 0x7fffffffu;
    double y[2];
    int n;
    if (ix <= 0x3fe921fbu)
        return ix < 0x3e400000u ? x : k_sin(x, 0.0, 0);
    if (ix >= 0x7ff00000u)
        return pm_nan();
    n = rem_pio2(x, y);
    switch (n & 3) {
    case 0:  return  k_sin(y[0], y[1], 1);
    case 1:  return  k_cos(y[0], y[1]);
    case 2:  return -k_sin(y[0], y[1], 1);
    default: return -k_cos(y[0], y[1]);
    }
}

static double pm_cos(double x)
{
    uint32_t ix = hi_of(x) & 0x7fffffffu;
    double y[2];
    int n;
    if (ix <= 0x3fe921fbu)
        return ix < 0x3e400000u ? 1.0 : k_cos(x, 0.0);
    if (ix >= 0x7ff00000u)
        return pm_nan();
    n = rem_pio2(x, y);
    switch (n & 3) {
    case 0:  return  k_cos(y[0], y[1]);
    case 1:  return -k_sin(y[0], y[1], 1);
    case 2:  return -k_cos(y[0], y[1]);
    default: return  k_sin(y[0], y[1], 1);
    }
}

static double pm_tan(double x)
{
    uint32_t ix = hi_of(x) & 0x7fffffffu;
    double y[2], s, c;
    int n;
    if (ix < 0x3e400000u)
        return x;
    if (ix >= 0x7ff00000u)
        return pm_nan();
    n = rem_pio2(x, y);
    s = k_sin(y[0], y[1], 1);
    c = k_cos(y[0], y[1]);
    return (n & 1) ? -c / s : s / c;
}

/* ---- atan, atan2 ------------------------------------------------------ */

static const double atanhi[] = {
    4.63647609000806093515e-01, 7.85398163397448278999e-01,
    9.82793723247329054082e-01, 1.57079632679489655800e+00,
};
static const double atanlo[] = {
    2.26987774529616870924e-17, 3.06161699786838301793e-17,
    1.39033110312309984516e-17, 6.12323399573676603587e-17,
};
static const double aT[] = {
     3.33333333333329318027e-01, -1.99999999998764832476e-01,
     1.42857142725034663711e-01, -1.11111104054623557880e-01,
     9.09088713343650656196e-02, -7.69187620504482999495e-02,
     6.66107313738753120669e-02, -5.83357013379057348645e-02,
     4.97687799461593236017e-02, -3.65315727442169155270e-02,
     1.62858201153657823623e-02,
};

static double pm_atan(double x)
{
    uint32_t hx = hi_of(x), ix = hx & 0x7fffffffu;
    double w, s1, s2, z;
    int id;

    if (ix >= 0x44100000u) {                    /* |x| >= 2^66 */
        if (ix > 0x7ff00000u || (ix == 0x7ff00000u && lo_of(x) != 0))
            return x + x;                       /* NaN */
        return (int32_t)hx > 0 ? atanhi[3] + atanlo[3] : -atanhi[3] - atanlo[3];
    }
    if (ix < 0x3fdc0000u) {                     /* |x| < 0.4375 */
        if (ix < 0x3e400000u)
            return x;
        id = -1;
    } else {
        x = pm_fabs(x);
        if (ix < 0x3ff30000u) {                 /* |x| < 1.1875 */
            if (ix < 0x3fe60000u) { id = 0; x = (2.0 * x - 1.0) / (2.0 + x); }
            else                  { id = 1; x = (x - 1.0) / (x + 1.0); }
        } else {
            if (ix < 0x40038000u) { id = 2; x = (x - 1.5) / (1.0 + 1.5 * x); }
            else                  { id = 3; x = -1.0 / x; }
        }
    }
    z = x * x;
    w = z * z;
    s1 = z * (aT[0] + w * (aT[2] + w * (aT[4] + w * (aT[6] + w * (aT[8] + w * aT[10])))));
    s2 = w * (aT[1] + w * (aT[3] + w * (aT[5] + w * (aT[7] + w * aT[9]))));
    if (id < 0)
        return x - x * (s1 + s2);
    z = atanhi[id] - ((x * (s1 + s2) - atanlo[id]) - x);
    return (int32_t)hx < 0 ? -z : z;
}

static const double
    pi     = 3.1415926535897931160e+00,
    pi_lo  = 1.2246467991473531772e-16,
    pi_o_2 = 1.5707963267948965580e+00,
    pi_o_4 = 7.8539816339744827900e-01;

static int pm_isnan(double x)
{
    uint32_t ix = hi_of(x) & 0x7fffffffu;
    return ix > 0x7ff00000u || (ix == 0x7ff00000u && lo_of(x) != 0);
}

static double pm_atan2(double y, double x)
{
    uint32_t hx = hi_of(x), hy = hi_of(y);
    uint32_t ix = hx & 0x7fffffffu, iy = hy & 0x7fffffffu;
    int m, k;
    double z;

    if (pm_isnan(x) || pm_isnan(y))
        return x + y;
    if (x == 1.0)
        return pm_atan(y);
    m = (int)(((hy >> 31) & 1) | ((hx >> 30) & 2));
    if ((iy | lo_of(y)) == 0) {                 /* y = +-0 */
        switch (m) {
        case 0: case 1: return y;
        case 2:         return pi;
        default:        return -pi;
        }
    }
    if ((ix | lo_of(x)) == 0)                   /* x = +-0 */
        return (int32_t)hy < 0 ? -pi_o_2 : pi_o_2;
    if (ix == 0x7ff00000u) {                    /* x = +-inf */
        if (iy == 0x7ff00000u) {
            switch (m) {
            case 0:  return  pi_o_4;
            case 1:  return -pi_o_4;
            case 2:  return  3.0 * pi_o_4;
            default: return -3.0 * pi_o_4;
            }
        }
        switch (m) {
        case 0:  return  0.0;
        case 1:  return -0.0;
        case 2:  return  pi;
        default: return -pi;
        }
    }
    if (iy == 0x7ff00000u)                      /* y = +-inf */
        return (int32_t)hy < 0 ? -pi_o_2 : pi_o_2;
    k = ((int)iy - (int)ix) >> 20;
    if (k > 60)
        z = pi_o_2 + 0.5 * pi_lo;
    else if ((int32_t)hx < 0 && k < -60)
        z = 0.0;
    else
        z = pm_atan(pm_fabs(y / x));
    switch (m) {
    case 0:  return z;
    case 1:  return -z;
    case 2:  return pi - (z - pi_lo);
    default: return (z - pi_lo) - pi;
    }
}

/* ---- log2 ------------------------------------------------------------- */

static const double
    ln2_hi = 6.93147180369123816490e-01,
    ln2_lo = 1.90821492927058770002e-10,
    two54  = 1.80143985094819840000e+16,
    ivln2  = 1.44269504088896338700e+00,
    Lg1 = 6.666666666666735130e-01,
    Lg2 = 3.999999999940941908e-01,
    Lg3 = 2.857142874366239149e-01,
    Lg4 = 2.222219843214978396e-01,
    Lg5 = 1.818357216161805012e-01,
    Lg6 = 1.531383769920937332e-01,
    Lg7 = 1.479819860511658591e-01;

static double pm_log(double x)
{
    double hfsq, f, s, z, R, w, t1, t2, dk;
    int32_t hx = (int32_t)hi_of(x), i, j;
    uint32_t lx = lo_of(x);
    int k = 0;

    if (hx < 0x00100000) {                      /* x < 2^-1022 */
        if (((hx & 0x7fffffff) | (int32_t)lx) == 0)
            return -pm_inf();                   /* log(+-0) */
        if (hx < 0)
            return pm_nan();                    /* log(-x) */
        k -= 54;
        x *= two54;                             /* subnormal: scale up */
        hx = (int32_t)hi_of(x);
    }
    if (hx >= 0x7ff00000)
        return x + x;
    k += (hx >> 20) - 1023;
    hx &= 0x000fffff;
    i = (hx + 0x95f64) & 0x100000;
    x = with_hi(x, (uint32_t)(hx | (i ^ 0x3ff00000)));   /* x or x/2 */
    k += (i >> 20);
    f = x - 1.0;
    if ((0x000fffff & (2 + hx)) < 3) {          /* -2^-20 <= f < 2^-20 */
        if (f == 0.0) {
            if (k == 0)
                return 0.0;
            dk = (double)k;
            return dk * ln2_hi + dk * ln2_lo;
        }
        R = f * f * (0.5 - 0.33333333333333333 * f);
        if (k == 0)
            return f - R;
        dk = (double)k;
        return dk * ln2_hi - ((R - dk * ln2_lo) - f);
    }
    s = f / (2.0 + f);
    dk = (double)k;
    z = s * s;
    i = hx - 0x6147a;
    w = z * z;
    j = 0x6b851 - hx;
    t1 = w * (Lg2 + w * (Lg4 + w * Lg6));
    t2 = z * (Lg1 + w * (Lg3 + w * (Lg5 + w * Lg7)));
    i |= j;
    R = t2 + t1;
    if (i > 0) {
        hfsq = 0.5 * f * f;
        if (k == 0)
            return f - (hfsq - s * (hfsq + R));
        return dk * ln2_hi - ((hfsq - (s * (hfsq + R) + dk * ln2_lo)) - f);
    }
    if (k == 0)
        return f - s * (f - R);
    return dk * ln2_hi - ((s * (f - R) - dk * ln2_lo) - f);
}

static double pm_log2(double x)
{
    uint32_t hx = hi_of(x);
    /* An exact power of two has an exact logarithm. */
    if ((hx & 0x000fffffu) == 0 && lo_of(x) == 0
        && hx >= 0x00100000u && hx < 0x7ff00000u)
        return (double)((int)(hx >> 20) - 1023);
    return pm_log(x) * ivln2;
}

/* ---- exp2 ------------------------------------------------------------- */

static const double
    halF[2]  = { 0.5, -0.5 },
    o_threshold = 7.09782712893383973096e+02,
    u_threshold = -7.45133219101941108420e+02,
    ln2HI[2] = { 6.93147180369123816490e-01, -6.93147180369123816490e-01 },
    ln2LO[2] = { 1.90821492927058770002e-10, -1.90821492927058770002e-10 },
    invln2   = 1.44269504088896338700e+00,
    ln2      = 6.93147180559945286227e-01,
    twom1000 = 9.33263618503218878990e-302,
    P1 =  1.66666666666666019037e-01,
    P2 = -2.77777777770155933842e-03,
    P3 =  6.61375632143793436117e-05,
    P4 = -1.65339022054652515390e-06,
    P5 =  4.13813679705723846039e-08;

static double pm_exp(double x)
{
    double y, hi = 0.0, lo = 0.0, c, t;
    int32_t k = 0, xsb;
    uint32_t hx = hi_of(x);

    xsb = (int32_t)((hx >> 31) & 1);
    hx &= 0x7fffffffu;
    if (hx >= 0x40862E42u) {                    /* |x| >= 709.78 */
        if (hx >= 0x7ff00000u) {
            if (((hx & 0xfffff) | lo_of(x)) != 0)
                return x + x;                   /* NaN */
            return xsb == 0 ? x : 0.0;          /* exp(+-inf) */
        }
        if (x > o_threshold)
            return pm_inf();
        if (x < u_threshold)
            return 0.0;
    }
    if (hx > 0x3fd62e42u) {                     /* |x| > 0.5 ln2 */
        if (hx < 0x3FF0A2B2u) {                 /* and < 1.5 ln2 */
            hi = x - ln2HI[xsb];
            lo = ln2LO[xsb];
            k = 1 - xsb - xsb;
        } else {
            k = (int32_t)(invln2 * x + halF[xsb]);
            t = (double)k;
            hi = x - t * ln2HI[0];
            lo = t * ln2LO[0];
        }
        x = hi - lo;
    } else if (hx < 0x3e300000u) {              /* |x| < 2^-28 */
        return 1.0 + x;
    } else {
        k = 0;
    }
    t = x * x;
    c = x - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
    if (k == 0)
        return 1.0 - ((x * c) / (c - 2.0) - x);
    y = 1.0 - ((lo - (x * c) / (2.0 - c)) - hi);
    if (k >= -1021)
        return with_hi(y, hi_of(y) + ((uint32_t)k << 20));
    return with_hi(y, hi_of(y) + ((uint32_t)(k + 1000) << 20)) * twom1000;
}

/* 2^n for -1022 <= n <= 1023, exactly. */
static double pow2i(int n)
{
    return from_words((uint32_t)(n + 1023) << 20, 0);
}

static double pm_exp2(double x)
{
    double f, y;
    int n;

    if (pm_isnan(x))
        return x + x;
    if (x >= 1024.0)
        return pm_inf();
    if (x < -1075.0)
        return 0.0;
    /* x = n + f, |f| <= 0.5, both exact: 2^x = 2^n * e^(f ln2). Rounding
     * f*ln2 costs a fraction of an ulp, where x*ln2 cost up to |x| ulps. */
    n = (int)(x < 0.0 ? x - 0.5 : x + 0.5);
    f = x - (double)n;
    y = pm_exp(f * ln2);
    if (n >= -1022)
        return n <= 1023 ? y * pow2i(n) : (y * pow2i(1023)) * pow2i(n - 1023);
    return (y * pow2i(-1022)) * pow2i(n + 1022 < -1022 ? -1022 : n + 1022);
}

/* ---- what the lifted code calls --------------------------------------- */

double recomp_fsin(double x)
{
    return recomp_portable_math_enabled() ? pm_sin(x) : sin(x);
}

double recomp_fcos(double x)
{
    return recomp_portable_math_enabled() ? pm_cos(x) : cos(x);
}

double recomp_ftan(double x)
{
    return recomp_portable_math_enabled() ? pm_tan(x) : tan(x);
}

double recomp_fatan2(double y, double x)
{
    return recomp_portable_math_enabled() ? pm_atan2(y, x) : atan2(y, x);
}

double recomp_flog2(double x)
{
    return recomp_portable_math_enabled() ? pm_log2(x) : log2(x);
}

double recomp_fexp2(double x)
{
    return recomp_portable_math_enabled() ? pm_exp2(x) : exp2(x);
}
