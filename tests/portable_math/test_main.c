/*
 * RECOMP_PORTABLE_MATH: the same bits under every compiler.
 *
 * Evaluates every portable transcendental (src/kernel/recomp_math.c) on a
 * fixed set of 2.4 million arguments -- game-sized angles, the edges of the
 * reduction ranges, logarithm and exponent ranges, special values -- and
 * prints one FNV hash of all the result bits. The hash is the test: it must
 * be identical for an MSVC build on Windows and a gcc or clang build on
 * Linux (EXPECTED_HASH below, or compare two runs' output). It also checks
 * the accuracy against the host's libm (a few ulps), so a typo in a constant
 * cannot hide behind determinism.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int    recomp_portable_math_enabled(void);
void   recomp_set_portable_math(int on);
double recomp_fsin(double x);
double recomp_fcos(double x);
double recomp_ftan(double x);
double recomp_fatan2(double y, double x);
double recomp_flog2(double x);
double recomp_fexp2(double x);

static uint64_t s_hash = 1469598103934665603ull;
static uint64_t s_rng = 0x9E3779B97F4A7C15ull;

static void mix(double d)
{
    uint64_t u;
    int i;
    memcpy(&u, &d, 8);
    for (i = 0; i < 8; i++) {
        s_hash ^= (u >> (8 * i)) & 0xFF;
        s_hash *= 1099511628211ull;
    }
}

/* A uniform double in [lo, hi), the same sequence everywhere. */
static double uni(double lo, double hi)
{
    s_rng ^= s_rng << 13;
    s_rng ^= s_rng >> 7;
    s_rng ^= s_rng << 17;
    return lo + (hi - lo) * ((double)(s_rng >> 11) * (1.0 / 9007199254740992.0));
}

static int64_t ulps(double a, double b)
{
    int64_t ia, ib;
    if (a == b || (a != a && b != b))
        return 0;
    memcpy(&ia, &a, 8);
    memcpy(&ib, &b, 8);
    if (ia < 0) ia = INT64_MIN - ia;
    if (ib < 0) ib = INT64_MIN - ib;
    return ia > ib ? ia - ib : ib - ia;
}

static int64_t s_worst[6];
static const char *const s_name[6] = { "sin", "cos", "tan", "atan2", "log2", "exp2" };

static void check(int f, double got, double ref)
{
    int64_t u = ulps(got, ref);
    /* Near a zero of the function (tan, atan2 and log2 around 0, 0 and 1)
     * a tiny absolute difference is many ulps; judge those absolutely. */
    if (fabs(ref) < 1e-6 && fabs(got - ref) < 1e-21)
        u = 0;
    if (u > s_worst[f])
        s_worst[f] = u;
    mix(got);
}

int main(void)
{
    static const double specials[] = {
        0.0, -0.0, 1.0, -1.0, 0.5, 2.0, 1e-300, -1e-300, 1e300, -1e300,
        3.141592653589793, 1.5707963267948966, 0.7853981633974483,
        6.283185307179586, 100.0, 1e6, 1e7, 4.9e-324,
    };
    size_t i;
    int k, fail = 0;
    const int N = 400000;

    recomp_set_portable_math(1);
    if (!recomp_portable_math_enabled()) {
        printf("portable math did not switch on\n");
        return 1;
    }
    for (i = 0; i < sizeof specials / sizeof specials[0]; i++) {
        double x = specials[i];
        mix(recomp_fsin(x)); mix(recomp_fcos(x)); mix(recomp_ftan(x));
        mix(recomp_fatan2(x, 1.0)); mix(recomp_fatan2(1.0, x)); mix(recomp_fatan2(x, -x));
        mix(recomp_flog2(fabs(x))); mix(recomp_fexp2(x > 2000 ? 2000 : x));
    }
    for (k = 0; k < N; k++) {
        double a = uni(-8.0, 8.0), b = uni(-1e5, 1e5), c = uni(-2e6, 2e6);
        double y = uni(-10.0, 10.0), x = uni(-10.0, 10.0);
        double l = uni(1e-6, 1e6), e = uni(-60.0, 60.0);
        check(0, recomp_fsin(a), sin(a));
        check(1, recomp_fcos(a), cos(a));
        check(2, recomp_ftan(a * 0.19), tan(a * 0.19));
        check(3, recomp_fatan2(y, x), atan2(y, x));
        check(4, recomp_flog2(l), log2(l));
        check(5, recomp_fexp2(e), exp2(e));
        mix(recomp_fsin(b)); mix(recomp_fcos(b)); mix(recomp_fsin(c));
    }
    printf("portable math hash %016llx\n", (unsigned long long)s_hash);
    for (k = 0; k < 6; k++) {
        printf("  %-5s worst %lld ulp vs this libm\n", s_name[k], (long long)s_worst[k]);
        if (s_worst[k] > 4)
            fail = 1;
    }
    recomp_set_portable_math(0);
    if (recomp_fcos(0.3) != cos(0.3)) {
        printf("switched off, recomp_fcos is not libm's cos\n");
        fail = 1;
    }
#ifdef EXPECTED_HASH
    if (s_hash != EXPECTED_HASH) {
        printf("hash differs from EXPECTED_HASH\n");
        fail = 1;
    }
#endif
    printf("%s\n", fail ? "FAILED" : "OK");
    return fail;
}
