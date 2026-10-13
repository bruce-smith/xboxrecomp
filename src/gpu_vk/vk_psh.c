/**
 * NV2A register combiners (and the texture stages in front of them) as a
 * GLSL fragment shader.
 *
 * A straight translation of the software path, operation for operation:
 * rc_stage_fetch (nv2a_pb_exec.c) for what each texture stage produces, and
 * nv2a_rc_eval_prep (nv2a_combiner.c) for the general and final combiners.
 * The combiner program -- ICWs, OCWs, the final combiner words, control and
 * the stage program -- is compiled in; the constants (factors, bump matrices,
 * fog colour, texture sizes) come from the draw's uniform block, so a
 * program shared by many draws is one shader.
 *
 * Register numbers (an input byte's low nibble): 0 zero, 1 c0, 2 c1, 3 fog,
 * 4 v0, 5 v1, 8-11 t0-t3, 12 r0, 13 r1, 14 v1+r0 sum and 15 E*F (final).
 */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vk_psh.h"

typedef struct {
    char *p;
    size_t n, cap;
} Str;

static void put(Str *s, const char *fmt, ...)
{
    va_list ap;
    int k;
    for (;;) {
        size_t room = s->cap - s->n;
        va_start(ap, fmt);
        k = vsnprintf(s->p + s->n, room, fmt, ap);
        va_end(ap);
        if (k >= 0 && (size_t)k < room) {
            s->n += (size_t)k;
            return;
        }
        s->cap = s->cap * 2 + (size_t)(k > 0 ? k : 256);
        s->p = (char *)realloc(s->p, s->cap);
    }
}

void vk_psh_key(const Nv2aDrawState *st, VkPshKey *k)
{
    int i, n;
    uint32_t prog = st->rc.stage_program;

    memset(k, 0, sizeof *k);
    k->rc_seen = st->rc_seen;
    if (!st->rc_seen)
        return;
    n = (int)(st->rc.control & 0xFF);
    if (n > 8)
        n = 8;
    k->control = st->rc.control;
    k->stage_program = prog;
    k->final0 = st->rc.final0;
    k->final1 = st->rc.final1;
    for (i = 0; i < n; i++) {
        k->color_icw[i] = st->rc.color_icw[i];
        k->color_ocw[i] = st->rc.color_ocw[i];
        k->alpha_icw[i] = st->rc.alpha_icw[i];
        k->alpha_ocw[i] = st->rc.alpha_ocw[i];
    }
    for (i = 0; i < 4; i++) {
        uint32_t mode = (prog >> (i * 5)) & 0x1F;
        if (!mode)
            continue;
        k->cube[i] = (uint8_t)(st->tex[i].valid && st->tex[i].cube);
        if (mode == 5)
            k->clip_plane_mode |= st->clip_plane_mode & (0xFu << (i * 4));
        if (mode == 6 || mode == 7) {
            int in = i == 1 ? 0 : (int)(st->other_stage_input >> (i == 2 ? 16 : 20)) & 3;
            k->bump_in[i] = (uint8_t)in;
            /* The input stage's B and G signed bits decide whether its
             * texel is already an offset in -1..1. */
            k->bump_signed[i] = (uint8_t)((st->tex[in].filter >> 30) & 3);
        }
    }
}

/* One input byte, RGB portion: the register (or its alpha replicated),
 * through the input mapping. */
static void in_rgb(Str *s, uint32_t byte)
{
    char src[32];
    unsigned r = byte & 0xF;
    if (byte & 0x10)
        snprintf(src, sizeof src, "vec3(R%u.a)", r);
    else
        snprintf(src, sizeof src, "R%u.rgb", r);
    switch ((byte >> 5) & 7) {
    case 0:  put(s, "max(%s, vec3(0.0))", src); break;
    case 1:  put(s, "(1.0 - clamp(%s, 0.0, 1.0))", src); break;
    case 2:  put(s, "(2.0 * max(%s, vec3(0.0)) - 1.0)", src); break;
    case 3:  put(s, "(-2.0 * max(%s, vec3(0.0)) + 1.0)", src); break;
    case 4:  put(s, "(max(%s, vec3(0.0)) - 0.5)", src); break;
    case 5:  put(s, "(-max(%s, vec3(0.0)) + 0.5)", src); break;
    case 6:  put(s, "(%s)", src); break;
    default: put(s, "(-%s)", src); break;
    }
}

/* Alpha portion: .a with the alpha bit, else .b. */
static void in_alpha(Str *s, uint32_t byte)
{
    char src[16];
    unsigned r = byte & 0xF;
    snprintf(src, sizeof src, "R%u.%c", r, (byte & 0x10) ? 'a' : 'b');
    switch ((byte >> 5) & 7) {
    case 0:  put(s, "max(%s, 0.0)", src); break;
    case 1:  put(s, "(1.0 - clamp(%s, 0.0, 1.0))", src); break;
    case 2:  put(s, "(2.0 * max(%s, 0.0) - 1.0)", src); break;
    case 3:  put(s, "(-2.0 * max(%s, 0.0) + 1.0)", src); break;
    case 4:  put(s, "(max(%s, 0.0) - 0.5)", src); break;
    case 5:  put(s, "(-max(%s, 0.0) + 0.5)", src); break;
    case 6:  put(s, "(%s)", src); break;
    default: put(s, "(-%s)", src); break;
    }
}

/* PS_COMBINEROUTPUT scale/bias as (x + bias) * scale (nv2a_combiner.c). */
static void out_map(uint32_t mapping, const char **bias, const char **scale)
{
    switch (mapping) {
    case 0x08: *bias = "-0.5"; *scale = "1.0"; break;
    case 0x10: *bias = "0.0";  *scale = "2.0"; break;
    case 0x18: *bias = "-0.5"; *scale = "2.0"; break;
    case 0x20: *bias = "0.0";  *scale = "4.0"; break;
    case 0x30: *bias = "0.0";  *scale = "0.5"; break;
    default:   *bias = "0.0";  *scale = "1.0"; break;
    }
}

static const char k_head[] =
"#version 450\n"
"layout(std140, set=0, binding=0) uniform U {\n"
"    vec4 vp, k, o, fogp, fogc;\n"
"    vec4 c0[8], c1[8];\n"
"    vec4 fc0, fc1;\n"
"    vec4 tsz[4], bm[4], bl[4];\n"
"    ivec4 at;\n"
"} u;\n"
"layout(location=0) noperspective in vec4 d0;\n"
"layout(location=1) noperspective in vec4 d1;\n"
"layout(location=2) noperspective in float fogf;\n"
"layout(location=3) in vec4 tc0;\n"
"layout(location=4) in vec4 tc1;\n"
"layout(location=5) in vec4 tc2;\n"
"layout(location=6) in vec4 tc3;\n"
"layout(location=0) out vec4 o;\n";

/* Alpha test (alpha_test_pass), then the output. */
static const char k_tail[] =
"    c = clamp(c, 0.0, 1.0);\n"
"    if (u.at.x != 0) {\n"
"        int v = int(c.a * 255.0 + 0.5), r = u.at.z;\n"
"        bool pass;\n"
"        switch (u.at.y) {\n"
"        case 0x200: pass = false; break;\n"
"        case 0x201: pass = v <  r; break;\n"
"        case 0x202: pass = v == r; break;\n"
"        case 0x203: pass = v <= r; break;\n"
"        case 0x204: pass = v >  r; break;\n"
"        case 0x205: pass = v != r; break;\n"
"        case 0x206: pass = v >= r; break;\n"
"        default:    pass = true; break;\n"
"        }\n"
"        if (!pass) discard;\n"
"    }\n"
"    o = c;\n"
"}\n";

char *vk_psh_glsl(const VkPshKey *k)
{
    Str s = { NULL, 0, 0 };
    int i, n;
    uint32_t flags = k->control >> 8;

    s.cap = 8192;
    s.p = (char *)malloc(s.cap);
    s.p[0] = 0;
    put(&s, "%s", k_head);
    for (i = 0; i < 4; i++)
        put(&s, "layout(set=0, binding=%d) uniform %s s%d;\n", i + 1,
            k->cube[i] ? "samplerCube" : "sampler2D", i);
    /* rc_texel: normalised coordinates as they are, linear ones (texels)
     * scaled by the texture's size. */
    put(&s, "vec4 fetch2(sampler2D t, int st, vec2 uv) {\n"
            "    return texture(t, u.tsz[st].z != 0.0 ? uv : uv * u.tsz[st].xy);\n"
            "}\n");
    put(&s, "void main() {\n    vec4 c;\n");

    if (!k->rc_seen) {
        /* No combiners programmed: the software path's "stage 0 texel x
         * diffuse" stand-in (its texture read is nearest, unfiltered). */
        put(&s, "    c = clamp(d0, 0.0, 1.0);\n"
                "    if (u.tsz[0].w != 0.0) {\n"
                "        float q = tc0.w != 0.0 ? tc0.w : 1.0;\n"
                "        c *= fetch2(s0, 0, tc0.xy / q);\n"
                "    }\n");
        put(&s, "%s", k_tail);
        return s.p;
    }

    /* Texture stages (rc_stage_fetch), in order: a bump stage reads an
     * earlier stage's result. */
    put(&s, "    vec4 t0 = vec4(0.0, 0.0, 0.0, 1.0), t1 = t0, t2 = t0, t3 = t0;\n");
    for (i = 0; i < 4; i++) {
        uint32_t mode = (k->stage_program >> (i * 5)) & 0x1F;
        switch (mode) {
        case 0:
            break;
        case 1: case 2:                           /* PROJECT2D / 3D */
            if (k->cube[i])
                put(&s, "    t%d = texture(s%d, tc%d.xyz);\n", i, i, i);
            else
                put(&s, "    t%d = fetch2(s%d, %d, tc%d.xy / (tc%d.w != 0.0 ? tc%d.w : 1.0));\n",
                    i, i, i, i, i, i);
            break;
        case 3:                                   /* CUBEMAP */
            if (k->cube[i])
                put(&s, "    t%d = texture(s%d, tc%d.xyz);\n", i, i, i);
            else
                put(&s, "    t%d = fetch2(s%d, %d, tc%d.xy);\n", i, i, i, i);
            break;
        case 4:                                   /* PASSTHRU */
            put(&s, "    t%d = clamp(tc%d, 0.0, 1.0);\n", i, i);
            break;
        case 5: {                                 /* CLIPPLANE */
            int j;
            static const char comp[] = "xyzw";
            for (j = 0; j < 4; j++) {
                int ge = (k->clip_plane_mode >> (i * 4 + j)) & 1;
                put(&s, "    if (tc%d.%c %s 0.0) discard;\n", i, comp[j], ge ? ">=" : "<");
            }
            put(&s, "    t%d = vec4(0.0);\n", i);
            break;
        }
        case 6: case 7: {                         /* BUMPENVMAP (_LUM) */
            int in = k->bump_in[i];
            put(&s, "    {\n        float ds = t%d.b, dt = t%d.g;\n", in, in);
            if (!(k->bump_signed[i] & 2))         /* B unsigned: 0..1 -> -1..1 */
                put(&s, "        ds = ds * 2.0 - 1.0;\n");
            if (!(k->bump_signed[i] & 1))         /* G */
                put(&s, "        dt = dt * 2.0 - 1.0;\n");
            put(&s, "        vec2 uv = tc%d.xy + vec2(u.bm[%d].x * ds + u.bm[%d].z * dt,\n"
                    "                                 u.bm[%d].y * ds + u.bm[%d].w * dt);\n",
                i, i, i, i, i);
            if (k->cube[i])
                put(&s, "        t%d = texture(s%d, vec3(uv, tc%d.z));\n", i, i, i);
            else
                put(&s, "        t%d = fetch2(s%d, %d, uv);\n", i, i, i);
            if (mode == 7)
                put(&s, "        t%d.rgb *= clamp(t%d.r * u.bl[%d].x + u.bl[%d].y, 0.0, 1.0);\n",
                    i, in, i, i);
            put(&s, "    }\n");
            break;
        }
        default:                                  /* dot-product modes: 0 */
            put(&s, "    t%d = vec4(0.0);\n", i);
            break;
        }
    }

    /* The registers. */
    put(&s, "    vec4 R0 = vec4(0.0), R1, R2, R3 = vec4(u.fogc.rgb, fogf);\n"
            "    vec4 R4 = d0, R5 = d1, R6 = vec4(0.0), R7 = vec4(0.0);\n"
            "    vec4 R8 = t0, R9 = t1, R10 = t2, R11 = t3;\n"
            "    vec4 R12 = vec4(0.0), R13 = vec4(0.0), R14 = vec4(0.0), R15 = vec4(0.0);\n");
    put(&s, "    R12.a = %s;\n", (k->stage_program & 0x1F) ? "t0.a" : "1.0");

    n = (int)(k->control & 0xFF);
    if (n > 8)
        n = 8;
    for (i = 0; i < n; i++) {
        uint32_t icw = k->color_icw[i], ocw = k->color_ocw[i];
        uint32_t aicw = k->alpha_icw[i], aocw = k->alpha_ocw[i];
        uint32_t fl = ocw >> 12, afl = aocw >> 12;
        int ab = ((ocw >> 4) & 0xF) != 0, cd = (ocw & 0xF) != 0, ms = ((ocw >> 8) & 0xF) != 0;
        int aab = ((aocw >> 4) & 0xF) != 0, acd = (aocw & 0xF) != 0, ams = ((aocw >> 8) & 0xF) != 0;
        const char *bias, *scale, *abias, *ascale;

        out_map((ocw >> 12) & 0x38, &bias, &scale);
        out_map((aocw >> 12) & 0x38, &abias, &ascale);
        put(&s, "    /* stage %d */\n    R1 = u.c0[%d]; R2 = u.c1[%d];\n", i,
            (flags & 0x010) ? i : 0, (flags & 0x100) ? i : 0);
        if (!(ab || cd || ms || aab || acd || ams))
            continue;
        put(&s, "    {\n");
        if ((ms && (fl & 4)) || (ams && (afl & 4))) {
            if (flags & 1)
                put(&s, "        bool mux = R12.a >= 0.5;\n");
            else
                put(&s, "        bool mux = (int(R12.a * 255.0) & 1) != 0;\n");
        }
        /* Both portions read before either writes. */
        if (ab || cd || ms) {
            int want_ab = ab || ms, want_cd = cd || ms;
            if (want_ab) {
                put(&s, "        vec3 A = "); in_rgb(&s, icw >> 24); put(&s, ";\n");
                put(&s, "        vec3 B = "); in_rgb(&s, icw >> 16); put(&s, ";\n");
                if (fl & 2)
                    put(&s, "        vec3 pab = vec3(A.x * B.x + A.y * B.y + A.z * B.z);\n");
                else
                    put(&s, "        vec3 pab = A * B;\n");
            } else {
                put(&s, "        vec3 pab = vec3(0.0);\n");
            }
            if (want_cd) {
                put(&s, "        vec3 C = "); in_rgb(&s, icw >> 8); put(&s, ";\n");
                put(&s, "        vec3 D = "); in_rgb(&s, icw); put(&s, ";\n");
                if (fl & 1)
                    put(&s, "        vec3 pcd = vec3(C.x * D.x + C.y * D.y + C.z * D.z);\n");
                else
                    put(&s, "        vec3 pcd = C * D;\n");
            } else {
                put(&s, "        vec3 pcd = vec3(0.0);\n");
            }
            if (ab)
                put(&s, "        vec3 ab = clamp((pab + %s) * %s, -1.0, 1.0);\n", bias, scale);
            if (cd)
                put(&s, "        vec3 cd = clamp((pcd + %s) * %s, -1.0, 1.0);\n", bias, scale);
            if (ms)
                put(&s, "        vec3 ms = clamp((%s + %s) * %s, -1.0, 1.0);\n",
                    (fl & 4) ? "(mux ? pcd : pab)" : "(pab + pcd)", bias, scale);
        }
        if (aab || acd || ams) {
            if (aab || ams) {
                put(&s, "        float apab = "); in_alpha(&s, aicw >> 24);
                put(&s, " * "); in_alpha(&s, aicw >> 16); put(&s, ";\n");
            } else {
                put(&s, "        float apab = 0.0;\n");
            }
            if (acd || ams) {
                put(&s, "        float apcd = "); in_alpha(&s, aicw >> 8);
                put(&s, " * "); in_alpha(&s, aicw); put(&s, ";\n");
            } else {
                put(&s, "        float apcd = 0.0;\n");
            }
            if (aab)
                put(&s, "        float aab = clamp((apab + %s) * %s, -1.0, 1.0);\n", abias, ascale);
            if (acd)
                put(&s, "        float acd = clamp((apcd + %s) * %s, -1.0, 1.0);\n", abias, ascale);
            if (ams)
                put(&s, "        float ams = clamp((%s + %s) * %s, -1.0, 1.0);\n",
                    (afl & 4) ? "(mux ? apcd : apab)" : "(apab + apcd)", abias, ascale);
        }
        /* RGB destinations; blue-to-alpha also writes the destination's .a. */
        if (ab) {
            unsigned d = (ocw >> 4) & 0xF;
            put(&s, "        R%u.rgb = ab;\n", d);
            if (fl & 0x80)
                put(&s, "        R%u.a = ab.b;\n", d);
        }
        if (cd) {
            unsigned d = ocw & 0xF;
            put(&s, "        R%u.rgb = cd;\n", d);
            if (fl & 0x40)
                put(&s, "        R%u.a = cd.b;\n", d);
        }
        if (ms)
            put(&s, "        R%u.rgb = ms;\n", (ocw >> 8) & 0xF);
        if (aab) put(&s, "        R%u.a = aab;\n", (aocw >> 4) & 0xF);
        if (acd) put(&s, "        R%u.a = acd;\n", aocw & 0xF);
        if (ams) put(&s, "        R%u.a = ams;\n", (aocw >> 8) & 0xF);
        put(&s, "        R0 = vec4(0.0);\n    }\n");
    }

    if (k->final0 || k->final1) {
        uint32_t f0 = k->final0, f1 = k->final1, ff = f1 & 0xFF;
        put(&s, "    /* final combiner */\n    R1 = u.fc0; R2 = u.fc1;\n");
        put(&s, "    {\n        vec3 a = %s, b = %s;\n",
            (ff & 0x40) ? "1.0 - R5.rgb" : "R5.rgb", (ff & 0x20) ? "1.0 - R12.rgb" : "R12.rgb");
        put(&s, "        R14 = vec4(%s, 0.0);\n",
            (ff & 0x80) ? "clamp(a + b, 0.0, 1.0)" : "a + b");
        put(&s, "        vec3 E = "); in_rgb(&s, f1 >> 24); put(&s, ";\n");
        put(&s, "        vec3 F = "); in_rgb(&s, f1 >> 16); put(&s, ";\n");
        put(&s, "        R15 = vec4(E * F, 0.0);\n");
        put(&s, "        vec3 A = "); in_rgb(&s, f0 >> 24); put(&s, ";\n");
        put(&s, "        vec3 B = "); in_rgb(&s, f0 >> 16); put(&s, ";\n");
        put(&s, "        vec3 C = "); in_rgb(&s, f0 >> 8); put(&s, ";\n");
        put(&s, "        vec3 D = "); in_rgb(&s, f0); put(&s, ";\n");
        put(&s, "        c.rgb = D + C * (1.0 - A) + B * A;\n");
        put(&s, "        c.a = "); in_alpha(&s, f1 >> 8); put(&s, ";\n    }\n");
    } else {
        put(&s, "    c = R12;\n");
    }
    put(&s, "%s", k_tail);
    return s.p;
}
