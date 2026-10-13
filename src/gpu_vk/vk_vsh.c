/**
 * NV2A vertex programs as GLSL vertex shaders.
 *
 * Instruction for instruction what nv2a_vsh_interp.c does (nv2a_vsh_run):
 * the same field table, both units reading their sources before either
 * writes, the ILU paired with a MAC op writing R1, R12 as oPos, a0-relative
 * constants with out-of-range reads going to c[0], RCP/RCC/RSQ/EXP/LOG/LIT
 * as the interpreter computes them, MIN/MAX as a < b ? a : b / a > b ? a : b,
 * and colour outputs clamped to [0,1] with NaN going to 0.
 *
 * Inputs are pulled, not bound: the draw's vertex data is one storage
 * buffer and each attribute's offset, stride and NV2A format are in the
 * draw's uniforms, so every vertex layout runs through the same pipeline and
 * the formats Vulkan has no name for (the packed 11:11:10 normal) are just
 * code. A disabled attribute reads its inline value, as on the GPU.
 *
 * The program's oPos is screen space (the D3D epilogue applies the viewport
 * and divides by w); the shared tail turns it back into clip space exactly
 * as the CPU-transformed path does.
 */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vk_vsh.h"

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

static uint32_t field(const uint32_t *ins, int dw, int lo, int width)
{
    return (ins[dw] >> lo) & ((1u << width) - 1u);
}

/* The uniform blocks and helpers every program shader shares. */
static const char k_head[] =
"#version 450\n"
"layout(std140, set=0, binding=0) uniform U {\n"
"    vec4 vp, k, o, fogp, fogc;\n"
"    vec4 c0[8], c1[8];\n"
"    vec4 fc0, fc1;\n"
"    vec4 tsz[4], bm[4], bl[4];\n"
"    ivec4 at;\n"
"    uvec4 attr[16];   /* byte offset of element imin, stride, type | size << 4 */\n"
"    vec4 imm[16];     /* inline values: what a disabled attribute reads */\n"
"    uvec4 imin;       /* x: lowest index the draw uses */\n"
"} u;\n"
"layout(std140, set=0, binding=5) uniform C { vec4 c[192]; } cst;\n"
"layout(std430, set=0, binding=6) readonly buffer Blob { uint w[]; } blob;\n"
"layout(location=0) noperspective out vec4 d0;\n"
"layout(location=1) noperspective out vec4 d1;\n"
"layout(location=2) noperspective out float fogf;\n"
"layout(location=3) out vec4 tc0;\n"
"layout(location=4) out vec4 tc1;\n"
"layout(location=5) out vec4 tc2;\n"
"layout(location=6) out vec4 tc3;\n"
"out gl_PerVertex { vec4 gl_Position; float gl_ClipDistance[2]; };\n"
"#define INF uintBitsToFloat(0x7F800000u)\n"
"uint rd32(uint b) {\n"
"    uint i = b >> 2u, sh = (b & 3u) * 8u;\n"
"    return sh == 0u ? blob.w[i] : (blob.w[i] >> sh) | (blob.w[i + 1u] << (32u - sh));\n"
"}\n"
"uint rd16(uint b) { return (rd32(b & ~1u) >> ((b & 1u) * 8u)) & 0xFFFFu; }\n"
"uint rd8(uint b) { return (rd32(b & ~3u) >> ((b & 3u) * 8u)) & 0xFFu; }\n"
"float s16(uint v) { return float(int(v << 16u) >> 16); }\n"
"/* One input, as fetch_input / fetch_attr read it. */\n"
"vec4 fetch(int a, uint vtx) {\n"
"    uvec4 d = u.attr[a];\n"
"    uint type = d.z & 15u, size = (d.z >> 4u) & 15u, n = min(size, 4u), k;\n"
"    vec4 r = vec4(0.0, 0.0, 0.0, 1.0);\n"
"    if (size == 0u) return u.imm[a];\n"
"    if (d.y == 0u || (d.z & 0x100u) != 0u) return r;   /* no stride / no array */\n"
"    uint b = d.x + (vtx - u.imin.x) * d.y;\n"
"    switch (type) {\n"
"    case 0u: { uint p = rd32(b);                     /* D3DCOLOR */\n"
"        return vec4(float((p >> 16u) & 255u), float((p >> 8u) & 255u),\n"
"                    float(p & 255u), float(p >> 24u)) / 255.0; }\n"
"    case 2u: for (k = 0u; k < n; k++) r[k] = uintBitsToFloat(rd32(b + 4u * k)); return r;\n"
"    case 4u: for (k = 0u; k < n; k++) r[k] = float(rd8(b + k)) / 255.0; return r;\n"
"    case 1u: for (k = 0u; k < n; k++) r[k] = s16(rd16(b + 2u * k)) / 32767.0; return r;\n"
"    case 5u: for (k = 0u; k < n; k++) r[k] = s16(rd16(b + 2u * k)); return r;\n"
"    case 6u: { uint v = rd32(b);                      /* packed 11:11:10 normal */\n"
"        r.x = float(int(v << 21u) >> 21) / 1023.0;\n"
"        r.y = float(int(v << 10u) >> 21) / 1023.0;\n"
"        r.z = float(int(v) >> 22) / 511.0;\n"
"        return r; }\n"
"    default: return r;\n"
"    }\n"
"}\n"
"vec4 c01(vec4 v) {             /* clamp01 of the interpreter: NaN -> 0 */\n"
"    v = mix(vec4(0.0), v, greaterThan(v, vec4(0.0)));\n"
"    return mix(vec4(1.0), v, lessThan(v, vec4(1.0)));\n"
"}\n"
"vec4 cread(int i) { return cst.c[(i < 0 || i >= 192) ? 0 : i]; }\n"
"float fog_factor(float d) {\n"
"    float f, px = u.fogp.z, py = u.fogp.w;\n"
"    int mode = int(u.fogp.y);\n"
"    if (u.fogp.x == 0.0) return 1.0;\n"
"    if (mode == 0x800 || mode == 0x802) f = px + exp2(d * py * 16.0) - 1.5;\n"
"    else if (mode == 0x801 || mode == 0x803) f = px + exp2(-d * d * py * py * 32.0) - 1.5;\n"
"    else f = px + d * py - 1.0;\n"
"    if (mode == 0x802 || mode == 0x803 || mode == 0x804) f = abs(f);\n"
"    if (isnan(f)) f = 1.0;\n"
"    return clamp(f, 0.0, 1.0);\n"
"}\n"
"vec4 ilu_rcc(float x) {\n"
"    float f = 1.0 / x, m = abs(f);\n"
"    if (m < 5.42101e-20) f = f < 0.0 ? -5.42101e-20 : 5.42101e-20;\n"
"    if (m > 1.884467e19) f = f < 0.0 ? -1.884467e19 : 1.884467e19;\n"
"    return vec4(f);\n"
"}\n"
"vec4 ilu_exp(float x) { float fl = floor(x); return vec4(exp2(fl), x - fl, exp2(x), 1.0); }\n"
"vec4 ilu_log(float x) {\n"
"    float ax = abs(x);\n"
"    if (ax == 0.0) return vec4(-INF, 1.0, -INF, 1.0);\n"
"    float e = floor(log2(ax));\n"
"    return vec4(e, ax / exp2(e), log2(ax), 1.0);\n"
"}\n"
"vec4 ilu_lit(vec4 c) {\n"
"    float nl = c.x > 0.0 ? c.x : 0.0, nh = c.y > 0.0 ? c.y : 0.0;\n"
"    float p = c.w < -127.9961 ? -127.9961 : (c.w > 127.9961 ? 127.9961 : c.w);\n"
"    /* powf(0, 0) is 1; GLSL's pow is undefined there (NaN in practice). */\n"
"    return vec4(1.0, nl, c.x > 0.0 ? (p == 0.0 ? 1.0 : pow(nh, p)) : 0.0, 1.0);\n"
"}\n"
"vec4 vmin(vec4 a, vec4 b) { return mix(b, a, lessThan(a, b)); }\n"
"vec4 vmax(vec4 a, vec4 b) { return mix(b, a, greaterThan(a, b)); }\n";

/* The CPU-transformed path's tail, from the program's outputs. */
static const char k_tail[] =
"    {\n"
"        float w = oPos.w;\n"
"        gl_Position = vec4((oPos.x * u.vp.x - 1.0) * w, (oPos.y * u.vp.y - 1.0) * w,\n"
"                           oPos.z * u.vp.z * w, w);\n"
"        float zc = u.k.z != 0.0 ? (oPos.z - u.o.z) / u.k.z * w : w;\n"
"        gl_ClipDistance[0] = zc;\n"
"        gl_ClipDistance[1] = w - 1e-3;\n"
"    }\n"
"    d0 = c01(o3);\n"
"    d1 = c01(o4);\n"
"    fogf = fog_factor(o5.x);\n"
"    tc0 = o9; tc1 = o10; tc2 = o11; tc3 = o12;\n";

/* Source operand k of `ins` as a GLSL expression. */
static void src(Str *s, const uint32_t *ins, int k)
{
    static const char sw[] = "xyzw";
    uint32_t neg, swz, mux, reg;
    char base[48];

    if (k == 0) {
        neg = field(ins, 1, 8, 1); swz = field(ins, 1, 0, 8);
        mux = field(ins, 2, 26, 2); reg = field(ins, 2, 28, 4);
    } else if (k == 1) {
        neg = field(ins, 2, 25, 1); swz = field(ins, 2, 17, 8);
        mux = field(ins, 2, 11, 2); reg = field(ins, 2, 13, 4);
    } else {
        neg = field(ins, 2, 10, 1); swz = field(ins, 2, 2, 8);
        mux = field(ins, 3, 28, 2);
        reg = (field(ins, 2, 0, 2) << 2) | field(ins, 3, 30, 2);
    }
    if (mux == 1) {
        if (reg == 12) snprintf(base, sizeof base, "oPos");
        else snprintf(base, sizeof base, "R%u", reg < 12 ? reg : 0);
    } else if (mux == 2) {
        snprintf(base, sizeof base, "v%u", field(ins, 1, 9, 4));
    } else {
        uint32_t idx = field(ins, 1, 13, 8);
        if (field(ins, 3, 1, 1))
            snprintf(base, sizeof base, "cread(%u + a0)", idx);
        else
            snprintf(base, sizeof base, "cst.c[%u]", idx < 192 ? idx : 0);
    }
    put(s, "%s%s.%c%c%c%c", neg ? "-" : "", base, sw[(swz >> 6) & 3], sw[(swz >> 4) & 3],
        sw[(swz >> 2) & 3], sw[swz & 3]);
}

/* A masked write of vec4 `v` to register `dst`. */
static void wr(Str *s, const char *dst, const char *v, uint32_t mask)
{
    static const char comp[] = "xyzw";
    int i;
    if (mask == 0xF) {
        put(s, "        %s = %s;\n", dst, v);
        return;
    }
    for (i = 0; i < 4; i++)
        if (mask & (8u >> i))
            put(s, "        %s.%c = %s.%c;\n", dst, comp[i], v, comp[i]);
}

/* Which inputs the program reads (as decode_program's `inputs`). */
static uint32_t inputs_read(const uint32_t (*w)[4], uint32_t n)
{
    uint32_t i, m = 0;
    for (i = 0; i < n; i++) {
        const uint32_t *ins = w[i];
        uint32_t mac = field(ins, 1, 21, 4), ilu = field(ins, 1, 25, 3), reads = 0, v;
        if (mac) reads |= 1;
        if (mac == 2 || (mac >= 4 && mac <= 12)) reads |= 2;
        if (mac == 3 || mac == 4 || ilu) reads |= 4;
        v = field(ins, 1, 9, 4);
        if ((reads & 1) && field(ins, 2, 26, 2) == 2) m |= 1u << v;
        if ((reads & 2) && field(ins, 2, 11, 2) == 2) m |= 1u << v;
        if ((reads & 4) && field(ins, 3, 28, 2) == 2) m |= 1u << v;
    }
    return m;
}

int vk_vsh_supported(const uint32_t (*w)[4], uint32_t n)
{
    uint32_t i;
    for (i = 0; i < n; i++)
        if (field(w[i], 3, 12, 4) && !field(w[i], 3, 11, 1))
            return 0;                    /* writes c[]: CXT_WRITE, per vertex */
    return n > 0 && field(w[n - 1], 3, 0, 1);
}

char *vk_vsh_glsl(const uint32_t (*w)[4], uint32_t n, const char *store)
{
    Str s = { NULL, 0, 0 };
    uint32_t i, in = inputs_read(w, n);
    static const char *const mac_name[16] = {
        "nop", "mov", "mul", "add", "mad", "dp3", "dph", "dp4",
        "dst", "min", "max", "slt", "sge", "arl", "mac14", "mac15"
    };

    s.cap = 16384;
    s.p = (char *)malloc(s.cap);
    s.p[0] = 0;
    put(&s, "%s", k_head);
    if (store)
        put(&s, "%s", store);
    put(&s, "void main() {\n    uint vtx = uint(gl_VertexIndex);\n");
    for (i = 0; i < 16; i++)
        if (in >> i & 1)
            put(&s, "    vec4 v%u = fetch(%u, vtx);\n", i, i);
    put(&s, "    vec4 R0 = vec4(0.0), R1 = R0, R2 = R0, R3 = R0, R4 = R0, R5 = R0;\n"
            "    vec4 R6 = R0, R7 = R0, R8 = R0, R9 = R0, R10 = R0, R11 = R0, oPos = R0;\n"
            "    vec4 o1 = R0, o2 = R0, o3 = vec4(0.0, 0.0, 0.0, 1.0), o4 = o3, o5 = R0;\n"
            "    vec4 o6 = R0, o7 = R0, o8 = R0, o9 = o3, o10 = o3, o11 = o3, o12 = o3;\n"
            "    int a0 = 0;\n");
    for (i = 0; i < n; i++) {
        const uint32_t *ins = w[i];
        uint32_t mac = field(ins, 1, 21, 4), ilu = field(ins, 1, 25, 3);
        uint32_t mac_mask = field(ins, 3, 24, 4), ilu_mask = field(ins, 3, 16, 4);
        uint32_t tdst = field(ins, 3, 20, 4), omask = field(ins, 3, 12, 4);
        uint32_t oaddr = field(ins, 3, 3, 8), out_ilu = field(ins, 3, 2, 1);
        uint32_t out_o = field(ins, 3, 11, 1);
        put(&s, "    { /* %u: %s ilu%u */\n", i, mac_name[mac], ilu);
        /* The operands the ops read (decode_program's `reads`); an unread
         * one cannot change a result. */
        {
            uint32_t reads = (mac ? 1u : 0u)
                           | ((mac == 2 || (mac >= 4 && mac <= 12)) ? 2u : 0u)
                           | ((mac == 3 || mac == 4 || ilu) ? 4u : 0u);
            static const char nm[3] = {'A', 'B', 'C'};
            int k;
            for (k = 0; k < 3; k++) {
                put(&s, "        vec4 %c = ", nm[k]);
                if (reads >> k & 1)
                    src(&s, ins, k);
                else
                    put(&s, "vec4(0.0)");
                put(&s, ";\n");
            }
        }
        put(&s, "        vec4 M = vec4(0.0), I = vec4(0.0);\n");
        switch (mac) {
        case 0:  break;
        case 1:  put(&s, "        M = A;\n"); break;
        case 2:  put(&s, "        M = A * B;\n"); break;
        case 3:  put(&s, "        M = A + C;\n"); break;
        case 4:  put(&s, "        M = A * B + C;\n"); break;
        case 5:  put(&s, "        M = vec4(A.x * B.x + A.y * B.y + A.z * B.z);\n"); break;
        case 6:  put(&s, "        M = vec4(A.x * B.x + A.y * B.y + A.z * B.z + B.w);\n"); break;
        case 7:  put(&s, "        M = vec4(A.x * B.x + A.y * B.y + A.z * B.z + A.w * B.w);\n"); break;
        case 8:  put(&s, "        M = vec4(1.0, A.y * B.y, A.z, B.w);\n"); break;
        case 9:  put(&s, "        M = vmin(A, B);\n"); break;
        case 10: put(&s, "        M = vmax(A, B);\n"); break;
        case 11: put(&s, "        M = vec4(lessThan(A, B));\n"); break;
        case 12: put(&s, "        M = vec4(greaterThanEqual(A, B));\n"); break;
        case 13: put(&s, "        M = A;\n"); break;
        default: put(&s, "        M = A;\n"); break;
        }
        switch (ilu) {
        case 0:  break;
        case 1:  put(&s, "        I = C;\n"); break;
        case 2:  put(&s, "        I = vec4(C.x == 0.0 ? INF : 1.0 / C.x);\n"); break;
        case 3:  put(&s, "        I = ilu_rcc(C.x);\n"); break;
        case 4:  put(&s, "        I = vec4(1.0 / sqrt(abs(C.x)));\n"); break;
        case 5:  put(&s, "        I = ilu_exp(C.x);\n"); break;
        case 6:  put(&s, "        I = ilu_log(C.x);\n"); break;
        case 7:  put(&s, "        I = ilu_lit(C);\n"); break;
        default: put(&s, "        I = C;\n"); break;
        }
        /* Writes, in the interpreter's order: a0, MAC temp, ILU temp, out. */
        if (mac == 13)
            put(&s, "        a0 = int(floor(A.x + 0.001));\n");
        if (mac && mac != 13 && mac_mask && tdst <= 12) {
            char d[8];
            snprintf(d, sizeof d, tdst == 12 ? "oPos" : "R%u", tdst);
            wr(&s, d, "M", mac_mask);
        }
        if (ilu && ilu_mask) {
            uint32_t it = mac ? 1u : tdst;
            if (it <= 12) {
                char d[8];
                snprintf(d, sizeof d, it == 12 ? "oPos" : "R%u", it);
                wr(&s, d, "I", ilu_mask);
            }
        }
        if (omask && out_o && oaddr < 13) {
            char d[8];
            snprintf(d, sizeof d, oaddr == 0 ? "oPos" : "o%u", oaddr);
            wr(&s, d, out_ilu ? "I" : "M", omask);
        }
        put(&s, "    }\n");
        if (field(ins, 3, 0, 1))
            break;
    }
    put(&s, "%s", k_tail);
    if (store)
        put(&s, "    test_store(vtx, oPos, c01(o3), c01(o4), o5, o9, o10, o11, o12);\n");
    put(&s, "}\n");
    return s.p;
}
