/**
 * NV2A vertex program interpreter. See nv2a_vsh_interp.h for why it exists.
 *
 * Encoding. An instruction is four dwords; dword 0 is unused. Fields, as
 * (dword, lowest bit, width):
 *
 *   ILU op        1 25 3      MAC op        1 21 4
 *   const index   1 13 8      input index   1  9 4   (shared by A, B, C)
 *   A negate      1  8 1      A swizzle     1  0 8   (x y z w, 2 bits each,
 *   A temp        2 28 4      A mux         2 26 2    x highest)
 *   B negate      2 25 1      B swizzle     2 17 8
 *   B temp        2 13 4      B mux         2 11 2
 *   C negate      2 10 1      C swizzle     2  2 8
 *   C temp        2 0 2 (high) : 3 30 2 (low)          C mux 3 28 2
 *   MAC temp mask 3 24 4      temp dest     3 20 4
 *   ILU temp mask 3 16 4      out mask      3 12 4
 *   out to o[]    3 11 1      out address   3  3 8   (0: to c[])
 *   out from ILU  3  2 1      a0.x relative 3  1 1   final 3 0 1
 *
 * Masks are x=8 y=4 z=2 w=1. A source mux of 1 reads a temp, 2 an input,
 * 3 a constant. Temp 12 is oPos. When an instruction carries both a MAC and
 * an ILU op, the ILU's temp write goes to R1 whatever the temp field says,
 * and both units read their sources before either writes.
 *
 * The layout is checked against real programs rather than taken on trust:
 * RECOMP_VSH_DUMP disassembles each program the title uploads, and an Xbox
 * D3D program is recognisable -- DP4s against the matrix constants, ending in
 * the runtime's screen-space epilogue (RCC of R12.w, then a MAD into oPos).
 */
#include "nv2a_vsh_interp.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

static uint32_t s_program[NV2A_VSH_SLOTS][4];
static float    s_const[NV2A_VSH_CONSTANTS][4];
static uint32_t s_load_slot, s_load_word, s_start_slot;
static uint32_t s_const_load, s_const_word;
static uint32_t s_cxt_write;
static int      s_dec_valid;            /* decoded program matches s_program */

static uint32_t field(const uint32_t *ins, int dw, int lo, int width)
{
    return (ins[dw] >> lo) & ((1u << width) - 1u);
}

/* ---- uploads ------------------------------------------------------------ */

void nv2a_vsh_set_load_slot(uint32_t slot)
{
    s_load_slot = slot;
    s_load_word = 0;
}

void nv2a_vsh_program_word(uint32_t word)
{
    if (s_load_slot < NV2A_VSH_SLOTS)
        s_program[s_load_slot][s_load_word] = word;
    s_dec_valid = 0;
    if (++s_load_word == 4) {
        s_load_word = 0;
        s_load_slot++;
    }
}

void nv2a_vsh_set_start_slot(uint32_t slot)
{
    if (slot != s_start_slot)
        s_dec_valid = 0;
    s_start_slot = slot;
}
void nv2a_vsh_set_cxt_write(uint32_t enable) { s_cxt_write = enable; }

void nv2a_vsh_set_constant_load(uint32_t index)
{
    s_const_load = index;
    s_const_word = 0;
}

void nv2a_vsh_constant_word(uint32_t word)
{
    if (s_const_load < NV2A_VSH_CONSTANTS)
        memcpy(&s_const[s_const_load][s_const_word], &word, 4);
    if (++s_const_word == 4) {
        s_const_word = 0;
        s_const_load++;
    }
}

void nv2a_vsh_set_constant(uint32_t index, const float v[4])
{
    if (index < NV2A_VSH_CONSTANTS)
        memcpy(s_const[index], v, sizeof s_const[index]);
}

void nv2a_vsh_set_instruction(uint32_t slot, const uint32_t words[4])
{
    if (slot < NV2A_VSH_SLOTS)
        memcpy(s_program[slot], words, sizeof s_program[slot]);
    s_dec_valid = 0;
}

/* ---- disassembly (RECOMP_VSH_DUMP) --------------------------------------- */

static const char *const k_mac[16] = {
    "nop", "mov", "mul", "add", "mad", "dp3", "dph", "dp4",
    "dst", "min", "max", "slt", "sge", "arl", "mac14", "mac15"
};
static const char *const k_ilu[8] = {
    "nop", "mov", "rcp", "rcc", "rsq", "exp", "log", "lit"
};

static void dis_src(char *b, size_t n, const uint32_t *ins, int which)
{
    static const char sw[] = "xyzw";
    uint32_t neg, swz, mux, reg;

    if (which == 0) {
        neg = field(ins, 1, 8, 1); swz = field(ins, 1, 0, 8);
        mux = field(ins, 2, 26, 2); reg = field(ins, 2, 28, 4);
    } else if (which == 1) {
        neg = field(ins, 2, 25, 1); swz = field(ins, 2, 17, 8);
        mux = field(ins, 2, 11, 2); reg = field(ins, 2, 13, 4);
    } else {
        neg = field(ins, 2, 10, 1); swz = field(ins, 2, 2, 8);
        mux = field(ins, 3, 28, 2);
        reg = (field(ins, 2, 0, 2) << 2) | field(ins, 3, 30, 2);
    }
    if (mux == 1)
        snprintf(b, n, "%sR%u", neg ? "-" : "", reg);
    else if (mux == 2)
        snprintf(b, n, "%sv%u", neg ? "-" : "", field(ins, 1, 9, 4));
    else
        snprintf(b, n, "%sc[%s%u]", neg ? "-" : "",
                 field(ins, 3, 1, 1) ? "a0.x+" : "", field(ins, 1, 13, 8));
    {
        size_t l = strlen(b);
        if (l + 6 < n)
            snprintf(b + l, n - l, ".%c%c%c%c", sw[(swz >> 6) & 3],
                     sw[(swz >> 4) & 3], sw[(swz >> 2) & 3], sw[swz & 3]);
    }
}

static void dump_program(uint32_t start)
{
    uint32_t s;
    fprintf(stderr, "  [VSH] program at slot %u:\n", start);
    for (s = start; s < NV2A_VSH_SLOTS; s++) {
        const uint32_t *ins = s_program[s];
        char a[40], bb[40], c[40];
        dis_src(a, sizeof a, ins, 0);
        dis_src(bb, sizeof bb, ins, 1);
        dis_src(c, sizeof c, ins, 2);
        fprintf(stderr, "  [VSH] %3u %-4s %-4s A=%s B=%s C=%s  R%u mac%X ilu%X"
                " out%s%u m%X%s%s\n", s,
                k_mac[field(ins, 1, 21, 4)], k_ilu[field(ins, 1, 25, 3)],
                a, bb, c, field(ins, 3, 20, 4), field(ins, 3, 24, 4),
                field(ins, 3, 16, 4), field(ins, 3, 11, 1) ? "o" : "c",
                field(ins, 3, 3, 8), field(ins, 3, 12, 4),
                field(ins, 3, 2, 1) ? " ilu" : " mac",
                field(ins, 3, 0, 1) ? " FINAL" : "");
        if (field(ins, 3, 0, 1))
            break;
    }
    fflush(stderr);
}

const float *nv2a_vsh_constant(uint32_t index)
{
    return s_const[index < NV2A_VSH_CONSTANTS ? index : 0];
}

void nv2a_vsh_constant_component(uint32_t index, uint32_t comp, uint32_t word)
{
    if (index < NV2A_VSH_CONSTANTS && comp < 4)
        memcpy(&s_const[index][comp], &word, 4);
}

/* ---- execution ----------------------------------------------------------- */

typedef struct { float v[4]; } vec4;

/* The program, decoded once rather than per vertex. Every field used to be
 * extracted from the instruction words again for each vertex -- about twenty
 * shifts and masks an instruction, and the interpreter was ~14% of the
 * pushbuffer executor in Blinx 2's hub. Invalidated by any program word or
 * start-slot write. */
typedef struct {
    uint8_t mux, reg, neg, rel;
    uint8_t sw[4];
    uint16_t idx;                           /* input or constant index */
    uint8_t ident;                          /* swizzle is .xyzw */
} VshSrc;

typedef struct {
    VshSrc src[3];
    uint8_t mac, ilu, mac_mask, ilu_mask, tdst;
    uint8_t omask, oaddr, out_ilu, out_o, final;
    uint8_t reads;                          /* bit k: source k is used */
} VshIns;

/* A decoded program, immutable once made: deduplicated by content and
 * never freed, so a pointer to one stays valid for a draw handed to another
 * thread (nv2a_vsh_snapshot). Titles load few distinct programs. */
struct Nv2aVshProg {
    VshIns ins[NV2A_VSH_SLOTS];
    uint32_t n, hash;
    uint16_t inputs;                        /* v[i] some used source reads */
    struct Nv2aVshProg *next;
};
#define VSH_PROG_BUCKETS 256
#define VSH_PROG_MAX     4096
static struct Nv2aVshProg *s_prog_tab[VSH_PROG_BUCKETS];
static unsigned s_prog_count;
static struct Nv2aVshProg s_prog_scratch;   /* when the table is full */
static const struct Nv2aVshProg *s_cur = &s_prog_scratch;
static int s_cur_shared;                    /* s_cur is a table entry */
#define s_dec        (s_cur->ins)
#define s_dec_n      (s_cur->n)

static void decode_src(const uint32_t *ins, int which, VshSrc *d)
{
    uint32_t swz;
    if (which == 0) {
        d->neg = (uint8_t)field(ins, 1, 8, 1); swz = field(ins, 1, 0, 8);
        d->mux = (uint8_t)field(ins, 2, 26, 2); d->reg = (uint8_t)field(ins, 2, 28, 4);
    } else if (which == 1) {
        d->neg = (uint8_t)field(ins, 2, 25, 1); swz = field(ins, 2, 17, 8);
        d->mux = (uint8_t)field(ins, 2, 11, 2); d->reg = (uint8_t)field(ins, 2, 13, 4);
    } else {
        d->neg = (uint8_t)field(ins, 2, 10, 1); swz = field(ins, 2, 2, 8);
        d->mux = (uint8_t)field(ins, 3, 28, 2);
        d->reg = (uint8_t)((field(ins, 2, 0, 2) << 2) | field(ins, 3, 30, 2));
    }
    d->sw[0] = (uint8_t)((swz >> 6) & 3); d->sw[1] = (uint8_t)((swz >> 4) & 3);
    d->sw[2] = (uint8_t)((swz >> 2) & 3); d->sw[3] = (uint8_t)(swz & 3);
    d->rel = (uint8_t)field(ins, 3, 1, 1);
    d->idx = (uint16_t)(d->mux == 2 ? field(ins, 1, 9, 4) : field(ins, 1, 13, 8));
    d->ident = d->sw[0] == 0 && d->sw[1] == 1 && d->sw[2] == 2 && d->sw[3] == 3;
}

/* The current program's hash, kept under RECOMP_VSH_DUMP only: frame
 * traces print it, to tell which dumped program a draw ran. */
uint32_t nv2a_vsh_cur_hash;

/* RECOMP_VSH_DUMP: disassemble each distinct program once, when decoded. */
static void maybe_dump(void)
{
    static int dump = -1;
    static uint32_t dumped[64];
    static int ndumped;

    if (dump < 0)
        dump = getenv("RECOMP_VSH_DUMP") != NULL;
    if (dump && ndumped < 64) {
        /* By content: titles reload different programs into the same slot. */
        uint32_t h = 2166136261u, t;
        int i, seen = 0;
        for (t = s_start_slot; t < NV2A_VSH_SLOTS; t++) {
            for (i = 0; i < 4; i++)
                h = (h ^ s_program[t][i]) * 16777619u;
            if (field(s_program[t], 3, 0, 1))
                break;
        }
        nv2a_vsh_cur_hash = h;
        for (i = 0; i < ndumped; i++)
            if (dumped[i] == h) seen = 1;
        if (!seen) {
            dumped[ndumped++] = h;
            fprintf(stderr, "  [VSH] hash %08X%c", h, 10);
            dump_program(s_start_slot);
        }
    }
}

unsigned long nv2a_vsh_decodes;               /* for RECOMP_VSH_TIME */

static void decode_program(void)
{
    static struct Nv2aVshProg t;
    struct Nv2aVshProg *e;
    uint32_t s, h = 2166136261u;
    size_t i;

    nv2a_vsh_decodes++;
    memset(&t, 0, sizeof t);
    for (s = s_start_slot; s < NV2A_VSH_SLOTS; s++) {
        const uint32_t *ins = s_program[s];
        VshIns *d = &t.ins[t.n++];
        int k;
        for (k = 0; k < 3; k++)
            decode_src(ins, k, &d->src[k]);
        d->mac = (uint8_t)field(ins, 1, 21, 4);
        d->ilu = (uint8_t)field(ins, 1, 25, 3);
        d->mac_mask = (uint8_t)field(ins, 3, 24, 4);
        d->ilu_mask = (uint8_t)field(ins, 3, 16, 4);
        d->tdst = (uint8_t)field(ins, 3, 20, 4);
        d->omask = (uint8_t)field(ins, 3, 12, 4);
        d->oaddr = (uint8_t)field(ins, 3, 3, 8);
        d->out_ilu = (uint8_t)field(ins, 3, 2, 1);
        d->out_o = (uint8_t)field(ins, 3, 11, 1);
        d->final = (uint8_t)field(ins, 3, 0, 1);
        /* Which sources the ops read: every MAC op reads A; B all but mov,
         * add and arl; C add and mad, and every ILU op. An unread source
         * cannot change a result, so the four-lane path skips fetching it. */
        d->reads = 0;
        if (d->mac)
            d->reads |= 1;
        if (d->mac == 2 || (d->mac >= 4 && d->mac <= 12))
            d->reads |= 2;
        if (d->mac == 3 || d->mac == 4 || d->ilu)
            d->reads |= 4;
        for (k = 0; k < 3; k++)
            if ((d->reads >> k & 1) && d->src[k].mux == 2)
                t.inputs |= (uint16_t)(1u << d->src[k].idx);
        if (d->final)
            break;
    }
    for (i = 0; i < t.n * sizeof t.ins[0]; i++)          /* FNV-1a */
        h = (h ^ ((const uint8_t *)t.ins)[i]) * 16777619u;
    t.hash = h;
    for (e = s_prog_tab[h % VSH_PROG_BUCKETS]; e; e = e->next)
        if (e->hash == h && e->n == t.n
            && !memcmp(e->ins, t.ins, t.n * sizeof t.ins[0]))
            break;
    if (!e && s_prog_count < VSH_PROG_MAX
        && (e = (struct Nv2aVshProg *)malloc(sizeof *e)) != NULL) {
        *e = t;
        e->next = s_prog_tab[h % VSH_PROG_BUCKETS];
        s_prog_tab[h % VSH_PROG_BUCKETS] = e;
        s_prog_count++;
    }
    if (e) {
        s_cur = e;
        s_cur_shared = 1;
    } else {
        s_prog_scratch = t;
        s_cur = &s_prog_scratch;
        s_cur_shared = 0;
    }
    s_dec_valid = 1;
    maybe_dump();
}

static vec4 read_src(const VshSrc *d, const float in[NV2A_VSH_INPUTS][4],
                     const vec4 *temp, const vec4 *opos, int a0)
{
    const float *s;
    vec4 r;

    if (d->mux == 1)
        s = (d->reg == 12) ? opos->v : (d->reg < 12 ? temp[d->reg].v : temp[0].v);
    else if (d->mux == 2)
        s = in[d->idx];
    else {
        int ci = d->idx;
        if (d->rel)
            ci += a0;
        if (ci < 0 || ci >= NV2A_VSH_CONSTANTS)
            ci = 0;
        s = s_const[ci];
    }
    r.v[0] = s[d->sw[0]];
    r.v[1] = s[d->sw[1]];
    r.v[2] = s[d->sw[2]];
    r.v[3] = s[d->sw[3]];
    if (d->neg) {
        r.v[0] = -r.v[0]; r.v[1] = -r.v[1];
        r.v[2] = -r.v[2]; r.v[3] = -r.v[3];
    }
    return r;
}

static void write_masked(float *dst, const vec4 *src, uint32_t mask)
{
    if (mask & 8) dst[0] = src->v[0];
    if (mask & 4) dst[1] = src->v[1];
    if (mask & 2) dst[2] = src->v[2];
    if (mask & 1) dst[3] = src->v[3];
}

static vec4 splat(float f)
{
    vec4 r;
    r.v[0] = r.v[1] = r.v[2] = r.v[3] = f;
    return r;
}

static vec4 run_mac(uint32_t op, vec4 a, vec4 b, vec4 c, int *a0)
{
    vec4 r = a;
    int i;

    switch (op) {
    case 1: break;                                            /* mov */
    case 2: for (i = 0; i < 4; i++) r.v[i] = a.v[i] * b.v[i]; break;
    case 3: for (i = 0; i < 4; i++) r.v[i] = a.v[i] + c.v[i]; break;
    case 4: for (i = 0; i < 4; i++) r.v[i] = a.v[i] * b.v[i] + c.v[i]; break;
    case 5: r = splat(a.v[0]*b.v[0] + a.v[1]*b.v[1] + a.v[2]*b.v[2]); break;
    case 6: r = splat(a.v[0]*b.v[0] + a.v[1]*b.v[1] + a.v[2]*b.v[2]
                      + b.v[3]); break;                        /* dph */
    case 7: r = splat(a.v[0]*b.v[0] + a.v[1]*b.v[1] + a.v[2]*b.v[2]
                      + a.v[3]*b.v[3]); break;
    case 8: r.v[0] = 1.0f; r.v[1] = a.v[1] * b.v[1];            /* dst */
            r.v[2] = a.v[2]; r.v[3] = b.v[3]; break;
    case 9: for (i = 0; i < 4; i++) r.v[i] = a.v[i] < b.v[i] ? a.v[i] : b.v[i]; break;
    /* max is a > b ? a : b -- b on a tie, so max(-0, 0) is 0. That is what
     * MSVC compiled the old `a >= b ? a : b` to anyway (maxss), and it is
     * exactly maxps, which the four-lane path uses. */
    case 10: for (i = 0; i < 4; i++) r.v[i] = a.v[i] > b.v[i] ? a.v[i] : b.v[i]; break;
    case 11: for (i = 0; i < 4; i++) r.v[i] = a.v[i] < b.v[i] ? 1.0f : 0.0f; break;
    case 12: for (i = 0; i < 4; i++) r.v[i] = a.v[i] >= b.v[i] ? 1.0f : 0.0f; break;
    case 13: *a0 = (int)floorf(a.v[0] + 0.001f); break;         /* arl */
    default: break;
    }
    return r;
}

static vec4 run_ilu(uint32_t op, vec4 c)
{
    float x = c.v[0];
    vec4 r = c;

    switch (op) {
    case 1: break;                                            /* mov */
    case 2: r = splat(x == 0.0f ? INFINITY : 1.0f / x); break;  /* rcp */
    case 3: {                                                  /* rcc */
        float f = 1.0f / x;
        if (fabsf(f) < 5.42101e-20f) f = f < 0 ? -5.42101e-20f : 5.42101e-20f;
        if (fabsf(f) > 1.884467e19f) f = f < 0 ? -1.884467e19f : 1.884467e19f;
        r = splat(f);
        break;
    }
    case 4: r = splat(1.0f / sqrtf(fabsf(x))); break;           /* rsq */
    case 5: {                                                  /* exp */
        float fl = floorf(x);
        r.v[0] = exp2f(fl); r.v[1] = x - fl; r.v[2] = exp2f(x); r.v[3] = 1.0f;
        break;
    }
    case 6: {                                                  /* log */
        float ax = fabsf(x);
        if (ax == 0.0f) {
            r.v[0] = r.v[2] = -INFINITY; r.v[1] = 1.0f;
        } else {
            float e = floorf(log2f(ax));
            r.v[0] = e; r.v[1] = ax / exp2f(e); r.v[2] = log2f(ax);
        }
        r.v[3] = 1.0f;
        break;
    }
    case 7: {                                                  /* lit */
        float nl = c.v[0] > 0.0f ? c.v[0] : 0.0f;
        float nh = c.v[1] > 0.0f ? c.v[1] : 0.0f;
        float p = c.v[3] < -127.9961f ? -127.9961f
                : (c.v[3] > 127.9961f ? 127.9961f : c.v[3]);
        r.v[0] = 1.0f; r.v[1] = nl;
        r.v[2] = c.v[0] > 0.0f ? powf(nh, p) : 0.0f; r.v[3] = 1.0f;
        break;
    }
    default: break;
    }
    return r;
}

int nv2a_vsh_run(const float in[NV2A_VSH_INPUTS][4], Nv2aVshOutput *out)
{
    vec4 temp[12], opos;
    float outregs[13][4];
    int a0 = 0;
    uint32_t s;

    memset(temp, 0, sizeof temp);
    memset(&opos, 0, sizeof opos);
    memset(outregs, 0, sizeof outregs);
    outregs[3][3] = outregs[4][3] = 1.0f;      /* colours default opaque */
    /* Unwritten texture coordinates are (0,0,0,1), as xemu initialises them:
     * a projective stage divides by q, and q = 0 would put every texel of an
     * unwritten stage at infinity. */
    outregs[9][3] = outregs[10][3] = outregs[11][3] = outregs[12][3] = 1.0f;

    if (!s_dec_valid)
        decode_program();
    for (s = 0; s < s_dec_n; s++) {
        const VshIns *ins = &s_dec[s];
        uint32_t mac = ins->mac, ilu = ins->ilu;
        uint32_t mac_mask = ins->mac_mask, ilu_mask = ins->ilu_mask;
        uint32_t tdst = ins->tdst;
        uint32_t omask = ins->omask, oaddr = ins->oaddr;
        vec4 a = read_src(&ins->src[0], in, temp, &opos, a0);
        vec4 b = read_src(&ins->src[1], in, temp, &opos, a0);
        vec4 c = read_src(&ins->src[2], in, temp, &opos, a0);
        vec4 mres = {{0, 0, 0, 0}}, ires = {{0, 0, 0, 0}};

        if (mac)
            mres = run_mac(mac, a, b, c, &a0);
        if (ilu)
            ires = run_ilu(ilu, c);

        if (mac && mac != 13 && mac_mask) {
            float *d = tdst == 12 ? opos.v : (tdst < 12 ? temp[tdst].v : NULL);
            if (d) write_masked(d, &mres, mac_mask);
        }
        if (ilu && ilu_mask) {
            /* Paired with any MAC op, the ILU can only write R1 -- even when
             * the MAC half writes no temp at all. The D3D epilogue is exactly
             * that: `mul o[0].xyz` beside `rcc R1.x`, then `mad` with R1.x.
             * Sending the rcc to the temp field (R7 there) left R1.x at 0 and
             * put every vertex of every 3D batch at c[59]. */
            uint32_t it = mac ? 1u : tdst;
            float *d = it == 12 ? opos.v : (it < 12 ? temp[it].v : NULL);
            if (d) write_masked(d, &ires, ilu_mask);
        }
        if (omask) {
            const vec4 *src = ins->out_ilu ? &ires : &mres;
            if (!ins->out_o) {                  /* to c[] */
                if (s_cxt_write && oaddr < NV2A_VSH_CONSTANTS)
                    write_masked(s_const[oaddr], src, omask);
            } else if (oaddr == 0) {
                write_masked(opos.v, src, omask);
            } else if (oaddr < 13) {
                write_masked(outregs[oaddr], src, omask);
            }
        }
        if (ins->final) {
            memcpy(out->pos, opos.v, sizeof out->pos);
            memcpy(out->d0, outregs[3], sizeof out->d0);
            memcpy(out->d1, outregs[4], sizeof out->d1);
            memcpy(out->fog, outregs[5], sizeof out->fog);
            memcpy(out->tex[0], outregs[9], sizeof out->tex[0]);
            memcpy(out->tex[1], outregs[10], sizeof out->tex[1]);
            memcpy(out->tex[2], outregs[11], sizeof out->tex[2]);
            memcpy(out->tex[3], outregs[12], sizeof out->tex[3]);
            return 1;
        }
    }
    return 0;
}

/* ---- four vertices at once ------------------------------------------------
 *
 * The same decoded program run over four vertices in lockstep, each register
 * held as four lanes per component in SSE registers. Decoding, dispatch and
 * operand routing are paid once per instruction instead of once per vertex,
 * and the arithmetic is one SSE op for four. Results are bit-identical to
 * nv2a_vsh_run: the same operations in the same order (dot products summed
 * left to right, correctly rounded divide and square root), compares written
 * as the scalar code's ?: rather than minps/maxps, and RCC/EXP/LOG/LIT done
 * per lane by the scalar code. A program that writes constants (CXT_WRITE)
 * runs per vertex instead, since a later vertex reads what an earlier one
 * wrote. */
#include <emmintrin.h>

typedef struct { __m128 c[4]; } vec4x4;   /* component k, lanes 0..3 */

/* Source operand d, as a pointer: straight at the register when the swizzle
 * is .xyzw and there is no negate, else swizzled into `scratch`. Operands
 * used to be built and returned by value -- three 64-byte copies a source --
 * which was most of the interpreter's time, not the arithmetic. */
static const vec4x4 *src4(const VshSrc *d, const vec4x4 *vin,
                          const vec4x4 *temp, const vec4x4 *opos,
                          const int a0[4], const float (*s_const)[4],
                          vec4x4 *scratch)
{
    const vec4x4 *s;
    vec4x4 g;
    int k;

    if (d->mux == 1) {
        s = (d->reg == 12) ? opos : &temp[d->reg < 12 ? d->reg : 0];
    } else if (d->mux == 2) {
        s = &vin[d->idx];
    } else if (!d->rel) {
        const float *c = s_const[d->idx < NV2A_VSH_CONSTANTS ? d->idx : 0];
        for (k = 0; k < 4; k++)
            scratch->c[k] = _mm_set1_ps(c[d->sw[k]]);
        if (d->neg) {
            const __m128 sign = _mm_set1_ps(-0.0f);
            for (k = 0; k < 4; k++)
                scratch->c[k] = _mm_xor_ps(scratch->c[k], sign);
        }
        return scratch;
    } else {
        int ci[4], l;
        for (l = 0; l < 4; l++) {
            ci[l] = d->idx + a0[l];
            if (ci[l] < 0 || ci[l] >= NV2A_VSH_CONSTANTS)
                ci[l] = 0;
        }
        for (k = 0; k < 4; k++)
            g.c[k] = _mm_set_ps(s_const[ci[3]][k], s_const[ci[2]][k],
                                s_const[ci[1]][k], s_const[ci[0]][k]);
        s = &g;
    }
    if (d->ident && !d->neg && s != &g)
        return s;
    for (k = 0; k < 4; k++)
        scratch->c[k] = s->c[d->sw[k]];
    if (d->neg) {
        const __m128 sign = _mm_set1_ps(-0.0f);
        for (k = 0; k < 4; k++)
            scratch->c[k] = _mm_xor_ps(scratch->c[k], sign);
    }
    return scratch;
}

static void wr4(vec4x4 *dst, const vec4x4 *src, uint32_t mask)
{
    if (mask & 8) dst->c[0] = src->c[0];
    if (mask & 4) dst->c[1] = src->c[1];
    if (mask & 2) dst->c[2] = src->c[2];
    if (mask & 1) dst->c[3] = src->c[3];
}

static __m128 sel4(__m128 m, __m128 a, __m128 b)
{
    return _mm_or_ps(_mm_and_ps(m, a), _mm_andnot_ps(m, b));
}

static void mac4(uint32_t op, const vec4x4 *a, const vec4x4 *b,
                 const vec4x4 *c, int a0[4], vec4x4 *rp)
{
    __m128 t, one = _mm_set1_ps(1.0f), zero = _mm_setzero_ps();
    int k;

    switch (op) {
    case 1: *rp = *a; break;
    case 2: for (k = 0; k < 4; k++) rp->c[k] = _mm_mul_ps(a->c[k], b->c[k]); break;
    case 3: for (k = 0; k < 4; k++) rp->c[k] = _mm_add_ps(a->c[k], c->c[k]); break;
    case 4: for (k = 0; k < 4; k++)
                rp->c[k] = _mm_add_ps(_mm_mul_ps(a->c[k], b->c[k]), c->c[k]);
            break;
    case 5: case 6: case 7:
        t = _mm_add_ps(_mm_add_ps(_mm_mul_ps(a->c[0], b->c[0]),
                                  _mm_mul_ps(a->c[1], b->c[1])),
                       _mm_mul_ps(a->c[2], b->c[2]));
        if (op == 6) t = _mm_add_ps(t, b->c[3]);
        if (op == 7) t = _mm_add_ps(t, _mm_mul_ps(a->c[3], b->c[3]));
        rp->c[0] = rp->c[1] = rp->c[2] = rp->c[3] = t;
        break;
    case 8:
        rp->c[0] = one;
        rp->c[1] = _mm_mul_ps(a->c[1], b->c[1]);
        rp->c[2] = a->c[2];
        rp->c[3] = b->c[3];
        break;
    /* minps/maxps are exactly a < b ? a : b and a > b ? a : b, NaN and
     * signed zeros included -- the scalar code's definitions. */
    case 9:  for (k = 0; k < 4; k++) rp->c[k] = _mm_min_ps(a->c[k], b->c[k]); break;
    case 10: for (k = 0; k < 4; k++) rp->c[k] = _mm_max_ps(a->c[k], b->c[k]); break;
    case 11: for (k = 0; k < 4; k++)
                 rp->c[k] = sel4(_mm_cmplt_ps(a->c[k], b->c[k]), one, zero);
             break;
    case 12: for (k = 0; k < 4; k++)
                 rp->c[k] = sel4(_mm_cmpge_ps(a->c[k], b->c[k]), one, zero);
             break;
    case 13: {
        float x[4];
        int l;
        *rp = *a;
        _mm_storeu_ps(x, a->c[0]);
        for (l = 0; l < 4; l++)
            a0[l] = (int)floorf(x[l] + 0.001f);
        break;
    }
    default: *rp = *a; break;
    }
}

static void ilu4(uint32_t op, const vec4x4 *c, vec4x4 *rp)
{
    int k, l;

    switch (op) {
    case 1: *rp = *c; break;
    case 2: {                                       /* rcp: 1/x, +inf for +-0 */
        __m128 x = c->c[0];
        __m128 v = _mm_div_ps(_mm_set1_ps(1.0f), x);
        v = sel4(_mm_cmpeq_ps(x, _mm_setzero_ps()), _mm_set1_ps(INFINITY), v);
        rp->c[0] = rp->c[1] = rp->c[2] = rp->c[3] = v;
        break;
    }
    case 3: {                                       /* rcc: 1/x, clamped */
        /* As the scalar code: the magnitude is held to [5.42101e-20,
         * 1.884467e19] keeping the sign (f < 0), NaN passes through. Done per
         * lane in scalar it was the D3D epilogue's cost on every vertex. */
        const __m128 sign = _mm_set1_ps(-0.0f), zero = _mm_setzero_ps();
        const __m128 lo = _mm_set1_ps(5.42101e-20f), hi = _mm_set1_ps(1.884467e19f);
        __m128 f = _mm_div_ps(_mm_set1_ps(1.0f), c->c[0]);
        __m128 neg = _mm_cmplt_ps(f, zero), mag = _mm_andnot_ps(sign, f);
        f = sel4(_mm_cmplt_ps(mag, lo), sel4(neg, _mm_or_ps(lo, sign), lo), f);
        f = sel4(_mm_cmpgt_ps(mag, hi), sel4(neg, _mm_or_ps(hi, sign), hi), f);
        rp->c[0] = rp->c[1] = rp->c[2] = rp->c[3] = f;
        break;
    }
    case 4: {                                       /* rsq: 1/sqrt(|x|) */
        __m128 x = _mm_andnot_ps(_mm_set1_ps(-0.0f), c->c[0]);
        __m128 v = _mm_div_ps(_mm_set1_ps(1.0f), _mm_sqrt_ps(x));
        rp->c[0] = rp->c[1] = rp->c[2] = rp->c[3] = v;
        break;
    }
    default: {                                      /* per lane, scalar */
        float cs[4][4], rs[4][4];
        for (k = 0; k < 4; k++)
            _mm_storeu_ps(cs[k], c->c[k]);
        for (l = 0; l < 4; l++) {
            vec4 cv, rv;
            for (k = 0; k < 4; k++)
                cv.v[k] = cs[k][l];
            rv = run_ilu(op, cv);
            for (k = 0; k < 4; k++)
                rs[k][l] = rv.v[k];
        }
        for (k = 0; k < 4; k++)
            rp->c[k] = _mm_loadu_ps(rs[k]);
        break;
    }
    }
}

/* Four vertices through program `p` with constants `cst`; touches no state
 * of its own, so any thread may run it. */
static int run4_with(const struct Nv2aVshProg *p, const float (*cst)[4],
                     const float (*const in[4])[4], Nv2aVshOutput *out[4])
{
    vec4x4 temp[12], opos, outregs[13], vin[NV2A_VSH_INPUTS];
    int a0[4] = {0, 0, 0, 0};
    uint32_t s;
    int k, l;

    memset(temp, 0, sizeof temp);
    memset(&opos, 0, sizeof opos);
    memset(outregs, 0, sizeof outregs);
    outregs[3].c[3] = outregs[4].c[3] = _mm_set1_ps(1.0f);
    outregs[9].c[3] = outregs[10].c[3] = outregs[11].c[3]
                    = outregs[12].c[3] = _mm_set1_ps(1.0f);
    /* Each input the program reads, as lanes, once -- not gathered from the
     * four vertices again at every read. */
    for (k = 0; k < NV2A_VSH_INPUTS; k++)
        if (p->inputs >> k & 1) {
            __m128 r0 = _mm_loadu_ps(in[0][k]), r1 = _mm_loadu_ps(in[1][k]);
            __m128 r2 = _mm_loadu_ps(in[2][k]), r3 = _mm_loadu_ps(in[3][k]);
            _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
            vin[k].c[0] = r0; vin[k].c[1] = r1;
            vin[k].c[2] = r2; vin[k].c[3] = r3;
        }

    for (s = 0; s < p->n; s++) {
        const VshIns *ins = &p->ins[s];
        vec4x4 sa, sb, sc, mres, ires;
        const vec4x4 *a = NULL, *b = NULL, *c = NULL;

        if (ins->reads & 1) a = src4(&ins->src[0], vin, temp, &opos, a0, cst, &sa);
        if (ins->reads & 2) b = src4(&ins->src[1], vin, temp, &opos, a0, cst, &sb);
        if (ins->reads & 4) c = src4(&ins->src[2], vin, temp, &opos, a0, cst, &sc);
        /* An idle unit's result is 0: an output may still name it. */
        if (ins->mac)
            mac4(ins->mac, a, b, c, a0, &mres);
        else
            memset(&mres, 0, sizeof mres);
        if (ins->ilu)
            ilu4(ins->ilu, c, &ires);
        else
            memset(&ires, 0, sizeof ires);
        if (ins->mac && ins->mac != 13 && ins->mac_mask) {
            vec4x4 *d = ins->tdst == 12 ? &opos
                      : (ins->tdst < 12 ? &temp[ins->tdst] : NULL);
            if (d) wr4(d, &mres, ins->mac_mask);
        }
        if (ins->ilu && ins->ilu_mask) {
            uint32_t it = ins->mac ? 1u : ins->tdst;   /* see nv2a_vsh_run */
            vec4x4 *d = it == 12 ? &opos : (it < 12 ? &temp[it] : NULL);
            if (d) wr4(d, &ires, ins->ilu_mask);
        }
        if (ins->omask && ins->out_o) {               /* c[] writes: s_cxt_write */
            const vec4x4 *src = ins->out_ilu ? &ires : &mres;
            if (ins->oaddr == 0)
                wr4(&opos, src, ins->omask);
            else if (ins->oaddr < 13)
                wr4(&outregs[ins->oaddr], src, ins->omask);
        }
        if (ins->final) {
            /* Back to one vertex per output: a 4x4 transpose per register. */
            static const int reg_of[8] = {0, 3, 4, 5, 9, 10, 11, 12};
            int j;
            for (j = 0; j < 8; j++) {
                const vec4x4 *rg = reg_of[j] ? &outregs[reg_of[j]] : &opos;
                __m128 r0 = rg->c[0], r1 = rg->c[1], r2 = rg->c[2], r3 = rg->c[3];
                size_t off = j == 0 ? offsetof(Nv2aVshOutput, pos)
                           : j == 1 ? offsetof(Nv2aVshOutput, d0)
                           : j == 2 ? offsetof(Nv2aVshOutput, d1)
                           : j == 3 ? offsetof(Nv2aVshOutput, fog)
                           : offsetof(Nv2aVshOutput, tex) + (size_t)(j - 4) * 16;
                _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
                _mm_storeu_ps((float *)((char *)out[0] + off), r0);
                _mm_storeu_ps((float *)((char *)out[1] + off), r1);
                _mm_storeu_ps((float *)((char *)out[2] + off), r2);
                _mm_storeu_ps((float *)((char *)out[3] + off), r3);
            }
            return 1;
        }
    }
    return 0;
}

int nv2a_vsh_run4(const float (*const in[4])[4], Nv2aVshOutput *out[4])
{
    int l;

    if (!s_dec_valid)
        decode_program();
    if (s_cxt_write) {
        for (l = 0; l < 4; l++)
            if (!nv2a_vsh_run(in[l], out[l]))
                return 0;
        return 1;
    }
    return run4_with(s_cur, (const float (*)[4])s_const, in, out);
}

const Nv2aVshProg *nv2a_vsh_snapshot(void)
{
    if (!s_dec_valid)
        decode_program();
    return s_cur_shared ? s_cur : NULL;
}

const float (*nv2a_vsh_constants(void))[4]
{
    return (const float (*)[4])s_const;
}

int nv2a_vsh_cxt_write(void)
{
    return s_cxt_write != 0;
}

int nv2a_vsh_transform(const Nv2aVshProg *p, const float (*cst)[4],
                       const float (*in)[NV2A_VSH_INPUTS][4], uint32_t n,
                       Nv2aVshOutput *out)
{
    uint32_t i, l;

    for (i = 0; i < n; i += 4) {
        const float (*ip[4])[4];
        Nv2aVshOutput tail[4], *op[4];
        for (l = 0; l < 4; l++) {
            uint32_t j = i + l < n ? i + l : n - 1;
            ip[l] = (const float (*)[4])in[j];
            op[l] = i + l < n ? &out[i + l] : &tail[l];
        }
        if (!run4_with(p, cst, ip, op))
            return 0;
    }
    return 1;
}
