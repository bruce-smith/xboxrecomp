/**
 * Register combiners to GLSL (vk_psh.c). The key is everything the shader
 * text depends on; equal keys give equal shaders.
 */
#ifndef XBOXRECOMP_VK_PSH_H
#define XBOXRECOMP_VK_PSH_H

#include <stdint.h>
#include "nv2a_backend.h"

typedef struct {
    uint32_t rc_seen;
    uint32_t control, stage_program, final0, final1;
    uint32_t color_icw[8], color_ocw[8], alpha_icw[8], alpha_ocw[8];
    uint32_t clip_plane_mode;           /* CLIPPLANE stages' bits only */
    uint8_t  cube[4];                   /* stage samples a cube map */
    uint8_t  bump_in[4];                /* BUMPENVMAP: the input stage */
    uint8_t  bump_signed[4];            /* its G (bit 0) / B (bit 1) signed */
} VkPshKey;

void  vk_psh_key(const Nv2aDrawState *st, VkPshKey *k);
char *vk_psh_glsl(const VkPshKey *k);   /* malloc'd; free() it */

/* The uniform block both shaders read (std140; vk_psh.c's k_head). */
typedef struct {
    float   vp[4];          /* 2/W, 2/H, 1/zmax, 0 */
    float   k[4], o[4];     /* viewport scale c[58], offset c[59] */
    float   fogp[4];        /* enable, mode, param0, param1 */
    float   fogc[4];        /* fog colour r g b */
    float   c0[8][4], c1[8][4];
    float   fc0[4], fc1[4];
    float   tsz[4][4];      /* 1/w, 1/h, normalised, valid */
    float   bm[4][4];       /* bump matrix 00 01 10 11 */
    float   bl[4][4];       /* bump luminance scale, offset */
    int32_t at[4];          /* alpha test enable, func, ref */
    /* Vertex pulling (vk_vsh.c), programs on the GPU only: */
    uint32_t attr[16][4];   /* byte offset of element imin in the blob, stride,
                               type | size << 4 | 0x100 no array */
    float   imm[16][4];     /* inline values */
    uint32_t imin[4];
} VkDrawUniforms;

#endif
