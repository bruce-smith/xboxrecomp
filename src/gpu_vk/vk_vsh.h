/**
 * NV2A vertex programs to GLSL (vk_vsh.c).
 */
#ifndef XBOXRECOMP_VK_VSH_H
#define XBOXRECOMP_VK_VSH_H

#include <stdint.h>

/* The program can run on the GPU: it ends in FINAL and writes no
 * constants (those run per vertex on the CPU, in order). */
int   vk_vsh_supported(const uint32_t (*w)[4], uint32_t n);
/* GLSL for program w[0..n). `store`, for tests: extra GLSL defining
 * test_store(uint vtx), called after the outputs are written. malloc'd. */
char *vk_vsh_glsl(const uint32_t (*w)[4], uint32_t n, const char *store);

#endif
