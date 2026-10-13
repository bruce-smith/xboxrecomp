# gpu_vk_vsh

NV2A vertex programs translated to GLSL (`src/gpu_vk/vk_vsh.c`) against the
CPU interpreter (`src/kernel/nv2a_vsh_interp.c`, `nv2a_vsh_run`).

Random programs (every MAC and ILU op, swizzles, negation, masks, temps,
a0-relative constants, all output registers) run on 64 vertices whose
inputs mix the vertex formats the shader pulls itself (float, D3DCOLOR,
unsigned byte, signed short normalised and not, the packed 11:11:10
normal) plus attributes read from their inline value. The vertex shader
stores the program's outputs (oPos, oD0/oD1 clamped, oFog, oT0-3) to a
buffer; each must equal the interpreter's within a relative 1e-4 (1e-3
after EXP/LOG/LIT/RSQ, which GPUs compute less exactly).

    cmake -S . -B build -G "Visual Studio 17 2022" -A x64
    cmake --build build --config Release
    ctest --test-dir build -C Release --output-on-failure
