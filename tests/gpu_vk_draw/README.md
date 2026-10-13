# gpu_vk_draw

Headless test of the Vulkan rendering backend (`src/gpu_vk/vk_backend.c`)
through the executor's backend interface (`src/kernel/nv2a_backend.h`),
without the executor: a fake guest memory stands in for the title's, and
`nv2a_exec_texel` for the software texel decoder.

It clears a 64x64 surface, draws triangles through the interface with
depth, culling, scissor, blending, alpha test and a texture, flips (the
backend reads the surface back into guest memory) and checks pixels
against what the software rasteriser would write. Also checks a
visibility report, and runs 500 random register-combiner programs (four
random textures, random constants, general and final combiners) through the
GPU and through the software combiners (`nv2a_rc_eval`): every pixel must
agree within 1/255.

    cmake -S . -B build -G "Visual Studio 17 2022" -A x64
    cmake --build build --config Release
    ctest --test-dir build -C Release --output-on-failure

Needs a Vulkan 1.3 driver; no window. `RECOMP_VK_DEVICE=<n|name>` picks
the GPU.
