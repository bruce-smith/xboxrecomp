# gpu_vk_present

Milestone 1 of the Vulkan backend (`src/gpu_vk`): opens a window, presents
a moving 640x480 test pattern through `xbox_vk_present_frame` for about two
seconds, resizes the window halfway (the swapchain must follow), minimises
it briefly (nothing shown, no error), and exits 0 if every presentable frame
was presented.

Needs a Vulkan 1.3 driver and a desktop session. Standalone CMake project:

    cmake -S . -B build -G "Visual Studio 17 2022" -A x64
    cmake --build build --config Release
    ctest --test-dir build -C Release --output-on-failure

`RECOMP_VK_DEVICE=<n|name>` picks the GPU (the log lists them).
