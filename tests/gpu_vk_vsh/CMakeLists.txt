cmake_minimum_required(VERSION 3.20)
project(gpu_vk_vsh_test C CXX)
set(CMAKE_C_STANDARD 11)
add_subdirectory(../../src/gpu_vk gpu_vk)
add_executable(gpu_vk_vsh_test test_main.c ../../src/kernel/nv2a_vsh_interp.c)
# The test drives Vulkan directly: the backend's internals.
target_link_libraries(gpu_vk_vsh_test PRIVATE xbox_gpu_vk xbox_gpu_vk_internal)
enable_testing()
add_test(NAME gpu_vk_vsh_test COMMAND gpu_vk_vsh_test)
