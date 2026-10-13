# Vulkan GPU backend (RECOMP_GPU=vulkan / RECOMP_VK_PRESENT=1).
#
# Fetched at configure time, so no Vulkan SDK is needed to build:
#   Vulkan-Headers (Apache-2.0 OR MIT): headers only.
#   volk (MIT): loads the Vulkan loader (vulkan-1.dll / libvulkan.so.1) at
#   run time, so there is no import library and an exe on a machine without
#   Vulkan still starts and keeps the software path.
#
# Platform-neutral C except vk_platform_win32.c, the one file that turns a
# native window into a VkSurfaceKHR.
include(FetchContent)

set(XBOXRECOMP_VK_SDK_TAG vulkan-sdk-1.4.350.0)
FetchContent_Declare(xbr_vulkan_headers
    URL https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/${XBOXRECOMP_VK_SDK_TAG}.tar.gz
    URL_HASH SHA256=70270d10bf2c1e074a06ee37a50b75d332993d1b80a1d9526eeed2da6d82ed22
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR do-not-add)
FetchContent_Declare(xbr_volk
    URL https://github.com/zeux/volk/archive/refs/tags/${XBOXRECOMP_VK_SDK_TAG}.tar.gz
    URL_HASH SHA256=a04f26f76e9a4f9acf936bd2c159f5c4c8348f8ebaf118ff72ba6a9637ad3e80
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR do-not-add)
# glslang (BSD-3-Clause / Apache-2.0): GLSL -> SPIR-V at run time, for the
# shaders generated per vertex program and combiner setup. Built static,
# without the optimizer, HLSL or the command-line tools.
FetchContent_Declare(xbr_glslang
    URL https://github.com/KhronosGroup/glslang/archive/refs/tags/${XBOXRECOMP_VK_SDK_TAG}.tar.gz
    URL_HASH SHA256=a6885b1631fd77c89cd689b939cf2b3032c5ec13ee99250270d34bcad1efc10c
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
set(ENABLE_OPT OFF CACHE BOOL "" FORCE)
set(ENABLE_HLSL OFF CACHE BOOL "" FORCE)
set(ENABLE_GLSLANG_BINARIES OFF CACHE BOOL "" FORCE)
set(ENABLE_SPVREMAPPER OFF CACHE BOOL "" FORCE)
set(GLSLANG_TESTS OFF CACHE BOOL "" FORCE)
set(GLSLANG_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BUILD_EXTERNAL OFF CACHE BOOL "" FORCE)
set(ENABLE_CTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(xbr_vulkan_headers xbr_volk xbr_glslang)

set(GPU_VK_SOURCES
    vk_device.c
    vk_present.c
    vk_shader.c
    vk_backend.c
    vk_psh.c
    ${xbr_volk_SOURCE_DIR}/volk.c
)
if(WIN32)
    list(APPEND GPU_VK_SOURCES vk_platform_win32.c)
    # The window-system extension is declared only where it is used: volk
    # loads it, the surface file calls it, nothing else sees <windows.h>.
    set_source_files_properties(${xbr_volk_SOURCE_DIR}/volk.c vk_platform_win32.c
        PROPERTIES COMPILE_DEFINITIONS VK_USE_PLATFORM_WIN32_KHR)
endif()

add_library(xbox_gpu_vk STATIC ${GPU_VK_SOURCES})
target_include_directories(xbox_gpu_vk
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${xbr_vulkan_headers_SOURCE_DIR}/include ${xbr_volk_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/../kernel)
# glslang's C interface; the backend also calls back into the executor
# (nv2a_exec_texel, guest memory), which links it in turn.
target_link_libraries(xbox_gpu_vk PRIVATE glslang glslang-default-resource-limits)
target_compile_definitions(xbox_gpu_vk PRIVATE VK_NO_PROTOTYPES)
if(MSVC)
    target_compile_options(xbox_gpu_vk PRIVATE /W3 /O2)
    target_compile_definitions(xbox_gpu_vk PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_link_libraries(xbox_gpu_vk PRIVATE ${CMAKE_DL_LIBS})
endif()
