cmake_minimum_required(VERSION 3.20)
project(xboxrecomp VERSION 0.9.0 LANGUAGES C)

# Xbox Static Recompilation Runtime Library
# Provides link-time libraries for statically recompiled Xbox games.
# Pattern: extract emulator subsystems (xemu) into reusable static libraries.
#
# Libraries:
#   xbox_kernel    - Xbox kernel replacement layer (147 imports → Win32)
#   xbox_d3d8      - D3D8 → D3D11 graphics compatibility
#   xbox_dsound    - DirectSound → software mixer compatibility
#   xbox_input     - Xbox gamepad → XInput compatibility
#   xbox_apu       - MCPX APU audio emulation (extracted from xemu)
#   xbox_nv2a      - NV2A GPU emulation (extracted from xemu)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Common definitions for all targets
add_compile_definitions(
    WIN32_LEAN_AND_MEAN
    NOMINMAX
)

# Option to build examples

# ── Runtime Libraries ─────────────────────────────────────────

# Platform: Win32/NT type vocabulary + POSIX implementation of generic
# Win32 host primitives. Carries no Xbox semantics; provides the foundation
# the kernel HLE and the graphics/audio/input layers build on.
add_subdirectory(src/platform)

# Core: Xbox kernel replacement (memory layout, file I/O, threading, sync)
add_subdirectory(src/kernel)

# Graphics: D3D8 → D3D11 compatibility layer
add_subdirectory(src/d3d)

# Audio: DirectSound → software mixer compatibility
add_subdirectory(src/audio)

# Audio: MCPX APU hardware emulation (voice processor, DSP)
add_subdirectory(src/apu)

# GPU: NV2A hardware emulation (register handlers, MMIO, push buffer)
add_subdirectory(src/nv2a)

# USB: MCPX OHCI host controllers. A title's XAPI is statically linked and
# drives these registers directly, so a pad only exists if they do.
add_subdirectory(src/usb)

# Settings (ini next to the exe) and the in-game menu (Escape; Dear ImGui)
add_subdirectory(src/ui)

# Input: Xbox gamepad → XInput
add_subdirectory(src/input)

# Network play: lockstep input over UDP (RECOMP_NET)
if(WIN32)
    add_library(xbox_net STATIC src/net/netplay.c)
    target_include_directories(xbox_net PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/net)
    target_link_libraries(xbox_net PUBLIC platform xbox_usb ws2_32)
endif()

# GPU: Vulkan backend (RECOMP_VK_PRESENT / RECOMP_GPU=vulkan). Fetches
# Vulkan-Headers, volk and glslang at configure time (src/gpu_vk); the loader
# is opened at run time, so the exe still starts (software path) where
# Vulkan is missing. -DXBOXRECOMP_VULKAN=OFF builds without it.
option(XBOXRECOMP_VULKAN "Build the Vulkan GPU backend" ON)
if(XBOXRECOMP_VULKAN)
    add_subdirectory(src/gpu_vk)
    # The executor installs the backend when RECOMP_GPU=vulkan.
    target_link_libraries(xbox_kernel PRIVATE xbox_gpu_vk)
    target_compile_definitions(xbox_kernel PRIVATE XBOXRECOMP_HAVE_VULKAN)
endif()

# Shared FMV player (Media Foundation -> D3D8 texture)
add_subdirectory(src/video)

# ── Umbrella target ───────────────────────────────────────────

# Convenience target that links all runtime libraries
add_library(xboxrecomp INTERFACE)
target_link_libraries(xboxrecomp INTERFACE
    xbox_kernel
    xbox_d3d8
    xbox_dsound
    xbox_input
    xbox_usb
    xbox_apu
    xbox_nv2a
    xbox_video
    xbox_settings
)
if(TARGET xbox_ui)
    target_link_libraries(xboxrecomp INTERFACE xbox_ui)
endif()
if(TARGET xbox_net)
    target_link_libraries(xboxrecomp INTERFACE xbox_net)
endif()

