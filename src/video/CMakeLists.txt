# Shared FMV player: Media Foundation decode onto a D3D8 texture.
add_library(xbox_video STATIC
    video_player.c
    video_pump.c
)
# The framebuffer window: GDI on Windows, SDL2 elsewhere.
if(WIN32)
    target_sources(xbox_video PRIVATE fb_present.c)
else()
    target_sources(xbox_video PRIVATE fb_present_sdl.c)
    find_package(SDL2 REQUIRED)
    target_link_libraries(xbox_video PRIVATE SDL2::SDL2)
endif()
target_include_directories(xbox_video PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_link_libraries(xbox_video PUBLIC xbox_d3d8)
if(TARGET xbox_ui)
    target_link_libraries(xbox_video PRIVATE xbox_ui)
    target_compile_definitions(xbox_video PRIVATE XBOXRECOMP_HAVE_UI)
endif()
if(WIN32)
    target_link_libraries(xbox_video PUBLIC mfplat mfreadwrite mfuuid mf ole32)
endif()
if(TARGET xbox_gpu_vk)
    target_link_libraries(xbox_video PRIVATE xbox_gpu_vk)
    target_compile_definitions(xbox_video PRIVATE XBOXRECOMP_HAVE_VULKAN)
endif()
