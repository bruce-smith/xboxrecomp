# The in-game menu (Dear ImGui, MIT) and the settings store it edits.
#
# xbox_settings is plain C with no dependencies, so the input layer can read
# its bindings without pulling in the menu. xbox_ui is the menu itself:
# ImGui's Win32 backend for input on Windows, its SDL2 backend elsewhere
# (fed by video/fb_present_sdl.c); output is the menu's own software overlay
# on both.

add_library(xbox_settings STATIC settings.c)
target_include_directories(xbox_settings PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

enable_language(CXX)
include(FetchContent)
FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.91.9b
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(imgui)

if(WIN32)
    set(XBOX_UI_BACKEND ${imgui_SOURCE_DIR}/backends/imgui_impl_win32.cpp)
else()
    set(XBOX_UI_BACKEND ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl2.cpp)
endif()
add_library(xbox_ui STATIC
    ui_menu.cpp
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${XBOX_UI_BACKEND}
)
target_include_directories(xbox_ui PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_compile_features(xbox_ui PRIVATE cxx_std_17)
target_link_libraries(xbox_ui PUBLIC xbox_settings xbox_input)
if(NOT WIN32)
    find_package(SDL2 REQUIRED)
    target_link_libraries(xbox_ui PRIVATE SDL2::SDL2 platform)
endif()
