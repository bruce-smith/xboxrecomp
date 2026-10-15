# The in-game menu (Dear ImGui, MIT) and the settings store it edits.
#
# xbox_settings is plain C with no dependencies, so the input layer can read
# its bindings without pulling in the menu. xbox_ui is the menu itself.

add_library(xbox_settings STATIC settings.c)
target_include_directories(xbox_settings PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(WIN32)
    enable_language(CXX)
    include(FetchContent)
    FetchContent_Declare(imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG        v1.91.9b
        GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(imgui)

    add_library(xbox_ui STATIC
        ui_menu.cpp
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_win32.cpp
    )
    target_include_directories(xbox_ui PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
        PRIVATE ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
    target_compile_features(xbox_ui PRIVATE cxx_std_17)
    target_link_libraries(xbox_ui PUBLIC xbox_settings xbox_input)
endif()
