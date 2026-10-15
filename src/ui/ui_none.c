/*
 * The in-game menu's API where the menu is not built (POSIX hosts for now):
 * it never opens, panels and widgets do nothing, and deferred guest work runs
 * at the next input poll as with the menu. Settings (settings.c) work as
 * everywhere, so blinx2_settings.ini and its environment variables apply.
 *
 * The menu itself is Dear ImGui with a Win32 backend (ui_menu.cpp); an SDL2
 * backend for video/fb_present_sdl.c is the follow-up.
 */
#include <stddef.h>
#include <stdint.h>
#include "xbox_ui.h"

extern int xbox_SchedMode(void);   /* kernel/sched.c */

int  xbox_ui_menu_open(void) { return 0; }
void xbox_ui_menu_toggle(void) { }
int  xbox_ui_wndproc(void *hwnd, unsigned msg, uintptr_t wparam, intptr_t lparam)
{ (void)hwnd; (void)msg; (void)wparam; (void)lparam; return 0; }
int  xbox_ui_render(void *hwnd, int client_w, int client_h)
{ (void)hwnd; (void)client_w; (void)client_h; return 0; }
int  xbox_ui_overlay(const uint32_t **px, int *w, int *h, uint32_t *serial)
{ (void)px; (void)w; (void)h; (void)serial; return 0; }

void xbox_ui_add_panel(const char *section, const char *title,
                       void (*draw)(void *ctx), void *ctx)
{ (void)section; (void)title; (void)draw; (void)ctx; }
int  xbox_ui_button(const char *label) { (void)label; return 0; }
int  xbox_ui_checkbox(const char *label, int *v) { (void)label; (void)v; return 0; }
int  xbox_ui_slider_int(const char *label, int *v, int lo, int hi)
{ (void)label; (void)v; (void)lo; (void)hi; return 0; }
int  xbox_ui_input_int(const char *label, int *v) { (void)label; (void)v; return 0; }
int  xbox_ui_combo(const char *label, int *cur, const char *const *items, int n)
{ (void)label; (void)cur; (void)items; (void)n; return 0; }
void xbox_ui_text(const char *fmt, ...) { (void)fmt; }
void xbox_ui_separator(void) { }
void xbox_ui_same_line(void) { }

/* One pending call is all a menu-less build can be asked for. */
static void (*volatile s_fn)(void *ctx);
static void *volatile s_ctx;

void xbox_ui_defer(void (*fn)(void *ctx), void *ctx)
{
    s_ctx = ctx;
    s_fn = fn;
}

void xbox_ui_run_deferred(void)
{
    void (*fn)(void *) = s_fn;
    if (fn) {
        s_fn = NULL;
        fn(s_ctx);
    }
}

int  xbox_ui_guest_writes_allowed(void) { return xbox_SchedMode() == 0; }
void xbox_ui_set_about(const char *text) { (void)text; }
