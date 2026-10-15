/*
 * xbox_ui.h -- the in-game menu (Escape) and the settings it edits.
 *
 * The menu is Dear ImGui, drawn by a small software rasteriser into an
 * overlay the size of the window: the framebuffer window composites it over
 * the frame (GDI or the Vulkan swapchain alike), and a renderer that presents
 * on its own can pull it with xbox_ui_overlay().
 *
 * Settings are declared in code (section, key, label, type, default) and live
 * in an ini file next to the exe. An environment variable named for a setting
 * wins over the file, so scripted runs behave as before.
 *
 * Menu input never reaches the guest: while the menu is open every emulated
 * pad reports nothing pressed (xbox_ui_menu_open()).
 */
#ifndef XBOX_UI_H
#define XBOX_UI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- settings ----------------------------------------------------------- */

#define XBOX_SETTING_RESTART 0x1u   /* takes effect at the next start */
#define XBOX_SETTING_HIDDEN  0x2u   /* kept in the file, not shown */

/* Load the file once, early (before anything reads the settings). A missing
 * file is fine: defaults apply and the file is written on the first change. */
void xbox_settings_load(const char *path);
void xbox_settings_save(void);

/* Declare an option (idempotent) and return its value. env, if not NULL and
 * set, overrides the file. Declared options show in the menu under their
 * section ("graphics", "controllers", ...) unless hidden. */
int  xbox_setting_bool(const char *section, const char *key, const char *label,
                       int def, const char *env, unsigned flags);
int  xbox_setting_int(const char *section, const char *key, const char *label,
                      int def, int lo, int hi, const char *env, unsigned flags);
int  xbox_setting_choice(const char *section, const char *key, const char *label,
                         const char *const *names, int n, int def,
                         const char *env, unsigned flags);

/* Current value of a declared or stored option; def if neither. */
int  xbox_setting_get(const char *section, const char *key, int def);
void xbox_setting_set(const char *section, const char *key, int value);

/* For a declared option with an env name: if that variable is unset, set it
 * from the stored value ("1"/"0" or the number) so code that reads the
 * variable sees the setting. Returns 1 if it set it. */
int  xbox_setting_to_env(const char *section, const char *key);

/* ---- the menu ------------------------------------------------------------- */

int  xbox_ui_menu_open(void);
void xbox_ui_menu_toggle(void);

/* The framebuffer window's message procedure calls this first; nonzero means
 * the menu consumed the message. */
int  xbox_ui_wndproc(void *hwnd, unsigned msg, uintptr_t wparam, intptr_t lparam);

/* Render the menu for this frame into the overlay; returns 1 when it is open.
 * Called on the window thread. */
int  xbox_ui_render(void *hwnd, int client_w, int client_h);

/* The overlay: premultiplied 0xAARRGGBB at the client size; returns 0 when
 * the menu is closed. serial changes whenever the pixels do. Callable from
 * any thread: when it returns 1 the pixels stay valid, and are held, until
 * xbox_ui_overlay_release() -- call that once done reading (keep it short:
 * the menu waits for it to publish its next frame). */
int  xbox_ui_overlay(const uint32_t **px, int *w, int *h, uint32_t *serial);
void xbox_ui_overlay_release(void);

/* Panels other modules add to a section ("cheats", "graphics", ...): draw is
 * called every frame while the menu shows that section, on the window
 * thread, using the widgets below. */
void xbox_ui_add_panel(const char *section, const char *title,
                       void (*draw)(void *ctx), void *ctx);

/* Widgets for panels; each returns 1 when pressed or changed. */
int  xbox_ui_button(const char *label);
int  xbox_ui_checkbox(const char *label, int *v);
int  xbox_ui_slider_int(const char *label, int *v, int lo, int hi);
int  xbox_ui_input_int(const char *label, int *v);
int  xbox_ui_combo(const char *label, int *cur, const char *const *items, int n);
void xbox_ui_text(const char *fmt, ...);
void xbox_ui_separator(void);
void xbox_ui_same_line(void);

/* Guest-visible changes from the menu (cheats): the window thread is not a
 * guest agent, so it must not write guest memory. xbox_ui_defer queues fn to
 * run on the guest's main thread at its next input poll, which calls
 * xbox_ui_run_deferred(). xbox_ui_guest_writes_allowed() is 0 during a
 * replay or a deterministic (virtual-mode) run; a recording is fine as long
 * as the change goes into the input recording (the cheat engine does). */
void xbox_ui_defer(void (*fn)(void *ctx), void *ctx);
void xbox_ui_run_deferred(void);
int  xbox_ui_guest_writes_allowed(void);

/* About box text from the game: name and build stamps. */
void xbox_ui_set_about(const char *text);

#ifdef __cplusplus
}
#endif

#endif /* XBOX_UI_H */
