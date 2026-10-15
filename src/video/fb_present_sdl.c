/**
 * Show the guest framebuffer in a window: the SDL2 version of fb_present.c,
 * for POSIX hosts (Linux, SteamOS). Same entry points, same behaviour:
 *
 *   - a finished frame is copied at each flip (xbox_FramebufferWindowPresent)
 *     and shown until the next, so the window never shows half a frame;
 *   - the window opens at 2x (RECOMP_FB_SCALE=<n>), scaled to the window with
 *     the aspect kept; F11 or Alt+Enter toggles fullscreen, F9 toggles
 *     nearest/linear (RECOMP_FB_FILTER=linear);
 *   - keys go to xbox_FramebufferKeyDown as Windows virtual-key codes, so the
 *     keyboard pad stand-in (src/input) is unchanged; a key already held when
 *     the window gets the focus is not a press until it is released;
 *   - closing the window ends the process;
 *   - RECOMP_FB_DUMP / RECOMP_FB_WINDOW_DUMP_EVERY write BMPs of what is shown.
 *
 *   - fullscreen from the start under gamescope (Steam Deck Game Mode), or
 *     with RECOMP_FB_FULLSCREEN=1.
 *
 * Off unless RECOMP_FB_WINDOW is set.
 */
#define _GNU_SOURCE                   /* strcasestr */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <SDL.h>
#if defined(XBOXRECOMP_HAVE_VULKAN)
#include "xbox_vk.h"
#endif

extern ptrdiff_t xbox_GetMemoryOffset(void);

/* RECOMP_VK_PRESENT=1: frames reach the screen through a Vulkan swapchain on
 * this window (src/gpu_vk) instead of the SDL renderer -- which lets the GPU
 * renderer present its own frame at its own resolution. Anything that fails
 * leaves the SDL renderer in charge, as GDI on Windows. */
static int s_vk;

static volatile int  s_fb_running;
static uint32_t      s_fb_va, s_fb_pitch, s_fb_width = 640, s_fb_height = 480;
static uint32_t     *s_rgb;           /* what the window shows (and dumps) */

/* A finished frame, taken at the flip; two buffers and an index, swapped
 * after the copy completes (see fb_present.c). */
static uint32_t     *s_present[2];
static volatile int  s_present_idx = -1;   /* -1 until the first flip */

static volatile unsigned char s_key_down[256];

static char          s_title[48] = "Xbox Recomp";
static volatile int  s_flips;
static volatile int  s_frame_draws;

void xbox_FramebufferWindowSet(uint32_t fb_va, uint32_t pitch)
{
    const char *pin = getenv("RECOMP_FB_VA");

    s_fb_va = pin ? (uint32_t)strtoul(pin, NULL, 0) : fb_va;
    if (pitch)
        s_fb_pitch = pitch;
}

static void convert_rows(uint32_t *dst_base, const uint8_t *src, uint32_t pitch,
                         uint32_t bpp)
{
    uint32_t x, y;

    for (y = 0; y < s_fb_height; y++) {
        const uint8_t *row = src + (size_t)y * pitch;
        uint32_t *dst = dst_base + (size_t)y * s_fb_width;

        if (bpp == 4) {
            memcpy(dst, row, (size_t)s_fb_width * 4);
        } else if (bpp == 2) {
            const uint16_t *p = (const uint16_t *)row;
            for (x = 0; x < s_fb_width; x++) {
                uint16_t v = p[x];
                uint32_t r = (uint32_t)((v >> 11) & 0x1F) * 255u / 31u;
                uint32_t g = (uint32_t)((v >>  5) & 0x3F) * 255u / 63u;
                uint32_t b = (uint32_t)( v        & 0x1F) * 255u / 31u;
                dst[x] = (r << 16) | (g << 8) | b;
            }
        } else {
            memset(dst, 0, (size_t)s_fb_width * 4);
        }
    }
}

/* Called by the pushbuffer executor when the title flips. */
void xbox_FramebufferWindowPresent(uint32_t fb_va, uint32_t pitch)
{
    int next;

    if (!s_fb_running || !fb_va || !pitch)
        return;
    if (getenv("RECOMP_FB_VA"))
        return;                       /* pinned: leave the old path alone */
    next = (s_present_idx == 0) ? 1 : 0;
    if (!s_present[next]) {
        s_present[next] = (uint32_t *)calloc((size_t)s_fb_width * s_fb_height, 4);
        if (!s_present[next])
            return;
    }
    convert_rows(s_present[next],
                 (const uint8_t *)((uintptr_t)fb_va + xbox_GetMemoryOffset()),
                 pitch, pitch / s_fb_width);
    /* Published only once it is whole. */
    __atomic_store_n(&s_present_idx, next, __ATOMIC_RELEASE);
}

void xbox_FramebufferWindowSetTitle(const uint16_t *name, int max_chars)
{
    int i;
    for (i = 0; i < max_chars && i < 47 && name[i]; i++)
        s_title[i] = name[i] < 128 ? (char)name[i] : '?';
    if (i)
        s_title[i] = 0;
}

void xbox_FramebufferWindowFrameStats(uint32_t draws)
{
    __atomic_add_fetch(&s_flips, 1, __ATOMIC_RELAXED);
    __atomic_store_n(&s_frame_draws, (int)draws, __ATOMIC_RELAXED);
}

int xbox_FramebufferKeyDown(int vk)
{
    if ((unsigned)vk > 255)
        return 0;
    return s_key_down[vk] != 0;
}

/* An SDL scancode as the Windows virtual-key code WM_KEYDOWN would carry:
 * letters and digits are their ASCII upper case, modifiers the generic
 * VK_SHIFT/VK_CONTROL/VK_MENU. 0 for keys nothing here reads. */
static int vk_of(SDL_Scancode sc)
{
    if (sc >= SDL_SCANCODE_A && sc <= SDL_SCANCODE_Z)
        return 'A' + (sc - SDL_SCANCODE_A);
    if (sc >= SDL_SCANCODE_1 && sc <= SDL_SCANCODE_9)
        return '1' + (sc - SDL_SCANCODE_1);
    if (sc >= SDL_SCANCODE_F1 && sc <= SDL_SCANCODE_F12)
        return 0x70 + (sc - SDL_SCANCODE_F1);
    if (sc >= SDL_SCANCODE_KP_1 && sc <= SDL_SCANCODE_KP_9)
        return 0x61 + (sc - SDL_SCANCODE_KP_1);          /* VK_NUMPAD1.. */
    switch (sc) {
    case SDL_SCANCODE_0:         return '0';
    case SDL_SCANCODE_KP_0:      return 0x60;            /* VK_NUMPAD0 */
    case SDL_SCANCODE_RETURN:
    case SDL_SCANCODE_KP_ENTER:  return 0x0D;            /* VK_RETURN */
    case SDL_SCANCODE_ESCAPE:    return 0x1B;
    case SDL_SCANCODE_BACKSPACE: return 0x08;            /* VK_BACK */
    case SDL_SCANCODE_TAB:       return 0x09;
    case SDL_SCANCODE_SPACE:     return 0x20;
    case SDL_SCANCODE_LEFT:      return 0x25;
    case SDL_SCANCODE_UP:        return 0x26;
    case SDL_SCANCODE_RIGHT:     return 0x27;
    case SDL_SCANCODE_DOWN:      return 0x28;
    case SDL_SCANCODE_LSHIFT:
    case SDL_SCANCODE_RSHIFT:    return 0x10;            /* VK_SHIFT */
    case SDL_SCANCODE_LCTRL:
    case SDL_SCANCODE_RCTRL:     return 0x11;            /* VK_CONTROL */
    case SDL_SCANCODE_LALT:
    case SDL_SCANCODE_RALT:      return 0x12;            /* VK_MENU */
    default:                     return 0;
    }
}

/* Write what the window is currently showing to a 24-bit BMP. */
int xbox_FramebufferDumpBmp(const char *path)
{
    FILE *f;
    uint32_t row = ((s_fb_width * 3u) + 3u) & ~3u;
    uint32_t img = row * s_fb_height, total = 54u + img, y, x;
    uint8_t hdr[54], *line;

    if (!s_rgb || !s_fb_va)
        return -1;
    f = fopen(path, "wb");
    if (!f)
        return -1;
    memset(hdr, 0, sizeof(hdr));
    hdr[0] = 'B'; hdr[1] = 'M';
    memcpy(hdr + 2, &total, 4);
    hdr[10] = 54; hdr[14] = 40;
    memcpy(hdr + 18, &s_fb_width, 4);
    memcpy(hdr + 22, &s_fb_height, 4);
    hdr[26] = 1; hdr[28] = 24;
    memcpy(hdr + 34, &img, 4);
    fwrite(hdr, 1, sizeof(hdr), f);

    line = (uint8_t *)calloc(1, row);
    for (y = 0; y < s_fb_height; y++) {
        const uint32_t *src = s_rgb + (size_t)(s_fb_height - 1 - y) * s_fb_width;
        for (x = 0; x < s_fb_width; x++) {
            line[x * 3 + 0] = (uint8_t)(src[x] & 0xFF);
            line[x * 3 + 1] = (uint8_t)((src[x] >> 8) & 0xFF);
            line[x * 3 + 2] = (uint8_t)((src[x] >> 16) & 0xFF);
        }
        fwrite(line, 1, row, f);
    }
    free(line);
    fclose(f);
    fprintf(stderr, "  [FBWIN] wrote %s (%ux%u from 0x%08X)\n",
            path, s_fb_width, s_fb_height, s_fb_va);
    return 0;
}

static void window_closed(void)
{
    extern void xbox_SchedClose(void);   /* kernel/sched.c */
    s_fb_running = 0;
    fprintf(stderr, "  [FBWIN] window closed; exiting\n");
    fflush(stderr);
    xbox_SchedClose();
    /* _exit, as TerminateProcess on Windows: guest threads are mid-flight
     * and exit()'s teardown under them can hang. */
    _exit(0);
}

static void handle_key(SDL_Window *win, const SDL_KeyboardEvent *k, int *linear)
{
    int vk = vk_of(k->keysym.scancode);
    int down = k->type == SDL_KEYDOWN;

    if (down) {
        /* Window keys, kept away from the pad: Enter is Start. */
        if (k->keysym.scancode == SDL_SCANCODE_F11
            || (k->keysym.scancode == SDL_SCANCODE_RETURN && (k->keysym.mod & KMOD_ALT))) {
            if (!k->repeat) {
                Uint32 fs = SDL_GetWindowFlags(win) & SDL_WINDOW_FULLSCREEN_DESKTOP;
                SDL_SetWindowFullscreen(win, fs ? 0 : SDL_WINDOW_FULLSCREEN_DESKTOP);
            }
            return;
        }
        if (k->keysym.scancode == SDL_SCANCODE_F9) {
            if (!k->repeat)
                *linear = !*linear;
            return;
        }
        /* A key already held when the window got the focus is not a press:
         * its auto-repeats arrive with nothing recorded as down. */
        if (vk && k->repeat && !s_key_down[vk]) {
            static int noted;
            if (noted++ < 8) {
                fprintf(stderr, "  [FBWIN] ignoring vk=0x%02X: held down before "
                        "the window had focus\n", (unsigned)vk);
                fflush(stderr);
            }
            return;
        }
        if (vk)
            s_key_down[vk] = 1;
        if (vk && getenv("RECOMP_KEY_TRACE")) {
            static unsigned n;
            if (n++ < 40) {
                fprintf(stderr, "  [KEY] down vk=0x%02X\n", (unsigned)vk);
                fflush(stderr);
            }
        }
    } else if (vk) {
        s_key_down[vk] = 0;
    }
}

/* Fullscreen from the start: RECOMP_FB_FULLSCREEN=1 asks, =0 refuses; unset,
 * it is on under gamescope -- a Steam Deck in Game Mode, or Steam's Big
 * Picture session -- where a window is shown fullscreen anyway and the
 * framebuffer should fill the panel (1280x800 on the Deck, letterboxed to
 * 4:3). */
static int start_fullscreen(void)
{
    const char *e = getenv("RECOMP_FB_FULLSCREEN");
    const char *de = getenv("XDG_CURRENT_DESKTOP");

    if (e && *e)
        return *e != '0';
    return getenv("GAMESCOPE_WAYLAND_DISPLAY") != NULL
        || (getenv("SteamDeck") && strcmp(getenv("SteamDeck"), "1") == 0
            && getenv("SteamGamepadUI") != NULL)
        || (de && strcasestr(de, "gamescope") != NULL);
}

static int fb_thread(void *unused)
{
    SDL_Window *win;
    SDL_Renderer *ren;
    SDL_Texture *tex;
    int k = 2, linear, shown_linear = -1;
    Uint32 t0 = 0;
    int f0 = 0;
    const char *e = getenv("RECOMP_FB_SCALE");
    const char *flt = getenv("RECOMP_FB_FILTER");

    (void)unused;
    linear = (flt && strcasecmp(flt, "linear") == 0) ? 1 : 0;
    /* Ctrl+C and SIGTERM end the process, as without SDL; closing the window
     * is the way to quit from the desktop. */
    SDL_SetHint(SDL_HINT_NO_SIGNAL_HANDLERS, "1");
    if (SDL_InitSubSystem(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "  [FBWIN] SDL video unavailable: %s\n", SDL_GetError());
        s_fb_running = 0;
        return 0;
    }
    if (e && atoi(e) > 0) {
        k = atoi(e);
    } else {
        SDL_Rect wa;
        if (SDL_GetDisplayUsableBounds(0, &wa) == 0)
            for (; k > 1; k--)
                if ((int)(s_fb_width * (uint32_t)k) <= wa.w
                    && (int)(s_fb_height * (uint32_t)k) <= wa.h)
                    break;
    }
    {
        Uint32 flags = SDL_WINDOW_RESIZABLE | SDL_WINDOW_SHOWN
                     | (start_fullscreen() ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0);
        int w = (int)(s_fb_width * (uint32_t)k), h = (int)(s_fb_height * (uint32_t)k);
        win = NULL;
#if defined(XBOXRECOMP_HAVE_VULKAN)
        if (getenv("RECOMP_VK_PRESENT")) {
            win = SDL_CreateWindow(s_title, SDL_WINDOWPOS_UNDEFINED,
                                   SDL_WINDOWPOS_UNDEFINED, w, h,
                                   flags | SDL_WINDOW_VULKAN);
            s_vk = win ? xbox_vk_present_init(win) : 0;
            if (!s_vk) {
                fprintf(stderr, "  [FBWIN] Vulkan presentation unavailable; using SDL\n");
                if (win)
                    SDL_DestroyWindow(win);
                win = NULL;
            }
        }
#endif
        if (!win)
            win = SDL_CreateWindow(s_title, SDL_WINDOWPOS_UNDEFINED,
                                   SDL_WINDOWPOS_UNDEFINED, w, h, flags);
    }
    ren = NULL;
    tex = NULL;
    if (!s_vk) {
        /* No vsync: the loop paces itself every 16 ms, as the GDI window does. */
        ren = win ? SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED) : NULL;
        if (!ren && win)
            ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_SOFTWARE);
        tex = ren ? SDL_CreateTexture(ren, SDL_PIXELFORMAT_ARGB8888,
                                      SDL_TEXTUREACCESS_STREAMING,
                                      (int)s_fb_width, (int)s_fb_height) : NULL;
    }
    if (!win || (!s_vk && !tex)) {
        fprintf(stderr, "  [FBWIN] cannot open the window: %s\n", SDL_GetError());
        s_fb_running = 0;
        return 0;
    }
    /* Letterboxed at the frame's aspect, black bars around it (the Vulkan
     * path does the same in xbox_vk_present_frame). */
    if (ren)
        SDL_RenderSetLogicalSize(ren, (int)s_fb_width, (int)s_fb_height);
    SDL_RaiseWindow(win);
    s_rgb = (uint32_t *)calloc((size_t)s_fb_width * s_fb_height, 4);
    fprintf(stderr, "  [FBWIN] framebuffer window open (%ux%u, SDL %s)\n",
            s_fb_width, s_fb_height, SDL_GetCurrentVideoDriver());

    while (s_fb_running) {
        SDL_Event ev;
        int have = 0;

        while (SDL_PollEvent(&ev)) {
            switch (ev.type) {
            case SDL_QUIT:
                window_closed();
                break;
            case SDL_WINDOWEVENT:
                if (ev.window.event == SDL_WINDOWEVENT_CLOSE)
                    window_closed();
                /* Alt-tabbing away with a key held would leave it held. */
                if (ev.window.event == SDL_WINDOWEVENT_FOCUS_LOST)
                    memset((void *)s_key_down, 0, sizeof s_key_down);
                break;
            case SDL_KEYDOWN:
            case SDL_KEYUP:
                handle_key(win, &ev.key, &linear);
                break;
            default:
                break;
            }
        }
        if (s_present_idx >= 0 && s_rgb) {
            int idx = __atomic_load_n(&s_present_idx, __ATOMIC_ACQUIRE);
            if (s_present[idx])
                memcpy(s_rgb, s_present[idx], (size_t)s_fb_width * s_fb_height * 4);
            have = 1;
        } else if (s_fb_va && s_fb_pitch && s_rgb) {
            /* No flip yet, or pinned with RECOMP_FB_VA: read guest memory. */
            convert_rows(s_rgb,
                         (const uint8_t *)((uintptr_t)s_fb_va + xbox_GetMemoryOffset()),
                         s_fb_pitch, s_fb_pitch / s_fb_width);
            have = 1;
        }
#if defined(XBOXRECOMP_HAVE_VULKAN)
        if (s_vk) {
            /* Skipped inside while the GPU renderer presents its own frame. */
            if (have)
                xbox_vk_present_frame(s_rgb, s_fb_width, s_fb_height, linear);
        } else
#endif
        {
            if (linear != shown_linear) {
                SDL_SetTextureScaleMode(tex, linear ? SDL_ScaleModeLinear
                                                    : SDL_ScaleModeNearest);
                shown_linear = linear;
            }
            SDL_SetRenderDrawColor(ren, 0, 0, 0, 255);
            SDL_RenderClear(ren);
            if (have) {
                SDL_UpdateTexture(tex, NULL, s_rgb, (int)s_fb_width * 4);
                SDL_RenderCopy(ren, tex, NULL, NULL);
            }
            SDL_RenderPresent(ren);
        }
        {
            const char *dump = getenv("RECOMP_FB_DUMP");
            const char *every = getenv("RECOMP_FB_WINDOW_DUMP_EVERY");
            static int frames;
            int period = every ? atoi(every) : 0;
            frames++;
            if (dump && period > 0) {
                if (frames % period == 0)
                    xbox_FramebufferDumpBmp(dump);
            } else if (dump && frames == 600) {
                xbox_FramebufferDumpBmp(dump);
            }
        }
        {
            Uint32 now = SDL_GetTicks();
            if (now - t0 >= 1000) {
                int f = s_flips;
                char tb[128];
                snprintf(tb, sizeof tb, "%s | FPS: %.1f | draws: %d", s_title,
                         t0 ? (f - f0) * 1000.0 / (now - t0) : 0.0, s_frame_draws);
                SDL_SetWindowTitle(win, tb);
                t0 = now;
                f0 = f;
            }
        }
        SDL_Delay(16);
    }
#if defined(XBOXRECOMP_HAVE_VULKAN)
    if (s_vk)
        xbox_vk_present_shutdown();
    s_vk = 0;
#endif
    if (tex)
        SDL_DestroyTexture(tex);
    if (ren)
        SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    free(s_rgb);
    s_rgb = NULL;
    return 0;
}

void xbox_FramebufferWindowStart(void)
{
    SDL_Thread *th;

    if (!getenv("RECOMP_FB_WINDOW"))
        return;
    if (!__sync_bool_compare_and_swap(&s_fb_running, 0, 1))
        return;
    /* The pad keeps working while another window has the focus, as XInput
     * does on Windows. */
    SDL_SetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, "1");
    th = SDL_CreateThread(fb_thread, "fb_window", NULL);
    if (th)
        SDL_DetachThread(th);
    else
        s_fb_running = 0;
}
