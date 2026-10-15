/**
 * Show the guest framebuffer in a window.
 *
 * The title renders into its own framebuffer in guest RAM and tells the kernel
 * where it is through AvSetDisplayMode; on hardware the CRTC scans that memory
 * out. Nothing here scans anything out, so however much of the GPU is
 * implemented, none of it is observable. This is the other half: a window that
 * reads that memory and puts it on screen.
 *
 * Deliberately plain GDI rather than the D3D8 layer. The point is to display
 * whatever the guest actually wrote, so the fewer stages between guest memory
 * and the screen the better -- and it must keep working while the D3D8 layer
 * is busy with something else, such as the FMV player's own window.
 *
 * Off unless RECOMP_FB_WINDOW is set.
 */
#include <stdint.h>

#if defined(_WIN32)
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(XBOXRECOMP_HAVE_VULKAN)
#include "xbox_vk.h"
#endif
#if defined(XBOXRECOMP_HAVE_UI)
#include "xbox_ui.h"
#endif

extern ptrdiff_t xbox_GetMemoryOffset(void);

/* RECOMP_VK_PRESENT=1: frames reach the screen through a Vulkan swapchain
 * (src/gpu_vk) instead of StretchDIBits. Set once the window thread has a
 * working swapchain; anything that fails leaves GDI in charge. */
static int s_vk;

static volatile LONG s_fb_running;
static uint32_t      s_fb_va, s_fb_pitch, s_fb_width = 640, s_fb_height = 480;
static uint32_t     *s_rgb;           /* converted 32-bit copy for GDI */

/* A finished frame, taken at the flip and shown until the next one.
 *
 * The window used to convert straight out of guest memory every 16 ms. Even
 * pointed at the buffer the title had just finished, that races the executor
 * drawing the next frame into the other one and, whenever the two swap, puts
 * a half-drawn image on the screen -- which is the flicker. Copying the
 * finished frame once per flip means the window never reads memory the
 * rasteriser is writing, so what it shows cannot be half of anything.
 *
 * Two buffers and an index, swapped after the copy completes, so the window
 * thread is never reading the one being filled. */
static uint32_t     *s_present[2];
static volatile LONG s_present_idx = -1;   /* -1 until the first flip */

void xbox_FramebufferWindowSet(uint32_t fb_va, uint32_t pitch)
{
    /* RECOMP_FB_VA pins the window to one guest address instead of following
     * whichever surface is being drawn into. A black window cannot distinguish
     * "the read path is broken" from "the title rendered black", and pointing
     * it at memory known to have content settles that. */
    const char *pin = getenv("RECOMP_FB_VA");

    s_fb_va = pin ? (uint32_t)strtoul(pin, NULL, 0) : fb_va;
    if (pitch)
        s_fb_pitch = pitch;
}

/* Called by the pushbuffer executor when the title flips. */
void xbox_FramebufferWindowPresent(uint32_t fb_va, uint32_t pitch)
{
    const uint8_t *src;
    LONG next;
    uint32_t bpp, x, y;

    if (!s_fb_running || !fb_va || !pitch)
        return;
    if (getenv("RECOMP_FB_VA"))
        return;                       /* pinned: leave the old path alone */
    next = (s_present_idx == 0) ? 1 : 0;
    if (!s_present[next]) {
        s_present[next] = (uint32_t *)calloc((size_t)s_fb_width * s_fb_height,
                                             4);
        if (!s_present[next])
            return;
    }
    bpp = pitch / s_fb_width;
    src = (const uint8_t *)((uintptr_t)fb_va + xbox_GetMemoryOffset());
    for (y = 0; y < s_fb_height; y++) {
        const uint8_t *row = src + (size_t)y * pitch;
        uint32_t *dst = s_present[next] + (size_t)y * s_fb_width;

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
    /* Published only once it is whole. */
    InterlockedExchange(&s_present_idx, next);
}

/* Which keys are down, for the pad stand-in in src/input.
 *
 * GetAsyncKeyState looked like the cheaper way to ask and does not work
 * here: it reads a state Wine keeps for the X server, and a guest process
 * drawing through GDI never sees it change. The window that has the focus
 * is the thing that receives the keys, so that is what has to remember
 * them.
 *
 * Reading this needs no lock. Each entry is written only by the window
 * thread and read only by the USB thread, one byte at a time, and a press
 * seen a frame late is indistinguishable from one made a frame later. */
static volatile unsigned char s_key_down[256];

/* Title bar, the way ps3recomp's window shows it. Written by the flip,
 * read once a second by the window thread; a torn read shows one stale
 * number for a second, which nobody can tell apart from a real one. */
static wchar_t       s_title[48] = L"Xbox Recomp";
static volatile LONG s_flips;
static volatile LONG s_frame_draws;

void xbox_FramebufferWindowSetTitle(const uint16_t *name, int max_chars)
{
    int i;
    for (i = 0; i < max_chars && i < 47 && name[i]; i++)
        s_title[i] = (wchar_t)name[i];
    if (i)
        s_title[i] = 0;
}

void xbox_FramebufferWindowFrameStats(uint32_t draws)
{
    InterlockedIncrement(&s_flips);
    InterlockedExchange(&s_frame_draws, (LONG)draws);
}

int xbox_FramebufferKeyDown(int vk)
{
    if ((unsigned)vk > 255)
        return 0;
    return s_key_down[vk] != 0;
}

/* Window sizing, scaling and fullscreen.
 *
 * The frame used to be blitted 1:1 into a 640x480 client area, which is tiny
 * on a modern desktop, and resizing the window only added empty space. Now:
 * the window opens at 2x (or the largest integer scale that fits the work
 * area; RECOMP_FB_SCALE=<n> picks one), the frame is scaled to fill the
 * client area with its aspect ratio kept and black bars around it, and
 * Alt+Enter or F11 toggles borderless fullscreen on the window's monitor.
 * Scaling is nearest-neighbour by default (crisp at integer scales);
 * RECOMP_FB_FILTER=linear or F9 switches to GDI's HALFTONE filter. */
static volatile LONG s_linear;
static int s_fullscreen;
static WINDOWPLACEMENT s_wp_saved;

static void fb_toggle_fullscreen(HWND h)
{
    DWORD style = (DWORD)GetWindowLongA(h, GWL_STYLE);

    if (!s_fullscreen) {
        MONITORINFO mi;
        mi.cbSize = sizeof mi;
        s_wp_saved.length = sizeof s_wp_saved;
        if (GetWindowPlacement(h, &s_wp_saved)
            && GetMonitorInfoA(MonitorFromWindow(h, MONITOR_DEFAULTTOPRIMARY), &mi)) {
            SetWindowLongA(h, GWL_STYLE, (LONG)((style & ~WS_OVERLAPPEDWINDOW) | WS_POPUP));
            SetWindowPos(h, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top,
                         mi.rcMonitor.right - mi.rcMonitor.left,
                         mi.rcMonitor.bottom - mi.rcMonitor.top,
                         SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
            s_fullscreen = 1;
        }
    } else {
        SetWindowLongA(h, GWL_STYLE, (LONG)((style & ~WS_POPUP) | WS_OVERLAPPEDWINDOW));
        SetWindowPlacement(h, &s_wp_saved);
        SetWindowPos(h, NULL, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER
                     | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
        s_fullscreen = 0;
    }
}

/* The frame, as large as the client area allows at its own aspect ratio,
 * centred, with the rest painted black. */
static void fb_blit(HWND hwnd, HDC hdc, const BITMAPINFO *bi, const uint32_t *px)
{
    RECT c;
    int cw, ch, dw, dh, dx, dy;

#if defined(XBOXRECOMP_HAVE_VULKAN)
    if (s_vk) {
        xbox_vk_present_frame(px, s_fb_width, s_fb_height, (int)s_linear);
        return;
    }
#endif
    GetClientRect(hwnd, &c);
    cw = c.right - c.left;
    ch = c.bottom - c.top;
    if (cw <= 0 || ch <= 0)
        return;
    if ((long long)cw * s_fb_height >= (long long)ch * s_fb_width) {
        dh = ch;
        dw = (int)((long long)ch * s_fb_width / s_fb_height);
    } else {
        dw = cw;
        dh = (int)((long long)cw * s_fb_height / s_fb_width);
    }
    dx = (cw - dw) / 2;
    dy = (ch - dh) / 2;
    if (s_linear) {
        SetStretchBltMode(hdc, HALFTONE);
        SetBrushOrgEx(hdc, 0, 0, NULL);
    } else {
        SetStretchBltMode(hdc, COLORONCOLOR);
    }
    StretchDIBits(hdc, dx, dy, dw, dh, 0, 0, (int)s_fb_width, (int)s_fb_height,
                  px, bi, DIB_RGB_COLORS, SRCCOPY);
    if (dy > 0) {
        PatBlt(hdc, 0, 0, cw, dy, BLACKNESS);
        PatBlt(hdc, 0, dy + dh, cw, ch - dy - dh, BLACKNESS);
    }
    if (dx > 0) {
        PatBlt(hdc, 0, dy, dx, dh, BLACKNESS);
        PatBlt(hdc, dx + dw, dy, cw - dx - dw, dh, BLACKNESS);
    }
}

/* The in-game menu (src/ui), when it is open: the frame scaled to the
 * client area as fb_blit would, dimmed, with the menu's overlay on top, at
 * the window's own resolution so the text is sharp. Returns 0 when the menu
 * is closed and the frame should go out as usual. */
static int fb_show_menu(HWND hwnd, HDC hdc, const uint32_t *px)
{
#if defined(XBOXRECOMP_HAVE_UI)
    static uint32_t *comp;
    static int comp_n;
    const uint32_t *ov;
    RECT c;
    int cw, ch, ow, oh, dw, dh, dx, dy, x, y;

    GetClientRect(hwnd, &c);
    cw = c.right - c.left;
    ch = c.bottom - c.top;
    if (!xbox_ui_render(hwnd, cw, ch) || !xbox_ui_overlay(&ov, &ow, &oh, NULL))
        return 0;
    if (ow != cw || oh != ch) {
        xbox_ui_overlay_release();
        return 0;
    }
    if (comp_n < cw * ch) {
        free(comp);
        comp = (uint32_t *)malloc((size_t)cw * ch * 4);
        comp_n = comp ? cw * ch : 0;
        if (!comp) {
            xbox_ui_overlay_release();
            return 0;
        }
    }
    if ((long long)cw * s_fb_height >= (long long)ch * s_fb_width) {
        dh = ch;
        dw = (int)((long long)ch * s_fb_width / s_fb_height);
    } else {
        dw = cw;
        dh = (int)((long long)cw * s_fb_height / s_fb_width);
    }
    dx = (cw - dw) / 2;
    dy = (ch - dh) / 2;
    for (y = 0; y < ch; y++) {
        uint32_t *row = comp + (size_t)y * cw;
        const uint32_t *orow = ov + (size_t)y * cw;
        int sy = y - dy;
        const uint32_t *srow = (sy >= 0 && sy < dh)
            ? px + (size_t)((long long)sy * s_fb_height / dh) * s_fb_width : NULL;
        for (x = 0; x < cw; x++) {
            uint32_t o = orow[x], a = o >> 24, g = 0, ia;
            int sx = x - dx;
            if (srow && sx >= 0 && sx < dw)
                g = (srow[(long long)sx * s_fb_width / dw] >> 1) & 0x7F7F7Fu;
            if (a == 255) {
                row[x] = o & 0xFFFFFFu;
                continue;
            }
            ia = 255 - a;
            row[x] = ((((g >> 16) & 0xFF) * ia / 255 + ((o >> 16) & 0xFF)) << 16)
                   | ((((g >> 8) & 0xFF) * ia / 255 + ((o >> 8) & 0xFF)) << 8)
                   | ((g & 0xFF) * ia / 255 + (o & 0xFF));
        }
    }
    xbox_ui_overlay_release();
#if defined(XBOXRECOMP_HAVE_VULKAN)
    if (s_vk) {
        xbox_vk_present_frame(comp, (uint32_t)cw, (uint32_t)ch, 0);
        return 1;
    }
#endif
    {
        BITMAPINFO bi;
        memset(&bi, 0, sizeof bi);
        bi.bmiHeader.biSize = sizeof bi.bmiHeader;
        bi.bmiHeader.biWidth = cw;
        bi.bmiHeader.biHeight = -ch;
        bi.bmiHeader.biPlanes = 1;
        bi.bmiHeader.biBitCount = 32;
        bi.bmiHeader.biCompression = BI_RGB;
        SetDIBitsToDevice(hdc, 0, 0, (DWORD)cw, (DWORD)ch, 0, 0, 0, (UINT)ch,
                          comp, &bi, DIB_RGB_COLORS);
    }
    return 1;
#else
    (void)hwnd; (void)hdc; (void)px;
    return 0;
#endif
}

static void fb_show(HWND hwnd, HDC hdc, const BITMAPINFO *bi, const uint32_t *px)
{
    if (!fb_show_menu(hwnd, hdc, px))
        fb_blit(hwnd, hdc, bi, px);
}

static LRESULT CALLBACK fb_wndproc(HWND h, UINT m, WPARAM w, LPARAM l)
{
#if defined(XBOXRECOMP_HAVE_UI)
    /* Escape, and everything while the menu is open, is the menu's. */
    if (m != WM_CLOSE && m != WM_DESTROY
        && xbox_ui_wndproc(h, m, (uintptr_t)w, (intptr_t)l))
        return 0;
#endif
    switch (m) {
    case WM_CLOSE:
        /* The window is the game: closing it ends the process. It used to
         * close only the window and leave the title running invisibly, so
         * a double-clicked exe had to be killed from Task Manager.
         * TerminateProcess rather than exit(): guest threads are mid-flight,
         * and CRT teardown under them can hang. */
        InterlockedExchange(&s_fb_running, 0);
        fprintf(stderr, "  [FBWIN] window closed; exiting\n");
        fflush(stderr);
        {
            extern void xbox_SchedClose(void);   /* kernel/sched.c */
            xbox_SchedClose();
        }
        TerminateProcess(GetCurrentProcess(), 0);
        return 0;
    case WM_DESTROY:
        InterlockedExchange(&s_fb_running, 0);
        return 0;

    case WM_SYSCHAR:
        if (w == 13)
            return 0;                   /* Alt+Enter: no beep */
        break;

    case WM_KEYDOWN:
    case WM_SYSKEYDOWN:
        /* Window keys, kept away from the pad: Enter is Start. */
        if ((w == VK_RETURN && (l & (1 << 29))) || w == VK_F11) {
            if (!(l & (1 << 30)))       /* not auto-repeat */
                fb_toggle_fullscreen(h);
            return 0;
        }
        if (w == VK_F9) {
            if (!(l & (1 << 30)))
                InterlockedExchange(&s_linear, !s_linear);
            return 0;
        }
        /* A key already held when the window got the focus is not a press.
         *
         * Windows sends a held key's auto-repeats (lParam bit 30: the key
         * was already down) to whichever window has the focus, so Enter
         * still held from launching the exe, or a key held through an
         * Alt+Tab, arrived here as Start held -- and a title skips its
         * movies on Start. Only a key whose first press this window saw
         * counts; the next real press after release does. */
        if ((unsigned)w < 256 && (l & (1 << 30)) && !s_key_down[w]) {
            static LONG noted;
            if (InterlockedIncrement(&noted) <= 8) {
                fprintf(stderr, "  [FBWIN] ignoring vk=0x%02X: held down before "
                        "the window had focus\n", (unsigned)w);
                fflush(stderr);
            }
            return m == WM_SYSKEYDOWN ? DefWindowProcA(h, m, w, l) : 0;
        }
        if ((unsigned)w < 256)
            s_key_down[w] = 1;
        /* RECOMP_KEY_TRACE: each key as it arrives, edge-triggered.
         *
         * The obvious diagnostic -- sampling which keys are held, once a
         * second, from the input path -- cannot tell a key that was never
         * pressed from one that was tapped: a 100 ms press is caught about
         * one time in ten. That ambiguity is expensive when the only way
         * to test is to ask someone to press a key and describe what
         * happened. This answers "did it arrive" on its own. */
        if (getenv("RECOMP_KEY_TRACE")) {
            static unsigned n;
            if (n++ < 40) {
                fprintf(stderr, "  [KEY] down vk=0x%02X\n", (unsigned)w);
                fflush(stderr);
            }
        }
        /* System keys still go to Windows, or Alt+F4 stops closing us. */
        return m == WM_SYSKEYDOWN ? DefWindowProcA(h, m, w, l) : 0;

    case WM_KEYUP:
    case WM_SYSKEYUP:
        if ((unsigned)w < 256)
            s_key_down[w] = 0;
        return m == WM_SYSKEYUP ? DefWindowProcA(h, m, w, l) : 0;

    /* Alt-tabbing away with a key held would leave it held for ever. */
    case WM_KILLFOCUS:
        memset((void *)s_key_down, 0, sizeof s_key_down);
        return 0;
    }
    return DefWindowProcA(h, m, w, l);
}

/* The display formats an Xbox front buffer is actually set to. The pitch says
 * how wide a row is in bytes, so pitch/width gives the pixel size; the exact
 * component layout only matters for 16-bit, where 5:6:5 and 1:5:5:5 differ. */
static void fb_convert(const uint8_t *src, uint32_t bpp)
{
    uint32_t x, y;

    for (y = 0; y < s_fb_height; y++) {
        const uint8_t *row = src + (size_t)y * s_fb_pitch;
        uint32_t *dst = s_rgb + (size_t)y * s_fb_width;

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

/* Write what the window is currently showing to a 24-bit BMP.
 *
 * A black window is ambiguous: it means either that the read path is wrong or
 * that the title really did render black. Dumping the same converted pixels
 * the window draws settles which, and does it without a screenshot. */
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

static DWORD WINAPI fb_thread(LPVOID unused)
{
    HWND hwnd;
    HDC hdc;
    BITMAPINFO bi;
    RECT r;

    (void)unused;

    {
        WNDCLASSA wc;
        memset(&wc, 0, sizeof(wc));
        wc.lpfnWndProc   = fb_wndproc;
        wc.hInstance     = GetModuleHandleA(NULL);
        wc.hCursor       = LoadCursorA(NULL, IDC_ARROW);
        wc.lpszClassName = "XboxRecompFramebuffer";
        RegisterClassA(&wc);
    }
    {
        const char *e = getenv("RECOMP_FB_SCALE");
        const char *f = getenv("RECOMP_FB_FILTER");
        RECT wa;
        int k = 2;

        s_linear = (f && _stricmp(f, "linear") == 0) ? 1 : 0;
        if (e && atoi(e) > 0) {
            k = atoi(e);
        } else if (SystemParametersInfoA(SPI_GETWORKAREA, 0, &wa, 0)) {
            for (; k > 1; k--) {
                RECT t;
                t.left = 0; t.top = 0;
                t.right = (LONG)(s_fb_width * (uint32_t)k);
                t.bottom = (LONG)(s_fb_height * (uint32_t)k);
                AdjustWindowRect(&t, WS_OVERLAPPEDWINDOW, FALSE);
                if (t.right - t.left <= wa.right - wa.left
                    && t.bottom - t.top <= wa.bottom - wa.top)
                    break;
            }
        }
        r.left = 0; r.top = 0;
        r.right = (LONG)(s_fb_width * (uint32_t)k);
        r.bottom = (LONG)(s_fb_height * (uint32_t)k);
    }
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    hwnd = CreateWindowExA(0, "XboxRecompFramebuffer", "Xbox Recomp - Framebuffer",
                           WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                           CW_USEDEFAULT, CW_USEDEFAULT,
                           r.right - r.left, r.bottom - r.top,
                           NULL, NULL, GetModuleHandleA(NULL), NULL);
    if (!hwnd) {
        InterlockedExchange(&s_fb_running, 0);
        return 0;
    }
    /* In front, titled from the start: it opens a few seconds after launch,
     * when something else may have the focus, and the keyboard pad
     * stand-in only sees keys sent to this window. */
    SetWindowTextW(hwnd, s_title);
    ShowWindow(hwnd, SW_SHOWNORMAL);
    SetForegroundWindow(hwnd);
    /* RECOMP_FB_FULLSCREEN=1 (the menu's "Start in fullscreen"): borderless
     * fullscreen from the start, as F11 / Alt+Enter toggle it. On a flip-model
     * Vulkan swapchain (RECOMP_VK_PRESENT) a window covering the monitor gets
     * independent flip, the closest Windows comes to exclusive fullscreen
     * today, and what frame-generating drivers want. */
    {
        const char *fs = getenv("RECOMP_FB_FULLSCREEN");
        if (fs && *fs && *fs != '0')
            fb_toggle_fullscreen(hwnd);
    }
    hdc = GetDC(hwnd);
#if defined(XBOXRECOMP_HAVE_VULKAN)
    if (getenv("RECOMP_VK_PRESENT")) {
        s_vk = xbox_vk_present_init(hwnd);
        if (!s_vk)
            fprintf(stderr, "  [FBWIN] Vulkan presentation unavailable; using GDI\n");
    }
#endif

    memset(&bi, 0, sizeof(bi));
    bi.bmiHeader.biSize        = sizeof(bi.bmiHeader);
    bi.bmiHeader.biWidth       = (LONG)s_fb_width;
    bi.bmiHeader.biHeight      = -(LONG)s_fb_height;   /* top-down */
    bi.bmiHeader.biPlanes      = 1;
    bi.bmiHeader.biBitCount    = 32;
    bi.bmiHeader.biCompression = BI_RGB;

    s_rgb = (uint32_t *)calloc((size_t)s_fb_width * s_fb_height, 4);

    fprintf(stderr, "  [FBWIN] framebuffer window open (%ux%u)\n",
            s_fb_width, s_fb_height);

    while (InterlockedCompareExchange(&s_fb_running, 1, 1)) {
        MSG msg;
        while (PeekMessageA(&msg, NULL, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageA(&msg);
        }
#if defined(XBOXRECOMP_HAVE_UI)
        /* Keys held when the menu opened are released, for the game. */
        if (xbox_ui_menu_open())
            memset((void *)s_key_down, 0, sizeof s_key_down);
#endif
        if (s_present_idx >= 0 && s_rgb) {
            /* A finished frame, published by the flip. Copied into s_rgb so
             * the dump path and GDI see one consistent image even if the
             * next flip lands mid-blit. */
            LONG idx = s_present_idx;
            if (s_present[idx])
                memcpy(s_rgb, s_present[idx],
                       (size_t)s_fb_width * s_fb_height * 4);
            fb_show(hwnd, hdc, &bi, s_rgb);
        } else if (s_fb_va && s_fb_pitch && s_rgb) {
            /* No flip yet, or pinned with RECOMP_FB_VA: read guest memory as
             * before, which is also what a title that never flips needs. */
            const uint8_t *src =
                (const uint8_t *)((uintptr_t)s_fb_va + xbox_GetMemoryOffset());
            fb_convert(src, s_fb_pitch / s_fb_width);
            fb_show(hwnd, hdc, &bi, s_rgb);
        }
        {
            /* One dump a few seconds in, so the title has had time to render
             * something rather than catching the first blank frame.
             *
             * RECOMP_FB_WINDOW_DUMP_EVERY=<frames> dumps repeatedly instead.
             * This window follows the address AvSetDisplayMode gave, which is
             * what the CRTC scans and therefore what a person sees; the
             * pushbuffer executor's own dump follows its draw surface. With
             * double buffering those are different buffers, and measuring
             * progress from the executor's dump reports a blank screen while
             * the window is showing the title's logo. Ask the window. */
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
            static DWORD t0;
            static LONG f0;
            DWORD now = GetTickCount();
            if (now - t0 >= 1000) {
                LONG f = s_flips;
                wchar_t tb[128];
                _snwprintf(tb, 127, L"%ls | FPS: %.1f | draws: %ld", s_title,
                           t0 ? (f - f0) * 1000.0 / (now - t0) : 0.0,
                           (long)s_frame_draws);
                tb[127] = 0;
                SetWindowTextW(hwnd, tb);
                t0 = now; f0 = f;
            }
        }
        Sleep(16);
    }

#if defined(XBOXRECOMP_HAVE_VULKAN)
    if (s_vk)
        xbox_vk_present_shutdown();
    s_vk = 0;
#endif
    ReleaseDC(hwnd, hdc);
    DestroyWindow(hwnd);
    free(s_rgb);
    s_rgb = NULL;
    return 0;
}

void xbox_FramebufferWindowStart(void)
{
    HANDLE th;

    if (!getenv("RECOMP_FB_WINDOW"))
        return;
    if (InterlockedCompareExchange(&s_fb_running, 1, 0) != 0)
        return;
    th = CreateThread(NULL, 0, fb_thread, NULL, 0, NULL);
    if (th)
        CloseHandle(th);
    else
        InterlockedExchange(&s_fb_running, 0);
}

#else
void xbox_FramebufferWindowSet(uint32_t fb_va, uint32_t pitch) { (void)fb_va; (void)pitch; }
void xbox_FramebufferWindowPresent(uint32_t fb_va, uint32_t pitch) { (void)fb_va; (void)pitch; }
void xbox_FramebufferWindowStart(void) {}
int xbox_FramebufferKeyDown(int vk) { (void)vk; return 0; }
void xbox_FramebufferWindowSetTitle(const uint16_t *n, int m) { (void)n; (void)m; }
void xbox_FramebufferWindowFrameStats(uint32_t draws) { (void)draws; }
#endif
