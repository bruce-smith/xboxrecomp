/* Milestone-1 test of the Vulkan presentation path: see README.md. */
#include <windows.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "xbox_vk.h"

static LRESULT CALLBACK wndproc(HWND h, UINT m, WPARAM w, LPARAM l)
{
    return DefWindowProcA(h, m, w, l);
}

static void pump(void)
{
    MSG msg;
    while (PeekMessageA(&msg, NULL, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessageA(&msg);
    }
}

int main(void)
{
    enum { W = 640, H = 480, FRAMES = 120 };
    WNDCLASSA wc;
    HWND hwnd;
    uint32_t *px = (uint32_t *)malloc(W * H * 4);
    int f, shown = 0, expected = 0;

    memset(&wc, 0, sizeof wc);
    wc.lpfnWndProc = wndproc;
    wc.hInstance = GetModuleHandleA(NULL);
    wc.lpszClassName = "GpuVkPresentTest";
    RegisterClassA(&wc);
    hwnd = CreateWindowExA(0, wc.lpszClassName, "gpu_vk_present_test",
                           WS_OVERLAPPEDWINDOW | WS_VISIBLE, CW_USEDEFAULT,
                           CW_USEDEFAULT, 1280 + 16, 960 + 39, NULL, NULL,
                           wc.hInstance, NULL);
    if (!hwnd || !px) {
        fprintf(stderr, "FAIL: no window\n");
        return 1;
    }
    pump();
    if (!xbox_vk_present_init(hwnd)) {
        fprintf(stderr, "FAIL: xbox_vk_present_init\n");
        return 1;
    }
    for (f = 0; f < FRAMES; f++) {
        int x, y, minimised;
        for (y = 0; y < H; y++)
            for (x = 0; x < W; x++)
                px[y * W + x] = ((uint32_t)((x + f * 4) & 0xFF) << 16)
                              | ((uint32_t)(y & 0xFF) << 8)
                              | (((x / 32 + y / 32) & 1) ? 0xC0u : 0x20u);
        if (f == FRAMES / 2)
            SetWindowPos(hwnd, NULL, 0, 0, 900, 500, SWP_NOMOVE | SWP_NOZORDER);
        if (f == FRAMES * 3 / 4)
            ShowWindow(hwnd, SW_MINIMIZE);
        if (f == FRAMES * 3 / 4 + 10)
            ShowWindow(hwnd, SW_RESTORE);
        pump();
        minimised = IsIconic(hwnd);
        if (!minimised)
            expected++;
        if (xbox_vk_present_frame(px, W, H, f & 1))
            shown++;
        Sleep(16);
    }
    xbox_vk_present_shutdown();
    DestroyWindow(hwnd);
    /* A frame right after a resize or restore may be dropped while the
     * swapchain is rebuilt; anything more is a failure. */
    printf("presented %d of %d frames (%d while not minimised)\n", shown, FRAMES,
           expected);
    if (shown < expected - 4) {
        fprintf(stderr, "FAIL: too many frames not presented\n");
        return 1;
    }
    printf("PASS\n");
    return 0;
}
