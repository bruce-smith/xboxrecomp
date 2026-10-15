/**
 * Xbox Input Compatibility Layer
 *
 * Translates the Xbox controller API to a host gamepad backend.
 * Handles the structural differences between the Xbox gamepad
 * (analog buttons as bytes, separate trigger channels) and the host
 * (digital face buttons, trigger axes).
 *
 *   _WIN32 -> Windows XInput
 *   POSIX  -> XInput on SDL2 GameController (xinput_host_sdl.c)
 */

#include "xinput_xbox.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ======================================================================== */
/* ====================  XInput backend  ================================== */
/* ======================================================================== */

/* XInput on every host: Windows' own, or on POSIX the same API answered from
 * SDL2's GameController (xinput_host_sdl.c, declared in win32_compat.h). */
#if defined(_WIN32)
#include <xinput.h>
#pragma comment(lib, "xinput.lib")
#endif

static BOOL  g_controller_connected[XBOX_MAX_CONTROLLERS] = { FALSE };
static DWORD g_last_packet[XBOX_MAX_CONTROLLERS] = { 0 };

/* ---- per-port mapping: host pads and the keyboard -------------------------
 *
 * Each Xbox port takes one host pad (or none) and, if enabled, the keyboard;
 * every Xbox control has one key and one host-pad source (xbox_input_map.h).
 * The table is kept in the settings file, section "controllers", and edited
 * from the in-game menu. The defaults are the mapping this layer always had:
 *
 *   arrows        d-pad             Enter      START
 *   Z X A S       A B X Y           Backspace  BACK
 *   Q E           white black       1 3        triggers
 *   numpad 8/2/4/6 left thumb       I/K/J/L    right thumb
 *   Shift Ctrl    left/right thumb click
 *
 * host pad n drives port n, LB is White and RB is Black (on the Controller S
 * White sits left of Black, as xemu maps them), and the keyboard drives port
 * 1 only.
 *
 * The keyboard as a whole stays behind RECOMP_KEYBOARD (main.c turns it on
 * for windowed play), because a keyboard silently acting as player 1 is
 * surprising in a scripted run. Keys come from the framebuffer window, which
 * only receives them while it has the focus, so typing in another window
 * does not drive the game.
 */
#include "xbox_input_map.h"
#include "xbox_ui.h"            /* settings */

static BOOL keyboard_enabled(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECOMP_KEYBOARD");
        on = (v && *v && *v != '0') ? 1 : 0;
    }
    return on ? TRUE : FALSE;
}

/* The framebuffer window records what is held; see src/video/fb_present.c.
 *
 * GetAsyncKeyState was the first attempt and reads nothing here -- under
 * Wine it answers about a state a GDI-drawing guest never touches. Going
 * through the window is also the better gate: a window only receives keys
 * while it has the focus, so there is no separate focus check to get wrong
 * and no way for typing in another application to drive the game.
 *
 * Declared rather than included so src/input does not depend on the video
 * module's include path. */
extern int xbox_FramebufferKeyDown(int vk);

static const char *const k_ctl_key[XIN_CONTROLS] = {
    "dpad_up", "dpad_down", "dpad_left", "dpad_right", "start", "back",
    "lthumb", "rthumb", "a", "b", "x", "y", "black", "white",
    "ltrigger", "rtrigger", "lx_neg", "lx_pos", "ly_neg", "ly_pos",
    "rx_neg", "rx_pos", "ry_neg", "ry_pos"
};
static const char *const k_ctl_label[XIN_CONTROLS] = {
    "D-pad up", "D-pad down", "D-pad left", "D-pad right", "Start", "Back",
    "Left thumb click", "Right thumb click", "A", "B", "X", "Y", "Black",
    "White", "Left trigger", "Right trigger", "Left stick left",
    "Left stick right", "Left stick down", "Left stick up",
    "Right stick left", "Right stick right", "Right stick down",
    "Right stick up"
};
static const char *const k_src_label[XSRC_COUNT] = {
    "-", "A", "B", "X", "Y", "LB", "RB", "Back", "Start", "LS click",
    "RS click", "D-pad up", "D-pad down", "D-pad left", "D-pad right", "LT",
    "RT", "LS left", "LS right", "LS down", "LS up", "RS left", "RS right",
    "RS down", "RS up"
};
static const int k_def_src[XIN_CONTROLS] = {
    XSRC_DUP, XSRC_DDOWN, XSRC_DLEFT, XSRC_DRIGHT, XSRC_START, XSRC_BACK,
    XSRC_LS, XSRC_RS, XSRC_A, XSRC_B, XSRC_X, XSRC_Y, XSRC_RB, XSRC_LB,
    XSRC_LT, XSRC_RT, XSRC_LX_NEG, XSRC_LX_POS, XSRC_LY_NEG, XSRC_LY_POS,
    XSRC_RX_NEG, XSRC_RX_POS, XSRC_RY_NEG, XSRC_RY_POS
};
static const int k_def_key[XIN_CONTROLS] = {
    VK_UP, VK_DOWN, VK_LEFT, VK_RIGHT, VK_RETURN, VK_BACK, VK_SHIFT,
    VK_CONTROL, 'Z', 'X', 'A', 'S', 'E', 'Q', '1', '3', VK_NUMPAD4,
    VK_NUMPAD6, VK_NUMPAD2, VK_NUMPAD8, 'J', 'L', 'K', 'I'
};
/* XInput button bits by source, 0 for analog sources. */
static const WORD k_src_bit[XSRC_COUNT] = {
    0, XINPUT_GAMEPAD_A, XINPUT_GAMEPAD_B, XINPUT_GAMEPAD_X, XINPUT_GAMEPAD_Y,
    XINPUT_GAMEPAD_LEFT_SHOULDER, XINPUT_GAMEPAD_RIGHT_SHOULDER,
    XINPUT_GAMEPAD_BACK, XINPUT_GAMEPAD_START, XINPUT_GAMEPAD_LEFT_THUMB,
    XINPUT_GAMEPAD_RIGHT_THUMB, XINPUT_GAMEPAD_DPAD_UP,
    XINPUT_GAMEPAD_DPAD_DOWN, XINPUT_GAMEPAD_DPAD_LEFT,
    XINPUT_GAMEPAD_DPAD_RIGHT, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};
/* Xbox digital buttons by control, 0 for the analog ones. */
static const WORD k_ctl_bit[XIN_CONTROLS] = {
    XBOX_GAMEPAD_DPAD_UP, XBOX_GAMEPAD_DPAD_DOWN, XBOX_GAMEPAD_DPAD_LEFT,
    XBOX_GAMEPAD_DPAD_RIGHT, XBOX_GAMEPAD_START, XBOX_GAMEPAD_BACK,
    XBOX_GAMEPAD_LEFT_THUMB, XBOX_GAMEPAD_RIGHT_THUMB
};
/* Analog button index by control (A..RT), -1 otherwise. */
static int ctl_analog(int c)
{
    switch (c) {
    case XIN_A: return XBOX_BUTTON_A;
    case XIN_B: return XBOX_BUTTON_B;
    case XIN_X: return XBOX_BUTTON_X;
    case XIN_Y: return XBOX_BUTTON_Y;
    case XIN_BLACK: return XBOX_BUTTON_BLACK;
    case XIN_WHITE: return XBOX_BUTTON_WHITE;
    case XIN_LTRIG: return XBOX_BUTTON_LTRIGGER;
    case XIN_RTRIG: return XBOX_BUTTON_RTRIGGER;
    }
    return -1;
}

const char *xbox_input_control_name(int c)
{
    return c >= 0 && c < XIN_CONTROLS ? k_ctl_label[c] : "?";
}

const char *xbox_input_source_name(int s)
{
    return s >= 0 && s < XSRC_COUNT ? k_src_label[s] : "?";
}

static XboxPortMap s_map[XBOX_MAX_CONTROLLERS];
static INIT_ONCE s_map_once = INIT_ONCE_STATIC_INIT;
static SRWLOCK s_map_lock = SRWLOCK_INIT;
static volatile LONG s_blocked;

static void map_defaults(int port, XboxPortMap *m)
{
    int c;
    memset(m, 0, sizeof *m);
    m->pad = port;
    m->keyboard = port == 0;
    m->dz_left = XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE;
    m->dz_right = XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE;
    m->trigger = XINPUT_GAMEPAD_TRIGGER_THRESHOLD;
    for (c = 0; c < XIN_CONTROLS; c++) {
        m->src[c] = k_def_src[c];
        m->key[c] = port == 0 ? k_def_key[c] : 0;
    }
}

static void map_key(char *buf, size_t n, int port, const char *what, const char *ctl)
{
    if (ctl)
        snprintf(buf, n, "port%d.%s.%s", port + 1, what, ctl);
    else
        snprintf(buf, n, "port%d.%s", port + 1, what);
}

static BOOL CALLBACK map_load(PINIT_ONCE o, PVOID p, PVOID *ctx)
{
    int port, c;
    char k[64];
    (void)o; (void)p; (void)ctx;
    for (port = 0; port < XBOX_MAX_CONTROLLERS; port++) {
        XboxPortMap *m = &s_map[port];
        map_defaults(port, m);
        map_key(k, sizeof k, port, "pad", NULL);
        m->pad = xbox_setting_get("controllers", k, m->pad);
        if (m->pad < -1 || m->pad > 3)
            m->pad = -1;
        map_key(k, sizeof k, port, "keyboard", NULL);
        m->keyboard = xbox_setting_get("controllers", k, m->keyboard) != 0;
        map_key(k, sizeof k, port, "dz_left", NULL);
        m->dz_left = xbox_setting_get("controllers", k, m->dz_left);
        map_key(k, sizeof k, port, "dz_right", NULL);
        m->dz_right = xbox_setting_get("controllers", k, m->dz_right);
        map_key(k, sizeof k, port, "trigger", NULL);
        m->trigger = xbox_setting_get("controllers", k, m->trigger);
        for (c = 0; c < XIN_CONTROLS; c++) {
            map_key(k, sizeof k, port, "key", k_ctl_key[c]);
            m->key[c] = xbox_setting_get("controllers", k, m->key[c]) & 0xFF;
            map_key(k, sizeof k, port, "src", k_ctl_key[c]);
            m->src[c] = xbox_setting_get("controllers", k, m->src[c]);
            if (m->src[c] < 0 || m->src[c] >= XSRC_COUNT)
                m->src[c] = XSRC_NONE;
        }
    }
    return TRUE;
}

static void map_ready(void)
{
    InitOnceExecuteOnce(&s_map_once, map_load, NULL, NULL);
}

void xbox_input_map_get(int port, XboxPortMap *m)
{
    map_ready();
    if (port < 0 || port >= XBOX_MAX_CONTROLLERS)
        return;
    AcquireSRWLockShared(&s_map_lock);
    *m = s_map[port];
    ReleaseSRWLockShared(&s_map_lock);
}

void xbox_input_map_set(int port, const XboxPortMap *m)
{
    int c;
    char k[64];
    map_ready();
    if (port < 0 || port >= XBOX_MAX_CONTROLLERS)
        return;
    AcquireSRWLockExclusive(&s_map_lock);
    s_map[port] = *m;
    ReleaseSRWLockExclusive(&s_map_lock);
    map_key(k, sizeof k, port, "pad", NULL);
    xbox_setting_set("controllers", k, m->pad);
    map_key(k, sizeof k, port, "keyboard", NULL);
    xbox_setting_set("controllers", k, m->keyboard);
    map_key(k, sizeof k, port, "dz_left", NULL);
    xbox_setting_set("controllers", k, m->dz_left);
    map_key(k, sizeof k, port, "dz_right", NULL);
    xbox_setting_set("controllers", k, m->dz_right);
    map_key(k, sizeof k, port, "trigger", NULL);
    xbox_setting_set("controllers", k, m->trigger);
    for (c = 0; c < XIN_CONTROLS; c++) {
        map_key(k, sizeof k, port, "key", k_ctl_key[c]);
        xbox_setting_set("controllers", k, m->key[c]);
        map_key(k, sizeof k, port, "src", k_ctl_key[c]);
        xbox_setting_set("controllers", k, m->src[c]);
    }
    xbox_settings_save();
}

void xbox_input_map_reset(int port)
{
    XboxPortMap m;
    map_defaults(port, &m);
    xbox_input_map_set(port, &m);
}

void xbox_input_set_blocked(int blocked)
{
    InterlockedExchange(&s_blocked, blocked ? 1 : 0);
}

int xbox_input_host_pad_connected(int pad)
{
    XINPUT_STATE st;
    return pad >= 0 && pad < 4 && XInputGetState((DWORD)pad, &st) == ERROR_SUCCESS;
}

int xbox_input_capture_source(int pad)
{
    XINPUT_STATE st;
    const XINPUT_GAMEPAD *g;
    int s;
    if (pad < 0 || pad > 3 || XInputGetState((DWORD)pad, &st) != ERROR_SUCCESS)
        return XSRC_NONE;
    g = &st.Gamepad;
    for (s = XSRC_A; s <= XSRC_DRIGHT; s++)
        if (g->wButtons & k_src_bit[s])
            return s;
    if (g->bLeftTrigger > 128)  return XSRC_LT;
    if (g->bRightTrigger > 128) return XSRC_RT;
    if (g->sThumbLX < -24000) return XSRC_LX_NEG;
    if (g->sThumbLX >  24000) return XSRC_LX_POS;
    if (g->sThumbLY < -24000) return XSRC_LY_NEG;
    if (g->sThumbLY >  24000) return XSRC_LY_POS;
    if (g->sThumbRX < -24000) return XSRC_RX_NEG;
    if (g->sThumbRX >  24000) return XSRC_RX_POS;
    if (g->sThumbRY < -24000) return XSRC_RY_NEG;
    if (g->sThumbRY >  24000) return XSRC_RY_POS;
    return XSRC_NONE;
}

/* Radial, like XInput's own guidance: inside the circle the stick is centred. */
static void stick_deadzone(SHORT x, SHORT y, int dz, int *ox, int *oy)
{
    if ((long long)x * x + (long long)y * y <= (long long)dz * dz)
        x = y = 0;
    *ox = x;
    *oy = y;
}

/* How far a host-pad source is pushed, 0..32767. */
static int src_amount(const XINPUT_GAMEPAD *g, int s, int trig, const int *st)
{
    int v;
    if (s <= XSRC_NONE || s >= XSRC_COUNT)
        return 0;
    if (k_src_bit[s])
        return (g->wButtons & k_src_bit[s]) ? 32767 : 0;
    switch (s) {
    case XSRC_LT:
        return g->bLeftTrigger > trig ? g->bLeftTrigger * 32767 / 255 : 0;
    case XSRC_RT:
        return g->bRightTrigger > trig ? g->bRightTrigger * 32767 / 255 : 0;
    }
    v = st[(s - XSRC_LX_NEG) / 2];             /* lx ly rx ry, deadzoned */
    if ((s - XSRC_LX_NEG) & 1)
        return v > 0 ? v : 0;                  /* positive half */
    return v < 0 ? (v < -32767 ? 32767 : -v) : 0;
}

static void log_mapping_once(void)
{
    static LONG done;
    XboxPortMap m;
    int c;
    if (InterlockedExchange(&done, 1))
        return;
    xbox_input_map_get(0, &m);
    fprintf(stderr, "  [INPUT] port 1: host pad %d, keyboard %s%s; settings file "
            "section [controllers] (in-game menu: Escape)\n", m.pad + 1,
            m.keyboard ? "on" : "off", keyboard_enabled() ? "" : " (RECOMP_KEYBOARD off)");
    for (c = 0; c < XIN_CONTROLS; c++)
        fprintf(stderr, "  [INPUT]   %-18s <- pad %-9s | key 0x%02X\n",
                k_ctl_label[c], k_src_label[m.src[c]], m.key[c]);
    fflush(stderr);
}

void xbox_InputInit(void)
{
    for (DWORD i = 0; i < XBOX_MAX_CONTROLLERS; i++) {
        XINPUT_STATE state;
        DWORD result = XInputGetState(i, &state);
        g_controller_connected[i] = (result == ERROR_SUCCESS);
    }
}

DWORD xbox_InputGetState(DWORD dwPort, XBOX_INPUT_STATE *pState)
{
    static DWORD packet[XBOX_MAX_CONTROLLERS];
    static XBOX_GAMEPAD last_out[XBOX_MAX_CONTROLLERS];
    XINPUT_STATE xi_state;
    XboxPortMap m;
    DWORD result = ERROR_DEVICE_NOT_CONNECTED;
    BOOL kbd;
    int st[4] = { 0, 0, 0, 0 }, c;

    if (dwPort >= XBOX_MAX_CONTROLLERS || !pState)
        return ERROR_DEVICE_NOT_CONNECTED;

    log_mapping_once();
    xbox_input_map_get((int)dwPort, &m);
    kbd = keyboard_enabled() && m.keyboard;
    memset(&xi_state, 0, sizeof xi_state);
    if (m.pad >= 0)
        result = XInputGetState((DWORD)m.pad, &xi_state);

    /* The host pad's first few changes, raw (before deadzones), with the
     * time since the first poll. Always on and capped, because the question
     * it answers comes from a player's own run with a real controller -- "the
     * intro movie skipped by itself" -- and no scripted run reproduces that
     * pad. The log then says whether, and what, the pad sent at that moment. */
    {
        static XINPUT_GAMEPAD last[XBOX_MAX_CONTROLLERS];
        static DWORD last_result[XBOX_MAX_CONTROLLERS] = {
            0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu };
        static ULONGLONG t0;
        static LONG logged;
        const XINPUT_GAMEPAD *g = &xi_state.Gamepad;
        if (!t0)
            t0 = GetTickCount64();
        if (result != last_result[dwPort]
            || (result == ERROR_SUCCESS && memcmp(g, &last[dwPort], sizeof *g))) {
            if (InterlockedIncrement(&logged) <= 24) {
                if (result == ERROR_SUCCESS)
                    fprintf(stderr, "  [HOSTPAD] t=%llu ms port %lu (host pad %d): "
                            "buttons=%04X LT=%u RT=%u L=(%d,%d) R=(%d,%d)\n",
                            GetTickCount64() - t0, (unsigned long)dwPort, m.pad + 1,
                            g->wButtons, g->bLeftTrigger, g->bRightTrigger,
                            g->sThumbLX, g->sThumbLY, g->sThumbRX, g->sThumbRY);
                else if (last_result[dwPort] == ERROR_SUCCESS
                         || last_result[dwPort] == 0xFFFFFFFFu)
                    fprintf(stderr, "  [HOSTPAD] t=%llu ms port %lu: no host pad "
                            "(%lu)\n", GetTickCount64() - t0,
                            (unsigned long)dwPort, (unsigned long)result);
                fflush(stderr);
            }
            last_result[dwPort] = result;
            if (result == ERROR_SUCCESS)
                last[dwPort] = *g;
        }
    }

    g_controller_connected[dwPort] = result == ERROR_SUCCESS;
    if (result != ERROR_SUCCESS && !kbd)
        return result;
    if (result != ERROR_SUCCESS)
        memset(&xi_state, 0, sizeof xi_state);

    memset(pState, 0, sizeof(XBOX_INPUT_STATE));
    /* The menu is open: the player is using it, not the game. */
    if (s_blocked) {
        memset(&last_out[dwPort], 0, sizeof last_out[dwPort]);
        pState->dwPacketNumber = packet[dwPort];
        return ERROR_SUCCESS;
    }

    /* XInput's deadzones, which the raw state does not apply. A stick at
     * rest is rarely at 0, and a trigger resting at a few counts is an
     * analog button half-pressed to a title -- either can read as input the
     * player never gave (a movie that skips on any press). */
    stick_deadzone(xi_state.Gamepad.sThumbLX, xi_state.Gamepad.sThumbLY,
                   m.dz_left, &st[0], &st[1]);
    stick_deadzone(xi_state.Gamepad.sThumbRX, xi_state.Gamepad.sThumbRY,
                   m.dz_right, &st[2], &st[3]);

    {
        int amount[XIN_CONTROLS];
        for (c = 0; c < XIN_CONTROLS; c++) {
            amount[c] = src_amount(&xi_state.Gamepad, m.src[c], m.trigger, st);
            /* Analog on the console, so a key is fully pressed rather than a
             * flag -- a title that reads a pressure never sees a press of 1. */
            if (kbd && m.key[c] && xbox_FramebufferKeyDown(m.key[c]))
                amount[c] = 32767;
        }
        for (c = 0; c < XIN_CONTROLS; c++) {
            int a = ctl_analog(c);
            if (c < XIN_A && amount[c] >= 16384)
                pState->Gamepad.wButtons |= k_ctl_bit[c];
            else if (a >= 0)
                pState->Gamepad.bAnalogButtons[a] = (BYTE)(amount[c] * 255 / 32767);
        }
        pState->Gamepad.sThumbLX = (SHORT)(amount[XIN_LX_POS] - amount[XIN_LX_NEG]);
        pState->Gamepad.sThumbLY = (SHORT)(amount[XIN_LY_POS] - amount[XIN_LY_NEG]);
        pState->Gamepad.sThumbRX = (SHORT)(amount[XIN_RX_POS] - amount[XIN_RX_NEG]);
        pState->Gamepad.sThumbRY = (SHORT)(amount[XIN_RY_POS] - amount[XIN_RY_NEG]);
    }

    /* The title's input layer looks for edges, so the packet number has to
     * move whenever the state does or a press is never noticed. */
    if (memcmp(&pState->Gamepad, &last_out[dwPort], sizeof last_out[dwPort])) {
        last_out[dwPort] = pState->Gamepad;
        packet[dwPort]++;
    }
    pState->dwPacketNumber = packet[dwPort];
    return ERROR_SUCCESS;
}

DWORD xbox_InputSetState(DWORD dwPort, const XBOX_VIBRATION *pVibration)
{
    XINPUT_VIBRATION xi_vib;

    if (dwPort >= XBOX_MAX_CONTROLLERS || !pVibration)
        return ERROR_DEVICE_NOT_CONNECTED;

    xi_vib.wLeftMotorSpeed = pVibration->wLeftMotorSpeed;
    xi_vib.wRightMotorSpeed = pVibration->wRightMotorSpeed;
    {   /* the host pad that drives this port */
        XboxPortMap m;
        xbox_input_map_get((int)dwPort, &m);
        if (m.pad < 0)
            return ERROR_SUCCESS;
        return XInputSetState((DWORD)m.pad, &xi_vib);
    }
}

BOOL xbox_InputIsConnected(DWORD dwPort)
{
    if (dwPort >= XBOX_MAX_CONTROLLERS) return FALSE;
    return g_controller_connected[dwPort];
}

DWORD xbox_InputGetCapabilities(DWORD dwPort, DWORD dwFlags, XBOX_INPUT_CAPABILITIES *pCaps)
{
    XINPUT_CAPABILITIES xi_caps;
    DWORD result;

    if (dwPort >= XBOX_MAX_CONTROLLERS || !pCaps)
        return ERROR_DEVICE_NOT_CONNECTED;

    result = XInputGetCapabilities(dwPort, dwFlags, &xi_caps);
    if (result != ERROR_SUCCESS) return result;

    memset(pCaps, 0, sizeof(XBOX_INPUT_CAPABILITIES));
    pCaps->Type = xi_caps.Type;
    pCaps->SubType = xi_caps.SubType;
    pCaps->Flags = xi_caps.Flags;
    return ERROR_SUCCESS;
}

/* ======================================================================== */
