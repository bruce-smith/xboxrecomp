/**
 * XInput on SDL2 (POSIX hosts): XInputGetState, XInputSetState and
 * XInputGetCapabilities answered from SDL's GameController, so the input
 * layer in xinput_device.c -- deadzones, the keyboard stand-in merged on top,
 * the [HOSTPAD] log -- is the same code on Linux as on Windows.
 *
 * SDL's controller database maps whatever is plugged in (a Steam Deck's
 * built-in pad, an Xbox or DualShock pad) to the Xbox layout XInput reports,
 * so the translation below is one-to-one. Controllers are opened lazily and
 * looked for again about once a second while a slot is empty, which covers
 * hot-plugging without an event loop on this thread.
 */
#include "platform/xbox_winnt.h"
#include <SDL.h>
#include <string.h>

#define HOST_PADS 4

static SDL_GameController *s_pad[HOST_PADS];
static SDL_mutex          *s_lock;
static DWORD               s_packet[HOST_PADS];
static XINPUT_GAMEPAD      s_last[HOST_PADS];
static Uint32              s_next_scan;

static int sdl_ready(void)
{
    static int state;          /* 0 untried, 1 up, -1 failed */
    if (state == 0) {
        /* Positions, as XInput reports them: the bottom face button is A on
         * every pad, Nintendo layouts included (SDL2 maps those by label by
         * default). And the pad keeps working while another window has the
         * focus, as XInput does. */
        SDL_SetHint(SDL_HINT_GAMECONTROLLER_USE_BUTTON_LABELS, "0");
        SDL_SetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, "1");
        /* SDL would turn SIGINT/SIGTERM into a quit event nobody reads. */
        SDL_SetHint(SDL_HINT_NO_SIGNAL_HANDLERS, "1");
        state = (SDL_InitSubSystem(SDL_INIT_GAMECONTROLLER) == 0) ? 1 : -1;
        if (state > 0)
            s_lock = SDL_CreateMutex();
        else
            fprintf(stderr, "  [HOSTPAD] SDL game controllers unavailable: %s\n",
                    SDL_GetError());
    }
    return state > 0 && s_lock;
}

/* Open every attached controller into a free slot, in device order. */
static void scan(void)
{
    int i, n = SDL_NumJoysticks();
    for (i = 0; i < n; i++) {
        SDL_JoystickID id;
        int k, have = 0, slot = -1;
        if (!SDL_IsGameController(i))
            continue;
        id = SDL_JoystickGetDeviceInstanceID(i);
        for (k = 0; k < HOST_PADS; k++) {
            if (s_pad[k] && SDL_JoystickInstanceID(
                    SDL_GameControllerGetJoystick(s_pad[k])) == id)
                have = 1;
            if (!s_pad[k] && slot < 0)
                slot = k;
        }
        if (!have && slot >= 0) {
            s_pad[slot] = SDL_GameControllerOpen(i);
            if (s_pad[slot])
                fprintf(stderr, "  [HOSTPAD] slot %d: %s\n", slot,
                        SDL_GameControllerName(s_pad[slot]));
        }
    }
}

/* The controller in slot idx, or NULL. Called with s_lock held. */
static SDL_GameController *pad(DWORD idx)
{
    SDL_GameController *c;
    Uint32 now = SDL_GetTicks();

    SDL_GameControllerUpdate();
    c = s_pad[idx];
    if (c && !SDL_GameControllerGetAttached(c)) {
        SDL_GameControllerClose(c);
        s_pad[idx] = c = NULL;
    }
    if (!c && (Sint32)(now - s_next_scan) >= 0) {
        s_next_scan = now + 1000;
        SDL_JoystickUpdate();
        scan();
        c = s_pad[idx];
    }
    return c;
}

static SHORT axis(SDL_GameController *c, SDL_GameControllerAxis a, int invert)
{
    int v = SDL_GameControllerGetAxis(c, a);
    /* SDL's Y axes point down, XInput's up; -1 - v keeps -32768 in range. */
    return (SHORT)(invert ? -1 - v : v);
}

DWORD XInputGetState(DWORD idx, XINPUT_STATE *state)
{
    static const struct { SDL_GameControllerButton b; WORD bit; } map[] = {
        { SDL_CONTROLLER_BUTTON_DPAD_UP,       XINPUT_GAMEPAD_DPAD_UP },
        { SDL_CONTROLLER_BUTTON_DPAD_DOWN,     XINPUT_GAMEPAD_DPAD_DOWN },
        { SDL_CONTROLLER_BUTTON_DPAD_LEFT,     XINPUT_GAMEPAD_DPAD_LEFT },
        { SDL_CONTROLLER_BUTTON_DPAD_RIGHT,    XINPUT_GAMEPAD_DPAD_RIGHT },
        { SDL_CONTROLLER_BUTTON_START,         XINPUT_GAMEPAD_START },
        { SDL_CONTROLLER_BUTTON_BACK,          XINPUT_GAMEPAD_BACK },
        { SDL_CONTROLLER_BUTTON_LEFTSTICK,     XINPUT_GAMEPAD_LEFT_THUMB },
        { SDL_CONTROLLER_BUTTON_RIGHTSTICK,    XINPUT_GAMEPAD_RIGHT_THUMB },
        { SDL_CONTROLLER_BUTTON_LEFTSHOULDER,  XINPUT_GAMEPAD_LEFT_SHOULDER },
        { SDL_CONTROLLER_BUTTON_RIGHTSHOULDER, XINPUT_GAMEPAD_RIGHT_SHOULDER },
        { SDL_CONTROLLER_BUTTON_A,             XINPUT_GAMEPAD_A },
        { SDL_CONTROLLER_BUTTON_B,             XINPUT_GAMEPAD_B },
        { SDL_CONTROLLER_BUTTON_X,             XINPUT_GAMEPAD_X },
        { SDL_CONTROLLER_BUTTON_Y,             XINPUT_GAMEPAD_Y },
    };
    SDL_GameController *c;
    XINPUT_GAMEPAD g;
    size_t i;

    if (!state)
        return 87;                              /* ERROR_INVALID_PARAMETER */
    memset(state, 0, sizeof *state);
    if (idx >= HOST_PADS || !sdl_ready())
        return ERROR_DEVICE_NOT_CONNECTED;

    SDL_LockMutex(s_lock);
    c = pad(idx);
    if (!c) {
        SDL_UnlockMutex(s_lock);
        return ERROR_DEVICE_NOT_CONNECTED;
    }
    memset(&g, 0, sizeof g);
    for (i = 0; i < sizeof map / sizeof map[0]; i++)
        if (SDL_GameControllerGetButton(c, map[i].b))
            g.wButtons |= map[i].bit;
    /* SDL triggers are 0..32767; XInput's 0..255. */
    g.bLeftTrigger  = (BYTE)(SDL_GameControllerGetAxis(c, SDL_CONTROLLER_AXIS_TRIGGERLEFT) >> 7);
    g.bRightTrigger = (BYTE)(SDL_GameControllerGetAxis(c, SDL_CONTROLLER_AXIS_TRIGGERRIGHT) >> 7);
    g.sThumbLX = axis(c, SDL_CONTROLLER_AXIS_LEFTX, 0);
    g.sThumbLY = axis(c, SDL_CONTROLLER_AXIS_LEFTY, 1);
    g.sThumbRX = axis(c, SDL_CONTROLLER_AXIS_RIGHTX, 0);
    g.sThumbRY = axis(c, SDL_CONTROLLER_AXIS_RIGHTY, 1);
    /* XInput's packet number moves only when the state does. */
    if (memcmp(&g, &s_last[idx], sizeof g)) {
        s_last[idx] = g;
        s_packet[idx]++;
    }
    state->dwPacketNumber = s_packet[idx];
    state->Gamepad = g;
    SDL_UnlockMutex(s_lock);
    return ERROR_SUCCESS;
}

DWORD XInputSetState(DWORD idx, XINPUT_VIBRATION *vibration)
{
    SDL_GameController *c;
    if (!vibration || idx >= HOST_PADS || !sdl_ready())
        return ERROR_DEVICE_NOT_CONNECTED;
    SDL_LockMutex(s_lock);
    c = pad(idx);
    if (c)
        /* XInput holds a level until it is changed; SDL wants a duration,
         * so a long one, refreshed by every call. */
        SDL_GameControllerRumble(c, vibration->wLeftMotorSpeed,
                                 vibration->wRightMotorSpeed, 10000);
    SDL_UnlockMutex(s_lock);
    return c ? ERROR_SUCCESS : ERROR_DEVICE_NOT_CONNECTED;
}

DWORD XInputGetCapabilities(DWORD idx, DWORD flags, XINPUT_CAPABILITIES *caps)
{
    SDL_GameController *c;
    (void)flags;
    if (!caps || idx >= HOST_PADS || !sdl_ready())
        return ERROR_DEVICE_NOT_CONNECTED;
    SDL_LockMutex(s_lock);
    c = pad(idx);
    SDL_UnlockMutex(s_lock);
    if (!c)
        return ERROR_DEVICE_NOT_CONNECTED;
    memset(caps, 0, sizeof *caps);
    caps->Type = XINPUT_DEVTYPE_GAMEPAD;
    caps->SubType = XINPUT_DEVSUBTYPE_GAMEPAD;
    return ERROR_SUCCESS;
}
