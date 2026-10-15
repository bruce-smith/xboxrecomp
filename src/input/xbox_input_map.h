/*
 * xbox_input_map.h -- which host input drives each Xbox control, per port.
 *
 * Each of the four Xbox ports takes one host pad (or none) and, if enabled,
 * the keyboard. Every Xbox control -- d-pad, Start/Back, thumb clicks, the
 * six analog buttons, both triggers, both sticks' four directions -- has one
 * key and one host-pad source. Stored in the settings file (section
 * "controllers"); the defaults are the mapping this layer always had.
 */
#ifndef XBOX_INPUT_MAP_H
#define XBOX_INPUT_MAP_H

#ifdef __cplusplus
extern "C" {
#endif

enum {
    XIN_DUP, XIN_DDOWN, XIN_DLEFT, XIN_DRIGHT, XIN_START, XIN_BACK,
    XIN_LTHUMB, XIN_RTHUMB, XIN_A, XIN_B, XIN_X, XIN_Y, XIN_BLACK, XIN_WHITE,
    XIN_LTRIG, XIN_RTRIG,
    XIN_LX_NEG, XIN_LX_POS, XIN_LY_NEG, XIN_LY_POS,
    XIN_RX_NEG, XIN_RX_POS, XIN_RY_NEG, XIN_RY_POS,
    XIN_CONTROLS
};

/* Host-pad sources (XInput). 0 is none. */
enum {
    XSRC_NONE, XSRC_A, XSRC_B, XSRC_X, XSRC_Y, XSRC_LB, XSRC_RB, XSRC_BACK,
    XSRC_START, XSRC_LS, XSRC_RS, XSRC_DUP, XSRC_DDOWN, XSRC_DLEFT, XSRC_DRIGHT,
    XSRC_LT, XSRC_RT, XSRC_LX_NEG, XSRC_LX_POS, XSRC_LY_NEG, XSRC_LY_POS,
    XSRC_RX_NEG, XSRC_RX_POS, XSRC_RY_NEG, XSRC_RY_POS,
    XSRC_COUNT
};

typedef struct {
    int pad;                    /* host pad 0-3, or -1 for none */
    int keyboard;               /* the keyboard drives this port too */
    int dz_left, dz_right;      /* radial stick deadzones, 0-32767 */
    int trigger;                /* trigger threshold, 0-255 */
    int key[XIN_CONTROLS];      /* virtual-key code, 0 none */
    int src[XIN_CONTROLS];      /* XSRC_* */
} XboxPortMap;

const char *xbox_input_control_name(int control);
const char *xbox_input_source_name(int src);

/* A copy of a port's map, and a change to it (kept and saved). */
void xbox_input_map_get(int port, XboxPortMap *m);
void xbox_input_map_set(int port, const XboxPortMap *m);
void xbox_input_map_reset(int port);        /* back to the defaults */

/* The first host-pad source held right now on host pad `pad` (for "press a
 * button to bind"), or XSRC_NONE. */
int  xbox_input_capture_source(int pad);
int  xbox_input_host_pad_connected(int pad);

/* While set (the menu is open), every port reports nothing pressed. */
void xbox_input_set_blocked(int blocked);

#ifdef __cplusplus
}
#endif

#endif /* XBOX_INPUT_MAP_H */
