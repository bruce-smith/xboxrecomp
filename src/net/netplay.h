/* netplay.h -- lockstep multiplayer (netplay.c). */
#ifndef XBOX_NETPLAY_H
#define XBOX_NETPLAY_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* RECOMP_NET=host:<port> | join:<ip>:<port>: wait for the session, then set
 * up virtual mode and the pads. Call before the schedule or USB start.
 * Returns 1 when a network game is on. */
int xbox_NetplayStart(void);
int xbox_NetplayActive(void);
int xbox_NetplayPlayer(void);     /* 0-based: the Xbox port this peer plays */
int xbox_NetplayPlayers(void);

/* The pad override (usb_gamepad.c): the 18 state bytes of `port` for the
 * current frame. */
int xbox_NetplayPadState(int port, uint8_t *state18);

#ifdef __cplusplus
}
#endif

#endif /* XBOX_NETPLAY_H */
