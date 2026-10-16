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
int xbox_NetplayStart(const char *game_dir);
/* The game's build identity, before xbox_NetplayStart: what must match for
 * two builds to play together (e.g. both repos' commits and a hash of the
 * generated code). A string containing "dirty", or none, falls back to
 * comparing the exe's bytes. The string must stay valid. */
void xbox_NetplaySetBuild(const char *id);
int xbox_NetplayActive(void);
void xbox_NetplayLeave(void);     /* tell the others this player left */
int xbox_NetplayPlayer(void);     /* 0-based: the Xbox port this peer plays */
int xbox_NetplayPlayers(void);

/* The pad override (usb_gamepad.c): the 18 state bytes of `port` for the
 * current frame. */
int xbox_NetplayPadState(int port, uint8_t *state18);

#ifdef __cplusplus
}
#endif

#endif /* XBOX_NETPLAY_H */
