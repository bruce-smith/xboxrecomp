/* netplay.h -- lockstep multiplayer (netplay.c). */
#ifndef XBOX_NETPLAY_H
#define XBOX_NETPLAY_H

#include <stddef.h>
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

/* Cheats in a network game (cheats.c): a request (22 bytes, as recorded in
 * IN_EVT_CHEAT) must apply on every peer at the same frame.
 * xbox_NetplayCheatSend: 1 if a network game took it -- then do not apply it
 * locally; it travels with this player's next frame (one a frame, the rest
 * queue). xbox_NetplayCheatNext: the next request that has come due, 1 if
 * one was returned; call it in a loop at the poll and apply each. */
int xbox_NetplayCheatSend(const uint8_t ev[22]);
int xbox_NetplayCheatNext(uint8_t ev[22]);

/* Journey Bags (netbag.c): a joiner's own team, made and returned by the
 * game's own code in headless passes, on a work copy of its saves. See the
 * header of netbag.c. root: xbox_NetBagRoot's folder. */
void xbox_NetBagRoot(const char *explicit_save_dir, char *out, size_t n);
/* The game folder (main.c), and this player's real saves and netbag folder. */
void xbox_NetplaySetGameDir(const char *dir);
void xbox_NetBagPaths(char *saves, size_t sn, char *root, size_t rn, int *explicit_dir);
int  xbox_NetBagPrepare(const char *saves, const char *root, int slot, char *why, size_t n);
int  xbox_NetBagReady(const char *root, char *bagdir, size_t n);
int  xbox_NetBagFinish(const char *saves, const char *root, int explicit_dir,
                       const char *session_bag, char *msg, size_t n);
int  xbox_NetBagHostBaseline(const char *saves, const char *root);
int  xbox_NetBagHostCommit(const char *saves, const char *root, int explicit_dir,
                           const char *session, char *msg, size_t n);
int  xbox_NetBagRestoreLast(const char *saves, const char *root, char *msg, size_t n);
uint64_t xbox_NetBagHashTree(const char *dir);
/* 1 (and the save's folder) if one of `saves` is the home of the bag in bagdir. */
int  xbox_NetBagHomeClash(const char *bagdir, const char *saves, char *id, size_t n);
/* The host ends the network game cleanly (the menu's End network game):
 * every peer stops at the same frame, compares, and joiners take their
 * Journey Bags home. 1 if an end was started. */
int  xbox_NetplayEnd(void);
/* The title's gate for that (Blinx 2: not during a match, so the Journey
 * Bags are saved): 1 when ending is safe now, else 0 and why. With a gate,
 * xbox_NetplayEnd waits for it; xbox_NetplayEndNow does not. Status for the
 * menu: NULL when no end is under way. */
void xbox_NetplaySetEndGate(int (*gate)(uint32_t frame, char *why, size_t n));
int  xbox_NetplayEndNow(void);
const char *xbox_NetplayEndStatus(void);

#ifdef __cplusplus
}
#endif

#endif /* XBOX_NETPLAY_H */
