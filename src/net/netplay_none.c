/*
 * Network play where netplay.c (Winsock) is not built -- POSIX hosts for
 * now: the same API, never active. RECOMP_NET is reported as unavailable.
 * A BSD-sockets port of netplay.c is the follow-up.
 */
#include <stdio.h>
#include <stdlib.h>
#include "netplay.h"

int xbox_NetplayStart(void)
{
    if (getenv("RECOMP_NET")) {
        fprintf(stderr, "[NET] network play is not available in this build yet\n");
        fflush(stderr);
    }
    return 0;
}

int xbox_NetplayActive(void)  { return 0; }
int xbox_NetplayPlayer(void)  { return 0; }
int xbox_NetplayPlayers(void) { return 1; }

int xbox_NetplayPadState(int port, uint8_t *state18)
{
    (void)port; (void)state18;
    return 0;
}
