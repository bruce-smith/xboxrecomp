/**
 * The APU's audio output on SDL2 (POSIX hosts): the xa2_* interface of
 * apu_xaudio2.h over an SDL audio device in queue (push) mode, with the same
 * format and the same back-pressure as the XAudio2 backend -- 48 kHz stereo
 * 16-bit, 1024-sample buffers, at most three queued -- so the APU's frame
 * thread paces itself identically on both hosts.
 *
 * RECOMP_AUDIO_MUTE=1 queues silence instead: timing as usual, no sound.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <SDL.h>
#include "apu_xaudio2.h"

#define SDL_A_SAMPLE_RATE 48000
#define SDL_A_CHANNELS    2
#define SDL_A_BUF_SAMPLES 1024   /* ~21 ms per submission */
#define SDL_A_NUM_BUFS    3
#define SDL_A_BUF_BYTES   (SDL_A_BUF_SAMPLES * SDL_A_CHANNELS * (int)sizeof(int16_t))

static SDL_AudioDeviceID s_dev;
static int               s_mute;
static int               s_frames_written;

int xa2_init(void)
{
    SDL_AudioSpec want, have;

    if (s_dev)
        return 1;
    /* SDL would turn SIGINT/SIGTERM into a quit event nobody reads. */
    SDL_SetHint(SDL_HINT_NO_SIGNAL_HANDLERS, "1");
    if (SDL_InitSubSystem(SDL_INIT_AUDIO) != 0) {
        fprintf(stderr, "[XA2] SDL audio unavailable: %s\n", SDL_GetError());
        return 0;
    }
    memset(&want, 0, sizeof want);
    want.freq     = SDL_A_SAMPLE_RATE;
    want.format   = AUDIO_S16SYS;
    want.channels = SDL_A_CHANNELS;
    want.samples  = SDL_A_BUF_SAMPLES;
    want.callback = NULL;                 /* SDL_QueueAudio */
    /* No allowed changes: SDL converts if the device differs, so the APU
     * always produces what it does on Windows. */
    s_dev = SDL_OpenAudioDevice(NULL, 0, &want, &have, 0);
    if (!s_dev) {
        fprintf(stderr, "[XA2] SDL_OpenAudioDevice failed: %s\n", SDL_GetError());
        return 0;
    }
    s_mute = getenv("RECOMP_AUDIO_MUTE") != NULL;
    s_frames_written = 0;
    SDL_PauseAudioDevice(s_dev, 0);
    fprintf(stderr, "[XA2] SDL audio initialized (%d Hz stereo 16-bit, %d x %d-sample "
            "buffers, driver %s)\n", SDL_A_SAMPLE_RATE, SDL_A_NUM_BUFS,
            SDL_A_BUF_SAMPLES, SDL_GetCurrentAudioDriver());
    return 1;
}

void xa2_shutdown(void)
{
    if (!s_dev)
        return;
    SDL_CloseAudioDevice(s_dev);
    s_dev = 0;
    fprintf(stderr, "[XA2] Shut down (%d frames written)\n", s_frames_written);
}

int xa2_is_active(void)
{
    return s_dev != 0;
}

int xa2_queued_buffers(void)
{
    Uint32 q;
    if (!s_dev)
        return -1;
    q = SDL_GetQueuedAudioSize(s_dev);
    return (int)((q + SDL_A_BUF_BYTES - 1) / SDL_A_BUF_BYTES);
}

int xa2_submit_samples(const int16_t *samples, int num_samples)
{
    static int16_t silence[SDL_A_BUF_SAMPLES * SDL_A_CHANNELS];
    int n;

    if (!s_dev)
        return 0;
    if (xa2_queued_buffers() >= SDL_A_NUM_BUFS)
        return 0;
    n = num_samples > SDL_A_BUF_SAMPLES ? SDL_A_BUF_SAMPLES : num_samples;
    if (SDL_QueueAudio(s_dev, s_mute ? silence : samples,
                       (Uint32)(n * SDL_A_CHANNELS * (int)sizeof(int16_t))) != 0)
        return 0;
    s_frames_written++;
    return 1;
}

int xa2_get_buffer_size(void)
{
    return SDL_A_BUF_SAMPLES;
}

int xa2_max_queued_buffers(void)
{
    return SDL_A_NUM_BUFS;
}
