/*
 * usb_gamepad.h -- the device on the other end of the wire.
 *
 * A host controller moves bytes; it does not know what they mean. This is the
 * thing that answers them: an Xbox controller, as the console's own USB stack
 * expects to find it -- standard descriptors over endpoint 0, and a 20-byte
 * report over interrupt endpoint 1.
 *
 * Kept apart from ohci.c because it is a different concern. The controller
 * walks descriptor lists and raises interrupts and would do the same for a
 * memory unit or a headset; this knows one device and nothing about lists.
 */
#ifndef XBOX_USB_GAMEPAD_H
#define XBOX_USB_GAMEPAD_H

#include <stdint.h>

/* USB setup packet, as it arrives in a SETUP transfer's buffer. */
typedef struct {
    uint8_t  bmRequestType;
    uint8_t  bRequest;
    uint16_t wValue;
    uint16_t wIndex;
    uint16_t wLength;
} UsbSetup;

/* Answer a control request.
 *
 * Returns the number of bytes written to `out` (at most `max`), or -1 if the
 * request is not one this device answers -- which the caller reports as a
 * stall rather than as a short transfer, because those mean different things
 * to a driver.
 */
int usb_gamepad_control(int pad, const UsbSetup *setup, uint8_t *out, int max);

/* Fill in pad `pad`'s 20-byte input report. Returns the byte count written. */
int usb_gamepad_report(int pad, uint8_t *out, int max);

/* The address the host assigned pad `pad` with SET_ADDRESS, 0 until it does,
 * and whether it has been configured. */
uint8_t usb_gamepad_address(int pad);
int usb_gamepad_configured(int pad);

/* Up to four pads, one per Xbox controller port. Pad n is driven by host
 * XInput pad n and by pad-script steps prefixed "p<n+1>-". */
#define USB_GAMEPAD_MAX 4

/* Network play (src/net): when set, fn supplies each pad's 18 state bytes
 * (buttons, analog buttons, sticks) instead of the host -- the lockstep
 * input of that port's player. usb_gamepad_sample_local reads this
 * machine's own player (the port-1 bindings: host pad, keyboard, script). */
void usb_gamepad_set_override(int (*fn)(int pad, uint8_t *state18));
void usb_gamepad_sample_local(uint8_t *state18);

/* A title-side filter on each live pad report (NULL: none, the default). fn
 * gets the pad (0-3) and its 18 state bytes -- byte 0/1 the digital buttons,
 * 2-9 the analog buttons (A, B, X, Y, Black, White, LT, RT), then four
 * little-endian int16 sticks (LX, LY, RX, RY) -- after the host pad,
 * keyboard and pad scripts and before the in-game menu's blanking, the
 * guest-CPU schedule, the input recording and the network game. In a network
 * game only this machine's own player is sampled, as pad 0
 * (usb_gamepad_sample_local); the filter maps it to xbox_NetplayPlayer()'s
 * port itself. It must decide from guest state alone, so every peer and
 * every replay presses the same thing. */
void usb_gamepad_set_filter(void (*fn)(int pad, uint8_t *state18));

#endif /* XBOX_USB_GAMEPAD_H */
