"""--yield-points puts RECOMP_YIELD_POINT(va) at a spin loop's head.

With the runtime's guest-CPU schedule one guest thread runs at a time, and it
lets go of the CPU only where it would block. A loop that waits in guest code
for another thread or a device -- Blinx 2's main loop waiting for the vblank
callback to count, D3D waiting for a busy bit -- never blocks, so it has to be
told where to let go. The call goes at the head of the loop, which is a branch
target and therefore already has a label; a function with no yield points
named carries none.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools.recomp import config  # noqa: E402
from tools.recomp.translator import FunctionTranslator  # noqa: E402

BASE = 0x00010000


def _translate(image, yield_points=None):
    config._install(
        [config.Section(".text", BASE, len(image), 0x0000, len(image), True)],
        entry_point=BASE, kernel_thunk_addr=BASE, origin="yield-point-test")
    db = {BASE: {"start": f"0x{BASE:08X}", "end": BASE + len(image),
                 "_addr": BASE, "size": len(image)}}
    return FunctionTranslator(image, db,
                              yield_points=yield_points).translate_function(
        BASE, db[BASE])


#   0: mov eax, [0x00001000]      A1 00 10 00 00
#   5: test eax, eax              85 C0
#   7: je 0                       74 F7
#   9: ret                        C3
SPIN = b"\xA1\x00\x10\x00\x00\x85\xC0\x74\xF7\xC3"


def test_no_yield_points_by_default():
    code = _translate(SPIN)
    assert "RECOMP_YIELD_POINT" not in code, code


def test_yield_point_at_the_loop_head():
    code = _translate(SPIN, {BASE})
    label = f"loc_{BASE:08X}: ;"
    assert label in code, code
    after = code.split(label, 1)[1]
    first = next(l for l in after.splitlines() if l.strip())
    assert first.strip() == f"RECOMP_YIELD_POINT(0x{BASE:08X}u);", code


def test_a_named_address_that_is_not_a_block_head_emits_nothing():
    code = _translate(SPIN, {BASE + 5})
    assert "RECOMP_YIELD_POINT" not in code, code


def _translate_loops(image, loop_heads):
    config._install(
        [config.Section(".text", BASE, len(image), 0x0000, len(image), True)],
        entry_point=BASE, kernel_thunk_addr=BASE, origin="loop-head-test")
    db = {BASE: {"start": f"0x{BASE:08X}", "end": BASE + len(image),
                 "_addr": BASE, "size": len(image)}}
    return FunctionTranslator(image, db,
                              loop_heads=loop_heads).translate_function(
        BASE, db[BASE])


def test_loop_head_counts_only_back_edge_targets():
    # The spin's head (BASE) is reached from the je after it: a loop head.
    code = _translate_loops(SPIN, True)
    after = code.split(f"loc_{BASE:08X}: ;", 1)[1]
    assert after.count("RECOMP_LOOP_HEAD();") == 1, code
    assert "RECOMP_LOOP_HEAD" not in _translate_loops(SPIN, False)
    #   0: test eax, eax   85 C0
    #   2: je 5            74 01
    #   4: nop             90
    #   5: ret             C3      (a forward target only)
    fwd = b"\x85\xC0\x74\x01\x90\xC3"
    assert "RECOMP_LOOP_HEAD" not in _translate_loops(fwd, True)
