"""MOVSX from a 16-bit register sign-extends, whichever register it is.

`_lift_movsx` listed ax/bx/cx/dx/si/di and left bp and sp out (movzx has
all eight), so `movsx eax, bp` fell through to a plain read of the low word:
a zero extension. Blinx 2's pad poll keeps the right stick's Y in bp
(0x0021C457), so a stick pushed down read as +1..+2 instead of -1..0 and the
first-person camera could not pitch. Nine sites in that title.

Against the unfixed lifter the bp and sp cases emit `eax = LO16(ebp);` and
fail.
"""

import pytest

from .disasm import Instruction, Operand
from .lifter import Lifter

_REGS16 = ("ax", "bx", "cx", "dx", "si", "di", "bp", "sp")


def _emit(mnemonic, src):
    ops = [Operand(type="reg", reg="eax"), Operand(type="reg", reg=src)]
    insn = Instruction(0, 3, mnemonic, f"eax, {src}", "", operands=ops)
    return " ".join(Lifter().lift_instruction(insn))


@pytest.mark.parametrize("src", _REGS16)
def test_movsx_from_every_16bit_register_sign_extends(src):
    assert "SX16(" in _emit("movsx", src)


@pytest.mark.parametrize("src", _REGS16)
def test_movzx_from_every_16bit_register_zero_extends(src):
    assert "ZX16(" in _emit("movzx", src)
