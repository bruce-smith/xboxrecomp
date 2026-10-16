"""fprem/fprem1 must leave the status word saying "complete". The CRT's fmod
loops `fprem; fnstsw ax; sahf; jp loop` until C2 clears, and the lifted jp
reads g_fp_cmp. fprem used to update neither, so after an unordered compare
(g_fp_cmp == 2, a NaN somewhere) the loop never ended: Blinx 2 hung entering
stage 4-3 in sub_002E0144."""
import unittest

from .disasm import BasicBlock, Instruction
from .lifter import Lifter, lift_basic_block, _make_condition


def _lift(mnemonic, raw):
    ins = Instruction(0, 2, mnemonic, "", raw)
    ins.operands = []
    return "\n".join(lift_basic_block(Lifter(), BasicBlock(start=0, instructions=[ins]))[0])


class FpremTest(unittest.TestCase):
    def test_fprem_reduces_and_says_complete(self):
        for m, raw, fn in (("fprem", "d9f8", "fmod"), ("fprem1", "d9f5", "remainder")):
            out = _lift(m, raw)
            self.assertIn("fp_top() = %s(fp_top(), fp_st1());" % fn, out)
            # C2 (0x0400) and the unordered C3/C0 cleared...
            self.assertIn("g_fp_cc = (uint16_t)(g_fp_cc & ~0x4500u);", out)
            # ...and the compare state no longer unordered, which is what the
            # loop's jp after fnstsw/sahf actually tests.
            self.assertIn("g_fp_cmp = 1;", out)
            self.assertIn("g_fp_cmp == 2", _make_condition("jp", "sahf", [])[0])


if __name__ == "__main__":
    unittest.main()
