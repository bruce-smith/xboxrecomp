"""_pass_pointer_after_end: a code pointer to a function that starts right
after another one ends.

Blinx 2 stopped five times as the game progressed, each time on an
"[ICALL] Failed to resolve" for a function reached only through a pointer,
each seeded by hand once someone had played that far:

    0x000EF540  mission summary load callback   mov ecx, offset
    0x0016A870  pause menu                      push offset
    0x001AE1E0  post-results handler            .data table
    0x001B04E0  post-results handler            .data table
    0x000AEC20  Tom-Tom stage                   mov [esi+6Ch], offset,
                                                right after a jump table

All five sit in gaps and all five failed probes_as_returning_body: its
64-instruction budget is shorter than their way to a ret (114 to 208
instructions), or they jump out to a helper first. These tests rebuild each
shape, and the false positives the pass has to refuse: an "and" mask that
falls in code, an unaligned address, and a nop-aligned loop head inside a
body that a jne branches back to (0x002A9000).

    py -3 tools/disasm/test_pointer_after_end.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools.disasm.functions import FunctionDetector  # noqa: E402

TEXT_LO, TEXT_HI = 0x00012000, 0x002DC000


class _Insn:
    def __init__(self, addr, size, mnemonic="mov", imm_ref=None, jump_target=None):
        self.address = addr
        self.size = size
        self.end_address = addr + size
        self.mnemonic = mnemonic
        self.imm_ref = imm_ref
        self.jump_target = jump_target
        self.is_ret = mnemonic == "ret"
        self.is_jump = mnemonic == "jmp"
        self.is_cond_jump = mnemonic.startswith("j") and mnemonic != "jmp"


class _Sec:
    def __init__(self, name, va, size, data=b""):
        self.name = name
        self.virtual_addr = va
        self.virtual_size = size
        self.executable = name == ".text"
        self._data = data


class _Func:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class _Image:
    def __init__(self, sections, dwords):
        self.sections = sections
        self._dwords = dwords

    def get_section_data(self, sec):
        return sec._data

    def read_u32_at_va(self, va):
        return self._dwords.get(va)


class _Engine:
    def __init__(self, insns, returning=(), body=()):
        self.instructions = {i.address: i for i in insns}
        self._returning = set(returning)
        self._body = set(body)

    def instruction_covering(self, addr):
        for i in self.instructions.values():
            if i.address < addr < i.end_address:
                return i
        return None

    def decode_at(self, addr):
        return 0

    def probes_as_returning_body(self, addr, max_insns=64):
        return addr in self._returning

    def probes_as_function_body(self, addr, max_insns=8192):
        return addr in self._body or addr in self._returning

    def get_instructions_in_range(self, start, end):
        return [i for i in self.instructions.values() if start <= i.address < end]


def _run(insns, functions, data=b"", dwords=None, returning=(), body=()):
    det = FunctionDetector.__new__(FunctionDetector)
    det.engine = _Engine(insns, returning, body)
    text = _Sec(".text", TEXT_LO, TEXT_HI - TEXT_LO)
    det.image = _Image([text, _Sec(".data", 0x00440000, len(data), data)], dwords or {})
    det.functions = {f.start: f for f in functions}
    det._candidates = {}
    det.added = []
    det._add_candidate = lambda addr, conf, why: det.added.append(addr)
    det._pass_pointer_after_end([text])
    return det.added


def _le(*values):
    return b"".join(v.to_bytes(4, "little") for v in values)


class PointerAfterEndTest(unittest.TestCase):
    def test_mov_ecx_offset_after_a_ret_in_a_gap(self):
        # 0x000EF540 follows sub_000EF4E0's ret; mov ecx, offset installs it.
        t = 0x000EF540
        insns = [_Insn(0x000EF53F, 1, "ret"), _Insn(t, 4, "mov"),
                 _Insn(0x000EFA48, 5, "mov", imm_ref=t)]
        funcs = [_Func(0x000EF4E0, t), _Func(0x000EFA00, 0x000EFB00)]
        self.assertEqual(_run(insns, funcs, body={t}), [t])

    def test_push_offset(self):
        t = 0x0016A870                      # the pause menu, pushed as an argument
        insns = [_Insn(t - 1, 1, "int3"), _Insn(t, 4, "sub"),
                 _Insn(0x0016AA3D, 5, "push", imm_ref=t)]
        funcs = [_Func(0x0016A800, t - 1), _Func(0x0016AA00, 0x0016AB00)]
        self.assertEqual(_run(insns, funcs, body={t}), [t])

    def test_data_table_entries(self):
        a, b = 0x001AE1E0, 0x001B04E0      # post-results handlers in .data
        insns = [_Insn(a - 1, 1, "ret"), _Insn(a, 1, "push"),
                 _Insn(b - 3, 3, "ret"), _Insn(b, 1, "push")]
        funcs = [_Func(0x001AE100, a), _Func(0x001B0400, b)]
        added = _run(insns, funcs, data=_le(0x12345678, a, b), body={a, b})
        self.assertEqual(sorted(added), [a, b])

    def test_after_a_jump_table(self):
        # 0x000AEC20: the dword before it is the last entry of the preceding
        # function's switch table, so no instruction ends at it.
        t = 0x000AEC20
        insns = [_Insn(t, 4, "mov"), _Insn(0x000ADF2E, 7, "mov", imm_ref=t)]
        funcs = [_Func(0x000AEB00, 0x000AEC10), _Func(0x000ADF00, 0x000ADF80)]
        self.assertEqual(_run(insns, funcs, dwords={t - 4: 0x000AEBF0}, body={t}), [t])

    def test_an_and_mask_is_not_a_pointer(self):
        t = 0x0001FFE0                      # "and ecx, 1FFE0h" at 0x002D8E7D
        insns = [_Insn(t - 1, 1, "ret"), _Insn(t, 5, "add"),
                 _Insn(0x002D8E7D, 6, "and", imm_ref=t)]
        self.assertEqual(_run(insns, [_Func(0x0001FF00, t)], body={t}), [])

    def test_unaligned_target(self):
        t = 0x000EF544
        insns = [_Insn(t - 1, 1, "ret"), _Insn(t, 4, "mov"),
                 _Insn(0x000EFA48, 5, "mov", imm_ref=t)]
        self.assertEqual(_run(insns, [_Func(0x000EF4E0, t)], body={t}), [])

    def test_loop_head_inside_a_body_is_not_split(self):
        # 0x002A9000: a strcpy loop, nop-aligned, "jne 0x2A9000" right below.
        t = 0x002A9000
        insns = [_Insn(t - 1, 1, "nop"), _Insn(t, 3, "mov"),
                 _Insn(t + 6, 2, "jne", jump_target=t),
                 _Insn(0x0029F000, 5, "mov", imm_ref=t)]
        funcs = [_Func(0x002A8F00, 0x002A9100)]
        self.assertEqual(_run(insns, funcs, returning={t}), [])

    def test_inside_a_body_after_a_ret_is_split(self):
        # A function _find_function_end walked into: after a ret, nothing in
        # the body branches to it, and it returns on its own.
        t = 0x000C6390
        insns = [_Insn(t - 1, 1, "int3"), _Insn(t, 3, "sub"),
                 _Insn(0x000C5CB0, 7, "mov", imm_ref=t)]
        funcs = [_Func(0x000C6300, 0x000C6500), _Func(0x000C5C00, 0x000C5D00)]
        self.assertEqual(_run(insns, funcs, returning={t}), [t])

    def test_inside_a_body_after_a_jmp_is_not_split(self):
        t = 0x000C6390
        insns = [_Insn(t - 2, 2, "jmp", jump_target=t + 0x40), _Insn(t, 3, "sub"),
                 _Insn(0x000C5CB0, 7, "mov", imm_ref=t)]
        funcs = [_Func(0x000C6300, 0x000C6500), _Func(0x000C5C00, 0x000C5D00)]
        self.assertEqual(_run(insns, funcs, returning={t}), [])


if __name__ == "__main__":
    unittest.main()
