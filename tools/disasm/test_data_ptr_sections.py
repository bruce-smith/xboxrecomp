"""_pass_data_ptr_targets trusts only conventional data sections.

A code pointer stored in a data section is a function whose address is only
ever taken at run time -- a vtable slot, a callback array -- and those live in
.rdata/.data. Every other section is either library code or a resource blob
(models, textures, environment data), and a dword inside one of those is not a
function address; it merely happens to fall in the code VA range.

Scanning them is not free. With --text-only, .text is the only disassembled
section, so every other section counted as "data": Blinx 2's MDL*/M*/G*
sections produced 14,549 such false hits, each becoming a callable alias whose
body starts mid-function and so cannot rebuild the frame its prologue never
pushed. The alias then ran with a garbage this/esi and crashed in the
destructor it reached.

    py -3 tools/disasm/test_data_ptr_sections.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools.disasm.functions import FunctionDetector, Function  # noqa: E402

CODE_LO, CODE_HI = 0x1000, 0x1100
# Both targets sit inside the one function and are real instruction
# boundaries, so the only thing that can separate them is where the pointer
# was stored.
TARGET_RDATA = 0x1020
TARGET_CONTENT = 0x1030


def _dwords(values):
    out = bytearray()
    for v in values:
        out += v.to_bytes(4, "little")
    return bytes(out)


class _Insn:
    def __init__(self, addr):
        self.address = addr
        self.end_address = addr + 1
        self.mnemonic = "nop"
        self.is_ret = False
        self.is_jump = False
        self.is_cond_jump = False
        self.is_branch = False
        self.jump_target = None


class _Sec:
    def __init__(self, name, va, size, data=b""):
        self.name = name
        self.virtual_addr = va
        self.virtual_size = size
        self.raw_size = size
        self.executable = True
        self._data = data


class _Image:
    def __init__(self, sections):
        self.sections = sections

    def get_section_data(self, sec):
        return sec._data

    def get_section_at_va(self, addr):
        for s in self.sections:
            if s.virtual_addr <= addr < s.virtual_addr + s.virtual_size:
                return s
        return None


class _Engine:
    def __init__(self, addrs):
        self.instructions = {a: _Insn(a) for a in addrs}


def _run(sections, code_sec, insn_addrs, funcs):
    det = FunctionDetector.__new__(FunctionDetector)
    det.engine = _Engine(insn_addrs)
    det.image = _Image(sections)
    det.functions = {s: Function(start=s, end=e, name=f"sub_{s:08X}")
                     for s, e in funcs}
    det._candidates = {}
    det._alias_entries = {}
    det._pass_data_ptr_targets([code_sec])
    return det._alias_entries


class DataPtrSectionTest(unittest.TestCase):
    def test_only_conventional_data_sections_are_scanned(self):
        code = _Sec(".text", CODE_LO, CODE_HI - CODE_LO)
        rdata = _Sec(".rdata", 0x2000, 0x20, _dwords([TARGET_RDATA]))
        content = _Sec("MDLBEN", 0x3000, 0x20, _dwords([TARGET_CONTENT]))

        aliases = _run(
            [code, rdata, content], code,
            [TARGET_RDATA, TARGET_CONTENT],
            [(CODE_LO, CODE_HI)])

        self.assertIn(TARGET_RDATA, aliases)
        self.assertNotIn(TARGET_CONTENT, aliases)

    def test_data_section_itself_is_not_scanned_as_data(self):
        # .data is passed as the code section here; a pointer stored in it must
        # not turn into an alias of its own bytes.
        code = _Sec(".data", CODE_LO, CODE_HI - CODE_LO)
        aliases = _run([code], code, [TARGET_RDATA], [(CODE_LO, CODE_HI)])
        self.assertEqual(aliases, {})


if __name__ == "__main__":
    unittest.main()
