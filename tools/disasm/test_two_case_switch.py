"""
A two-case switch table is data too (resync_jump_tables).

Run: py -3 tools/disasm/test_two_case_switch.py

resync_jump_tables skipped any table shorter than three entries: too short to
tell from code that merely looks like pointers. Blinx 2 has two-case switches,
though -- `jmp [ecx*4 + 0x1CE000]` at 0x1CDFCC with arms 0x1CDFD3/0x1CDFE8 --
and left alone the sweep decoded that table as `call 0x1CFCE8`: a "function"
in the middle of an instruction of the shop's state machine sub_001CFCA0. The
split turned the state machine's own `jp` into an unresolved stub, and
Carlos's dialogue never advanced.

A two-entry table now counts when its dispatch site and both arms lie within
ARM_WINDOW of it, which is the shape of a real switch. A two-entry run
pointing far away stays undecided, as before.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools.disasm.engine import DisasmEngine  # noqa: E402

BASE = 0x00010000


class _Section:
    def __init__(self, va, data):
        self.virtual_addr = va
        self.virtual_size = len(data)
        self.executable = True
        self.name = ".text"
        self.data = data


class _Image:
    def __init__(self, sec):
        self._sec = sec
        self.base_address = sec.virtual_addr
        self.image_size = sec.virtual_size

    def get_section_at_va(self, va):
        s = self._sec
        return s if s.virtual_addr <= va < s.virtual_addr + s.virtual_size else None

    def get_section_data(self, section):
        return section.data

    def read_bytes_at_va(self, va, size):
        s = self._sec
        off = va - s.virtual_addr
        if off < 0 or off >= len(s.data):
            return None
        return s.data[off:off + size]

    def read_u32_at_va(self, va):
        b = self.read_bytes_at_va(va, 4)
        return int.from_bytes(b, "little") if b and len(b) == 4 else None


def _le(v):
    return v.to_bytes(4, "little")


def _swept(data):
    sec = _Section(BASE, data)
    eng = DisasmEngine(_Image(sec))
    eng.linear_sweep(sec)
    eng.resync_jump_tables()
    return eng


def test_local_two_case_switch_is_data():
    tbl = BASE + 0x10
    data = (b"\xff\x24\x8d" + _le(tbl)          # 0x10000 jmp [ecx*4 + tbl]
            + b"\xb8\x01\x00\x00\x00\xc3"       # 0x10007 arm 1: mov eax, 1; ret
            + b"\x33\xc0\xc3"                   # 0x1000D arm 2: xor eax, eax; ret
            + _le(BASE + 0x07) + _le(BASE + 0x0D)   # 0x10010 the table
            + b"\x90\xc3")                      # 0x10018 code after it
    eng = _swept(data)
    assert tbl in eng.jump_tables, "a local two-case table should be resynced"
    inside = [a for a in eng.instructions if tbl <= a < tbl + 8]
    assert not inside, f"instructions decoded over the table: {inside}"
    assert (BASE + 0x18) in eng.instructions, "code after the table"


def test_far_two_entry_run_is_left_alone():
    tbl = BASE + 0x10
    far = BASE + 0x800                          # beyond ARM_WINDOW of the table
    data = bytearray(b"\x90" * 0x900)
    data[0:7] = b"\xff\x24\x8d" + _le(tbl)
    data[0x10:0x18] = _le(far) + _le(far + 2)
    data[0x800:0x804] = b"\x33\xc0\xc3\x90"
    eng = _swept(bytes(data))
    assert tbl not in eng.jump_tables, "a far two-entry run must stay undecided"


if __name__ == "__main__":
    test_local_two_case_switch_is_data()
    test_far_two_entry_run_is_left_alone()
    print("ok")
