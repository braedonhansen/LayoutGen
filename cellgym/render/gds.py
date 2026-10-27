"""Minimal GDSII stream writer: one cell of rectangles and text labels.

Units: 1 database unit = 1 nm (user unit 1 um). Coordinates are integers in nm
and are snapped to the 5 nm manufacturing grid on write.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

GRID_NM = 5

# record type, data type
_HEADER, _BGNLIB, _LIBNAME, _UNITS = 0x0002, 0x0102, 0x0206, 0x0305
_ENDLIB, _BGNSTR, _STRNAME, _ENDSTR = 0x0400, 0x0502, 0x0606, 0x0700
_BOUNDARY, _TEXT, _LAYER, _DATATYPE = 0x0800, 0x0C00, 0x0D02, 0x0E02
_XY, _ENDEL, _TEXTTYPE, _STRING = 0x1003, 0x1100, 0x1602, 0x1906


def snap(v: int | float) -> int:
    """Nearest multiple of the 5 nm grid."""
    return int(round(v / GRID_NM)) * GRID_NM


@dataclass
class GdsCell:
    name: str
    rects: list[tuple[tuple[int, int], int, int, int, int]] = field(default_factory=list)
    texts: list[tuple[tuple[int, int], int, int, str]] = field(default_factory=list)

    def rect(self, layer: tuple[int, int], x0: int, y0: int, x1: int, y1: int) -> None:
        x0, x1 = sorted((snap(x0), snap(x1)))
        y0, y1 = sorted((snap(y0), snap(y1)))
        if x1 > x0 and y1 > y0:
            self.rects.append((layer, x0, y0, x1, y1))

    def text(self, layer: tuple[int, int], x: int, y: int, s: str) -> None:
        self.texts.append((layer, snap(x), snap(y), s))


def _real8(v: float) -> bytes:
    """IBM excess-64 base-16 double."""
    if v == 0:
        return b"\0" * 8
    sign = 0x80 if v < 0 else 0
    v = abs(v)
    exp = 64
    while v >= 1:
        v /= 16
        exp += 1
    while v < 1 / 16:
        v *= 16
        exp -= 1
    mant = int(round(v * (1 << 56)))
    if mant >= 1 << 56:
        mant >>= 4
        exp += 1
    return bytes([sign | exp]) + mant.to_bytes(7, "big")


def _rec(rtype: int, data: bytes = b"") -> bytes:
    return struct.pack(">HH", 4 + len(data), rtype) + data


def _i2(*vals: int) -> bytes:
    return struct.pack(f">{len(vals)}h", *vals)


def _i4(*vals: int) -> bytes:
    return struct.pack(f">{len(vals)}i", *vals)


def _ascii(s: str) -> bytes:
    b = s.encode("ascii")
    return b + b"\0" if len(b) % 2 else b


def dumps(cell: GdsCell, lib: str = "cellgym") -> bytes:
    """Serialize one cell as a GDSII library (fixed timestamps: deterministic)."""
    date = _i2(2026, 1, 1, 0, 0, 0) * 2
    out = [_rec(_HEADER, _i2(600)), _rec(_BGNLIB, date), _rec(_LIBNAME, _ascii(lib)),
           _rec(_UNITS, _real8(1e-3) + _real8(1e-9)),
           _rec(_BGNSTR, date), _rec(_STRNAME, _ascii(cell.name))]
    for (layer, dt), x0, y0, x1, y1 in cell.rects:
        out += [_rec(_BOUNDARY), _rec(_LAYER, _i2(layer)), _rec(_DATATYPE, _i2(dt)),
                _rec(_XY, _i4(x0, y0, x1, y0, x1, y1, x0, y1, x0, y0)), _rec(_ENDEL)]
    for (layer, tt), x, y, s in cell.texts:
        out += [_rec(_TEXT), _rec(_LAYER, _i2(layer)), _rec(_TEXTTYPE, _i2(tt)),
                _rec(_XY, _i4(x, y)), _rec(_STRING, _ascii(s)), _rec(_ENDEL)]
    out += [_rec(_ENDSTR), _rec(_ENDLIB)]
    return b"".join(out)
