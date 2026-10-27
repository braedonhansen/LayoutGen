"""Cell height and track layout from the widest devices (SPEC.md section 1).

Values are SKY130 rules (sky130A_mr.drc, sky130A.tech) and hd tech-LEF
conventions; y is in um from the VSS rail center.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

MET1_PITCH = 0.34           # hd tech LEF, horizontal tracks
MET1_OFFSET = 0.17
MET1_HALF_WIDTH = 0.07      # m1.1 0.14
MET1_SPACE = 0.14           # m1.2
RAIL_HALF_WIDTH = 0.24      # hd rails, met1 0.48 wide
ROW_MARGIN = 0.135          # half of difftap.3 (0.27) to the abutting row
NDIFF_TO_TRACK = 0.275      # licon.14 0.19 + half licon 0.085
TRACK_TO_PDIFF = 0.32       # licon.9 + psdm.5a 0.235 + half licon 0.085
CHANNEL_TRACKS = 3
EPS = 1e-9

CHANNEL_NAMES = tuple(f"c{i}" for i in range(CHANNEL_TRACKS))


def _track_y(k: int) -> float:
    return MET1_OFFSET + MET1_PITCH * k


@dataclass(frozen=True)
class Layout:
    height: float
    channel_k: tuple[int, ...]     # met1 grid index of c0, c1, c2
    n_tracks: int                  # n0 .. n{n_tracks-1}, n0 next to the channel
    p_tracks: int                  # p0 .. p{p_tracks-1}, p0 next to the channel

    def has_track(self, name: str) -> bool:
        region, idx = name[0], name[1:]
        if not idx.isdigit():
            return False
        limit = {"c": CHANNEL_TRACKS, "n": self.n_tracks, "p": self.p_tracks}.get(region)
        return limit is not None and int(idx) < limit


def layout(w_n: float, w_p: float) -> Layout:
    """Smallest height on the met1 grid that fits w_n, w_p and the channel."""
    ndiff_top = ROW_MARGIN + w_n
    k0 = math.ceil((ndiff_top + NDIFF_TO_TRACK - MET1_OFFSET) / MET1_PITCH - EPS)
    channel_k = tuple(range(k0, k0 + CHANNEL_TRACKS))
    pdiff_top = _track_y(channel_k[-1]) + TRACK_TO_PDIFF + w_p
    pitches = math.ceil((pdiff_top + ROW_MARGIN) / MET1_PITCH - EPS)
    height = round(pitches * MET1_PITCH, 3)
    clear = RAIL_HALF_WIDTH + MET1_SPACE + MET1_HALF_WIDTH
    usable = [k for k in range(pitches + 1)
              if _track_y(k) >= clear - EPS and _track_y(k) <= height - clear + EPS]
    n_tracks = sum(1 for k in usable if k < channel_k[0])
    p_tracks = sum(1 for k in usable if k > channel_k[-1])
    return Layout(height, channel_k, n_tracks, p_tracks)


def layout_for_netlist(netlist: dict) -> Layout:
    devs = netlist["devices"]
    return layout(max(d["w"] for d in devs if d["type"] == "nmos"),
                  max(d["w"] for d in devs if d["type"] == "pmos"))
