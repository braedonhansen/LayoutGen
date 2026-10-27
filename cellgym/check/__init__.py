"""CellGym M3 fast checker: DRC subset + LVS on rendered rectangles.

    from cellgym.check import check
    res = check(rendered, netlist)        # or check(shapes, netlist, labels=...)

Independent of the renderer's rule tables and compactor (SPEC.md section 8):
it reads only rectangles (layer name or GDS (layer, datatype), x0, y0, x1, y1
in nm) and labels (net, kind, rect).
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field

from .drc import GDS_LAYERS, LAYER_NAMES, RULES, Geometry, Violation, run_drc
from .lvs import run_lvs

# half of each layer's own spacing rule: the abutment margin to the cell edge
_HALF_SPACE = {"poly": RULES["poly.2"][0] // 2, "diff": RULES["difftap.3"][0] // 2,
               "licon": RULES["MR_licon.SP.1"][0] // 2, "li": RULES["li.3"][0] // 2,
               "mcon": RULES["ct.2"][0] // 2, "met1": RULES["m1.2"][0] // 2}


@dataclass
class CheckResult:
    drc_violations: list[Violation] = field(default_factory=list)
    lvs_ok: bool = False
    lvs_errors: list[str] = field(default_factory=list)
    width_um: float = 0.0
    ms: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.drc_violations and self.lvs_ok

    def summary(self) -> str:
        from collections import Counter
        c = Counter(v.rule for v in self.drc_violations)
        drc = ", ".join(f"{k} x{n}" for k, n in sorted(c.items())) or "clean"
        lvs = "ok" if self.lvs_ok else "; ".join(self.lvs_errors)
        return f"DRC {drc} | LVS {lvs} | width {self.width_um:.3f} um | {self.ms:.1f} ms"


def _layer_name(layer) -> str | None:
    if isinstance(layer, str):
        return layer if layer in LAYER_NAMES else None
    return GDS_LAYERS.get(tuple(layer))


def _width(rects: dict, labels) -> int:
    bands = [lab.rect for lab in labels if lab.kind == "met1" and lab.net in ("VDD", "VSS")]
    from .geometry import Region
    lo = hi = None
    for layer, half in _HALF_SPACE.items():
        rs = rects.get(layer, [])
        if layer in ("li", "met1", "mcon"):
            rs = Region.from_rects(rs).rects() if layer != "mcon" else rs
            rs = [r for r in rs if not any(b[1] <= r[1] and r[3] <= b[3] for b in bands)]
        for r in rs:
            a, b = r[0] - half, r[2] + half
            lo = a if lo is None or a < lo else lo
            hi = b if hi is None or b > hi else hi
    if lo is None:
        return 0
    return -(-(hi - lo) // 5) * 5


def check(rendered_or_shapes, netlist: dict, labels=None) -> CheckResult:
    """Fast DRC + LVS of one layout. GC is paused for the call (it only
    allocates short-lived tuples), which keeps the timing stable."""
    enabled = gc.isenabled()
    gc.disable()
    try:
        return _check(rendered_or_shapes, netlist, labels)
    finally:
        if enabled:
            gc.enable()


def _check(rendered_or_shapes, netlist: dict, labels=None) -> CheckResult:
    t0 = time.perf_counter()
    shapes = getattr(rendered_or_shapes, "shapes", rendered_or_shapes)
    if labels is None:
        labels = getattr(rendered_or_shapes, "labels", [])
    rects: dict[str, list] = {k: [] for k in LAYER_NAMES}
    for s in shapes:
        name = _layer_name(s.layer)
        if name is not None:
            rects[name].append((s.x0, s.y0, s.x1, s.y1))
    g = Geometry(rects)
    viol = run_drc(g)
    ok, errs = run_lvs(g, labels, netlist)
    w = _width(rects, labels)
    return CheckResult(viol, ok, errs, w / 1000, (time.perf_counter() - t0) * 1000)


__all__ = ["check", "CheckResult", "Violation", "RULES"]
