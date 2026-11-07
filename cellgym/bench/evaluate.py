"""One plan -> parse, validate, render, fast check (shared by the scorer and the viewer)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cellgym.check import CheckResult, check
from cellgym.dsl.plan import ROWS, Plan, PlanSyntaxError, parse
from cellgym.dsl.validate import validate
from cellgym.render.render import Rendered, RenderError, render

STAGES = ("missing", "parse", "validate", "render", "check", "pass")


@dataclass
class Evaluation:
    cell: str
    stage: str                     # first failing stage, or "pass"
    detail: str = ""
    width_nm: int | None = None
    sites: int | None = None
    breaks: int | None = None
    check_ms: float | None = None
    rendered: Rendered | None = None
    checked: CheckResult | None = None

    @property
    def ok(self) -> bool:
        return self.stage == "pass"


def diffusion_breaks(plan: Plan) -> int:
    """Gate-column neighbours (in column order) that do not share a diffusion
    island in some row where both have a device. Island rule as the renderer:
    an empty slot, a gate with no device, or two adjacent sd columns with
    different nets break a row. Euler: = its extra sd columns; naive: pairs - 1."""
    island: dict[tuple[str, int], int] = {}
    for row in ROWS:
        cur, prev_sd, n = None, None, 0
        for c in plan.columns:
            net = c.nets[row]
            if c.kind == "g":
                if net and cur is not None:
                    island[(row, c.index)] = cur
                    prev_sd = None
                else:
                    cur = None
                continue
            if net is None:
                cur = None
                continue
            if cur is None or (prev_sd is not None and prev_sd != net):
                cur, n = n, n + 1
            prev_sd = net
    gates = [c.index for c in plan.columns if c.kind == "g"]
    breaks = 0
    for a, b in zip(gates, gates[1:]):
        if any((r, a) in island and (r, b) in island and island[(r, a)] != island[(r, b)] for r in ROWS):
            breaks += 1
    return breaks


def evaluate(plan_text: str | None, netlist: dict, keep: bool = False) -> Evaluation:
    """keep=True keeps the Rendered and CheckResult (viewer, ground truth)."""
    cid = netlist["id"]
    if plan_text is None:
        return Evaluation(cid, "missing", "no plan file")
    try:
        plan = parse(plan_text)
    except PlanSyntaxError as e:
        return Evaluation(cid, "parse", str(e))
    rep = validate(plan, netlist)
    if not rep.ok:
        return Evaluation(cid, "validate", str(rep).splitlines()[0])
    breaks = diffusion_breaks(plan)
    try:
        cell = render(plan, netlist)
    except (RenderError, ValueError, KeyError, IndexError) as e:
        return Evaluation(cid, "render", f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}",
                          breaks=breaks)
    res = check(cell, netlist)
    ev = Evaluation(cid, "pass" if res.ok else "check", "" if res.ok else res.summary(),
                    cell.width_nm, cell.sites, breaks, res.ms)
    if keep:
        ev.rendered, ev.checked = cell, res
    return ev


def read_plan(plans_dir: Path, cid: str) -> str | None:
    f = Path(plans_dir) / f"{cid}.plan"
    return f.read_text() if f.exists() else None
