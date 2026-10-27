"""Plan data model, line-format parser and writer (SPEC.md section 9)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

ROWS = ("nmos", "pmos")
LAYERS = ("li1", "met1")
_TRACK = re.compile(r"[cnp]\d+$")
_ENDPOINT = re.compile(r"(nmos|pmos|gate):(\d+)([np]?)$")
_NET = re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")


class PlanSyntaxError(ValueError):
    def __init__(self, line: int, message: str):
        super().__init__(f"line {line}: {message}")
        self.line = line
        self.message = message


@dataclass(frozen=True)
class Endpoint:
    kind: str            # "nmos" | "pmos" | "gate"
    col: int
    side: str = ""       # "n" | "p" for a split gate column, else ""

    def __str__(self) -> str:
        return f"{self.kind}:{self.col}{self.side}"


@dataclass
class Column:
    index: int
    kind: str                       # "sd" | "g"
    nets: dict[str, str | None]     # row -> net (sd) or input (g); None = "-"
    contact: dict[str, bool]        # row -> contacted (sd only)
    stage: str | None = None
    line: int = 0


@dataclass
class Route:
    net: str
    layer: str
    track: str
    endpoints: list[Endpoint]
    line: int = 0


@dataclass
class Pin:
    net: str
    landing: Endpoint
    track: str | None = None
    line: int = 0


@dataclass
class Plan:
    name: str = ""
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    columns: list[Column] = field(default_factory=list)
    routes: list[Route] = field(default_factory=list)
    pins: list[Pin] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "cell": self.name, "inputs": self.inputs, "outputs": self.outputs,
            "columns": [{"col": c.index, "kind": c.kind,
                         **{r: c.nets[r] for r in ROWS},
                         **({"contact": [r for r in ROWS if c.contact[r]]} if c.kind == "sd" else {}),
                         **({"stage": c.stage} if c.stage else {})} for c in self.columns],
            "routes": [{"net": r.net, "layer": r.layer, "track": r.track,
                        "endpoints": [str(e) for e in r.endpoints]} for r in self.routes],
            "pins": [{"net": p.net, "landing": str(p.landing),
                      **({"track": p.track} if p.track else {})} for p in self.pins],
        }


def _endpoint(tok: str, line: int) -> Endpoint:
    m = _ENDPOINT.match(tok)
    if not m:
        raise PlanSyntaxError(line, f"bad endpoint '{tok}' (want nmos:<col>, pmos:<col> or gate:<col>[n|p])")
    kind, col, side = m.groups()
    if side and kind != "gate":
        raise PlanSyntaxError(line, f"bad endpoint '{tok}': only gate endpoints take n/p")
    return Endpoint(kind, int(col), side)


def _track(tok: str, line: int) -> str:
    if not _TRACK.match(tok):
        raise PlanSyntaxError(line, f"bad track '{tok}' (want c<i>, n<i> or p<i>)")
    return tok


def _net(tok: str, line: int) -> str:
    if not _NET.match(tok):
        raise PlanSyntaxError(line, f"bad net name '{tok}'")
    return tok


def _column(toks: list[str], line: int, expected: int) -> Column:
    if len(toks) < 5:
        raise PlanSyntaxError(line, "col needs: col <i> sd|g n=<..> p=<..> [st=<..>]")
    if not toks[1].isdigit() or int(toks[1]) != expected:
        raise PlanSyntaxError(line, f"column index '{toks[1]}', expected {expected}")
    kind = toks[2]
    if kind not in ("sd", "g"):
        raise PlanSyntaxError(line, f"column kind '{kind}', expected sd or g")
    fields = {}
    for tok in toks[3:]:
        key, eq, val = tok.partition("=")
        if not eq or key not in ("n", "p", "st") or key in fields or not val:
            raise PlanSyntaxError(line, f"bad column field '{tok}'")
        fields[key] = val
    if "n" not in fields or "p" not in fields:
        raise PlanSyntaxError(line, "column needs both n= and p=")
    nets, contact = {}, {}
    for row, key in zip(ROWS, ("n", "p")):
        val = fields[key]
        starred = val.endswith("*")
        if starred and kind == "g":
            raise PlanSyntaxError(line, f"'*' on a gate column ({key}={val})")
        val = val.rstrip("*") if starred else val
        if val == "-":
            if starred:
                raise PlanSyntaxError(line, f"contact on empty slot ({key}={fields[key]})")
            nets[row] = None
        else:
            nets[row] = _net(val, line)
        contact[row] = starred
    return Column(expected, kind, nets, contact, fields.get("st"), line)


def parse(text: str) -> Plan:
    """Parse the line format. Raises PlanSyntaxError on the first bad line."""
    plan = Plan()
    seen_header = set()
    for line, raw in enumerate(text.splitlines(), start=1):
        toks = raw.split("#", 1)[0].split()
        if not toks:
            continue
        cmd = toks[0]
        if cmd in ("cell", "in", "out"):
            if cmd in seen_header:
                raise PlanSyntaxError(line, f"duplicate '{cmd}' line")
            seen_header.add(cmd)
            if cmd == "cell":
                if len(toks) != 2:
                    raise PlanSyntaxError(line, "cell needs one name")
                plan.name = toks[1]
            else:
                names = [_net(t, line) for t in toks[1:]]
                if not names:
                    raise PlanSyntaxError(line, f"'{cmd}' needs at least one net")
                setattr(plan, "inputs" if cmd == "in" else "outputs", names)
        elif cmd == "col":
            plan.columns.append(_column(toks, line, len(plan.columns)))
        elif cmd == "route":
            if len(toks) < 6:
                raise PlanSyntaxError(line, "route needs: route <net> <layer> <track> <endpoint> <endpoint>...")
            if toks[2] not in LAYERS:
                raise PlanSyntaxError(line, f"bad layer '{toks[2]}' (want li1 or met1)")
            plan.routes.append(Route(_net(toks[1], line), toks[2], _track(toks[3], line),
                                     [_endpoint(t, line) for t in toks[4:]], line))
        elif cmd == "pin":
            if len(toks) not in (3, 4):
                raise PlanSyntaxError(line, "pin needs: pin <net> <landing> [track]")
            plan.pins.append(Pin(_net(toks[1], line), _endpoint(toks[2], line),
                                 _track(toks[3], line) if len(toks) == 4 else None, line))
        else:
            raise PlanSyntaxError(line, f"unknown statement '{cmd}'")
    for cmd in ("cell", "in", "out"):
        if cmd not in seen_header:
            raise PlanSyntaxError(0, f"missing '{cmd}' line")
    return plan


def dumps(plan: Plan) -> str:
    """Write a plan back to the line format (single-space separated)."""
    out = [f"cell {plan.name}", "in " + " ".join(plan.inputs), "out " + " ".join(plan.outputs)]
    for c in plan.columns:
        parts = [f"col {c.index} {c.kind}"]
        for row, key in zip(ROWS, ("n", "p")):
            net = c.nets[row] or "-"
            parts.append(f"{key}={net}{'*' if c.contact[row] else ''}")
        if c.stage:
            parts.append(f"st={c.stage}")
        out.append(" ".join(parts))
    for r in plan.routes:
        out.append(f"route {r.net} {r.layer} {r.track} " + " ".join(map(str, r.endpoints)))
    for p in plan.pins:
        out.append(f"pin {p.net} {p.landing}" + (f" {p.track}" if p.track else ""))
    return "\n".join(out) + "\n"
