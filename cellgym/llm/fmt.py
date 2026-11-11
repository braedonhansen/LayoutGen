"""Plain-text formats for fine-tuning a pretrained LM on cell layout (no custom tokens).

Two pipelines, both reading the netlist as text:

  oneshot  prompt = netlist text; target = the whole plan body (col/route/pin lines, as in
           the DSL, without the cell/in/out header), then the tokenizer's EOS.

  step     placement only, one column per step, from a state the environment computes:

               left nmos: A OUT-n1 | B n1-VSS        unplaced devices: gate drain-source
               left pmos: A VDD-p1 | B p1-OUT
               last: sd n=VSS p=VDD | g A | sd n=OUT p=p1      last LAST_K columns
               next:

           target = one column ("g A" or "sd n=n1 p=p2"). The environment lists the legal
           moves (Placer.legal), stops when every device is placed and the last column is
           sd, and hands the columns to the Euler baseline's router (contacts, routes,
           pins). Nothing in the state grows with the cell except the shrinking device
           lists; no column numbers, no plan length.

Augmentation (training only): order-preserving renaming of inputs onto random letters A-Z
and of internal nets onto random distinct numbers, consistently in prompt and target.
"""

from __future__ import annotations

import random
import re
from collections import Counter

from cellgym.baselines.common import outputs_of

LAST_K = 3
LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
NET_MAX = 40
_INPUT_WORD = re.compile(r"(?<![A-Za-z0-9])([A-Z])(?![A-Za-z0-9])")
_NET_WORD = re.compile(r"(?<![A-Za-z0-9])([np])(\d+)(?![A-Za-z0-9.])")


# -- netlist text ---------------------------------------------------------------------

def netlist_text(nl: dict) -> str:
    lines = ["inputs: " + " ".join(nl["inputs"]), "outputs: " + " ".join(outputs_of(nl)), "devices:"]
    for d in nl["devices"]:
        lines.append(f"{d['type']} {d['gate']} {d['drain']} {d['source']} {d['w']:g}")
    return "\n".join(lines) + "\n"


def plan_body(plan: str) -> str:
    return "".join(l + "\n" for l in plan.splitlines() if l.split()[0] not in ("cell", "in", "out"))


def plan_header(nl: dict) -> str:
    return f"cell {nl['id']}\nin {' '.join(nl['inputs'])}\nout {' '.join(outputs_of(nl))}\n"


def oneshot_prompt(nl: dict) -> str:
    return netlist_text(nl) + "plan:\n"


# -- renaming (augmentation) --------------------------------------------------------------

def rename_map(nl: dict, rng: random.Random) -> tuple[dict, dict]:
    ins = list(nl["inputs"])
    letters = dict(zip(ins, sorted(rng.sample(LETTERS, len(ins)))))
    nets = sorted({t for d in nl["devices"] for t in (d["drain"], d["source"]) if _NET_WORD.fullmatch(t)})
    nm = {}
    for fam in "np":
        own = [x for x in nets if x[0] == fam]
        for x, k in zip(own, rng.sample(range(1, NET_MAX + 1), len(own))):
            nm[x] = f"{fam}{k}"
    return letters, nm


def apply_names(text: str, letters: dict, nets: dict, plan_side: bool = False) -> str:
    """Rename inputs and internal nets in text. On plan text, tracks (n0.., p0..) share the
    net spelling: only n=/p= values and route nets are nets there."""
    text = _INPUT_WORD.sub(lambda m: letters.get(m[1], m[1]), text)
    if not plan_side:
        return _NET_WORD.sub(lambda m: nets.get(m[0], m[0]), text)
    out = []
    for line in text.splitlines():
        w = line.split()
        if not w:
            continue
        if w[0] == "col":
            w = [re.sub(r"^([np]=)([np]\d+)", lambda m: m[1] + nets.get(m[2], m[2]), x) for x in w]
        elif w[0] in ("route", "pin"):
            w[1] = nets.get(w[1], w[1])
        out.append(" ".join(w))
    return "".join(l + "\n" for l in out)


def renamed_netlist(nl: dict, letters: dict, nets: dict) -> dict:
    f = lambda x: letters.get(x, nets.get(x, x))
    out = dict(nl)
    out["inputs"] = [f(x) for x in nl["inputs"]]
    out["devices"] = [dict(d, gate=f(d["gate"]), drain=f(d["drain"]), source=f(d["source"])) for d in nl["devices"]]
    return out


# -- step-wise placement ------------------------------------------------------------------

def plan_columns(plan: str) -> list[tuple]:
    """("sd", n, p) / ("g", x) per col line, contacts dropped."""
    cols = []
    for line in plan.splitlines():
        w = line.split()
        if not w or w[0] != "col":
            continue
        n = w[3][2:].rstrip("*")
        p = w[4][2:].rstrip("*")
        cols.append(("g", n) if w[2] == "g" else ("sd", n, p))
    return cols


def col_text(c: tuple) -> str:
    return f"g {c[1]}" if c[0] == "g" else f"sd n={c[1]} p={c[2]}"


def parse_col(s: str) -> tuple | None:
    w = s.strip().split()
    if len(w) == 2 and w[0] == "g":
        return ("g", w[1])
    if len(w) == 3 and w[0] == "sd" and w[1].startswith("n=") and w[2].startswith("p="):
        return ("sd", w[1][2:], w[2][2:])
    return None


class Placer:
    """Environment state for step-wise placement of one netlist."""

    def __init__(self, nl: dict):
        self.nl = nl
        self.left = {"nmos": Counter(), "pmos": Counter()}
        for d in nl["devices"]:
            self.left[d["type"]][(d["gate"], frozenset((d["drain"], d["source"])))] += 1
        self.cols: list[tuple] = []
        self.pending = None        # gate of the last g column, waiting for its sd

    # devices as text
    def _row(self, kind: str) -> str:
        items = []
        for (g, t), n in sorted(self.left[kind].items(), key=lambda kv: (kv[0][0], sorted(kv[0][1]))):
            a, b = sorted(t) if len(t) == 2 else (next(iter(t)),) * 2
            items += [f"{g} {a}-{b}"] * n
        return " | ".join(items) if items else "none"

    def state(self) -> str:
        last = " | ".join(col_text(c) for c in self.cols[-LAST_K:]) or "start"
        return (f"left nmos: {self._row('nmos')}\nleft pmos: {self._row('pmos')}\n"
                f"last: {last}\nnext:")

    def done(self) -> bool:
        return (not +self.left["nmos"] and not +self.left["pmos"] and bool(self.cols)
                and self.cols[-1][0] == "sd")

    def _terms(self, kind):
        return sorted({x for (g, t), n in self.left[kind].items() if n > 0 for x in t})

    def legal(self) -> list[tuple]:
        out = []
        if self.cols and self.cols[-1][0] == "g":
            g = self.cols[-1][1]
            fn, fp = self.cols[-2][1], self.cols[-2][2]
            ns = [x for (gg, t), n in self.left["nmos"].items() if n > 0 and gg == g and fn in t
                  for x in (t - {fn} or {fn})]
            ps = [x for (gg, t), n in self.left["pmos"].items() if n > 0 and gg == g and fp in t
                  for x in (t - {fp} or {fp})]
            return [("sd", a, b) for a in sorted(set(ns)) for b in sorted(set(ps))]
        if self.cols:   # after an sd: a gate continuing both rows, or a break
            fn, fp = self.cols[-1][1], self.cols[-1][2]
            gn = {g for (g, t), n in self.left["nmos"].items() if n > 0 and fn in t}
            gp = {g for (g, t), n in self.left["pmos"].items() if n > 0 and fp in t}
            out = [("g", g) for g in sorted(gn & gp)]
        if +self.left["nmos"] or +self.left["pmos"]:
            out += [("sd", a, b) for a in self._terms("nmos") for b in self._terms("pmos")
                    if not self.cols or ("sd", a, b) != self.cols[-1]]
        return out

    def step(self, c: tuple) -> bool:
        if c not in self.legal():
            return False
        if c[0] == "sd" and self.cols and self.cols[-1][0] == "g":
            g = self.cols[-1][1]
            for kind, prev, new in (("nmos", self.cols[-2][1], c[1]), ("pmos", self.cols[-2][2], c[2])):
                self.left[kind][(g, frozenset((prev, new)))] -= 1
        self.cols.append(c)
        return True


def step_examples(nl: dict, plan: str, rng: random.Random, per_plan: int) -> list[tuple[str, str]]:
    """(state, target) pairs along an Euler plan's columns; per_plan sampled steps (all if <= 0).
    Returns [] if the plan's columns don't replay legally (should not happen)."""
    cols = plan_columns(plan)
    p = Placer(nl)
    pairs = []
    for c in cols:
        pairs.append((p.state(), col_text(c)))
        if not p.step(c):
            return []
    if not p.done():
        return []
    if per_plan > 0 and len(pairs) > per_plan:
        pairs = rng.sample(pairs, per_plan)
    return pairs
