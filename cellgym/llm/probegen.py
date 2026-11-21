"""Fast generators for probe cells far outside training (used by cellgym.llm.probe).

Single stage (sample_networks_fast): a random series-parallel shape is built directly, top
down, inside the two stack budgets, then labeled so every input appears, and kept only if the
exact irredundancy test passes (forcing any one device open or shorted changes the function;
this implies the function depends on every input). No shape enumeration and no canonical-form
search over input permutations, which is what made cellgym.synth.devset.sample_networks stall.

A series-parallel network with PDN depth a and PUN depth b has at most a*b devices per side
(by induction: a series node adds PDN depths and takes the max PUN depth). So 20 or 24 devices
per side cannot exist with both stacks <= 4; min_stack(per_side) is the smallest symmetric limit.

Multi stage (sample_multistage): 2-3 single-stage cells wired into a DAG. Stage inputs are
primary inputs or earlier stages' outputs, every sink is a cell output, every primary input and
stage net is used, no inverter drives an inverter, and every device is necessary at cell level
(forcing it open or shorted changes some output).

check_netlist is a switch-level simulation of the transistor netlist alone (any number of
stages), compared with the truth tables computed from the trees.
"""

from __future__ import annotations

import hashlib
import random

from cellgym.netlists.enumerate import (INPUT_NAMES, Network, _label, _strip, depth, make,
                                        num_leaves, relabel, signatures, tree_str, truth)
from cellgym.netlists.transistors import Sizing, build_cell

SUPPLY = ("VDD", "VSS", "VPB", "VNB")
EXHAUSTIVE_INPUTS = 12       # check_netlist simulates every row up to this many inputs
SAMPLED_ROWS = 2048          # ... and this many random rows (plus all-0, all-1) beyond it


# -- shapes ---------------------------------------------------------------------------

def min_stack(per_side: int) -> int:
    """Smallest s such that a network with per_side devices and both stacks <= s exists."""
    s = 1
    while s * s < per_side:
        s += 1
    return s


def _ok(n: int, prim: int, sec: int) -> bool:
    """A node with n leaves exists whose children share `prim` additively (stack on its own
    side) and are each bounded by `sec` on the other side."""
    return n <= prim * sec and (n == 1 or prim >= 2)


def _shape(n: int, prim: int, sec: int, kind: str, rng: random.Random):
    if n == 1:
        return ("L", 0)
    other = "P" if kind == "S" else "S"
    kids, rem_n, rem_d = [], n, prim
    while rem_n:
        opts = []
        for ni in range(1, rem_n + 1):
            if ni == n:
                continue                      # at least two children
            for di in range(1, rem_d + 1):
                if not _ok(ni, sec, di):
                    continue
                rn, rd = rem_n - ni, rem_d - di
                if rn == 0 or (rd >= 1 and rn <= rd * sec):
                    opts.append((ni, di))
        ni, di = rng.choice(opts)
        kids.append(_shape(ni, sec, di, other, rng))
        rem_n -= ni
        rem_d -= di
    return make(kind, kids)


def random_shape(n: int, max_pdn: int, max_pun: int, rng: random.Random):
    """Unlabeled SP tree with exactly n leaves, PDN depth <= max_pdn, PUN depth <= max_pun."""
    if n == 1:
        return ("L", 0)
    kinds = [k for k, (p, s) in (("S", (max_pdn, max_pun)), ("P", (max_pun, max_pdn))) if _ok(n, p, s)]
    if not kinds:
        raise ValueError(f"no network with {n} devices per side and stacks <= {max_pdn}/{max_pun} "
                         f"(at most {max_pdn * max_pun})")
    kind = rng.choice(kinds)
    return _shape(n, *((max_pdn, max_pun) if kind == "S" else (max_pun, max_pdn)), kind, rng)


# -- truth tables ---------------------------------------------------------------------

def fast_masks(k: int) -> tuple[list[int], int]:
    """Same values as enumerate.var_masks, without the per-row loop (usable at 16+ inputs)."""
    full = (1 << (1 << k)) - 1
    return [(full // ((1 << (1 << i)) + 1)) << (1 << i) for i in range(k)], full


def irredundant(t, masks, full) -> bool:
    base = truth(t, masks, full)
    return all(truth(t, masks, full, (i, v)) != base for i in range(num_leaves(t)) for v in (0, 1))


def _first_use_order(t, k: int):
    """Rename inputs in order of first appearance (repeated until stable): a fixed spelling."""
    for _ in range(8):
        seen: list[int] = []

        def walk(x):
            if x[0] == "L":
                if x[1] not in seen:
                    seen.append(x[1])
            else:
                for c in x[1]:
                    walk(c)

        walk(t)
        t2 = relabel(t, {old: new for new, old in enumerate(seen)})
        if t2 == t:
            break
        t = t2
    return t


def _iso_key(t, k: int):
    """Equal for any two networks that differ only by renaming inputs (and for some that differ
    more: the sampler then drops a distinct cell, never keeps a duplicate)."""
    return _strip(t), tuple(sorted(signatures(t, k)))


# -- single stage ---------------------------------------------------------------------

def sample_networks_fast(n_inputs: int, n_devices: int, max_stack: int, count: int, seed: int = 0,
                         tries: int = 200_000, labelings: int = 20) -> list[Network]:
    """Random irredundant networks, distinct up to input renaming, with exactly n_inputs inputs
    and n_devices devices per side, both stacks <= max_stack."""
    if not 1 <= n_inputs <= n_devices:
        raise ValueError("need 1 <= inputs <= devices per side")
    if n_inputs > len(INPUT_NAMES):
        raise ValueError("more inputs than input letters")
    rng = random.Random(seed)
    masks, full = fast_masks(n_inputs)
    seen: dict = {}
    for _ in range(tries):
        if len(seen) >= count:
            break
        shp = random_shape(n_devices, max_stack, max_stack, rng)
        for _ in range(labelings):
            lab = list(range(n_inputs)) + [rng.randrange(n_inputs) for _ in range(n_devices - n_inputs)]
            rng.shuffle(lab)
            t = _label(shp, iter(lab))
            if irredundant(t, masks, full):
                key = _iso_key(t, n_inputs)
                if key not in seen:
                    seen[key] = _first_use_order(t, n_inputs)
                break
    out = sorted(seen.values())
    for t in out:
        assert num_leaves(t) == n_devices and max(depth(t), depth(t, dual=True)) <= max_stack
    return [Network(t, n_inputs) for t in out]


# -- multi stage ----------------------------------------------------------------------

def _random_stage(rng: random.Random, max_inputs: int, max_stack: int, max_devices: int) -> Network:
    k = rng.choice([i for i in (1, 2, 2, 2, 3, 3, 3, 4, 4, 5) if i <= max_inputs])
    n = min(max_devices, k + rng.choice((0, 0, 0, 1, 2))) if k > 1 else 1
    while True:
        nets = sample_networks_fast(k, n, max_stack, 1, seed=rng.randrange(1 << 30), tries=40)
        if nets:
            return nets[0]
        n = k                                   # read-once always labels irredundantly


def _stage_fns(stages, pins, pmasks, full, force=None):
    """Truth table (over the primary inputs) of every stage output. pins[j][v] is a primary
    input index (int) or ("s", i) for stage i's output. force = (stage, leaf, 0|1)."""
    fn = []
    for j, net in enumerate(stages):
        m = [pmasks[p] if isinstance(p, int) else fn[p[1]] for p in pins[j]]
        f = force[1:] if force is not None and force[0] == j else None
        fn.append(~truth(net.tree, m, full, f) & full)
    return fn


def _compose(stages: list[Network], rng: random.Random, max_primary: int):
    """Random wiring of the stages (in topological order). Returns (pins, n_primary) or None."""
    S = len(stages)
    drivers = [[]]
    for j in range(1, S):
        m = rng.randint(1, min(j, stages[j].n_inputs))
        drivers.append(sorted(rng.sample(range(j), m)))
    q = [stages[j].n_inputs - len(drivers[j]) for j in range(S)]
    if max(q) > max_primary or sum(q) == 0:
        return None
    for j in range(1, S):                       # no inverter driven by an inverter
        if stages[j].n_devices == 1 and stages[drivers[j][0]].n_devices == 1:
            return None
    P = rng.randint(max(max(q), 1), min(max_primary, sum(q)))
    pins = []
    for j in range(S):
        prim = rng.sample(range(P), q[j])
        slots = [("s", i) for i in drivers[j]] + prim
        rng.shuffle(slots)
        pins.append(slots)
    order: list[int] = []                       # primary inputs named in order of first use
    for row in pins:
        for p in row:
            if isinstance(p, int) and p not in order:
                order.append(p)
    if len(order) != P:
        return None                             # an unused primary input
    ren = {old: new for new, old in enumerate(order)}
    return [[ren[p] if isinstance(p, int) else p for p in row] for row in pins], P


def sinks_of(pins) -> list[int]:
    used = {p[1] for row in pins for p in row if not isinstance(p, int)}
    return [j for j in range(len(pins)) if j not in used]


def multistage_netlist(stages: list[Network], pins, n_primary: int, sizing: Sizing = Sizing(),
                       stage_prefix: str = "x") -> dict:
    """Transistor netlist of the composed cell, in the out/cells/*.json schema. Outputs are OUT
    (one sink) or OUT1.. ; other stage outputs are <stage_prefix>1.. ; internal nets n*/p* are
    numbered across stages. All NMOS devices come first (stage order), then all PMOS."""
    S = len(stages)
    snk = sinks_of(pins)
    inputs = list(INPUT_NAMES[:n_primary])
    out_net, xi = {}, 0
    for j in range(S):
        if j in snk:
            out_net[j] = "OUT" if len(snk) == 1 else f"OUT{snk.index(j) + 1}"
        else:
            xi += 1
            out_net[j] = f"{stage_prefix}{xi}"
    pmasks, full = fast_masks(n_primary)
    fn = _stage_fns(stages, pins, pmasks, full)
    count = {"n": 0, "p": 0}
    ndev, pdev = [], []
    for j, net in enumerate(stages):
        cell = build_cell(net, sizing)
        m = {INPUT_NAMES[v]: (inputs[p] if isinstance(p, int) else out_net[p[1]]) for v, p in enumerate(pins[j])}
        m["OUT"] = out_net[j]
        for x in cell.nets:
            if x not in m and x not in SUPPLY and x[0] in "np":
                count[x[0]] += 1
                m[x] = f"{x[0]}{count[x[0]]}"
        for d in cell.devices:
            dd = d.as_dict()
            for key in ("gate", "drain", "source"):
                dd[key] = m.get(dd[key], dd[key])
            (ndev if d.type == "nmos" else pdev).append(dd)
    for i, d in enumerate(ndev):
        d["name"] = f"MN{i + 1}"
    for i, d in enumerate(pdev):
        d["name"] = f"MP{i + 1}"
    outs = [out_net[j] for j in snk]
    desc = [{"out": out_net[j], "tree": tree_str(net.tree),
             "pins": [inputs[p] if isinstance(p, int) else out_net[p[1]] for p in pins[j]]}
            for j, net in enumerate(stages)]
    h = hashlib.sha1(repr([(s["tree"], s["pins"], s["out"]) for s in desc]).encode()).hexdigest()[:8]
    internal = [f"{stage_prefix}{i + 1}" for i in range(xi)] + \
               [f"{f}{i + 1}" for f in "np" for i in range(count[f])]
    rows = 1 << n_primary
    return {
        "id": f"cg_s{S}o{len(outs)}_{n_primary}i{len(ndev)}d_{h}",
        "inputs": inputs,
        "ports": inputs + outs + list(SUPPLY[:2]) + list(SUPPLY[2:]),
        "function": {out_net[j]: format(fn[j], "x").zfill(max(1, rows // 4)) for j in snk},
        "function_note": "hex truth table per output; bit r = row r, input A = LSB of r",
        "stages": desc,
        "n_stages": S,
        "pdn_depth": max(depth(n.tree) for n in stages),
        "pun_depth": max(depth(n.tree, dual=True) for n in stages),
        "n_devices": len(ndev) + len(pdev),
        "max_w_n": max(d["w"] for d in ndev),
        "max_w_p": max(d["w"] for d in pdev),
        "fits": True,
        "devices": ndev + pdev,
        "nets": inputs + outs + ["VDD", "VSS", "VPB", "VNB"] + internal,
    }


def sample_multistage(n_stages: int, count: int, per_side: tuple[int, int], outputs: tuple[int, int] = (1, 1),
                      seed: int = 0, max_primary: int = 5, stage_inputs: int = 5, stage_stack: int = 4,
                      stage_devices: int = 7, sizing: Sizing = Sizing(), tries: int = 200_000) -> list[dict]:
    """Random multi-stage netlists (see module docstring); every stage is itself a cell with
    <= stage_inputs inputs, stacks <= stage_stack and <= stage_devices devices per side."""
    rng = random.Random(seed)
    out: dict[str, dict] = {}
    for _ in range(tries):
        if len(out) >= count:
            break
        stages = [_random_stage(rng, stage_inputs, stage_stack, stage_devices) for _ in range(n_stages)]
        total = sum(s.n_devices for s in stages)
        if not per_side[0] <= total <= per_side[1]:
            continue
        wired = _compose(stages, rng, max_primary)
        if wired is None:
            continue
        pins, P = wired
        snk = sinks_of(pins)
        if not outputs[0] <= len(snk) <= outputs[1]:
            continue
        pmasks, full = fast_masks(P)
        fn = _stage_fns(stages, pins, pmasks, full)
        if any(f in (0, full) for f in fn) or len(set(fn)) < len(fn):
            continue                                        # a constant or duplicated stage
        base = [fn[j] for j in snk]
        if any([_stage_fns(stages, pins, pmasks, full, (j, i, v))[s] for s in snk] == base
               for j, net in enumerate(stages) for i in range(net.n_devices) for v in (0, 1)):
            continue                                        # a device no output needs
        nl = multistage_netlist(stages, pins, P, sizing)
        out.setdefault(nl["id"], nl)
    return [out[k] for k in sorted(out)]


# -- switch-level check ---------------------------------------------------------------

def simulate(nl: dict, vec: dict[str, bool]) -> dict[str, bool]:
    """Values of every net that is fully driven under the input vector. Stages are found from
    the devices alone: groups connected through channels, evaluated once their gates are known."""
    val = dict(vec, VDD=True, VSS=False)
    group: dict[str, str] = {}

    def find(p, x):
        p.setdefault(x, x)
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    for d in nl["devices"]:
        a, b = (x for x in (d["drain"], d["source"]))
        for x in (a, b):
            if x not in ("VDD", "VSS"):
                find(group, x)
        if a not in ("VDD", "VSS") and b not in ("VDD", "VSS"):
            group[find(group, a)] = find(group, b)
    comps: dict[str, list] = {}
    for d in nl["devices"]:
        x = d["drain"] if d["drain"] not in ("VDD", "VSS") else d["source"]
        comps.setdefault(find(group, x), []).append(d)
    todo = list(comps.values())
    while todo:
        ready = [c for c in todo if all(d["gate"] in val for d in c)]
        if not ready:
            break
        for c in ready:
            todo.remove(c)
            p: dict[str, str] = {}
            for d in c:
                find(p, d["drain"]), find(p, d["source"])
                if val[d["gate"]] == (d["type"] == "nmos"):
                    p[find(p, d["drain"])] = find(p, d["source"])
            up, dn = find(p, "VDD"), find(p, "VSS")
            for x in list(p):
                if x in ("VDD", "VSS"):
                    continue
                r = find(p, x)
                if (r == up) != (r == dn):
                    val[x] = r == up
    return val


def check_netlist(nl: dict, expect: dict[str, int], seed: int = 0) -> list[str]:
    """Problems (empty = pass): every output must be driven to exactly the expected value
    (expect[out] = truth table, bit r = row r, first input = LSB) on every simulated row."""
    k = len(nl["inputs"])
    if k <= EXHAUSTIVE_INPUTS:
        rows = range(1 << k)
    else:
        rng = random.Random(seed)
        rows = [0, (1 << k) - 1] + [rng.getrandbits(k) for _ in range(SAMPLED_ROWS)]
    problems = []
    for r in rows:
        val = simulate(nl, {x: bool((r >> i) & 1) for i, x in enumerate(nl["inputs"])})
        for o, f in expect.items():
            if o not in val:
                problems.append(f"row {r}: {o} floating, shorted or undriven")
            elif val[o] != bool((f >> r) & 1):
                problems.append(f"row {r}: {o}={int(val[o])}, expected {(f >> r) & 1}")
        if len(problems) > 5:
            break
    return problems


def expected_single(net: Network) -> dict[str, int]:
    masks, full = fast_masks(net.n_inputs)
    return {"OUT": ~truth(net.tree, masks, full) & full}


def stats(nl: dict) -> dict:
    """Axes of one netlist, from its devices (not from generator bookkeeping)."""
    gates = {d["gate"] for d in nl["devices"]}
    outs = [p for p in nl["ports"] if p not in nl["inputs"] and p not in SUPPLY]
    stage_nets = {x for x in gates if x not in nl["inputs"]} | set(outs)
    return {"inputs": len(nl["inputs"]), "per_side": sum(d["type"] == "nmos" for d in nl["devices"]),
            "stack": max(nl["pdn_depth"], nl["pun_depth"]), "stages": len(stage_nets), "outputs": len(outs)}
