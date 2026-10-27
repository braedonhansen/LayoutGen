"""Fast LVS: extract devices and nets from rectangles, compare to the netlist.

Extraction (own code path, geometry only):
- gate = poly AND diff (one device per gate component); s/d = diff NOT poly,
  one terminal per s/d component touching the gate on its two current sides.
- type from implant (nsdm -> nmos, psdm -> pmos); pmos model is the hvt one
  when the gate is inside hvtp. W = gate extent along the poly, L = across.
- conductors: s/d, poly, li, met1 components; licon joins li to s/d or poly,
  mcon joins li to met1. Nets are named by text labels on li (67/5) and met1
  (68/5) at the label point.
- bulk: pmos -> the n-well component under the gate (named by its VPB label),
  nmos -> substrate (VNB).

Compare: ports must be labelled once each with no two on one net; then an
exact device-graph match (named nets fixed, internal nets renamed, s/d
swappable) with W and L within 1 nm.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict

from .drc import is_rect
from .geometry import Region

TOL = 1  # nm
NMOS_MODEL = "sky130_fd_pr__nfet_01v8"
PMOS_MODELS = {True: "sky130_fd_pr__pfet_01v8_hvt", False: "sky130_fd_pr__pfet_01v8"}
SUPPLY = ("VDD", "VSS", "VPB", "VNB")


class _UF:
    def __init__(self):
        self.p: dict = {}

    def find(self, a):
        p = self.p
        p.setdefault(a, a)
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


class _Index:
    """Rect -> component lookup for one layer."""

    def __init__(self, layer: str, reg: Region):
        self.items = []
        for ci, comp in enumerate(reg.components()):
            for r in comp:
                self.items.append((r, (layer, ci)))
        self.items.sort()
        self.maxw = max((r[2] - r[0] for r, _ in self.items), default=0)
        self.keys = [r[0] for r, _ in self.items]

    def hits(self, q, touch: bool = False):
        x0, y0, x1, y1 = q
        i = bisect_left(self.keys, x0 - self.maxw - 1)
        j = bisect_right(self.keys, x1)
        out = set()
        for r, cid in self.items[i:j]:
            if touch:
                ox = min(r[2], x1) - max(r[0], x0)
                oy = min(r[3], y1) - max(r[1], y0)
                if ox >= 0 and oy >= 0 and (ox > 0 or oy > 0):
                    out.add(cid)
            elif min(r[2], x1) > max(r[0], x0) and min(r[3], y1) > max(r[1], y0):
                out.add(cid)
        return out

    def overlap_len(self, q) -> dict:
        """Component -> overlap length with a 1-unit-thick strip q (along its long side)."""
        x0, y0, x1, y1 = q
        i = bisect_left(self.keys, x0 - self.maxw - 1)
        j = bisect_right(self.keys, x1)
        out: dict = {}
        for r, cid in self.items[i:j]:
            ox = min(r[2], x1) - max(r[0], x0)
            oy = min(r[3], y1) - max(r[1], y0)
            if ox > 0 and oy > 0:
                out[cid] = out.get(cid, 0) + max(ox, oy)
        return out

    def at(self, x, y):
        out = set()
        for r, cid in self.items:
            if r[0] <= x <= r[2] and r[1] <= y <= r[3]:
                out.add(cid)
        return out


def extract(g, labels) -> tuple[list[dict], dict, list[str]]:
    """Returns (devices, net_names, errors). Devices: dict(type, model, g, sd
    (frozenset of 1-2 nets), b, w, l, box). net_names: net -> set(label names)."""
    errors: list[str] = []
    uf = _UF()
    idx = {k: _Index(k, g[k]) for k in ("sd", "poly", "li", "met1", "nwell")}
    for k in ("sd", "poly", "li", "met1"):
        for _, cid in idx[k].items:
            uf.find(cid)
    # licon: li <-> s/d | poly ; mcon: li <-> met1
    for comp in g["licon"].components():
        for r in comp:
            lis = idx["li"].hits(r)
            lower = idx["sd"].hits(r) | idx["poly"].hits(r)
            for a in lis | lower:
                for b in lis | lower:
                    uf.union(a, b)
    for comp in g["mcon"].components():
        for r in comp:
            group = idx["li"].hits(r) | idx["met1"].hits(r)
            first = next(iter(group), None)
            for b in group:
                uf.union(first, b)

    # bulk nets
    nwell_names: dict = defaultdict(set)
    sub_names: set = set()
    names: dict = defaultdict(set)
    for lab in labels:
        kind, net = lab.kind, lab.net
        x0, y0, x1, y1 = lab.rect
        x, y = (x0 + x1) // 2, (y0 + y1) // 2
        if kind in ("li", "met1"):
            hit = idx[kind].at(x, y)
            if not hit:
                errors.append(f"label {net}: not on {kind} at ({x}, {y})")
            for cid in hit:
                names[uf.find(cid)].add(net)
        elif kind == "nwell":
            hit = idx["nwell"].at(x, y)
            if not hit:
                errors.append(f"label {net}: not on nwell")
            for cid in hit:
                nwell_names[cid].add(net)
        elif kind == "sub":
            sub_names.add(net)
    for cid, ns in nwell_names.items():
        names[("well", cid)] |= ns
    names[("sub",)] |= sub_names

    nsdm, psdm, hvtp, poly, nwell = g["nsdm"], g["psdm"], g["hvtp"], g["poly"], g["nwell"]
    devices = []
    for comp in g["gate"].components():
        bb = (min(r[0] for r in comp), min(r[1] for r in comp), max(r[2] for r in comp), max(r[3] for r in comp))
        x0, y0, x1, y1 = bb
        if not is_rect(comp):
            errors.append(f"device at {bb}: non-rectangular gate")
            continue
        vert = poly.overlaps((x0, y1, x1, y1 + 1)) or poly.overlaps((x0, y0 - 1, x1, y0))
        if vert:
            w, l = y1 - y0, x1 - x0
            sides = [(x0 - 1, y0, x0, y1), (x1, y0, x1 + 1, y1)]
        else:
            w, l = x1 - x0, y1 - y0
            sides = [(x0, y1, x1, y1 + 1), (x0, y0 - 1, x1, y0)]
        terms, shared = [], 0
        for s in sides:
            lens = idx["sd"].overlap_len(s)
            terms += sorted(lens)
            shared += sum(lens.values())
        if terms:   # Magic: W = mean length of the gate edges shared with s/d
            w = shared / len(sides)
        gate_poly = idx["poly"].hits(bb)
        n_in, p_in = nsdm.covers(bb), psdm.covers(bb)
        if n_in == p_in:
            errors.append(f"device at {bb}: gate not under exactly one of nsdm/psdm")
            continue
        typ = "nmos" if n_in else "pmos"
        wells = idx["nwell"].hits(bb)
        if typ == "pmos":
            if not nwell.covers(bb):
                errors.append(f"device at {bb}: pmos gate not inside nwell")
                continue
            bulk = ("well", next(iter(wells)))
            hv = hvtp.covers(bb)
            if not hv and hvtp.overlaps(bb):
                errors.append(f"device at {bb}: gate partly under hvtp")
            model = PMOS_MODELS[hv]
        else:
            if wells:
                errors.append(f"device at {bb}: nmos gate inside nwell")
                continue
            bulk = ("sub",)
            model = NMOS_MODEL
        if len(set(terms)) != 2:
            errors.append(f"device at {bb}: {len(set(terms))} s/d terminals (need 2)")
            continue
        devices.append(dict(type=typ, model=model, g=uf.find(next(iter(gate_poly))),
                            sd=frozenset(uf.find(t) for t in terms), b=bulk, w=w, l=l, box=bb))
    return devices, dict(names), errors


def _ref(netlist: dict):
    devs = []
    for d in netlist["devices"]:
        bulk = "VPB" if d["type"] == "pmos" else "VNB"
        devs.append(dict(type=d["type"], model=d["model"], g=d["gate"],
                         sd=frozenset((d["drain"], d["source"])), b=bulk,
                         w=int(round(d["w"] * 1000)), l=int(round(d["l"] * 1000)), name=d["name"]))
    outputs = [p for p in netlist["ports"] if p not in netlist["inputs"] and p not in SUPPLY]
    ports = list(netlist["inputs"]) + outputs + list(SUPPLY)
    return devs, ports


def _sig(d) -> tuple:
    return (d["type"], d["model"], d["w"], d["l"])


def _fmt(d) -> str:
    return f"{d['type']} {d['model'].split('__')[-1]} W={d['w'] / 1000:g} L={d['l'] / 1000:g}"


def compare(devices, net_names, netlist) -> list[str]:
    errs: list[str] = []
    ref, ports = _ref(netlist)
    port_set = set(ports)
    # -- names / ports
    nets_of: dict[str, list] = defaultdict(list)
    for net, ns in net_names.items():
        for n in ns:
            nets_of[n].append(net)
        named = sorted(ns & port_set)
        if len(named) > 1:
            errs.append(f"short: ports {', '.join(named)} on one net")
    for n, nets in nets_of.items():
        if n not in port_set:
            errs.append(f"extra label {n} (not a port)")
        elif len(nets) > 1:
            errs.append(f"open: port {n} labelled on {len(nets)} unconnected nets")
    for p in ports:
        if p not in nets_of:
            errs.append(f"missing pin label {p}")
    # extracted net -> name (only unambiguous port names fix a net)
    fixed: dict = {}
    for n, nets in nets_of.items():
        if n in port_set and len(nets) == 1 and len(net_names[nets[0]] & port_set) == 1:
            fixed[nets[0]] = n

    # -- W/L within TOL nm of a reference value count as that value
    rw = {(d["type"], d["w"]) for d in ref}
    rl = {(d["type"], d["l"]) for d in ref}
    for d in devices:
        for key, vals in (("w", rw), ("l", rl)):
            near = [v for t, v in vals if t == d["type"] and abs(v - d[key]) <= TOL]
            d[key] = near[0] if near else d[key]
    # -- device multiset
    ce = Counter(_sig(d) for d in devices)
    cr = Counter(_sig(d) for d in ref)
    if ce != cr:
        extra = ce - cr
        missing = cr - ce
        by_type_e = defaultdict(list)
        for s, k in extra.items():
            by_type_e[(s[0], s[1], s[3])] += [s] * k
        for s, k in missing.items():
            for _ in range(k):
                cand = by_type_e.get((s[0], s[1], s[3]))
                if cand:
                    e = cand.pop()
                    extra[e] -= 1
                    errs.append(f"wrong W: {s[0]} expected W={s[2] / 1000:g}, got W={e[2] / 1000:g}")
                else:
                    errs.append(f"missing device: {s[0]} {s[1].split('__')[-1]} W={s[2] / 1000:g} L={s[3] / 1000:g}")
        for s, k in (+extra).items():
            errs.append(f"extra device x{k}: {s[0]} {s[1].split('__')[-1]} W={s[2] / 1000:g} L={s[3] / 1000:g}")
        return errs
    if errs:
        return errs

    # -- exact graph match (named nets fixed, internal nets bijective)
    ext_nets = {n for d in devices for n in (d["g"], d["b"], *d["sd"])}
    ref_nets = {n for d in ref for n in (d["g"], d["b"], *d["sd"])}
    internal_e = [n for n in ext_nets if n not in fixed]
    internal_r = [n for n in ref_nets if n not in port_set]
    if len(internal_e) != len(internal_r):
        kind = "short (fewer nets)" if len(internal_e) < len(internal_r) else "open (more nets)"
        errs.append(f"{kind}: {len(internal_e)} internal nets extracted, {len(internal_r)} expected")
    m = _match(devices, ref, fixed)
    if m is None:
        if not errs:
            errs.append("connectivity mismatch: no device-graph isomorphism")
        errs += _diagnose(devices, ref, fixed, port_set)
    return errs


def _match(devices, ref, fixed):
    """Backtracking device assignment. Returns net map or None."""
    fwd = dict(fixed)                       # ext net -> ref net
    bwd = {v: k for k, v in fixed.items()}  # ref net -> ext net
    n = len(devices)
    used = [False] * len(ref)

    def score(d):
        return -sum(1 for x in (d["g"], d["b"], *d["sd"]) if x in fixed)
    order = sorted(range(n), key=lambda i: score(devices[i]))
    cands = {i: [j for j, r in enumerate(ref) if _sig(r) == _sig(devices[i])] for i in range(n)}

    def bind(pairs, undo):
        for e, r in pairs:
            if e in fwd:
                if fwd[e] != r:
                    return False
            elif r in bwd:
                return False
            else:
                fwd[e] = r
                bwd[r] = e
                undo.append(e)
        return True

    def rec(k):
        if k == n:
            return True
        d = devices[order[k]]
        sd = list(d["sd"])
        for j in cands[order[k]]:
            if used[j]:
                continue
            r = ref[j]
            rsd = list(r["sd"])
            opts = [list(zip(sd, rsd))]
            if len(sd) == 2 and len(rsd) == 2:
                opts.append(list(zip(sd, rsd[::-1])))
            elif len(sd) != len(rsd):
                continue
            for sdp in opts:
                undo: list = []
                if bind([(d["g"], r["g"]), (d["b"], r["b"])] + sdp, undo):
                    used[j] = True
                    if rec(k + 1):
                        return True
                    used[j] = False
                for e in undo:
                    del bwd[fwd.pop(e)]
        return False

    return fwd if rec(0) else None


def _diagnose(devices, ref, fixed, port_set) -> list[str]:
    """Per named net: terminal counts extracted vs expected."""
    out = []
    inv = {v: k for k, v in fixed.items()}

    def counts(devs, net):
        g = sum(1 for d in devs if d["g"] == net)
        s = sum(1 for d in devs if net in d["sd"])
        return g, s
    for p in sorted(port_set):
        e = counts(devices, inv.get(p, object()))
        r = counts(ref, p)
        if e != r:
            out.append(f"net {p}: {e[0]} gates/{e[1]} s-d extracted, expected {r[0]}/{r[1]}")
    return out


def run_lvs(g, labels, netlist) -> tuple[bool, list[str]]:
    devices, names, errs = extract(g, labels)
    errs = errs + compare(devices, names, netlist)
    return not errs, errs
