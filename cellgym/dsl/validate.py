"""Static plan validator (SPEC.md "Validator"): runs before rendering.

Checks a parsed Plan against its M1 netlist (the JSON dict). Every problem is
reported as an Issue with the plan line it belongs to and a stable code; the
agent reads these during repair. Near-misses (adjacent objects of different
nets on one track) are legal and only counted: compaction turns them into
width.
"""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .plan import ROWS, Endpoint, Plan
from .template import CHANNEL_NAMES, Layout, layout_for_netlist

RAILS = {"nmos": "VSS", "pmos": "VDD"}
SUPPLY = {"VDD", "VSS", "VPB", "VNB"}


@dataclass(frozen=True)
class Issue:
    line: int
    code: str
    message: str

    def __str__(self) -> str:
        where = f"line {self.line}" if self.line else "plan"
        return f"{where} [{self.code}] {self.message}"


@dataclass
class Report:
    layout: Layout
    issues: list[Issue] = field(default_factory=list)
    near_misses: int = 0
    gate_tracks: dict[tuple[int, str], str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.issues

    def codes(self) -> set[str]:
        return {i.code for i in self.issues}

    def __str__(self) -> str:
        if self.ok:
            return f"ok (height {self.layout.height} um, {self.near_misses} near-misses)"
        return "\n".join(map(str, self.issues))


@dataclass
class _Island:
    id: int
    row: str
    net: str
    cols: list[int]
    contacted: bool
    line: int


class _Checker:
    def __init__(self, plan: Plan, netlist: dict):
        self.plan = plan
        self.netlist = netlist
        self.report = Report(layout_for_netlist(netlist))
        self.cols = plan.columns
        self.islands: list[_Island] = []
        self.island_at: dict[tuple[str, int], _Island] = {}
        self.gate_contacts: dict[tuple[int, str], str] = {}   # (col, side) -> net
        self.uf: dict = {}
        self.bad_nets: set[str] = set()   # nets with a rejected route or pin: no cascade errors

    def err(self, line: int, code: str, message: str, net: str | None = None) -> None:
        self.report.issues.append(Issue(line, code, message))
        if net:
            self.bad_nets.add(net)

    # -- structure -----------------------------------------------------------

    def check_io(self) -> None:
        net_inputs = list(self.netlist["inputs"])
        net_outputs = [p for p in self.netlist["ports"] if p not in net_inputs and p not in SUPPLY]
        if sorted(self.plan.inputs) != sorted(net_inputs):
            self.err(0, "io", f"inputs {self.plan.inputs} != netlist inputs {net_inputs}")
        if sorted(self.plan.outputs) != sorted(net_outputs):
            self.err(0, "io", f"outputs {self.plan.outputs} != netlist outputs {net_outputs}")
        if not 1 <= len(self.plan.outputs) <= 3:
            self.err(0, "io", f"{len(self.plan.outputs)} outputs; a cell has 1-3")

    def check_columns(self) -> None:
        cols = self.cols
        if not cols:
            self.err(0, "columns", "plan has no columns")
            return
        for c in (cols[0], cols[-1]):
            if c.kind != "sd":
                self.err(c.line, "ends", f"column {c.index}: a row must start and end with an sd column")
        known = set(self.netlist["nets"])
        for c in cols:
            for row in ROWS:
                net = c.nets[row]
                if net is None:
                    continue
                if net not in known:
                    self.err(c.line, "unknown-net", f"column {c.index}: '{net}' is not a netlist net")
                if c.kind == "sd" and net in RAILS.values() and net != RAILS[row]:
                    self.err(c.line, "rail-row", f"column {c.index}: {net} in the {row} row")
                if c.kind == "g" and net in SUPPLY:
                    self.err(c.line, "rail-gate", f"column {c.index}: gate tied to {net}")
            if c.kind == "g":
                if c.nets["nmos"] is None and c.nets["pmos"] is None:
                    self.err(c.line, "empty-gate", f"column {c.index}: gate column with no device")
                for nb in (c.index - 1, c.index + 1):
                    if 0 <= nb < len(cols) and cols[nb].kind == "g":
                        self.err(c.line, "gate-gate", f"columns {c.index} and {nb}: two gate columns need an sd column between them")
                        break
                for row in ROWS:
                    if c.nets[row] is None:
                        continue
                    for nb in (c.index - 1, c.index + 1):
                        if 0 <= nb < len(cols) and cols[nb].kind == "sd" and cols[nb].nets[row] is None:
                            self.err(c.line, "gate-diffusion", f"column {c.index}: {row} device has no diffusion at column {nb}")

    def build_islands(self) -> None:
        for row in ROWS:
            cur = None
            for c in self.cols:
                net = c.nets[row] if c.kind == "sd" else None
                if net is None:
                    cur = None
                    continue
                if cur is None or cur.net != net:
                    cur = _Island(len(self.islands), row, net, [], False, c.line)
                    self.islands.append(cur)
                cur.cols.append(c.index)
                cur.contacted |= c.contact[row]
                self.island_at[(row, c.index)] = cur

    def check_devices(self) -> None:
        want = Counter((d["type"], d["gate"], frozenset((d["drain"], d["source"])))
                       for d in self.netlist["devices"])
        have: dict = defaultdict(list)
        for c in self.cols:
            if c.kind != "g" or not 0 < c.index < len(self.cols) - 1:
                continue
            for row in ROWS:
                g = c.nets[row]
                a = self.cols[c.index - 1].nets[row]
                b = self.cols[c.index + 1].nets[row]
                if g is None or a is None or b is None:
                    continue
                have[(row, g, frozenset((a, b)))].append(c)
        for key, cs in have.items():
            for c in cs[want.get(key, 0):]:
                row, g, terms = key
                self.err(c.line, "device-extra",
                         f"column {c.index}: {row} gate {g} between {'/'.join(sorted(terms))} is not in the netlist (or appears too often)")
        for key, n in want.items():
            for _ in range(n - len(have.get(key, []))):
                row, g, terms = key
                self.err(0, "device-missing", f"netlist {row} gate {g} between {'/'.join(sorted(terms))} is not placed")

    def collect_gate_contacts(self) -> None:
        for c in self.cols:
            if c.kind != "g":
                continue
            n, p = c.nets["nmos"], c.nets["pmos"]
            if n and p and n != p:
                self.gate_contacts[(c.index, "n")] = n
                self.gate_contacts[(c.index, "p")] = p
            elif n or p:
                self.gate_contacts[(c.index, "")] = n or p

    def check_contacts(self) -> None:
        islands_per_net = Counter(i.net for i in self.islands)
        gate_nets = set(self.gate_contacts.values())
        for isl in self.islands:
            needed = (isl.net in SUPPLY or isl.net in self.plan.outputs
                      or islands_per_net[isl.net] > 1 or isl.net in gate_nets)
            if needed and not isl.contacted:
                self.err(isl.line, "contact-missing",
                         f"{isl.row} {isl.net} at column {isl.cols[0]} needs a contact")

    # -- routes, pins, gate tracks ------------------------------------------

    def _sd_endpoint(self, ep: Endpoint, net: str, line: int):
        if not 0 <= ep.col < len(self.cols) or self.cols[ep.col].kind != "sd":
            self.err(line, "endpoint", f"{ep}: not an sd column", net)
            return None
        col = self.cols[ep.col]
        if col.nets[ep.kind] != net:
            self.err(line, "endpoint", f"{ep} is {col.nets[ep.kind] or 'empty'}, not {net}", net)
            return None
        if not col.contact[ep.kind]:
            self.err(line, "endpoint", f"{ep} has no contact", net)
            return None
        return self.island_at[(ep.kind, ep.col)]

    def _gate_endpoint(self, ep: Endpoint, net: str, line: int):
        key = (ep.col, ep.side)
        if key not in self.gate_contacts:
            split = (ep.col, "n") in self.gate_contacts
            if split and not ep.side:
                self.err(line, "gate-side", f"{ep}: split column, use gate:{ep.col}n or gate:{ep.col}p", net)
            else:
                self.err(line, "endpoint", f"{ep}: no gate contact there", net)
            return None
        if self.gate_contacts[key] != net:
            self.err(line, "endpoint", f"{ep} is {self.gate_contacts[key]}, not {net}", net)
            return None
        return key

    def _set_gate_track(self, key, track: str, line: int) -> None:
        old = self.report.gate_tracks.get(key)
        if old and old != track:
            self.err(line, "gate-track-conflict", f"gate:{key[0]}{key[1]} is on {old} and {track}")
        else:
            self.report.gate_tracks[key] = track

    def _union(self, a, b) -> None:
        self.uf[self._find(a)] = self._find(b)

    def _find(self, x):
        while self.uf.get(x, x) != x:
            x = self.uf[x]
        return x

    def check_routes(self) -> None:
        lay = self.report.layout
        for r in self.plan.routes:
            if not lay.has_track(r.track):
                self.err(r.line, "track-missing", f"track {r.track} does not exist at height {lay.height}", r.net)
                continue
            if r.layer == "li1" and r.track not in CHANNEL_NAMES:
                self.err(r.line, "li-off-channel", f"li1 runs only on channel tracks, not {r.track}", r.net)
                continue
            if r.net in SUPPLY:
                self.err(r.line, "rail-routed", f"{r.net} is strapped by the renderer, never routed")
                continue
            nodes = []
            for ep in r.endpoints:
                if ep.kind == "gate":
                    key = self._gate_endpoint(ep, r.net, r.line)
                    if key is None:
                        continue
                    if r.track not in CHANNEL_NAMES:
                        self.err(r.line, "gate-track", f"{ep}: gate contacts sit on channel tracks, not {r.track}", r.net)
                        continue
                    self._set_gate_track(key, r.track, r.line)
                    nodes.append(("gc", key))
                else:
                    isl = self._sd_endpoint(ep, r.net, r.line)
                    if isl is None:
                        continue
                    if r.track[0] not in ("c", ep.kind[0]):
                        self.err(r.line, "endpoint-track", f"{ep} cannot reach {r.track} over the other row", r.net)
                        continue
                    nodes.append(("isl", isl.id))
            for a, b in zip(nodes, nodes[1:]):
                self._union(a, b)

    def check_pins(self) -> None:
        lay = self.report.layout
        count = Counter(p.net for p in self.plan.pins)
        for net in self.plan.inputs + self.plan.outputs:
            if count[net] == 0:
                self.err(0, "pin-missing", f"no pin for {net}")
        for p in self.plan.pins:
            if p.net not in self.plan.inputs and p.net not in self.plan.outputs:
                self.err(p.line, "pin-net", f"{p.net} is not a cell input or output")
                continue
            if count[p.net] > 1:
                self.err(p.line, "pin-dup", f"{p.net} has more than one pin")
            is_input = p.net in self.plan.inputs
            if is_input != (p.landing.kind == "gate"):
                self.err(p.line, "pin-landing", f"{'input' if is_input else 'output'} {p.net} cannot land on {p.landing}", p.net)
                continue
            if p.landing.kind == "gate":
                key = self._gate_endpoint(p.landing, p.net, p.line)
                if key is not None and p.track:
                    if not lay.has_track(p.track) or p.track not in CHANNEL_NAMES:
                        self.err(p.line, "gate-track", f"pin track {p.track} is not a channel track", p.net)
                    else:
                        self._set_gate_track(key, p.track, p.line)
            else:
                if p.track:
                    self.err(p.line, "pin-landing", "a track is only given for a gate landing")
                self._sd_endpoint(p.landing, p.net, p.line)

    def check_gate_tracks(self) -> None:
        tracks = self.report.gate_tracks
        for key, net in self.gate_contacts.items():
            if key not in tracks and net not in self.bad_nets:
                line = self.cols[key[0]].line
                self.err(line, "gate-untracked",
                         f"gate:{key[0]}{key[1]} ({net}) has no track: route it or give its pin a track")
        for (col, side), t in tracks.items():
            if side == "n" and (col, "p") in tracks:
                # poly pads on adjacent tracks are 0.07 apart (poly.2 needs 0.21)
                if CHANNEL_NAMES.index(tracks[(col, "p")]) - CHANNEL_NAMES.index(t) < 2:
                    self.err(self.cols[col].line, "split-order",
                             f"column {col}: split contacts must be NMOS on c0, PMOS on c2")

    def check_met1_pads(self) -> None:
        """met1 mcon pads (0.23 tall) on adjacent tracks in one sd column are 0.11
        apart (m1.2 needs 0.14); compaction cannot separate them."""
        lay = self.report.layout
        pads = defaultdict(list)    # col -> (track k, net, line)
        for r in self.plan.routes:
            if r.layer != "met1" or not lay.has_track(r.track):
                continue
            for ep in r.endpoints:
                if ep.kind != "gate":
                    pads[ep.col].append((lay.track_k(r.track), r.net, r.line))
        for col, ps in pads.items():
            for a, b in itertools.combinations(ps, 2):
                if a[1] != b[1] and abs(a[0] - b[0]) == 1:
                    self.err(max(a[2], b[2]), "met1-pad",
                             f"column {col}: met1 pads of {a[1]} and {b[1]} on adjacent tracks")

    def check_connectivity(self) -> None:
        groups: dict[str, set] = defaultdict(set)
        for isl in self.islands:
            if isl.contacted and isl.net not in SUPPLY:
                groups[isl.net].add(("isl", isl.id))
        for key, net in self.gate_contacts.items():
            groups[net].add(("gc", key))
        for net, nodes in sorted(groups.items()):
            if net in self.bad_nets:
                continue
            comps = defaultdict(list)
            for nd in nodes:
                comps[self._find(nd)].append(nd)
            if len(comps) > 1:
                parts = []
                for comp in comps.values():
                    names = []
                    for kind, ref in sorted(comp, key=str):
                        if kind == "isl":
                            isl = self.islands[ref]
                            names.append(f"{isl.row}:{isl.cols[0]}")
                        else:
                            names.append(f"gate:{ref[0]}{ref[1]}")
                    parts.append("{" + " ".join(names) + "}")
                self.err(0, "open", f"net {net} is split into {len(comps)} pieces: " + " ".join(parts))

    # -- overlaps --------------------------------------------------------------

    def check_overlaps(self) -> None:
        li = defaultdict(list)       # track -> (lo, hi, net, line)
        m1 = defaultdict(list)
        pads = defaultdict(list)
        for r in self.plan.routes:
            if r.net in SUPPLY or not self.report.layout.has_track(r.track):
                continue
            if r.layer == "li1" and r.track not in CHANNEL_NAMES:
                continue
            cs = [ep.col for ep in r.endpoints]
            (li if r.layer == "li1" else m1)[r.track].append((min(cs), max(cs), r.net, r.line))
            if r.track not in CHANNEL_NAMES:
                continue
            i = CHANNEL_NAMES.index(r.track)
            for ep in r.endpoints:
                if ep.kind == "gate":
                    continue
                crossed = CHANNEL_NAMES[: i + 1] if ep.kind == "nmos" else CHANNEL_NAMES[i:]
                for t in crossed:
                    li[t].append((ep.col, ep.col, r.net, r.line))
        for (col, side), t in self.report.gate_tracks.items():
            net = self.gate_contacts[(col, side)]
            li[t].append((col, col, net, self.cols[col].line))
            pads[t].append((col, col, net, self.cols[col].line))
        groups = [(f"li1 {t}", objs, 1) for t, objs in li.items()]
        groups += [(f"poly {t}", objs, 2) for t, objs in pads.items()]
        groups += [(f"met1 {t}", objs, 1) for t, objs in m1.items()]
        reported = set()
        for where, objs, gap in groups:
            for a, b in itertools.combinations(objs, 2):
                if a[2] == b[2]:
                    continue
                if a[0] <= b[1] and b[0] <= a[1]:
                    key = (where, frozenset((a[2], b[2])))
                    if key not in reported:
                        reported.add(key)
                        lo, hi = max(a[0], b[0]), min(a[1], b[1])
                        self.err(max(a[3], b[3]), "overlap",
                                 f"{a[2]} and {b[2]} overlap on {where} at columns {lo}-{hi}")
                elif a[0] - gap <= b[1] and b[0] - gap <= a[1]:
                    self.report.near_misses += 1

    def run(self) -> Report:
        self.check_io()
        self.check_columns()
        self.build_islands()
        self.check_devices()
        self.collect_gate_contacts()
        self.check_contacts()
        self.check_routes()
        self.check_pins()
        self.check_gate_tracks()
        self.check_met1_pads()
        self.check_connectivity()
        self.check_overlaps()
        self.report.issues.sort(key=lambda i: (i.line, i.code))
        return self.report


def validate(plan: Plan, netlist: dict) -> Report:
    return _Checker(plan, netlist).run()
