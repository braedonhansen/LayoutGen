"""Tree -> transistor-level netlist, with sizing."""

from __future__ import annotations

from dataclasses import dataclass, field

from .enumerate import INPUT_NAMES, Network, depth, dual_tree, tree_str, num_leaves

NMOS_MODEL = "sky130_fd_pr__nfet_01v8"
PMOS_MODEL = "sky130_fd_pr__pfet_01v8_hvt"
GRID_UM = 0.005


@dataclass(frozen=True)
class Sizing:
    wn_unit: float = 0.42       # um, SKY130 minimum
    l: float = 0.15             # um
    wp_wn_ratio: float = 1.5
    nmos_model: str = NMOS_MODEL
    pmos_model: str = PMOS_MODEL
    max_diffusion_width: float | None = None  # um; None = no fit filtering

    @property
    def wp_unit(self) -> float:
        return self.wn_unit * self.wp_wn_ratio


@dataclass(frozen=True)
class Device:
    name: str
    type: str       # "nmos" | "pmos"
    model: str
    gate: str
    drain: str
    source: str
    bulk: str
    w: float
    l: float

    def as_dict(self) -> dict:
        return {"name": self.name, "type": self.type, "model": self.model,
                "gate": self.gate, "drain": self.drain, "source": self.source,
                "bulk": self.bulk, "w": self.w, "l": self.l}


@dataclass
class Cell:
    id: str
    network: Network
    devices: list[Device]
    nets: list[str]
    inputs: list[str] = field(default_factory=list)

    @property
    def max_w_n(self) -> float:
        return max(d.w for d in self.devices if d.type == "nmos")

    @property
    def max_w_p(self) -> float:
        return max(d.w for d in self.devices if d.type == "pmos")


def snap(w: float) -> float:
    return round(round(w / GRID_UM) * GRID_UM, 3)


def _series_multipliers(t, extra=0, out=None):
    """Per leaf (in tree order): length of the longest series path through it."""
    if out is None:
        out = []
    if t[0] == "L":
        out.append(1 + extra)
    elif t[0] == "P":
        for c in t[1]:
            _series_multipliers(c, extra, out)
    else:
        total = sum(depth(c) for c in t[1])
        for c in t[1]:
            _series_multipliers(c, extra + total - depth(c), out)
    return out


def _build(t, top, bottom, kind, names, counters, devs, mults, sizing):
    """Place devices of tree t between nets top and bottom."""
    if t[0] == "L":
        i = counters["dev"] = counters["dev"] + 1
        m = mults.pop(0)
        gate = INPUT_NAMES[t[1]]
        if kind == "nmos":
            devs.append(Device(f"MN{i}", "nmos", sizing.nmos_model, gate, top, bottom,
                               "VSS", snap(sizing.wn_unit * m), sizing.l))
        else:
            devs.append(Device(f"MP{i}", "pmos", sizing.pmos_model, gate, bottom, top,
                               "VDD", snap(sizing.wp_unit * m), sizing.l))
        return
    if t[0] == "P":
        for c in t[1]:
            _build(c, top, bottom, kind, names, counters, devs, mults, sizing)
        return
    nets = [top]
    for _ in t[1][:-1]:
        counters["net"] += 1
        nets.append(f"{names}{counters['net']}")
    nets.append(bottom)
    for c, a, b in zip(t[1], nets, nets[1:]):
        _build(c, a, b, kind, names, counters, devs, mults, sizing)


def cell_id(net: Network) -> str:
    import hashlib
    h = hashlib.sha1(tree_str(net.tree).encode()).hexdigest()[:8]
    return f"cg_{net.n_inputs}i{net.n_devices}d_{h}"


def build_cell(net: Network, sizing: Sizing = Sizing()) -> Cell:
    """PDN between OUT (top) and VSS; PUN (dual) between VDD (top) and OUT."""
    devs: list[Device] = []
    pdn = net.tree
    pun = dual_tree(pdn)
    _build(pdn, "OUT", "VSS", "nmos", "n", {"dev": 0, "net": 0}, devs,
           _series_multipliers(pdn), sizing)
    pdev: list[Device] = []
    _build(pun, "VDD", "OUT", "pmos", "p", {"dev": 0, "net": 0}, pdev,
           _series_multipliers(pun), sizing)
    devs += pdev
    inputs = list(INPUT_NAMES[:net.n_inputs])
    internal = sorted({x for d in devs for x in (d.drain, d.source)}
                      - {"OUT", "VDD", "VSS"}, key=lambda s: (s[0], int(s[1:])))
    nets = inputs + ["OUT", "VDD", "VSS"] + internal
    return Cell(cell_id(net), net, devs, nets, inputs)


def fits(cell: Cell, sizing: Sizing) -> bool:
    mx = sizing.max_diffusion_width
    return mx is None or max(cell.max_w_n, cell.max_w_p) <= mx + 1e-9
