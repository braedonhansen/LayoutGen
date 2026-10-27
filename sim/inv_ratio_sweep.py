"""Find the SKY130 W_p/W_n ratio for the netlist generator's sizing rule.

Sweeps the PMOS width of a minimum-length inverter (W_n fixed) at the tt
corner and reports, per W_p, the switching threshold V_M and the FO4 rise
and fall delays. The ratio is read off where V_M = VDD/2 and where
t_rise = t_fall.

Run inside WSL (needs ngspice and a volare-installed SKY130 PDK):
    python3 sim/inv_ratio_sweep.py [--wn 0.42] [--pfet pfet_01v8_hvt]
"""

import argparse
import csv
import os
import re
import subprocess
import tempfile
from pathlib import Path

PDK_ROOT = Path(os.environ.get("PDK_ROOT", Path.home() / ".volare"))
MODEL_LIB = PDK_ROOT / "sky130A/libs.tech/ngspice/sky130.lib.spice"
SIM_DIR = Path(__file__).resolve().parent  # holds .spiceinit

VDD = 1.8
L = 0.15  # minimum channel length, um
FANOUT = 4

HEADER = """* SKY130 inverter ratio sweep, W_n={wn}
.lib "{lib}" {corner}
.temp {temp}

Vdd vdd 0 {vdd}
Vin in 0 PULSE(0 {vdd} 100p 20p 20p 1.5n 3.2n)
Vdc dcin 0 0
"""

# One copy per W_p, all in a single netlist so the SKY130 models load once.
POINT = """.subckt inv{k} a y vdd vss
XN y a vss vss sky130_fd_pr__nfet_01v8 w={wn} l={l}
XP y a vdd vdd sky130_fd_pr__{pfet} w={wp} l={l}
.ends
* Transient: ideal edge -> shaping inverter -> DUT -> FO4 of DUT-sized
* inverters, each itself loaded by FO4 so the DUT's load sees realistic edges.
X0_{k} in a{k} vdd 0 inv{k}
X1_{k} a{k} b{k} vdd 0 inv{k}
XL1_{k} b{k} c{k} vdd 0 inv{k} M={fo}
XLL1_{k} c{k} d{k} vdd 0 inv{k} M={fo2}
* DC: an isolated inverter for the voltage transfer curve.
XDC_{k} dcin dcout{k} vdd 0 inv{k}
"""

DC_MEAS = """meas dc vm{k} when v(dcout{k})=v(dcin)
echo RESULT {k} vm=$&vm{k}
"""

TRAN_MEAS = """meas tran tfall{k} trig v(a{k}) val={half} rise=1 targ v(b{k}) val={half} fall=1
meas tran trise{k} trig v(a{k}) val={half} fall=1 targ v(b{k}) val={half} rise=1
echo RESULT {k} tfall=$&tfall{k} trise=$&trise{k}
"""


def run_sweep(wps, args):
    common = dict(wn=args.wn, l=L, pfet=args.pfet, vdd=VDD, half=VDD / 2,
                  fo=FANOUT, fo2=FANOUT * FANOUT)
    net = [HEADER.format(lib=MODEL_LIB, corner=args.corner, temp=args.temp,
                         **common)]
    net += [POINT.format(k=k, wp=wp, **common) for k, wp in enumerate(wps)]
    net.append(f".control\nop\ndc Vdc 0 {VDD} 1m\n")
    net += [DC_MEAS.format(k=k) for k in range(len(wps))]
    net.append("tran 1p 3.3n\n")
    net += [TRAN_MEAS.format(k=k, **common) for k in range(len(wps))]
    net.append("quit\n.endc\n.end\n")

    with tempfile.NamedTemporaryFile("w", suffix=".spice", delete=False) as f:
        f.write("".join(net))
        path = f.name
    try:
        out = subprocess.run(
            ["ngspice", "-b", path], cwd=SIM_DIR,
            capture_output=True, text=True, timeout=1800,
        ).stdout
    finally:
        os.unlink(path)

    rows = []
    for k, wp in enumerate(wps):
        m_vm = re.search(rf"RESULT {k} vm=(\S+)", out)
        m_t = re.search(rf"RESULT {k} tfall=(\S+) trise=(\S+)", out)
        if not (m_vm and m_t):
            raise RuntimeError(f"W_p={wp}: no result\n{out[-2000:]}")
        tf, tr = (float(x) for x in m_t.groups())
        rows.append({"wp": wp, "ratio": wp / args.wn,
                     "vm": float(m_vm.group(1)),
                     "tfall_ps": tf * 1e12, "trise_ps": tr * 1e12})
    return rows


def crossing(rows, key, target):
    """Linear interpolation of the W_p where rows[key] crosses target."""
    for a, b in zip(rows, rows[1:]):
        ya, yb = a[key] - target, b[key] - target
        if ya == 0:
            return a["wp"]
        if ya * yb < 0:
            return a["wp"] + (b["wp"] - a["wp"]) * ya / (ya - yb)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wn", type=float, default=0.42, help="NMOS width, um")
    p.add_argument("--pfet", default="pfet_01v8_hvt",
                   help="pfet_01v8_hvt (std-cell PMOS) or pfet_01v8")
    p.add_argument("--corner", default="tt")
    p.add_argument("--temp", type=float, default=27)
    p.add_argument("--ratios", default="1.0:6.0:0.25",
                   help="start:stop:step of W_p/W_n")
    p.add_argument("--csv", help="write the sweep table here")
    args = p.parse_args()

    lo, hi, step = (float(x) for x in args.ratios.split(":"))
    n = int(round((hi - lo) / step)) + 1
    wps = [round(args.wn * (lo + i * step), 4) for i in range(n)]

    rows = run_sweep(wps, args)

    print(f"{args.pfet}  W_n={args.wn}um  L={L}um  {args.corner} {args.temp}C")
    print(f"{'W_p':>7} {'ratio':>6} {'V_M':>7} {'t_fall':>8} {'t_rise':>8}")
    for r in rows:
        print(f"{r['wp']:7.3f} {r['ratio']:6.2f} {r['vm']:7.4f} "
              f"{r['tfall_ps']:7.1f}p {r['trise_ps']:7.1f}p")

    for r in rows:
        r["skew_ps"] = r["trise_ps"] - r["tfall_ps"]
        r["tavg_ps"] = (r["trise_ps"] + r["tfall_ps"]) / 2
    wp_vm = crossing(rows, "vm", VDD / 2)
    wp_dly = crossing(rows, "skew_ps", 0)
    wp_avg = min(rows, key=lambda r: r["tavg_ps"])["wp"]
    for label, wp in (("V_M = VDD/2", wp_vm), ("t_rise = t_fall", wp_dly),
                      ("min (t_rise+t_fall)/2", wp_avg)):
        if wp is None:
            print(f"{label}: outside sweep range")
        else:
            print(f"{label}: W_p = {wp:.3f}um  ratio = {wp / args.wn:.2f}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
