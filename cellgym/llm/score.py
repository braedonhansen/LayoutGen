"""Score cellgym.llm.infer samples with the repo pipeline (parse, validate, render, fast check).

    python -m cellgym.llm.score out/llm/step05/samples.jsonl [--jobs 14]

oneshot samples: header + body -> evaluate. step samples: columns -> the Euler baseline's router
(route_best: contacts, routes, pins) -> evaluate; a placement that ended illegal / stuck / at the
column cap, or that the router cannot route, fails with that stage. Writes <samples>.summary.json
(per set: per-sample pass, pass@k, W/W_euler of passes, stage counts) and prints a table.

--tracks 3 (default) routes at the fixed 3-track channel; min at the fewest tracks that route (area
against Euler width x the 3-track height); area at the smallest-area track count (route_min_area),
area against the Euler placement routed the same way (the cellgym.llm.arearef cache, "area/Euler-area";
"-" for sets with no reference, e.g. probes).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from cellgym.llm import fmt as F
from cellgym.llm.infer import load_sets

_CELLS = {}
_TRACKS = "3"      # "3" (fixed template) | "min" (smallest channel that routes) | "area" | an integer
_REFS = {}         # area mode: set -> {cell: Euler reference area, nm^2}


def _init(sets, tracks="3", refs=None):
    global _CELLS, _TRACKS, _REFS
    _CELLS, _TRACKS, _REFS = sets, tracks, refs or {}


def _route(cols, nl):
    """Route one placement under the --tracks policy; (plan text, tracks)."""
    from cellgym.baselines import common as C
    if _TRACKS == "3":
        return C.route_best([cols], nl).text, 3
    if _TRACKS == "min":
        rt = C.route_min_tracks([cols], nl)
    elif _TRACKS == "area":
        rt = C.route_min_area([cols], nl)
    else:
        rt = C.route_best([cols], nl, tracks=int(_TRACKS))
    return rt.text, rt.tracks


def ref_height_nm(nl):
    """Height of the fixed 3-track template: the Euler reference's height."""
    from cellgym.dsl.template import layout_for_netlist
    return round(layout_for_netlist(nl).height * 1000)


def _score(r):
    from cellgym.baselines.common import BaselineError, route_best
    from cellgym.bench.evaluate import evaluate
    cells, w = _CELLS[r["set"]]
    nl = cells[r["cell"]]
    try:
        if "text" in r:
            ev = evaluate(F.plan_header(nl) + r["text"], nl)
        else:
            if r["status"] != "done":
                return r["set"], r["cell"], r["status"], None, None, None
            try:
                text, tracks = _route([tuple(c) for c in r["cols"]], nl)
            except BaselineError as e:
                return r["set"], r["cell"], "route:" + e.reason, None, None, None
            ev = evaluate(text, nl)
    except Exception as e:  # noqa: BLE001
        return r["set"], r["cell"], "error:" + type(e).__name__, None, None, None
    if ev.stage != "pass":
        return r["set"], r["cell"], ev.stage + ":" + ev.detail.split("]")[0].split("[")[-1][:30], None, None, None
    we = w.get(r["cell"])
    if "text" in r:
        return r["set"], r["cell"], "pass", (ev.width_nm / we if we else None), None, None
    if _TRACKS == "area":
        ref = _REFS.get(r["set"], {}).get(r["cell"])
        return (r["set"], r["cell"], "pass", (ev.width_nm / we if we else None), tracks,
                (ev.width_nm * ev.height_nm / ref if ref else None))
    h3 = ref_height_nm(nl)
    h = h3 + 340 * (tracks - 3)
    return (r["set"], r["cell"], "pass", (ev.width_nm / we if we else None), tracks,
            (ev.width_nm * h / (we * h3) if we else None))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("samples", type=Path)
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--extra", nargs="*", default=[], help="name=dir:part, as given to infer")
    ap.add_argument("--tracks", default="3", help="channel tracks: 3 (fixed template), min (smallest that routes), "
                    "area (smallest area, ratios against the arearef cache), or N")
    ap.add_argument("--arearef", type=Path, default=Path("out/arearef"), help="Euler area reference cache (--tracks area)")
    ap.add_argument("--sets", nargs="*", default=None, help="score only these sets")
    ap.add_argument("--tag", default="", help="suffix for the summary file")
    a = ap.parse_args(argv)
    rows = [json.loads(l) for l in a.samples.read_text().splitlines()]
    if a.sets:
        rows = [r for r in rows if r["set"] in a.sets]
    extra = [tuple(x.split("=", 1)) for x in a.extra]
    sets = load_sets(sorted({r["set"] for r in rows}), extra)
    from cellgym.llm.congest import density
    dens = defaultdict(list)
    for r in rows:
        if r.get("cols") and r.get("status") == "done":
            dens[r["set"]].append(density(r["cols"]))
    refs = {}
    if a.tracks == "area":
        from cellgym.llm import arearef
        for s in sets:
            x = arearef.load(s, a.arearef)
            if x is None:
                print(f"no area reference for {s} ({arearef.path_of(s, a.arearef)}): ratios not reported", file=sys.stderr)
            refs[s] = arearef.areas_nm2(x)
    with ProcessPoolExecutor(a.jobs, initializer=_init, initargs=(sets, a.tracks, refs)) as ex:
        res = list(ex.map(_score, rows, chunksize=8))
    per = defaultdict(lambda: defaultdict(list))
    stages = defaultdict(Counter)
    trk, area = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for s, c, st, wr, t, ar in res:
        per[s][c].append((st, wr))
        stages[s][st] += 1
        if t is not None:
            trk[s].append(t)
        if ar is not None:
            area[s][c].append(ar)
    summ = {}
    order = [x for x, _ in __import__("cellgym.llm.infer", fromlist=["SETS"]).SETS] + [n for n, _ in extra]
    for s in [x for x in order if x in per]:
        cs = per[s]
        n = sum(len(v) for v in cs.values())
        ok = [wr for v in cs.values() for st, wr in v if st == "pass"]
        wrs = [x for x in ok if x]
        summ[s] = {"cells": len(cs), "samples": n, "sample_pass": len(ok) / n,
                   "pass_at_k": sum(any(st == "pass" for st, _ in v) for v in cs.values()) / len(cs),
                   "w_over_euler": sum(wrs) / len(wrs) if wrs else None,
                   "best_of_k_w_over_euler": (lambda b: sum(b) / len(b) if b else None)(
                       [min(wr for st, wr in v if st == "pass" and wr) for v in cs.values()
                        if any(st == "pass" and wr for st, wr in v)]),
                   "tracks_mean": sum(trk[s]) / len(trk[s]) if trk[s] else None,
                   "tracks_hist": dict(sorted(Counter(trk[s]).items())),
                   "area_over_euler": (lambda x: sum(x) / len(x) if x else None)(
                       [y for v in area[s].values() for y in v]),
                   "best_of_k_area_over_euler": (lambda b: sum(b) / len(b) if b else None)(
                       [min(v) for v in area[s].values() if v]),
                   **({"area_ref": "euler-area", "area_ref_cells": sum(c in refs.get(s, {}) for c in cs)}
                      if a.tracks == "area" else {}),
                   "density_mean": sum(dens[s]) / len(dens[s]) if dens[s] else None,
                   "density_le3": sum(d <= 3 for d in dens[s]) / len(dens[s]) if dens[s] else None,
                   "stages": dict(stages[s].most_common(6))}
    out = a.samples.with_suffix((f".{a.tag}" if a.tag else "") + ".summary.json")
    out.write_text(json.dumps(summ, indent=1) + "\n")
    an = "area/Euler-area" if a.tracks == "area" else "area/Euler3"
    print(f"| set | cells | per-sample | pass@k | W/W_euler | best-of-k W/W_euler | tracks mean (hist) | {an} | best-of-k {an if a.tracks == 'area' else 'area'} | top failure |")
    for s, v in summ.items():
        top = next((f"{k} ({c})" for k, c in v["stages"].items() if k != "pass"), "-")
        we = f"{v['w_over_euler']:.3f}" if v["w_over_euler"] else "-"
        bw = f"{v['best_of_k_w_over_euler']:.3f}" if v["best_of_k_w_over_euler"] else "-"
        f3 = lambda x: f"{x:.3f}" if x else "-"
        th = f"{v['tracks_mean']:.2f} {v['tracks_hist']}" if v["tracks_mean"] else "-"
        print(f"| {s} | {v['cells']} | {100 * v['sample_pass']:.1f} | {100 * v['pass_at_k']:.1f} | {we} | {bw} | {th} | "
              f"{f3(v['area_over_euler'])} | {f3(v['best_of_k_area_over_euler'])} | {top} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
