"""Euler area references: the placement of the Euler label, routed at its smallest-area channel.

    python -m cellgym.llm.arearef [--sets val ood10_all ... | --all] [--corpus] [--extra name=dir:part]
                                  [--out out/arearef] [--jobs 14] [--t-max 8]

The width reference of every cell is one Euler plan at the fixed 3-track channel. A sample scored
with a free channel-track count (cellgym.baselines.common.route_min_area) needs a reference with
the same freedom: the columns of that same Euler plan, re-routed by route_min_area and checked by
the repo pipeline. One cache file per source:

    out/arearef/<set>.json, out/arearef/corpus.json
    {cell_id: {"area_um2", "tracks", "width_nm", "height_nm"} | null}     null: no track count routes

Reference plan per source: benchmark sets -> out/plans/<id>.plan (the plan behind
bench/euler_widths_v0.json); devset-style sets -> <dir>/plans/<id>.plan (the plan behind
w_euler_nm in index.json); corpus -> the narrowest plan of the cell in plans.jsonl (the one
cellgym.llm.rl.corpus_cells takes the width from). Sets without stored plans (probes) get an
empty cache. Resumable: cells already in the file are kept.

Sets without stored plans (the probes):

    python -m cellgym.llm.arearef --sets stage3 --extra stage3=out/probe/stage3:all --search --heur
                                  [--timeout 300] [--mem-mb 1500] [--retry-null] [--heur-n 400] [--heur-secs 60]

--search runs the Euler placement search (cellgym.baselines.euler) per cell in its own process
under a hard time and memory limit, routes each of its (up to 4) minimal-break placements with
route_min_area and keeps the smallest area; the plan goes to <dir>/plans/<id>.plan. The search is
exponential in the devices per side: a cell that does not finish is null in <set>.json, with the
reason in <set>.status.json (timeout | memory | no-routing | unpaired). --heur writes a SECOND-TIER
reference to a separate file, <set>.heur.json (+ heur_plans/<set>/<id>.plan): the smallest
route_min_area area over random gate-preferring placements of cellgym.llm.fmt.Placer. It is an
upper bound from a weak placer, not an Euler reference, and nothing reads it by default: load()
returns it only with kind="heur" / "euler+heur" (Euler where it exists, heuristic elsewhere) or
with the environment variable CELLGYM_AREAREF set to one of those (how to point
`cellgym.llm.score --tracks area` at it; the summary still labels the reference "euler-area").
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from cellgym.llm import fmt as F

OUT = Path("out/arearef")
CORPUS = "corpus"
BENCH_PLANS = Path("out/plans")


ENV_KIND = "CELLGYM_AREAREF"      # which reference load() returns: euler (default) | heur | euler+heur
KINDS = ("euler", "heur", "euler+heur")


def area_key(rec: dict) -> tuple:
    """Order of reference records: smaller area first, then fewer tracks."""
    return (rec["width_nm"] * rec["height_nm"], rec["tracks"])


def pick_best(recs: list) -> int | None:
    """Index of the smallest-area record (ties: fewer tracks, then the earlier one); None if all are None."""
    best = None
    for i, r in enumerate(recs):
        if r is not None and (best is None or area_key(r) < area_key(recs[best])):
            best = i
    return best


def route_rec(cols, nl, t_max: int = 8):
    """(reference record, plan text) of one placement routed at its smallest-area channel and
    checked by the repo pipeline; None if no track count routes or the plan does not pass."""
    from cellgym.baselines.common import BaselineError, route_min_area
    from cellgym.bench.evaluate import evaluate
    try:
        rt = route_min_area([[tuple(c) for c in cols]], nl, t_max=t_max)
        ev = evaluate(rt.text, nl)
    except BaselineError:
        return None
    except Exception:  # noqa: BLE001
        return None
    if ev.stage != "pass":
        return None
    return {"area_um2": round(ev.width_nm * ev.height_nm / 1e6, 6), "tracks": rt.tracks,
            "width_nm": ev.width_nm, "height_nm": ev.height_nm}, rt.text


def area_ref(job):
    """(cell id, reference record or None) for job = (cell id, netlist, columns, t_max)."""
    cid, nl, cols, t_max = job
    r = route_rec(cols, nl, t_max)
    return cid, (r[0] if r else None)


def corpus_placements(plans: Path) -> dict[str, tuple[dict, list]]:
    """{cell id: (netlist, columns of its narrowest Euler plan)}: the choice of rl.corpus_cells."""
    best = {}
    with open(plans, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            w = round(r["width_um"] * 1000)
            c = best.get(r["id"])
            if c is None or w < c[0]:
                best[r["id"]] = (w, r["netlist"], F.plan_columns(r["plan"]))
    return {c: (nl, cols) for c, (_, nl, cols) in best.items()}


def set_placements(name: str, extra=()) -> dict[str, tuple[dict, list]]:
    """{cell id: (netlist, columns of the stored Euler plan)}; cells without a stored plan are left out."""
    from cellgym.llm.infer import SETS, load_sets
    spec = dict(list(SETS) + list(extra))[name]
    src = spec.split(":")[0]
    pdir = BENCH_PLANS if src == "bench" else Path(src) / "plans"
    cells, _ = load_sets([name], extra)[name]
    out = {}
    for cid, nl in cells.items():
        p = pdir / f"{cid}.plan"
        if p.exists():
            out[cid] = (nl, F.plan_columns(p.read_text()))
    return out


def kind_of(kind: str | None = None) -> str:
    """The reference kind: the argument, else $CELLGYM_AREAREF, else "euler"."""
    k = kind or os.environ.get(ENV_KIND) or "euler"
    if k not in KINDS:
        raise ValueError(f"area reference kind {k!r}: one of {KINDS}")
    return k


def path_of(name: str, out: Path = OUT, kind: str | None = None) -> Path:
    """Cache file of one source: <name>.json (Euler), <name>.heur.json (kind "heur")."""
    return out / (f"{name}.heur.json" if kind_of(kind) == "heur" else f"{name}.json")


def merge_refs(euler: dict | None, heur: dict | None) -> dict | None:
    """Euler records, with the cells Euler has none for filled from the heuristic file."""
    if euler is None and heur is None:
        return None
    out = dict(heur or {})
    out.update({c: v for c, v in (euler or {}).items() if v or c not in out})
    return out


def load(name: str, out: Path = OUT, kind: str | None = None) -> dict | None:
    """The cache of one source, or None when it was never computed. kind (default: the environment
    variable CELLGYM_AREAREF, else "euler"): "euler" -> <name>.json; "heur" -> the second-tier
    <name>.heur.json; "euler+heur" -> Euler where it has a record, the heuristic one elsewhere."""
    k = kind_of(kind)
    if k == "euler+heur":
        return merge_refs(load(name, out, "euler"), load(name, out, "heur"))
    p = path_of(name, out, k)
    return json.loads(p.read_text()) if p.exists() else None


def areas_nm2(refs: dict | None) -> dict[str, int]:
    """{cell id: reference area in nm^2} of the cells that have one."""
    return {c: v["width_nm"] * v["height_nm"] for c, v in (refs or {}).items() if v}


def _save(p: Path, refs: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(refs, sort_keys=True) + "\n")
    os.replace(tmp, p)


def build(name: str, items: dict[str, tuple[dict, list]], pool, out: Path = OUT, t_max: int = 8,
          save_every: int = 2000) -> dict:
    """Compute the missing references of one source through `pool` (an executor); returns the cache."""
    p = path_of(name, out, "euler")
    refs = load(name, out, "euler") or {}
    jobs = [(c, nl, cols, t_max) for c, (nl, cols) in sorted(items.items()) if c not in refs]
    n = 0
    for cid, rec in pool.map(area_ref, jobs, chunksize=max(1, min(16, len(jobs) // 256))):
        refs[cid] = rec
        n += 1
        if n % save_every == 0:
            _save(p, refs)
    if jobs or not p.exists():
        _save(p, refs)
    return refs


# -- sets without stored plans: Euler search under a hard limit, heuristic second tier ---------

def _child(fn_name: str, job, conn) -> None:
    """Process body: run a worker, piping ("placed"|"best"|"info"|"done", data) messages."""
    def send(kind, data=None):
        conn.send((kind, data))
    try:
        status = globals()[fn_name](job, send)
    except MemoryError:
        status = "memory"
    except RecursionError:
        status = "crash"
    send("done", status)
    conn.close()


def run_limited(fn_name: str, jobs: list, n_jobs: int, timeout: float, route_timeout: float,
                mem_mb: int = 0, floor_mb: int = 0):
    """Run worker `fn_name` on every job (job[0] = cell id), each in its own process, n_jobs at a
    time; yields (cell id, payload or None, status, seconds). A worker is killed `timeout` seconds
    after its start unless it reported "placed", and `route_timeout` seconds after that report;
    with psutil, also when its memory passes mem_mb, or (the largest one) when the free memory of
    the machine falls under floor_mb. A killed worker keeps the last payload it sent: status
    "timeout"/"memory" with no payload means no reference."""
    import multiprocessing as mp
    from multiprocessing.connection import wait
    try:
        import psutil
    except ImportError:
        psutil = None
    ctx = mp.get_context("spawn")
    todo = list(reversed(jobs))
    live = {}      # conn -> state

    def finish(conn, status):
        st = live.pop(conn)
        if st["proc"].is_alive():
            st["proc"].terminate()
        st["proc"].join()
        conn.close()
        return st["cid"], st["payload"], status, round(time.time() - st["t0"], 1)

    while todo or live:
        while todo and len(live) < n_jobs:
            job = todo.pop()
            rx, tx = ctx.Pipe(duplex=False)
            proc = ctx.Process(target=_child, args=(fn_name, job, tx), daemon=True)
            proc.start()
            tx.close()
            now = time.time()
            live[rx] = {"cid": job[0], "proc": proc, "t0": now, "deadline": now + timeout, "payload": None}
        for conn in wait(list(live), timeout=0.5):
            st = live[conn]
            try:
                kind, data = conn.recv()
            except (EOFError, OSError):
                yield finish(conn, "crash")
                continue
            if kind == "placed":
                st["deadline"] = time.time() + route_timeout
            elif kind == "best":
                st["payload"] = data
            elif kind == "info" and st["payload"] is not None:
                st["payload"].update(data)
            elif kind == "done":
                yield finish(conn, data)
        now = time.time()
        for conn in [c for c, st in live.items() if now > st["deadline"]]:
            yield finish(conn, "timeout")
        if psutil and live and (mem_mb or floor_mb):
            rss = {}
            for conn, st in live.items():
                try:
                    rss[conn] = psutil.Process(st["proc"].pid).memory_info().rss >> 20
                except psutil.Error:
                    pass
            over = [c for c, m in rss.items() if mem_mb and m > mem_mb]
            if not over and floor_mb and rss and psutil.virtual_memory().available >> 20 < floor_mb:
                big = max(rss, key=rss.get)
                over = [big] if rss[big] > 200 else []
            for conn in over:
                yield finish(conn, "memory")


def euler_placements(nl: dict, limit: int | None = None) -> list[list[tuple]]:
    """Columns of the minimal-break placements of the Euler baseline (the ones euler_solve routes
    at 3 tracks), without routing. Raises BaselineError("unpaired") like the baseline."""
    from cellgym.baselines import euler as E
    return [E._columns(s) for s in E._Placer(nl).sequences(limit or E.MAX_PLACEMENTS)]


def _euler_worker(job, send) -> str:
    """Euler search, then every placement at its smallest-area channel; reports the smallest."""
    from cellgym.baselines.common import BaselineError
    cid, nl, t_max = job
    try:
        places = euler_placements(nl)
    except BaselineError as e:
        return e.reason
    send("placed", len(places))
    recs, best = [], None
    for cols in places:
        r = route_rec(cols, nl, t_max)
        recs.append(r[0] if r else None)
        i = pick_best(recs)
        if r and i == len(recs) - 1 and i != best:
            best = i
            send("best", {**r[0], "placement": i, "placements": len(places), "plan": r[1]})
    return "ok" if best is not None else "no-routing"


def _gates_after(p, a: str, b: str) -> set:
    gn = {g for (g, t), n in p.left["nmos"].items() if n > 0 and a in t}
    gp = {g for (g, t), n in p.left["pmos"].items() if n > 0 and b in t}
    return gn & gp


def _continues(p, c: tuple) -> bool:
    """Would the sd column c (legal now) leave a gate column legal right after it, or finish the cell?"""
    if not (p.cols and p.cols[-1][0] == "g"):
        return bool(_gates_after(p, c[1], c[2]))
    g = p.cols[-1][1]
    keys = [("nmos", (g, frozenset((p.cols[-2][1], c[1])))), ("pmos", (g, frozenset((p.cols[-2][2], c[2]))))]
    for kind, k in keys:
        p.left[kind][k] -= 1
    try:
        return bool(_gates_after(p, c[1], c[2])) or not (+p.left["nmos"] or +p.left["pmos"])
    finally:
        for kind, k in keys:
            p.left[kind][k] += 1


def gate_pref_placement(nl: dict, rng) -> list[tuple] | None:
    """One random legal placement of cellgym.llm.fmt.Placer that avoids diffusion breaks greedily:
    a gate column whenever one is legal; after a gate, an sd that lets a gate follow (or ends the
    cell) when there is one; a break column only when no gate is legal, and then one a gate can
    follow. None if the walk gets stuck (unpaired gates)."""
    p = F.Placer(nl)
    limit = 6 * len(nl["devices"]) + 8
    while not p.done():
        legal = p.legal()
        if not legal or len(p.cols) > limit:
            return None
        gates = [c for c in legal if c[0] == "g"]
        after_gate = bool(p.cols) and p.cols[-1][0] == "g"
        if gates:
            c = rng.choice(gates)
        else:
            good = [c for c in legal if _continues(p, c)]
            if not good and not after_gate:
                return None
            c = rng.choice(good or legal)
        if not p.step(c):
            return None
    return list(p.cols)


def heur_candidates(nl: dict, n: int, rng) -> list[list[tuple]]:
    """The distinct placements among n gate_pref_placement draws, fewest columns first (stable)."""
    seen = {}
    for _ in range(n):
        cols = gate_pref_placement(nl, rng)
        if cols:
            seen.setdefault(tuple(cols), None)
    return [list(c) for c in sorted(seen, key=len)]


def _heur_worker(job, send) -> str:
    """Heuristic reference: the smallest route_min_area area over random gate-preferring placements."""
    import random
    import zlib
    cid, nl, t_max, n, secs, seed = job
    cands = heur_candidates(nl, n, random.Random(zlib.crc32(cid.encode()) ^ seed))
    send("placed", len(cands))
    t0, best, routed = time.time(), None, 0
    for cols in cands:
        if routed and time.time() - t0 > secs:
            break
        r = route_rec(cols, nl, t_max)
        routed += 1
        if r and (best is None or area_key(r[0]) < area_key(best)):
            best = r[0]
            send("best", {**best, "ref": "heur", "columns": len(cols), "candidates": len(cands),
                          "routed": routed, "plan": r[1]})
    send("info", {"routed": routed})
    return "ok" if best else "no-routing"


def status_path(name: str, out: Path = OUT) -> Path:
    return out / f"{name}.status.json"


def search(name: str, cells: dict[str, dict], plan_dir: Path | None, out: Path = OUT, *, jobs: int = 12,
           timeout: float = 300, mem_mb: int = 1500, floor_mb: int = 500, t_max: int = 8,
           retry_null: bool = False) -> dict:
    """Euler references of the cells of a set that have neither a cache record nor a stored plan:
    Euler search (hard time and memory limit per cell), each of its placements routed by
    route_min_area, the smallest area kept. A cell whose search does not finish is recorded null
    (out/arearef/<name>.status.json says why). The reference plan goes to plan_dir/<id>.plan."""
    p = path_of(name, out, "euler")
    refs = load(name, out, "euler") or {}
    sp = status_path(name, out)
    status = json.loads(sp.read_text()) if sp.exists() else {}
    todo = [(c, nl, t_max) for c, nl in sorted(cells.items())
            if c not in refs or (retry_null and refs[c] is None)]
    for cid, rec, st, secs in run_limited("_euler_worker", todo, jobs, timeout, timeout, mem_mb, floor_mb):
        plan = rec.pop("plan", None) if rec else None
        refs[cid] = rec
        status[cid] = {"status": st if rec is None or st == "ok" else f"partial-{st}", "secs": secs}
        if plan and plan_dir is not None:
            plan_dir.mkdir(parents=True, exist_ok=True)
            (plan_dir / f"{cid}.plan").write_text(plan, newline="\n")
        _save(p, refs)
        _save(sp, status)
    return refs


def heur(name: str, cells: dict[str, dict], out: Path = OUT, *, jobs: int = 12, n: int = 400,
         secs: float = 60, t_max: int = 8, seed: int = 0, mem_mb: int = 1500, floor_mb: int = 500) -> dict:
    """Second-tier references out/arearef/<name>.heur.json (never read unless asked for, see load):
    per cell the smallest route_min_area area over the distinct placements of n gate-preferring
    random walks, fewest columns first, routing for at most `secs` seconds. Its plan goes to
    out/arearef/heur_plans/<name>/<id>.plan."""
    p = path_of(name, out, "heur")
    refs = load(name, out, "heur") or {}
    todo = [(c, nl, t_max, n, secs, seed) for c, nl in sorted(cells.items()) if c not in refs]
    for cid, rec, _, _ in run_limited("_heur_worker", todo, jobs, 120, 3 * secs + 120, mem_mb, floor_mb):
        plan = rec.pop("plan", None) if rec else None
        refs[cid] = rec
        if plan:
            d = out / "heur_plans" / name
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{cid}.plan").write_text(plan, newline="\n")
        _save(p, refs)
    return refs


def mean_area(refs: dict | None) -> float | None:
    ok = [v["area_um2"] for v in (refs or {}).values() if v]
    return round(sum(ok) / len(ok), 3) if ok else None


def describe_search(name: str, out: Path = OUT) -> str:
    refs = load(name, out, "euler") or {}
    sp = status_path(name, out)
    st = Counter(v["status"] for c, v in (json.loads(sp.read_text()) if sp.exists() else {}).items()
                 if refs.get(c) is None)
    s = f"{describe(refs)}  null-by-status {dict(sorted(st.items()))}  mean-area {mean_area(refs)}"
    h = load(name, out, "heur")
    if h is not None:
        both = [h[c]["area_um2"] / v["area_um2"] for c, v in refs.items() if v and h.get(c)]
        s += (f"  | heur: refs {sum(1 for v in h.values() if v)}/{len(h)}  mean-area {mean_area(h)}"
              f"  heur/euler {round(sum(both) / len(both), 3) if both else None} over {len(both)}")
    return s


def describe(refs: dict) -> str:
    ok = [v for v in refs.values() if v]
    hist = dict(sorted(Counter(v["tracks"] for v in ok).items()))
    return f"cells {len(refs)}  null {len(refs) - len(ok)}  tracks {hist}"


def main(argv=None) -> int:
    from cellgym.llm.infer import SETS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="*", default=[], help="evaluation sets (names of cellgym.llm.infer.SETS or --extra)")
    ap.add_argument("--all", action="store_true", help="every set of cellgym.llm.infer.SETS and the corpus")
    ap.add_argument("--corpus", action="store_true")
    ap.add_argument("--plans", type=Path, default=Path("out/synth5/plans.jsonl"))
    ap.add_argument("--extra", nargs="*", default=[], help="name=dir:part devset-style sets")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--t-max", type=int, default=8)
    ap.add_argument("--search", action="store_true",
                    help="cells without a stored plan: run the Euler search (per-cell limits), store its plan")
    ap.add_argument("--timeout", type=float, default=300, help="--search: seconds per cell (search; again for routing)")
    ap.add_argument("--mem-mb", type=int, default=1500, help="--search/--heur: memory limit per cell process")
    ap.add_argument("--floor-mb", type=int, default=500, help="kill the largest cell process under this much free memory")
    ap.add_argument("--retry-null", action="store_true", help="--search: try the null cells of the cache again")
    ap.add_argument("--heur", action="store_true", help="also write the second-tier <set>.heur.json")
    ap.add_argument("--heur-n", type=int, default=400, help="--heur: random placements per cell")
    ap.add_argument("--heur-secs", type=float, default=60, help="--heur: routing seconds per cell")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    extra = [tuple(x.split("=", 1)) for x in a.extra]
    names = list(dict.fromkeys(list(a.sets) + ([n for n, _ in SETS] if a.all else [])))
    rc = 0
    with ProcessPoolExecutor(a.jobs) as pool:
        for name in names + ([CORPUS] if a.corpus or a.all else []):
            t0 = time.time()
            try:
                items = corpus_placements(a.plans) if name == CORPUS else set_placements(name, extra)
            except (OSError, KeyError) as e:
                print(f"{name}: skipped ({type(e).__name__}: {e})", flush=True)
                rc = 1
                continue
            had = len(load(name, a.out, "euler") or {})
            refs = build(name, items, pool, a.out, a.t_max)
            print(f"{name}: {describe(refs)}  (computed {len(refs) - had}, {time.time() - t0:.1f}s)", flush=True)
            if name == CORPUS or not (a.search or a.heur):
                continue
            from cellgym.llm.infer import load_sets
            src = dict(list(SETS) + extra)[name].split(":")[0]
            cells = load_sets([name], extra)[name][0]
            if a.search:
                search(name, cells, None if src == "bench" else Path(src) / "plans", a.out, jobs=a.jobs,
                       timeout=a.timeout, mem_mb=a.mem_mb, floor_mb=a.floor_mb, t_max=a.t_max,
                       retry_null=a.retry_null)
            if a.heur:
                heur(name, cells, a.out, jobs=a.jobs, n=a.heur_n, secs=a.heur_secs, t_max=a.t_max,
                     seed=a.seed, mem_mb=a.mem_mb, floor_mb=a.floor_mb)
            print(f"{name}: {describe_search(name, a.out)}  ({time.time() - t0:.1f}s total)", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
