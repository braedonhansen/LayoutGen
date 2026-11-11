"""Sample plans from a fine-tuned LM with vLLM, for every evaluation set (scored by cellgym.llm.score).

    python -m cellgym.llm.infer --model out/llm/step05/model --mode step --out out/llm/step05/samples.jsonl [--k 8]

mode oneshot: k samples of the whole plan body per cell.
mode step:    k independent placements per cell; each step the model sees Placer.state() and its
              continuation is restricted to Placer.legal() (vLLM guided choice; if this vLLM has no
              guided decoding, sample and retry up to --retries times, an illegal column ends the
              sample). The environment stops a placement when every device is placed and the last
              column is sd, or after 4 x devices + 4 columns.
Output: one json line per (set, cell, sample): oneshot -> "text" (plan body); step -> "cols".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from cellgym.llm import fmt as F

SETS = [("val", "bench:val"), ("test_id", "bench:test_id"), ("ood6_dev", "out/ood6:dev"),
        ("ood7_dev", "out/ood7:dev"), ("ood8_dev", "out/ood8:dev"), ("ood9_all", "out/ood9:all"),
        ("ood10_all", "out/ood10:all"), ("oodstack5_dev", "out/oodstack5:dev"),
        ("oodstack6_all", "out/oodstack6:all"), ("ooddev10_dev", "out/ooddev10:dev"),
        ("ooddev11_dev", "out/ooddev11:dev"), ("ooddev12_all", "out/ooddev12:all"), ("adder", "out/adder:dev")]


def load_sets(names=None, extra=()):
    from cellgym.bench.common import EULER_WIDTHS_FILE, load_cells
    from cellgym.bench.splits import load as load_splits, split_of
    from cellgym.synth.devset import load as load_set
    out = {}
    bench = None
    for name, spec in list(SETS) + list(extra):
        if names and name not in names:
            continue
        src, part = spec.split(":")
        if src == "bench":
            if bench is None:
                cells = load_cells(Path("out/cells"))
                sp = split_of(load_splits())
                w = json.loads(Path(EULER_WIDTHS_FILE).read_text())
                w = w.get("cells", w)
                bench = (cells, sp, w)
            cells, sp, w = bench
            ids = sorted(c for c in cells if sp.get(c) == part)
            out[name] = ({c: cells[c] for c in ids}, {c: _w_nm(w.get(c)) for c in ids})
        else:
            out[name] = load_set(Path(src), part)
    return out


def _w_nm(v):
    if v is None:
        return None
    return v["width_nm"] if isinstance(v, dict) else v


def _guided(choices, temperature, max_tokens):
    from vllm import SamplingParams
    try:
        from vllm.sampling_params import StructuredOutputsParams
        return SamplingParams(temperature=temperature, max_tokens=max_tokens,
                              structured_outputs=StructuredOutputsParams(choice=choices))
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams
        return SamplingParams(temperature=temperature, max_tokens=max_tokens,
                              guided_decoding=GuidedDecodingParams(choice=choices))


def run_step(llm, sets, k, temperature, retries, guided, out):
    from vllm import SamplingParams
    jobs = []      # (set, cid, sample, placer)
    for s, (cells, _) in sets.items():
        for cid, nl in cells.items():
            for j in range(k):
                jobs.append([s, cid, j, F.Placer(nl), 4 * len(nl["devices"]) + 4, "run"])
    rounds = 0
    while True:
        act = [jb for jb in jobs if jb[5] == "run"]
        if not act:
            break
        rounds += 1
        if guided:
            prompts, params = [], []
            for jb in act:
                legal = jb[3].legal()
                if not legal:
                    jb[5] = "stuck"
                    continue
                prompts.append(jb[3].state())
                params.append(_guided([" " + F.col_text(c) for c in legal], temperature, 16))
            live = [jb for jb in act if jb[5] == "run"]
            res = llm.generate(prompts, params, use_tqdm=False) if prompts else []
            for jb, r in zip(live, res):
                c = F.parse_col(r.outputs[0].text)
                if c is None or not jb[3].step(c):
                    jb[5] = "illegal"
        else:
            sp = SamplingParams(temperature=temperature, max_tokens=16, stop=["\n"], n=1)
            pending = list(act)
            for _ in range(retries + 1):
                if not pending:
                    break
                res = llm.generate([jb[3].state() for jb in pending], sp, use_tqdm=False)
                nxt = []
                for jb, r in zip(pending, res):
                    c = F.parse_col(r.outputs[0].text)
                    if c is None or not jb[3].step(c):
                        nxt.append(jb)
                pending = nxt
            for jb in pending:
                jb[5] = "illegal"
        for jb in act:
            if jb[5] != "run":
                continue
            if jb[3].done():
                jb[5] = "done"
            elif len(jb[3].cols) >= jb[4]:
                jb[5] = "cap"
        print(f"round {rounds}: {sum(jb[5] == 'run' for jb in jobs)} running", flush=True)
    with open(out, "w") as f:
        for s, cid, j, p, _, st in jobs:
            f.write(json.dumps({"set": s, "cell": cid, "sample": j, "status": st, "cols": p.cols}) + "\n")


def run_oneshot(llm, sets, k, temperature, max_tokens, out):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=k)
    keys, prompts = [], []
    for s, (cells, _) in sets.items():
        for cid, nl in cells.items():
            keys.append((s, cid))
            prompts.append(F.oneshot_prompt(nl))
    res = llm.generate(prompts, sp)
    with open(out, "w") as f:
        for (s, cid), r in zip(keys, res):
            for j, o in enumerate(r.outputs):
                f.write(json.dumps({"set": s, "cell": cid, "sample": j, "status": o.finish_reason,
                                    "text": o.text}) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["oneshot", "step"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--no-guided", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--sets", nargs="*", default=None)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    a = ap.parse_args(argv)
    from vllm import LLM
    t0 = time.time()
    sets = load_sets(a.sets)
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem, max_model_len=4096,
              enable_prefix_caching=True, seed=0)
    if a.mode == "oneshot":
        run_oneshot(llm, sets, a.k, a.temperature, a.max_tokens, a.out)
    else:
        guided = not a.no_guided
        if guided:
            try:
                _guided([" a"], 1.0, 4)
            except Exception as e:  # noqa: BLE001
                print(f"guided decoding unavailable ({e}); sampling with retries", flush=True)
                guided = False
        run_step(llm, sets, a.k, a.temperature, a.retries, guided, a.out)
    print(f"done in {time.time() - t0:.0f}s -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
