"""Text datasets for LM fine-tuning from a synth plans.jsonl (prompt/target jsonl).

    python -m cellgym.llm.build --plans out/synth5/plans.jsonl --out out/llm/data [--rename 0.9] [--steps-per-plan 4]

Writes {oneshot,step}_{train,val}.jsonl. Val = 5% of topologies (sha256 bucket, as train.py).
Only the given corpus is used: <= 5 inputs, <= 9 devices per side, stack <= 4 for synth5.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from cellgym.llm import fmt as F
from cellgym.synth.train import split_topologies


def build(plans: Path, out: Path, rename: float, steps_per_plan: int, seed: int = 0) -> dict:
    rows = [json.loads(l) for l in plans.read_text().splitlines()]
    _, va = split_topologies([r["topology_id"] for r in rows], 0.05)
    va = set(va)
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    files = {f"{k}_{s}": open(out / f"{k}_{s}.jsonl", "w", newline="\n")
             for k in ("oneshot", "step") for s in ("train", "val")}
    n = {k: 0 for k in files}
    for r in rows:
        split = "val" if r["topology_id"] in va else "train"
        nl, plan = r["netlist"], r["plan"]
        if split == "train" and rng.random() < rename:
            L, N = F.rename_map(nl, rng)
            nl = F.renamed_netlist(nl, L, N)
            plan = F.apply_names(plan, L, N, plan_side=True)
        files[f"oneshot_{split}"].write(json.dumps({"prompt": F.oneshot_prompt(nl), "target": F.plan_body(plan)}) + "\n")
        n[f"oneshot_{split}"] += 1
        for s, t in F.step_examples(nl, plan, rng, steps_per_plan):
            files[f"step_{split}"].write(json.dumps({"prompt": s, "target": " " + t}) + "\n")
            n[f"step_{split}"] += 1
    for f in files.values():
        f.close()
    m = {"plans": str(plans), "rename": rename, "steps_per_plan": steps_per_plan, "counts": n}
    (out / "manifest.json").write_text(json.dumps(m, indent=1) + "\n")
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", type=Path, default=Path("out/synth5/plans.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("out/llm/data"))
    ap.add_argument("--rename", type=float, default=0.9)
    ap.add_argument("--steps-per-plan", type=int, default=4)
    a = ap.parse_args(argv)
    print(json.dumps(build(a.plans, a.out, a.rename, a.steps_per_plan), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
