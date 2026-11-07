"""RL environment: netlist prompt -> sampled plan ids -> reward from the repo's own checker.

    env = make_env("train", synth_plans="out/synth/plans.jsonl")
    sampler = CellSampler(env, seed=0)
    cids = sampler.next_groups(n_groups=8)            # distinct cells; each gets group_size samples in grpo
    infos = env.reward_batch(cids_per_sample, samples, jobs=4)

Reward (existing cellgym.synth.eval.score_sample -> cellgym.bench.score.reward): 0 for any failing plan,
1 + (W_euler - W)/W_euler for a passing one, floored at MIN_PASS_REWARD so a pass always beats a fail.
Held-out splits (val, test_id, test_ood) are refused unless allow_heldout=True.
"""

from __future__ import annotations

import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from cellgym.bench import reference, splits
from cellgym.bench.common import EULER_WIDTHS_FILE, SPLITS_FILE, load_cells
from cellgym.synth import tokens as T
from cellgym.synth.eval import score_sample

MIN_PASS_REWARD = 0.01
HELD_OUT = ("val", "test_id", "test_ood")
DEFAULT_CELLS = Path("out/cells")
DEFAULT_CACHE = Path("out/rl/w_euler_cache.json")


@dataclass
class RewardInfo:
    reward: float
    stage: str
    passed: bool
    width_um: float | None
    w_ratio: float | None      # W / W_euler for passing plans


def _score(args) -> RewardInfo:
    netlist, ids, w_euler_nm = args
    s = score_sample(netlist, ids, w_euler_nm)
    r = s["reward"] or 0.0
    if s["pass"]:
        r = max(r, MIN_PASS_REWARD)
    w = s["width_nm"]
    return RewardInfo(float(r), s["stage"], bool(s["pass"]), w / 1000 if w else None, s["w_over_euler"])


class PlanEnv:
    def __init__(self, cells: dict[str, dict], w_euler: dict[str, int] | None = None,
                 split_of: dict[str, str] | None = None, allow_heldout: bool = False):
        if split_of is None:
            split_of = splits.split_of(splits.load(SPLITS_FILE))
        if w_euler is None:
            w_euler = {c: v["width_nm"] for c, v in reference.load(EULER_WIDTHS_FILE)["cells"].items()}
        bad = sorted(c for c in cells if split_of.get(c) in HELD_OUT)
        if bad and not allow_heldout:
            raise ValueError(f"{len(bad)} held-out cells refused (e.g. {bad[0]} in {split_of[bad[0]]}); "
                             "pass allow_heldout=True only for evaluation")
        missing = sorted(c for c in cells if c not in w_euler)
        if missing:
            raise ValueError(f"no W_euler for {len(missing)} cells (e.g. {missing[0]})")
        self.cells, self.w_euler, self.split_of = dict(cells), dict(w_euler), split_of
        self._prompts: dict[str, list[int]] = {}
        self._pool: ProcessPoolExecutor | None = None
        self._pool_jobs = 0

    def ids(self) -> list[str]:
        return sorted(self.cells)

    def prompt_ids(self, cell_id: str) -> list[int]:
        if cell_id not in self._prompts:
            self._prompts[cell_id] = T.ids(T.encode_netlist(self.cells[cell_id]))
        return list(self._prompts[cell_id])

    def _task(self, cid, ids):
        return (self.cells[cid], list(ids), self.w_euler[cid])

    def reward(self, cell_id: str, ids: list[int]) -> RewardInfo:
        return _score(self._task(cell_id, ids))

    def reward_batch(self, cell_ids: list[str], samples: list[list[int]], jobs: int = 1) -> list[RewardInfo]:
        assert len(cell_ids) == len(samples)
        tasks = [self._task(c, s) for c, s in zip(cell_ids, samples)]
        t0 = time.time()
        if jobs <= 1 or len(tasks) < 2 * jobs:
            out = [_score(t) for t in tasks]
        else:
            if self._pool is None or self._pool_jobs != jobs:
                self.close()
                self._pool, self._pool_jobs = ProcessPoolExecutor(jobs), jobs
            out = list(self._pool.map(_score, tasks, chunksize=max(1, len(tasks) // (4 * jobs))))
        self.last_throughput = len(tasks) / max(time.time() - t0, 1e-9)    # samples / s
        return out

    def close(self):
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None


def _euler_nm(netlist: dict) -> int:
    from cellgym.baselines.euler import euler_solve
    return round(euler_solve(netlist).width_um * 1000)


def load_w_euler(cells: dict[str, dict], cache: Path | None = DEFAULT_CACHE) -> dict[str, int]:
    """Benchmark W_euler where available, else euler_solve once per cell, cached on disk as JSON."""
    known = {c: v["width_nm"] for c, v in reference.load(EULER_WIDTHS_FILE)["cells"].items()}
    cached = json.loads(Path(cache).read_text()) if cache and Path(cache).exists() else {}
    out, dirty = {}, False
    for cid, nl in cells.items():
        if cid in known:
            out[cid] = known[cid]
        else:
            if cid not in cached:
                cached[cid] = _euler_nm(nl)
                dirty = True
            out[cid] = cached[cid]
    if dirty and cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        Path(cache).write_text(json.dumps(cached, indent=0, sort_keys=True) + "\n")
    return out


def read_synth_netlists(path: Path) -> dict[str, dict]:
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out.setdefault(r["id"], r["netlist"])
    return out


def make_env(split: str = "train", synth_plans: str | Path | None = None, cells_dir: Path = DEFAULT_CELLS,
             cache: Path | None = DEFAULT_CACHE, allow_heldout: bool = False) -> PlanEnv:
    split_of = splits.split_of(splits.load(SPLITS_FILE))
    if split in HELD_OUT and not allow_heldout:
        raise ValueError(f"split {split!r} is held out; pass allow_heldout=True")
    all_cells = load_cells(cells_dir)
    cells = {c: all_cells[c] for c in sorted(split_of) if split_of[c] == split and c in all_cells}
    if synth_plans:
        for cid, nl in read_synth_netlists(Path(synth_plans)).items():
            if split_of.get(cid) in HELD_OUT and not allow_heldout:
                raise ValueError(f"synthetic cell {cid} is in held-out split {split_of[cid]}")
            cells.setdefault(cid, nl)
    return PlanEnv(cells, load_w_euler(cells, cache), split_of, allow_heldout)


class CellSampler:
    """Seeded draws of cells for GRPO. weights: cell id -> weight (default 1); set_weights() is the
    curriculum hook. Each draw is without replacement within a step, so groups are distinct cells."""

    def __init__(self, env_or_ids, seed: int = 0, weights: dict[str, float] | None = None):
        self.ids = env_or_ids.ids() if hasattr(env_or_ids, "ids") else sorted(env_or_ids)
        self.weights = dict(weights or {})
        self.rng = random.Random(seed)

    def set_weights(self, weights: dict[str, float]):
        self.weights = dict(weights)

    def next_groups(self, n_groups: int) -> list[str]:
        pool = list(self.ids)
        w = [max(self.weights.get(c, 1.0), 0.0) for c in pool]
        out = []
        for _ in range(min(n_groups, len(pool))):
            if sum(w) <= 0:
                w = [1.0] * len(pool)
            i = self.rng.choices(range(len(pool)), w)[0]
            out.append(pool.pop(i))
            w.pop(i)
        return out

    @staticmethod
    def expand(cell_ids: list[str], group_size: int) -> list[str]:
        """One entry per sample, samples of a cell adjacent: [a,a,a,a,b,b,b,b,...]."""
        return [c for c in cell_ids for _ in range(group_size)]
