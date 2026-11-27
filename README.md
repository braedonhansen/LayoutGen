# CellGym (LayoutGen)

CellGym is an environment for learning standard-cell layout in SKY130. It
enumerates static-CMOS transistor netlists (323 "core" cells, 1 to 6 inputs),
defines a small plan language in which a model chooses the topology (column
order, diffusion sharing, contacts, route tracks), renders plans to DRC-clean
geometry, and scores them with a fast checker (DRC subset + LVS, ~7 ms) that is
audited against full ground truth (KLayout DRC + Magic DRC + Netgen LVS). The
reward is the exact rendered cell width against an Euler-path baseline.

On top of the environment sits a learned placement policy (`cellgym/llm`): a
0.5B-parameter language model that places a cell one column at a time, with a
classical router finishing the layout. Results, limits and how to reproduce
them are in [Learned placement model](#learned-placement-model) below.

## Install

Python 3.10+, standard library only for everything except ground truth. DRC/LVS
needs the WSL2 environment in [`env/SETUP.md`](env/SETUP.md) (conda env
`cellgym` with Magic and Netgen, KLayout, sky130A PDK pinned to open_pdks
`792ef67`). Run the commands below from the repo root inside that env.

## Quickstart

```bash
python -m cellgym.netlists.generate --out out/cells                 # 1. netlists (323 cells)
python -m cellgym.netlists.schematic --out out/schematics           # 2. schematics (SVG gallery)
python -m cellgym.baselines.euler --out out/plans                   # 3. plans: Euler baseline
python -m cellgym.baselines.naive --out out/plans_naive             #    plans: naive baseline
python -m cellgym.groundtruth.verify out/plans/cg_3i3d_29e68bd3.plan out/cells/cg_3i3d_29e68bd3.json
                                                                    # 4. render + full DRC/LVS
python -m cellgym.check out/plans --cells out/cells                 # 5. fast check
python -m cellgym.bench.score out/plans --name euler --confirm      # 6. score (+ ground truth)
python -m cellgym.viewer.static out/plans --out out/gallery/euler   # 7. layout gallery (index.html)
```

`python -m pytest tests` runs the test suite (EDA tests skip without the
tools; `--runslow` adds the all-cell sweeps).

## Baseline results (v0, ground-truth confirmed)

| split | baseline | n | pass@1 (GT) | mean sites | mean W/W_euler | mean reward | mean breaks |
|---|---|---|---|---|---|---|---|
| test_id | naive | 25 | 1.000 | 14.40 | 1.578 | 0.422 | 5.00 |
| test_id | euler | 25 | 1.000 | 9.32 | 1.000 | 1.000 | 0.32 |
| test_ood | naive | 172 | 1.000 | 16.91 | 1.581 | 0.419 | 6.22 |
| test_ood | euler | 172 | 1.000 | 10.90 | 1.000 | 1.000 | 0.44 |
| all | naive | 323 | 1.000 | 15.91 | 1.588 | 0.412 | 5.72 |
| all | euler | 323 | 1.000 | 10.21 | 1.000 | 1.000 | 0.33 |

W/W_euler and reward use exact widths from `bench/euler_widths_v0.json`;
reward = 1 + (W_euler - W) / W_euler for a passing plan, 0 otherwise. Full
table with medians, breaks and timing: [`bench/results_v0.md`](bench/results_v0.md).
Against hand-made sky130_fd_sc_hd cells matched by function:
[`bench/human_v0.md`](bench/human_v0.md) (20 core cells match hd `_1` cells; mean width hd 5.45 sites, Euler 6.40, naive 8.70, with hd at 2.72 um height and ours at 3.06 to 5.10 um).

## Learned placement model

A Qwen2.5-0.5B model fine-tuned to choose the next column of a placement from a
text description of the cell and of what is placed so far. Deployment recipe:
sample 32 placements at T = 0.7, keep those with the fewest columns, route each
at its area-optimal track count, keep the smallest layout the fast checker
passes. Only placement is learned; routing is the classical router the Euler
baseline also uses.

Trained on generated cells with at most 5 inputs, at most 9 devices per side,
series stack depth at most 4, 1 to 2 stages, 1 output. Tested on cells it was
not trained on, most of them outside those limits. Area / Euler area (below 1
is smaller than the baseline), two training seeds, pass rate 100% on every row:

| tested on | cells | area / Euler area, seed 1 / seed 2 |
|---|---|---|
| val, test_id (inside the training limits) | 15, 25 | 1.004 / 0.989, 0.986 / 0.986 |
| 6 / 7 / 8 / 9 / 10 inputs | 167 / 205 / 120 / 200 / 145 | 0.984 to 0.995 / 0.983 to 0.993 |
| stack depth 5 / 6 | 75 / 174 | 0.983, 0.990 / 0.979, 0.984 |
| 10 / 11 / 12 devices per side | 87 / 19 / 15 | 0.996, 0.998, 1.003 / 0.991, 1.001, 0.997 |
| 2 stages / 3 stages | 30 / 30 | 0.995, 1.014 / 1.001, 1.012 |

Columns against the exact optimum: on the 1,234 of these single-stage cells
where an exact search finishes (all but the 15 at 12 per side), the deployed
placement of seed 1 has the minimum possible number of columns on 1,177
(95.4%), and on 207 of 207 inside the training limits and at 6 inputs.

Beyond the reach of the baseline's search (8 to 16 inputs at 14 to 24 devices
per side, 70 cells, where the Euler search did not finish in 60 s per cell at 14 per side)
every cell is placed, routed and passes; the column count is within 2.4 to 4.1%
of a lower bound that needs no search, for both seeds.

Ground truth: all 4,137 layouts the recipe deploys on these sets for three
models (both seeds of the release model and its predecessor) are clean under KLayout
DRC, Magic DRC and Netgen LVS, with no disagreement with the fast checker.

What this does not show:

- The model matches the Euler baseline on area (within about 2% either way on
  every set); it does not clearly beat it, and on columns there is nothing
  left to win on these sets. What it adds is that it keeps working where the
  search stops.
- A single greedy placement is 2 to 20% larger than the baseline (3 to 5% on
  the 6-to-10-input sets); the result above needs 32 samples (about 10 s per
  cell on an RTX 3090 at 14 to 24 devices per side).
- Hand-made sky130_fd_sc_hd cells are narrower and shorter than both the model
  and the baseline ([`bench/human_v0.md`](bench/human_v0.md)).
- Cells are single-output static CMOS without taps; no timing, power or
  parasitics are measured.

Reproduce (24 GB GPU; training about 3.5 h, evaluation about 3 h; the script
has not yet been run end to end in one go): `bash cellgym/llm/reproduce.sh`.
Weights are not published yet. Every run, with what it was trained on and
tested on: [`bench/llm_runs.md`](bench/llm_runs.md).

## Benchmark splits

[`bench/splits_v0.json`](bench/splits_v0.json) (committed, hashed, one entry
per cell with its netlist sha256):

| split | cells | contents |
|---|---|---|
| train | 111 | <=5-input cells, stratified by input count and read-once |
| val | 15 | same strata |
| test_id | 25 | same strata |
| test_ood | 172 | all 160 six-input cells + 12 anchor cells: every core realization of AOI21, AOI22, OAI21, OAI211, NAND3, NOR3 and NOT MAJ3 (matched by truth table) |

`python -m cellgym.bench.splits --check` regenerates and compares; the scorer
refuses netlists whose sha256 differs from the split file.

## Status

v0.1 = environment + baselines: netlist generator (M1), plan DSL, validator
and renderer (M2), fast checker audited against ground truth (M3), benchmark
splits, scorer, baselines, human reference and static gallery (M4).

Branch `m4-benchmark` adds the learned placement policy (`cellgym/llm`),
channel tracks as a per-plan parameter (`tracks N`, default 3; see
[`cellgym/dsl/SPEC.md`](cellgym/dsl/SPEC.md)) and the area objective against
an Euler reference routed the same way.

Docs: [`cellgym/netlists/README.md`](cellgym/netlists/README.md) (netlists),
[`cellgym/dsl/SPEC.md`](cellgym/dsl/SPEC.md) (plan format and renderer rules).
License: Apache-2.0.
