# CellGym (LayoutGen)

CellGym is an environment for learning standard-cell layout in SKY130. It
enumerates static-CMOS transistor netlists (323 "core" cells, 1 to 6 inputs),
defines a small plan language in which a model chooses the topology (column
order, diffusion sharing, contacts, route tracks), renders plans to DRC-clean
geometry, and scores them with a fast checker (DRC subset + LVS, ~7 ms) that is
audited against full ground truth (KLayout DRC + Magic DRC + Netgen LVS). The
reward is the exact rendered cell width against an Euler-path baseline. v0.1 is
the environment, the benchmark splits and two classical baselines; there is no
learned model yet.

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

v0.1 = environment + baselines, no ML. Done: netlist generator (M1), plan DSL,
validator and renderer (M2), fast checker audited against ground truth (M3),
benchmark splits, scorer, baselines, human reference and static gallery (M4).
Next: a learned plan generator trained against the fast checker.

Docs: [`cellgym/netlists/README.md`](cellgym/netlists/README.md) (netlists),
[`cellgym/dsl/SPEC.md`](cellgym/dsl/SPEC.md) (plan format and renderer rules).
License: Apache-2.0.
