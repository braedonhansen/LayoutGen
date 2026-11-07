# Tiny-model pretraining runs (imitation on synthetic Euler-labeled plans)

Model: GPT 31.8M (d512, 10 layers, 8 heads), vocab 203, trained bf16 on an RTX 3050 4 GB.
Eval: benchmark val + test_id (40 cells), k=8 T=0.8 sampled, plus greedy. All numbers use
grammar-constrained decoding (`--constrained`, parse rate 100%) except run0.
Cell = 2.5 points, so differences of 1-3 cells are noise.

## Runs

| run | corpus | plans / topologies | best val loss | greedy pass@1 | pass@8 | W/W_euler (best) |
|---|---|---|---|---|---|---|
| run0 | out/synth (stack<=3) | 1,825 / 200 | 0.23 | 10% | 25% | 1.05 (unconstrained) |
| run1 | out/synth4 (stack<=4) | 6,467 / 686 | 0.079 | 50% | 67.5% | 1.009 |
| **run2** | out/synth5 (stack<=4, +4 dev, <=9 dev, 6 renames) | 31,184 / 1,934 | 0.031 | **70%** | **95%** | 0.989 |
| run3 | out/synth6 (same topologies, 10 renames, 3 placements) | 39,563 / 1,934 | 0.0305 | 67.5% | 87.5% | 1.009 |

Best checkpoint: `out/synth5/run2/best.pt` (4,000 steps, ~46 min).

## Findings
- Topology count drives generalization (run1 -> run2). More renamings/placements of the same
  topologies (run3) gave no gain.
- Sampled pass@1 (~48-52%) is well below greedy (70%) and pass@8 (95%): the valid plan is
  usually in the distribution but not reliably picked. This is the gap RL should close.
- Constrained decoding fixes syntax only; remaining failures are semantic (coverage,
  diffusion, routing), which the grammar does not check.
- Generation cost: ~2.4 h for run2's corpus at 3 jobs; ~64 min for run3's at 18 jobs when
  reusing the cached topologies. Enumerating >9 devices is a slow serial step.

## Commands (repo root, `PYTHONPATH=.`)
```
python -m cellgym.synth.generate --out DIR --cache out/synth5/topologies.json --max-inputs 5 \
  --max-stack 4 --extra-devices 4 --max-devices 9 --renames 6 --placements 3 --jobs 18
python -m cellgym.synth.tokens DIR/plans.jsonl          # round-trip check
python -m cellgym.synth.train --data DIR/plans.jsonl --out DIR/runN --steps 4000 --device cuda --amp
python -m cellgym.synth.eval --ckpt DIR/runN/best.pt --splits val test_id --k 8 --temperature 0.8 --constrained
python -m cellgym.synth.eval --ckpt DIR/runN/best.pt --splits val test_id --k 1 --temperature 0 --constrained --tag greedy_constrained
```
Training has no resume. Raw results: `out/*/run*/eval_*.json`, `log.jsonl` (gitignored).

## RL (GRPO from run2)

Pre-RL probe of run2 at T=1.0, 8 samples/cell: benchmark train 71% pass (unconstrained) / 74%
(constrained), ~80% of groups mixed; synthetic cells 42%, ~81-86% mixed. 126 of 142 held-out
failures were `device-missing`.

rl/run0: `python -m cellgym.rl.grpo --ckpt out/synth5/run2/best.pt --out out/rl/run0 --steps 200
--synth out/synth5/plans.jsonl --bf16 --jobs 6` (defaults beta 0.02, clip 0.2, group 8, 16 groups,
lr 1e-5, T=1.0, unconstrained sampler, inner-epochs 1). ~26 s/step (sampling ~2/3), 200 steps ~1.4 h.
Train pass rate (25-step means): 38, 45, 45, 45, 47, 52, 52, 50%. KL rose to 0.004. clipfrac is 0
by construction with one inner epoch.

| eval (val+test_id, constrained) | run2 | rl/run0 |
|---|---|---|
| sampled k=8 T=0.8: per-sample pass | 55.6% | **67.5%** |
| sampled pass@1 | 47.5% | 62.5% |
| pass@8 | 95% | **100%** |
| greedy pass@1 | 70% (val 73, test_id 68) | 72.5% (val 100, test_id 56) |
| W/W_euler best | 0.989 | 0.985 |

RL made sampling more reliable (+12 pts per sample) and every cell now has a pass in 8. Greedy is flat
overall: val went up, test_id went down 3 cells — watch that, it may be noise (2.5 pts/cell).
Results: `out/rl/run0/eval_val-test_id.json`, `eval_greedy_constrained.json`.

rl/run1: same as run0 but `--lr 3e-5 --inner-epochs 2 --steps 300`, plus batched sampling
(`model.sample_many`: all rows across cells, left-padded with per-row positions and a pad mask,
preallocated KV cache, chunk 32; greedy output identical to per-cell sampling). ~16.7 s/step vs ~25.9.
Stopped at step 241 by decision. Train pass tracked run0 within ~2 pts, but more all-fail groups
(17% vs 10%), KL 0.02-0.03 (~8x run0), and one spike at step 144 (KL 2.6, grad norm 27.8; the k3
estimator exp(ref - logp) is unbounded).

| eval (val+test_id, constrained) | run2 | run0 @200 | run1 @100 | run1 @200 |
|---|---|---|---|---|
| per-sample pass (k=8, T=0.8) | 55.6% | **67.5%** | 57.8% | 60.0% |
| sampled pass@1 | 47.5% | 62.5% | 52.5% | 62.5% |
| pass@8 | 95% | **100%** | 87.5% | 90% |
| greedy pass@1 | 70% | **72.5%** | - | 62.5% |

Bigger updates hurt held-out results (greedy below run2); train pass rate did not show it.
Best checkpoint remains `out/rl/run0/last.pt`. Run1 files: `out/rl/run1/step{100,200}.pt`, evals.

From run2 on, model selection uses **val only** (`--eval-every 25`: 16 unconstrained samples at
T=0.8 + greedy per val cell; `best.pt` = best val per-sample pass). test_id/test_ood were evaluated
once at the end and not used for any choice. (run0/run1 were compared on val+test_id.)

rl/run2: run0 settings for 500 steps, `--kl-clamp 3 --eval-every 25`. Val per-sample pass 59.6% (step 0)
-> 81.2% (best, step 450), still rising slowly; KL 0.009 at the end, no spikes.

rl/run3: run2 + `--weighting bucket` (cells drawn by (inputs, devices) in benchmark-train proportions,
10% uniform mix). Motivation: 86% of synthetic cells have 16-18 devices, which the benchmark core never
has, and run0 was weakest on small cells (4-10 devices). Val rose faster (78.7% at step 150 vs 66% for
run2) and peaked at 83.3% (step 300); KL grew ~2x faster than run2. Stopped at step 400 (4 checks
without a new best).

Final eval (once; constrained; k=8 T=0.8 and greedy):

| test_id (25 cells) | pretrain run2 | rl/run0 | rl/run2 best | rl/run3 best |
|---|---|---|---|---|
| per-sample pass | 56.0% | 58.5% | **64.0%** | 61.0% |
| sampled pass@1 | 52% | 64% | **72%** | 64% |
| pass@8 | 88% | **96%** | 92% | 80% |
| greedy pass@1 | 68% | 56% | **72%** | 68% |
| best W/W_euler | 0.991 | 0.987 | 0.987 | 0.956 |

| test_ood (172 cells, all 6-input) | pretrain | rl/run0 | rl/run2 | rl/run3 |
|---|---|---|---|---|
| per-sample pass | 0.8% | 1.2% | 1.2% | 0.9% |
| pass@8 | 2.3% | 2.9% | 2.9% | 2.9% |

Findings:
- RL helps in-distribution, modestly: best (rl/run2) +8 pts per sample and +4 pts greedy on test_id
  (200 samples / 25 cells; roughly 1.5-2 standard errors). Val gains (+22 pts) greatly overstate it.
- Val (15 cells) is too small to select on: run3 beat run2 on val by 2-14 pts at every check from
  step 100, but is worse on test_id (lower pass@8 than the pretrained model). Picking the max of ~20
  noisy checks also inflates the val number.
- test_ood is ~1% for every checkpoint; failures are almost all `device-missing` (1,339 of 1,363 for
  rl/run3). RL on <=5-input cells does nothing for 6-input cells: OOD needs pretraining data.

## Toward 6 inputs without 6-input training data

6-input measurement set `out/ood6` (`python -m cellgym.synth.devset build`): 968 six-input cells with a
4-deep stack on one side (the core never has one), <= 8 devices per side, dropping any whose function
equals ANY core cell's under renaming. Split by hash: dev 167 (measuring and choosing) / pool 801 (for
later training). Not benchmark cells; test_ood is untouched. All runs below train on <= 5 inputs only.

Why test_ood was ~1%: in the corpus an n-input cell always uses letters A..n, so `F` never appeared as an
input. The model never wrote it (max p(F) 0.3% at any position, 0 of 160 samples). Editing F's embedding
(copy E, extrapolate E+(E-D), mean of A-E plus noise) left renamed-val at 0/120; even renaming a <=4-input
cell to B.. without F halved its pass rate. The naming itself was learned, so it is fixed in training.

pretrain run4 = run2 + `--relabel 0.5`: half the training sequences get their inputs renamed, order kept,
onto a random subset of A-F (prompt and target tokens together; checked on 300 plans: all still pass at
identical width). Val loss 0.0331 (run2 0.0331).

| eval (constrained) | run2 | run4 |
|---|---|---|
| val per-sample / pass@8 / greedy | 61.7% / 100% / 73.3% | 59.2% / 93.3% / 80.0% |
| val renamed so the last input is F, per-sample / pass@8 | 0% / 0% | 58.3% / 100% |
| ood6 dev per-sample / pass@8 | 0% / 0% | 0% / 0% |

run4's 6-input failures are still `device-missing`, but now from writing too many gate columns (8-11 for 6-8
devices, repeating letters). On held-out 5-input cells its gate count is right 112/115. Cause: the `in` line
has one more letter, which shifts every device line by one position (learned absolute positions). With the
`in` line cut to 5 letters (the gates still name all 6), right gate count goes 3/40 -> 34/40, greedy pass
0/40 -> 6/40, and sampled ood6 dev reaches 7.3% per sample, 27.5% pass@8, 46/167 cells with mixed groups.
So the layout skill mostly transfers; the prompt layout did not.

pretrain run5 = run4 + `--prompt noin` (no `in` line; the inputs are the device gates, so device lines sit at
the same positions for any input count). Val loss 0.0307.

| eval (constrained) | run4 | run5 |
|---|---|---|
| val per-sample / pass@8 / greedy | 59.2% / 93.3% / 80.0% | 70.0% / 93.3% / 66.7% |
| val renamed (last input F), per-sample / pass@8 | 58.3% / 100% | 62.5% / 93.3% |
| **ood6 dev per-sample / pass@8** | 0% / 0% | **41.6% / 80.8%** (122/167 cells mixed) |

No 6-input cell was trained on. By devices per side: 6 -> 29%, 7 -> 44%, 8 -> 42% per sample. Of 132 failed
samples (60 cells x 4), 106 had the right gate count and letters but wrong diffusion nets (a gate's
neighbouring sd nets are not that device's terminals), 24 had too many gates. With distinct letters that is a
pure lookup (find the device line with this gate, copy its other terminal); the model guesses nets from the
generator's naming pattern instead.

pretrain run6 = run5 + `--rename-nets 0.5`: half the sequences get their internal nets renamed to random
distinct names of the same family (n1..n16 / p1..p16) in the netlist and the plan (sd nets and route nets;
tracks untouched). Checked on 300 plans: all pass at identical width and token length. Val loss 0.0312.

| eval (constrained) | run5 | run6 |
|---|---|---|
| val per-sample / pass@8 / greedy | 70.0% / 93.3% / 66.7% | 65.8% / 100% / 66.7% |
| val renamed, per-sample / pass@8 | 62.5% / 93.3% | 55.8% / 100% |
| ood6 dev per-sample / pass@8 | **41.6% / 80.8%** | 36.7% / 79.0% |
| ood6 dev failures (60 cells x 4): right gates, wrong nets / too many gates | 106 / 24 | 108 / 40 |

Negative: renaming nets did not fix the wrong-net failures and added over-counting. The net errors are not
name memorization; getting the diffusion chain consistent in both rows is the hard part itself. Best
pretrained checkpoint: `out/synth5/run5/best.pt` (`--relabel 0.5 --prompt noin`).

Lessons: two "naming" variables that change with input count (the letter F; the `in` line's length, which
moves every device line) took 6-input cells from 0% to 42% per sample with no 6-input data. A third naming
variable (internal nets) was not a bottleneck. Remaining errors are about chain consistency, which GRPO can
now work on: 122 of 167 ood6 dev cells give mixed groups at k=8.

## Status / open
- Bigger selection set: hold out synthetic topologies (not renames) as a few-hundred-cell selection
  split, so best.pt is not chosen on 15 cells.
- 6 inputs: run5 already reaches 42% per sample on ood6 dev with no 6-input data. Next: GRPO from run5
  on the ood6 pool (801 cells), choosing checkpoints on ood6 dev; test_ood scored once at the end.
- RL knobs not yet tried: constrained sampler in rollouts; k1/k2 KL estimator.
- Hold out some stack-4 topologies so Deep stays an OOD axis.
