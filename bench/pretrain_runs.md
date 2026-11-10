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

### run5 zero-shot on test_ood (scored once, before any 6-input RL)

| test_ood (172 cells; constrained) | pretrain run2 | rl/run2 best | **pretrain run5** |
|---|---|---|---|
| per-sample pass (k=8, T=0.8) | 0.8% | 1.2% | **54.1%** |
| pass@8 | 2.3% | 2.9% | **86.0%** |
| greedy pass | - | - | **66.3%** |

By input count: the 160 six-input cells 56.9% per sample / 88.7% pass@8; the 12 anchors (3-4 inputs,
e.g. NOT MAJ3) 12.5-28%. No 6-input cell was trained on; the run was chosen on ood6 dev (stack-4 cells),
never on test_ood.

### rl/ood_run0: GRPO on 6-input cells

`python -m cellgym.rl.grpo --ckpt out/synth5/run5/best.pt --out out/rl/ood_run0 --steps 500
--synth out/synth5/plans.jsonl --ood6 out/ood6 --ood6-frac 0.5 --bf16 --jobs 3 --kl-clamp 3
--eval-every 25 --eval-k 8`: half the groups from the 801 ood6 pool cells, half from benchmark train +
synthetic <=5-input cells; best.pt by ood6 dev (167 cells, unconstrained k=8 + greedy), benchmark val
logged alongside. Step 0: ood6 dev 37.3% per sample / 77.2% pass@8 / 51.5% greedy; val 65.8%. ~14 s/step.
(`--jobs 6` died with MemoryError: every scoring worker imports torch.)

Best ood6 dev at step 475: 71.3% per sample / 92.2% pass@8 / 74.9% greedy (unconstrained); val 79.2%.
Final eval of `out/rl/ood_run0/best.pt`, scored once (constrained; k=8 T=0.8 and greedy):

| split | per-sample | pass@8 | greedy | vs run5 zero-shot (per-sample / pass@8 / greedy) |
|---|---|---|---|---|
| test_ood (172) | **72.4%** | **91.9%** | **77.3%** | 54.1% / 86.0% / 66.3% |
| test_id (25) | 67.0% | 96% | 76% | - |
| val (15) | 85.8% | 100% | 86.7% | - |
| ood6 dev (167, constrained) | 70.4% | 92.8% | - | 41.6% / 80.8% |

### Input letters G-L (vocab 209 -> 211 incl. stack-5 widths; ids of older tokens unchanged)
Letters G-L are appended to the vocab; old checkpoints grow their embedding on load (new letter rows =
mean of A-F + noise). Relabeling now draws from A-L. Reference plans renamed onto G-L pass the scorer
(20/20), so only the model has to learn the names.

- ood_run0 on ood6 dev renamed onto G-L (`devset eval --shift 6`): 0% (never seen those letters).
- ft_al = ood_run0 + 1000 SL steps on the <=5-input corpus, `--relabel 0.5 --lr 1e-4` (97 s on a 4090):
  G-L 6.5% per sample / 29.9% pass@8, but A-F ood6 dev fell 70.4% -> 45.4%: SL on <=5 inputs undoes
  the 6-input RL gains.
- ft_al2 = ood_run0 + 4000 SL steps, `--relabel 0.9 --lr 3e-4 --warmup 100` (7 min on a 4090):
  ood6 dev A-F 48.2% / 88.6% pass@8, renamed onto G-L 38.5% / 80.8%. Letters are now interchangeable;
  6-input skill is back near run5 level (SL on <=5 inputs again pulled it back).

### Zero-shot on more OOD axes (constrained, k=8 T=0.8; dev parts only)
Sets (`python -m cellgym.synth.devset build ...`; none overlaps the corpus envelope of <=5 inputs,
stack <=4, <=9 devices per side):
- ood7: 7 inputs, stack <=4, 7-9 devices/side (1200 of 4694 candidates, hash-picked; dev 205). Lengths
  288-424 tokens (ood6 max 379).
- ood8: 8 inputs, stack <=4, 8-9 devices/side (700 of 1812; dev 120). Lengths 329-422.
- oodstack5: 5 inputs, deepest stack exactly 5, <=9 devices/side (401 routable of 592; dev 75). Needs
  width tokens w2.1 / w3.15 (vocab appended).
- adder: the two logic stages of a mirror full adder as single cells: carry = NOT MAJ3 (3 inputs,
  5 devices/side) and sum = P(S(P(A,B,C),D),S(A,B,C)) (4 inputs incl. the inverted carry, 7/side, PUN
  stack 4).

| dev set | run5 (pretrain) | ood_run0 (RL on 6) | ft_al2 (A-L) |
|---|---|---|---|
| ood7 (205) | 0.0% / 0.0% | 0.1% / 0.5% | **3.7% / 16.1%** |
| ood8 (120) | 0.0% / 0.0% | 0.0% / 0.0% | 0.5% / 3.3% |
| adder stages (2) | 68.8% / 2 of 2 | 75.0% / 2 of 2 | 68.8% / 2 of 2 |
(per-sample / pass@8)

7 and 8 inputs need letters G/H: run5 and ood_run0 never saw them, so 0% is the vocabulary wall,
not a layout failure. With the letters learned (ft_al2), 7 inputs is at 16% pass@8: enough mixed
groups to RL on. 8 inputs (two steps out) is barely above zero.

Stack 5 with the new width rows set three ways (`cellgym/synth/widthedit.py`):

| oodstack5 dev (75) | mean (generic init) | nearest (copy w1.68 / w2.52) | extrapolate (2e(last) - e(prev)) |
|---|---|---|---|
| ood_run0 | 6.7% / 16.0% | 5.7% / 18.7% | 5.2% / 16.0% |
| ft_al2 | 4.3% / 16.0% | 3.5% / 17.3% | 3.5% / 14.7% |

### rl/ood67_run0: GRPO on 6 + 7 inputs from ft_al2
`out/ood67` = ood6 + ood7 (pool 1796, dev 372). `--ckpt out/synth5/ft_al2/best.pt --steps 400
--ood6 out/ood67 --ood6-frac 0.6 --jobs 12` on an RTX 4090 pod. Combined dev 23.0% / 43.8% / 29.8%
(step 0) -> 63.4% / 83.6% / 66.4% (best, step 400; still rising). Val 73% -> 73%.

Scored once (constrained, k=8 T=0.8; greedy):

| split | per-sample | pass@8 | greedy |
|---|---|---|---|
| test_ood (172) | 73.4% | 91.3% | 79.7% |
| test_id (25) | 81.5% | 92% | 80% |
| val (15) | 74.2% | 86.7% | 80% |

Every axis (dev parts; probe-only sets: all cells), per-sample / pass@8:

| set (cells) | run5 | ood_run0 | ft_al2 | **ood67_run0** |
|---|---|---|---|---|
| ood6 dev (167) | 41.6 / 80.8 | 70.4 / 92.8 | 48.2 / 88.6 | 70.3 / 86.8 |
| ood7 dev (205) | 0 / 0 | 0.1 / 0.5 | 3.7 / 16.1 | **44.5 / 60.5** |
| ood8 dev (120), never trained | 0 / 0 | 0 / 0 | 0.5 / 3.3 | **31.5 / 54.2** |
| ood9 (200), 9 inputs 10/side | - | - | 0 / 0 | 0 / 0 |
| ood10 (145), 10 inputs 10/side | - | - | 0 / 0 | 0 / 0 |
| oodstack5 dev (75) | - | 6.7 / 16.0 | 4.3 / 16.0 | 7.5 / 21.3 |
| oodstack6 (174) | 1.7 / 6.9 | 5.0 / 13.8 | 2.9 / 12.6 | 4.0 / 12.6 |
| ooddev10 dev (87), 5 inputs 10/side | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| ooddev11 dev (19) | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| ooddev12 (15) | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| adder stages (2) | 68.8 / 2 of 2 | 75.0 / 2 of 2 | 68.8 / 2 of 2 | 50.0 / 1 of 2 |

ood9/ood10/ooddev10-12 are built with the random sampler (`devset build --sample N`); the exhaustive
enumerator does not finish for 10+ devices. All sequences fit in 512 tokens (ood9 411-465,
ooddev10 415-464, ooddev11 454-505).

Where it fails: every sample on a 10+ device-per-side cell is `device-missing` -- the plan stops
one gate column short. Training never had more than 9 gates per row, and those cells are also the
first ones longer than ~425 tokens, so "count past 9" and "positions past ~425" are confounded;
the position-shift probe below separates them.

Position probe (`devset eval --pos-shift N`: same tokens, every position id + N): ood67_run0 on
ood7 dev 44.5% -> 0.0% (+30) -> 0.0% (+60); ood8 dev 31.5% -> 0.0% -> 0.0%. The model is tied to
absolute positions; a shift of 30 is enough to break it. This does not by itself prove the 10-device
wall is positional, but absolute learned positions are the most fragile part of the architecture:
the next scaling step should use relative (rotary) positions and train lengths past 512.

### rl/oodsd_run0: second axis (stack 5 + 10 devices/side) from ood67_run0
`out/oodsd` = oodstack5 + ooddev10 (dev 162). 300 steps, same settings. Combined dev 3.2% / 9.3%
(step 0) -> 11.0% / 19.1% (best, step 300). Scored once:

| set | ood67_run0 | oodsd_run0 |
|---|---|---|
| oodstack5 dev (trained) | 7.5 / 21.3 | **16.5 / 30.7** |
| oodstack6 (never trained) | 4.0 / 12.6 | 5.5 / 16.1 |
| ooddev10 dev (trained) | 0 / 0 | 0 / 0 |
| ooddev11, ooddev12, ood9 | 0 / 0 | 0 / 0 |
| ood7 dev / ood8 dev | 44.5 / 60.5, 31.5 / 54.2 | 45.1 / 60.0, 32.3 / 50.0 |
| test_ood (172) | 73.4 / 91.3 | **77.7 / 94.2** |
| test_id (25) / val (15) | 81.5 / 92, 74.2 / 86.7 | 74.5 / 88, 73.3 / 86.7 |

RL grows an axis only where zero-shot pass rate is above zero: 10-device groups are all-fail, so they
give no gradient and stay at 0%.

### Cost: search vs verification (`out/pod/search_vs_verify.json`; 10 cells per size, 1 CPU core)

| devices/side | Euler reference search | verify one plan | ratio |
|---|---|---|---|
| 6 | 25 ms | 6.3 ms | 4x |
| 7 | 57 ms | 9.3 ms | 6x |
| 8 | 189 ms | 11.7 ms | 16x |
| 9 | 369 ms | 13.4 ms | 28x |
| 10 | 6.67 s | 15.6 ms | 427x |
| 11 | 27.8 s | 18.7 ms | 1488x |

Model sampling on the 4090: 5.9 s per 128 samples (46 ms each, batched); scoring in the RL loop 800-1300
plans/s with 12 workers. Of ood67_run0's 65 solved 8-input dev cells, the narrowest is 0.89x the width
of the single Euler reference plan (the reference is one optimal-sequence placement, not a proven optimum).

The width edit makes no difference: widths appear only in the prompt and the width embeddings are
nearly orthogonal (cosines 0.02-0.18), so there is no direction to extend. The hard part of stack 5
is the deeper series chain itself; zero-shot sits at ~16% pass@8 (RL-able, like 7 inputs).

## Status / open
- Bigger selection set: hold out synthetic topologies (not renames) as a few-hundred-cell selection
  split, so best.pt is not chosen on 15 cells.
- 6 inputs: run5 already reaches 42% per sample on ood6 dev with no 6-input data. Next: GRPO from run5
  on the ood6 pool (801 cells), choosing checkpoints on ood6 dev; test_ood scored once at the end.
- RL knobs not yet tried: constrained sampler in rollouts; k1/k2 KL estimator.
- Hold out some stack-4 topologies so Deep stays an OOD axis.

## The 10-per-side wall, re-diagnosed (2026-11-08)

Two checks on the "10 devices per side is 0% for every model" result:

- **Generation budget.** `eval.MAX_NEW` was 320 and the constrained decoder forces a plan to close within
  its budget; Euler plans for 10/11/12 per side are 292-341 / 319-370 / 348-383 target tokens. Raising the
  budget to 480 (capped by max_len 512) changed nothing: oodsd_run0 and ood67_run0 still 0/696 on ooddev10
  dev, 0/1600 on ood9. The default budget is now 640 everywhere (capped per prompt by max_len).
- **Column numbers never seen.** The pretraining corpus never uses a column index above 19 (max col: 19
  in 9,820 plans, nothing higher); every 10-per-side cell needs col 20+, and 8 of the 700 ood8 Euler plans
  do too. Sampled 10-per-side plans mostly end cleanly (EOS) at col 18-19 with 9 of 10 gates: the model
  writes the longest plan it has seen. Column numbers are separate tokens 0..127, so "20" is an untrained
  embedding in that role, the same failure G-L letters had before relabeling.

So the absolute-position probe (shift +30 -> 0%) is real, but the wall itself is most likely the column-
number ceiling. Two pretrains to separate them, both `--relabel 0.9` over A-P, `--prompt noin`,
`--max-len 1024`, 4000 steps on the synth5 corpus (<=5 inputs, <=9 per side), on a 4090:

- synth7/ropeA: `--pos rope` (rotary positions, no absolute position embedding)
- synth7/ropeB: ropeA + `--num-embed` (column numbers and widths embedded from their value through
  sinusoidal features, model.ValueEmbed; shifting a value is a fixed rotation, so col 20 is "col 19 + 1"
  the same way col 5 is "col 4 + 1", and w3.15 / w3.78 sit on the same curve as the trained widths)

### synth7 results (zero-shot, pretrain only; constrained k=8 T=0.8, budget 640; per-sample / pass@8)

| set | run5 (learned pos) | ropeA | ropeB (+num-embed) |
|---|---|---|---|
| val pass@1 / test_id pass@1 | - | 53.3 / 56.0 | **73.3 / 68.0** |
| ood6 dev (167) | 41.6 / 80.8 | 32.4 / 71.9 | 24.0 / 67.7 |
| ood7 dev (205) | 0 / 0 | **3.0 / 15.6** | 3.0 / 12.2 |
| ood8 dev (120) | 0 / 0 | 0.4 / 2.5 | 0 / 0 |
| oodstack5 dev (75) | - | 0.5 / 2.7 | **1.8 / 6.7** |
| oodstack6 all (174) | - | 0 / 0 | 0.1 / 0.6 (1 pass) |
| ooddev10 dev / ooddev11 dev / ooddev12 all | 0 | 0 | 0 |
| ood9 all / ood10 all | 0 | 0 | 0 |
| adder dev (2) | - | 93.8 / 100 | 50.0 / 100 |

- Rotary positions remove the position dependence: `--pos-shift 30` on ood7 / ood8 gives identical scores
  (ood67_run0 fell 44.5% -> 0%). Zero-shot 7 inputs goes 0 -> 3% / 16%, about what ft_al2 got with SL on A-L.
- Value embeddings help in-distribution and on deeper stacks (the axis they target), cost on more inputs.
- The wall holds in both: sampled 10-per-side plans still stop at col 17-21 (one ropeA sample reached 67)
  and fail `device-missing`. A smooth number representation does not lift the length prior learned from
  data, and with zero passing samples RL has no signal there.

## Next steps (2026-11-09)

1. **Decoder: no early close.** The constrained decoder knows the netlist; mask plan-closing / EOS while any
   device is unplaced. A checkable constraint, not a hint. Re-run ooddev10 / ood9 zero-shot on ropeA and
   ropeB; any passing sample gives GRPO a signal.
2. **Data: columns past 19.** Add synthetic 10-12-per-side plans (Euler-generated, A-P relabeled) to a
   small share of the SL corpus so col 20+ appear in training; hold out ooddev10/11/12 dev and ood9/10.
3. **Pick the base** (ropeA vs ropeB) from 1-2, then GRPO stepwise: 6-7 inputs, then stack 5 + 10/side,
   then test 11/12 per side, stack 6, 9-10 inputs zero-shot.
4. **Read-once dev sets** as the default (108/120 ood8 dev cells repeat an input); report both splits.
5. **Correct the write-ups**: report and portfolio attribute the wall to absolute positions; it is the
   column/length ceiling (positions were a separate, now fixed, dependence).
6. Pods: results pull + stop is part of every job script (synth7 pod sat idle ~11 h). Old pod 2wswp8tfjldciv
   and glji7howwap4rw are stopped with disks billing; terminate on request.

## synth8: decoder cover, columns past 19, first GRPO stage (2026-11-10)

**Decoder cover** (`constrain.PlanGrammar(cover=True)`, now the default): the col phase cannot be left
until every netlist device with an input gate appears as an sd-g-sd triple. Re-scoring synth7 with it
changes nothing on 10+ per side (ooddev10/11/12, ood9/10 all 0): the model can no longer close early,
so it keeps writing columns past 19 and drifts (col 67, gate after gate) until the budget runs out.
The constraint is sound, but the model can't write a long row. `grpo --constrained` samples rollouts
with this decoder (not used yet).

**Data** (`cellgym/synth/wide.py`): sampled 10-12-per-side topologies at 4-5 inputs, stack <= 4, gave only
49 plans (785 sampled, 730 dropped because ooddev10/11/12 already cover almost every 5-input function
at that size; 11/12 per side route 28% / 15%). Added the ooddev10/11 **pool** cells (Euler plans), dropping
any whose function equals a dev cell's (316 of 445 and 36 of 110 dropped): 252 wide plans in all,
oversampled x6 into synth5 (32,696 records, ~5% wide). Dev parts and ooddev12 stay unseen.
synth8/ropeA and ropeB = synth7's recipes on this corpus (4000 steps, `--relabel 0.9 --prompt noin
--max-len 1024`, rope; ropeB `--num-embed`).

**Base pick** (`out/synth8/pick_base.py`, rule fixed before the run: max sum of pass@8 on ood6/ood7/
oodstack5/ooddev10 dev with val pass@1 >= 0.5): synth8/ropeB 1.040, synth8/ropeA 1.038, synth7/ropeA
0.941, synth7/ropeB 0.828, so it's a tie between the synth8 pair.

**rl/s8_ood67**: GRPO from synth8/ropeB, ood67 recipe (`--ood6 out/ood67 --ood6-frac 0.6 --synth
out/synth5/plans.jsonl --bf16 --jobs 12 --kl-clamp 3 --eval-every 25 --eval-k 8`), 600 steps (~10 s/step).
Combined ood67 dev (unconstrained) 2.0% / 7.0% / 3.0% at step 0 (the constrained score is far higher:
the pretrain relies on the grammar) -> 43.2% / 60.8% / 48.7% at best (step 500), flat after ~450.

Constrained k=8 T=0.8, cover on; per-sample / pass@8 (val/test_id row: pass@1):

| set | synth7 ropeA (cover) | synth8 ropeA | synth8 ropeB | **rl/s8_ood67** | ood67_run0 (prior best) |
|---|---|---|---|---|---|
| val / test_id pass@1 | 66.7 / 56.0 | 73.3 / 32.0 | 73.3 / 64.0 | 80.0 / 76.0 | - |
| ood6 dev (167) | 34.1 / 74.9 | 26.3 / 64.1 | 21.2 / 59.9 | 56.5 / 77.8 | 70.3 / 86.8 |
| ood7 dev (205) | 3.7 / 16.6 | 4.6 / 17.6 | 5.2 / 19.0 | 35.1 / 56.1 | 44.5 / 60.5 |
| ood8 dev (120), never trained | 0.2 / 1.7 | 1.8 / 8.3 | 0.8 / 2.5 | 12.5 / 20.0 | 31.5 / 54.2 |
| oodstack5 dev (75) | 0.3 / 2.7 | 0.5 / 2.7 | 1.5 / 6.7 | 1.3 / 4.0 | 7.5 / 21.3 |
| oodstack6 (174) | 0 / 0 | 0.1 / 1.1 | 0.1 / 0.6 | 0.1 / 1.1 | 4.0 / 12.6 |
| **ooddev10 dev (87)** | 0 / 0 | **4.9 / 19.5** | 4.7 / 18.4 | 8.0 / 19.5 | 0 / 0 |
| ooddev11 dev (19) | 0 / 0 | 0.7 / 5.3 (1 cell) | 0 / 0 | 0 / 0 | 0 / 0 |
| ooddev12 (15), ood9 (200), ood10 (145) | 0 | 0 | 0 | 0 | 0 |
| adder stages (2) | 93.8 / 2 of 2 | 68.8 / 2 of 2 | 75.0 / 2 of 2 | 87.5 / 2 of 2 | 50.0 / 1 of 2 |

rl/s8_ood67 scored once on test: test_ood (172) 66.7% per sample / 90.7% pass@8 / 74.4% greedy;
test_id (25) 64.0 / 88.0 / 68.0 (ood67_run0: test_ood 73.4 / 91.3 / 79.7).

- **The 10-per-side wall was the data.** With ~250 wide cells in SL, ooddev10 dev goes 0 -> ~19% pass@8
  zero-shot (both architectures). The first row ever passing at 10 per side means RL now has signal there.
  11 per side is 1 cell; 12 per side and 9-10 inputs are still 0.
- Stage-1 GRPO grew 6-8 inputs as before (ood7 5 -> 35% per sample, ood8 1 -> 12.5% untrained) but ends
  below ood67_run0, which started from a much stronger ood6 base (ft_al2: 48 / 89 vs 21 / 60 here).
  It did not grow 10 per side (no wide cells in its pool) and did not grow stack 5.
- Read-once split (step 4): the dev sets are almost all repeated-input. ooddev10 dev has 0 of 87 read-once
  cells, ood7 dev 2 of 205, ood8 dev 12 of 120. On ood8, s8_ood67 gets 25.0% pass@8 read-once (12 cells)
  vs 19.4% repeated (108). Read-once dev sets need building from scratch; they can't be cut from these.

## Next steps (2026-11-10)

1. **Stage 2 GRPO: stack 5 + 10 per side** from rl/s8_ood67 on `out/oodsd` (oodstack5 + ooddev10 pools),
   now that 10 per side starts above zero; try `--constrained` rollouts there (the unconstrained
   step-0 score is ~0 on long rows).
2. 6-8 inputs lag ood67_run0. Either a short SL refresh on A-P before RL (as ft_al2 did) or more stage-1
   steps; check that the wide oversampling isn't what costs ood6 (synth8 ood6 26 / 64 vs synth7 34 / 75).
3. 11-12 per side: still only ~75 training cells at 11 and 0 at 12. Get more from 4-input / stack-3
   topologies or a cheaper labeler; 12 per side Euler search is ~2 min per cell.
4. Build read-once dev sets (current ones are 0-10% read-once).
5. Write-ups still attribute the wall to absolute positions; correct to "column / length ceiling in
   the data" (confirmed here).
6. Pods: new pod 86f89101cuzuy3 (EUR-IS-1) is stopped and holds every synth8/RL artifact. glji7howwap4rw
   and 2wswp8tfjldciv are stopped; their hosts had no free GPU, so they couldn't restart.
