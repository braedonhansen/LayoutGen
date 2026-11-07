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

## Status / open
- Next RL knobs: keep lr 1e-5 / 1 inner epoch and run longer; bound the KL term (clamp
  ref - logp or use k1/k2); constrained sampler in rollouts. Evaluate held-out every ~50 steps.
- New-topology pretraining run is mainly for test_ood (6-input cells; corpus stops at 5 inputs).
  Also: hold out some stack-4 topologies so Deep stays an OOD axis.
- Uncommitted code: vocab extension (tokens.py), `--constrained` (eval.py), synth/constrain.py,
  cellgym/rl/, tests/synth/test_constrain.py, tests/rl/.
