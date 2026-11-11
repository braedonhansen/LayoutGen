# Pretrained-LM runs: abstraction over OOD axes (2026-11-11)

Question: can the system succeed on sizes it was never trained on, if the representation carries no
size-dependent symbol and the environment holds the bookkeeping? Everything below trains only on
the synth5 corpus (<= 5 inputs, <= 9 devices per side, stack <= 4) and is tested zero-shot on every axis.
No 6+-input, 10+-per-side or stack-5+ cell is in training. (synth8 in pretrain_runs.md trained on
10-12-per-side cells and does not count as OOD on that axis.)

Code: `cellgym/llm/` (fmt, build, train, infer, score, probe; run scripts `run*_2026_11_11.sh`).
Base: Qwen2.5-0.5B (base), full fine-tune, plain text in Qwen's own tokenizer (numbers split into
digits). Augmentation: inputs renamed order-preserving onto random A-Z letters and internal nets onto
random numbers <= 40 (90% of train rows), so names are an invariance, not an axis; input count stays
untrained.

## Two pipelines

- **one-shot** (`oneshot05`): prompt = netlist text, target = the whole plan body (col/route/pin lines).
  2 epochs (653 steps, 28 min on a 4090), unconstrained sampling k=4 T=0.7.
- **step-wise placement** (`step05`): each step the model sees only an environment-computed state:
  ```
  left nmos: B OUT-n1 | B VSS-n3 | C OUT-n1 ...      unplaced devices (gate drain-source)
  left pmos: ...
  last: sd n=n2 p=p3 | g A | sd n=n3 p=OUT          last 3 columns
  next:
  ```
  and writes one column (`g B` / `sd n=n1 p=p2`). The environment (`fmt.Placer`) lists the legal
  columns and generation is restricted to them (token trie), stops when every device is placed, and the
  Euler baseline's router (`route_best`) adds contacts, routes and pins. No column numbers, no plan
  length, no netlist size in the input apart from the shrinking device lists. 1 epoch over every step
  of every corpus plan (539k examples, 944 steps, 32 min). k=8 T=0.7.
- **random-legal baseline**: the same environment and router, uniform over legal columns. It shows how
  much the environment alone solves.

## Results (per-sample / pass@k; W/W_euler = mean width of passing samples over the Euler reference,
best-of-k = narrowest passing sample per cell)

| set (cells) | step05 | step05 W/W_e (mean / best-of-8) | random legal | oneshot05 (k=4) | oneshot05 W/W_e | tiny one-shot synth7 ropeA (k=8, constrained) |
|---|---|---|---|---|---|---|
| val (15) | 100 / 100 | 1.108 / 0.994 | 7.5 / 33 | 66.7 / 93.3 | 1.019 | pass@1 66.7 |
| test_id (25) | 100 / 100 | 1.092 / 0.999 | 15 / 48 | 70.0 / 96.0 | 1.026 | pass@1 56.0 |
| ood6 dev (167) | 100 / 100 | 1.106 / 1.008 | 1.4 / 10 | 46.7 / 79.6 | 1.019 | 34.1 / 74.9 |
| ood7 dev (205) | 100 / 100 | 1.091 / 1.004 | 0.3 / 2.4 | 34.5 / 68.8 | 1.010 | 3.7 / 16.6 |
| ood8 dev (120) | 100 / 100 | 1.092 / 1.008 | 0.3 / 2.5 | 24.8 / 55.8 | 0.997 | 0.2 / 1.7 |
| ood9, 9 inputs (200) | **100 / 100** | 1.092 / 1.010 | 0 / 0 | 0.2 / 0.5 | - | 0 / 0 |
| ood10, 10 inputs (145) | **100 / 100** | 1.069 / 1.005 | 0 / 0 | 0 / 0 | - | 0 / 0 |
| oodstack5 dev (75) | 77.2 / 100 | 1.131 / 1.026 | 0.7 / 5.3 | 17.3 / 46.7 | 1.015 | 0.3 / 2.7 |
| oodstack6 (174) | 82.3 / 100 | 1.118 / 1.018 | 0.1 / 0.6 | 13.9 / 34.5 | 1.018 | 0 / 0 |
| ooddev10, 10/side (87) | 56.8 / **97.7** | 1.128 / 1.041 | 0 / 0 | 0.3 / 1.1 | - | 0 / 0 |
| ooddev11, 11/side (19) | 35.5 / **89.5** | 1.141 / 1.068 | 0 / 0 | 0 / 0 | - | 0 / 0 |
| ooddev12, 12/side (15) | 20.8 / **73.3** | 1.122 / 1.062 | 0 / 0 | 0 / 0 | - | 0 / 0 |
| probe 8 inputs 14/side (30, no Euler label) | 27.5 / 70.0 (k=4) | - | 0 / 0 | - | - | - |
| adder stages (2) | 100 / 100 | 1.181 / 1.000 | 18.8 / 100 | 62.5 / 100 | 1.005 | 93.8 / 100 |

- **Step-wise generalizes on every axis tried.** Every step05 placement finishes; every failure is
  `route:no-routing`: the router finds no legal routing for that placement. (The Euler reference fails to
  route about half of the 10-12-per-side candidate cells too; those cells are not in the sets.) The
  random-legal baseline is ~0 on every OOD set (it wanders into breaks until the column cap), so the
  result is the learned policy, not the environment.
- Placement quality: ~9-14% wider than Euler per sample, at or near Euler with best-of-8 (1.00-1.07),
  up to 12 per side. Euler search is exponential in devices per side (2 min per 12-per-side cell); the
  policy is linear (one short decision per column).
- **One-shot with a pretrained LM helps on inputs but hits the length wall.** ood7 16.6 -> 68.8 and
  ood8 1.7 -> 55.8 pass@k vs the tiny one-shot model, at Euler width (it picks contacts and routes
  itself). It still drops to 0 at 9-10 inputs and 10+ per side, almost all `device-missing`: the plan
  closes early, the same length prior as before. Digit tokens and pretraining alone don't fix it.
- **Not yet separated: pretraining vs format.** The scratch-init control (same step data, 8x512
  transformer) crashed at model construction (`layer_types=None`; fixed in train.py, not rerun). Without
  it, the step result shows the format + environment works with a pretrained 0.5B. It does not show the
  pretraining is needed.
- Probes i10d16 / i12d20 weren't built in time (sampling large irredundant networks is slow); only
  i8d14 ran.

## Caveats

- In step mode the router is the Euler baseline's router, so the learned part is placement only.
  One-shot learns everything, which is a different task.
- Pass rates for step mode are bounded by the router, not the model; W/W_euler is the policy's measure.
- Pod note: all runs finished at 11:09 UTC; the pod then sat idle ~7 h because the local watcher died
  with a PC crash (total 9.2 h, ~$8.16).

## Next

1. Rerun the scratch control (format vs pretraining), ~40 min.
2. Width: GRPO on the step policy with the width reward (each step is a short, checkable action; the
   reward is final width), or learn contact choice too.
3. Routing as the next learned stage in the same step style (route one net per step from an
   environment state), since the router is now the failure point at 10+ per side and stack 5-6.
4. Bigger probes (14-20 per side, 10-12 inputs) once built on CPU ahead of time.

## Control, replicate and RL (2026-11-11, second pod, A40)

**Trained on** (every model below): <= 5 inputs, <= 9 devices per side, stack <= 4. Supervised on Euler
step decisions; the RL model adds group-relative policy-gradient RL on the same in-distribution cells
(corpus TRAIN topologies; checkpoint chosen on corpus VAL topologies). **Tested on** sets never seen in SL
or RL: 6-10 inputs, 10-12 devices per side, stack 5-6, and an 8-input / 14-per-side probe.

Per-sample / pass@k (k=8; probe k=4), width = passing samples' W / W_euler (mean, best-of-k):

| set | trained up to -> tested | scratch 102M, SL | Qwen 0.5B SL, seed 1 | **Qwen 0.5B SL + RL** |
|---|---|---|---|---|
| val / test_id | in distribution | 100 / 100, 1.23 / 1.06 | 100 / 100, 1.08 / 0.99 | 100 / 100, 1.065 / 1.004 |
| ood6-8 dev | 5 -> 6-8 inputs | 100 / 100, 1.21 / 1.03 | 100 / 100, 1.08-1.10 / 1.01 | 100 / 100, 1.06-1.07 / 1.00-1.01 |
| ood9 / ood10 | 5 -> 9 / 10 inputs | 100 / 100, 1.21 / 1.04 | 100 / 100, 1.09 / 1.01 | **100 / 100, 1.05-1.07 / 1.01** |
| oodstack5 dev | stack 4 -> 5 | 72 / 100, 1.22 / 1.07 | 75 / 100, 1.13 / 1.03 | 81 / 97, 1.08 / 1.01 |
| oodstack6 | stack 4 -> 6 | 84 / 100, 1.25 / 1.06 | 85 / 100, 1.11 / 1.02 | 87 / 100, 1.08 / 1.02 |
| ooddev10 dev | 9 -> 10 per side | 49 / 98, 1.24 / 1.10 | 55 / 97, 1.12 / 1.04 | 59 / 94, 1.11 / 1.05 |
| ooddev11 dev | 9 -> 11 per side | 28 / 79, 1.26 / 1.13 | 32 / 84, 1.16 / 1.10 | 42 / 90, 1.13 / 1.10 |
| ooddev12 | 9 -> 12 per side | 20 / 73, 1.26 / 1.19 | 24 / 87, 1.15 / 1.12 | 33 / 73, 1.07 / 1.07 |
| probe i8d14 (20) | 5 & 9 -> 8 inputs & 14 per side | 24 / 60 | 43 / 80 | 33 / 75 |
| adder stages | composed cells | 94 / 100 | 94 / 100 | 88 / 100, best 0.99 |

- **Pretraining is not what generalizes; the format is.** The scratch control (no pretraining) reaches the
  same pass rates on every OOD axis. Pretraining buys placement quality: width 1.21-1.26x -> 1.08-1.16x per sample.
- **Seed replicate** reproduces the first run (step05) within noise.
- **RL** (80 steps x 32 cells x 8 placements, width reward, in-distribution cells only) moves width toward
  Euler on every set, including the never-seen ones: per sample 1.08-1.10 -> 1.05-1.07 at 6-10 inputs,
  1.13 -> 1.08 at stack 5-6. It also raises the per-sample pass rate at 10-12 per side (55/32/24 ->
  59/42/33), because routable placements earn reward. Best-of-8 stays at or just above Euler
  (1.00-1.01 at <= 10 inputs). Not under Euler on average yet. In-distribution val moved slowly (reward
  0.731 -> 0.744): the reward varies little within a group, so more steps / higher lr is the next knob.
- RL implementation note: the first attempt used full-vocabulary log-probs while sampling over legal
  tokens; negative advantages pushed mass onto illegal tokens (log-prob drift of -2 nats in 40 steps,
  no gain). Fixed by renormalizing over the legal-token set at each position (`rl.Policy._logps`).
- Cost: A40 pod 4.9 h, about $2.90. Models saved locally: `out/llm/rl2_best`, `out/llm/step05s1_model`.
