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
