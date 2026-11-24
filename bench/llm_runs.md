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

## RL for routability: channel-density penalty (2026-11-12, A40, rl3)

**Trained on**: <= 5 inputs, <= 9 per side, stack <= 4. SL as step05s1 (seed 1, retrained), then RL with
the rl2 settings (80 steps x 32 cells x 8, lr 2e-6, beta 0.02, in-distribution TRAIN topologies only) and
reward = width reward - 0.15 * max(0, D - 2), D = channel density (`cellgym/llm/congest.py`: max number of
channel nets, i.e. nets touching both rows or a gate, whose column span covers one column). **Tested on**
the same never-seen sets as rl2.

Why density: on step05s1's OOD samples, D <= 3 routes 100%, D = 4 routes 34-73%, D = 5 10-40%, D = 6 0-25%.

| set | trained up to -> tested | rl2 (width only) per-sample / pass@8, mean D | **rl3 (width + density)** per-sample / pass@8, mean D | rl3 W/W_e mean / best |
|---|---|---|---|---|
| ood6-10 | 5 -> 6-10 inputs | 100 / 100, 1.0-2.6 | 100 / 100, 1.0-2.6 | 1.05-1.07 / 1.00-1.02 |
| oodstack5 / 6 | stack 4 -> 5 / 6 | 81 / 97, 87 / 100; D 3.5 | 81 / 99, 87 / 100; D 3.8 / 3.5 | 1.09 / 1.02-1.03 |
| ooddev10 | 9 -> 10 per side | 59 / 94, D 4.20 | 63 / 97, D 4.21 | 1.10 / 1.04 |
| ooddev11 | 9 -> 11 per side | 42 / 90, D 4.45 | 41 / 84, D 4.32 | 1.10 / 1.05 |
| ooddev12 | 9 -> 12 per side | 33 / 73, D 4.78 | 28 / 80, D 4.79 | 1.09 / 1.05 |
| probe i8d14 (k=4) | 5 & 9 -> 8 inputs & 14 per side | 33 / 75 | 34 / 70 | - |

- **Negative result: the density penalty did not lower density, in distribution or out.** In-distribution
  val D stayed at 3.43 through RL; OOD D is unchanged (4.2-4.8). Pass rates are within noise of rl2.
- Reason: in the training range D is almost fixed by the cell (always <= 3, little spread within a group
  of 8 placements of one cell), so the penalty gives the policy almost no contrast to learn from. A
  reward that only bites outside the training range can't be learned inside it.
- The OOD cells are dense by nature: the Euler reference plans for the same cells have D = 4-6 too
  (ooddev10: 113 at 3, 319 at 4, 99 at 5), and they route because Euler searches many placements and keeps a
  routable one. So the policy's placements are not more congested than Euler's; routing at D >= 4 depends
  on finer structure (which nets share columns, stub and pad positions) than the max-density number.
- Cost: A40 3.2 h ~= $1.90 (+ ~$0.05 for an H200 pod that billed $3.79/h, not the listed $0.50; killed in 1 min).

## Free channel tracks and an area objective (2026-11-18/19, A40, run7: step05s1 retrained + rl4)

**Trained on** (both models): <= 5 inputs, <= 9 devices per side, stack <= 4, 1 stage, 1 output. SL as step05s1
(seed 1, retrained). rl4 = SL + 80 steps of RL (32 cells x 8, lr 4e-6, beta 0.02) on in-distribution TRAIN
topologies only, with the area reward below. **Tested on** sets never seen in SL or RL: 6-10 inputs, 10-12 per
side, stack 5-6, and six probe sets built for this run (8/10/12/16 inputs at 14/16/20/24 per side; 2-stage and
3-stage cells, the first test of composition).

### What changed in the formulation

Every remaining OOD miss was `route:no-routing` in a fixed 3-track channel (the Euler reference fails on about
half of the 10-12-per-side candidates too), so those pass rates measured the template, not the policy.

- **Channel tracks are a per-plan parameter T** (1-8, default 3; optional `tracks N` plan line; SPEC.md). Height
  moves 0.34 um per track, as it already does with stack depth. At T = 3 everything is byte-identical (all 323
  Euler plans). Ground truth (KLayout DRC + Magic DRC + Netgen LVS): 199 / 199 routed layouts pass at T = 1-6
  (40 cells), no disagreement with the fast checker; split contacts keep the "PMOS index - NMOS index >= 2" rule
  (7 hand-built variants pass). T = 7-8 are allowed but not audited. One renderer fix, independent of T: a met1
  route with all endpoints in one column drew a pad under the met1 minimum area.
- **Objective = area.** `route_min_area` routes a placement at the track count with the smallest
  width x height (first T that routes, then up to 3 more while width still shrinks; matches an exhaustive scan
  on 283 cells). First-T-that-routes is not the area optimum for ~45% of cells.
- **The Euler reference gets the same freedom** (`cellgym.llm.arearef`): its placement re-routed with
  `route_min_area`. Area ratios below are against that. RL reward = max(0.05, 1 + (A_ref - A) / A_ref).

### Results (k = 8, T = 0.7; per-sample / pass@8)

| set | trained up to -> tested | SL, fixed 3 tracks | rl4, fixed 3 tracks | SL, free tracks | rl4, free tracks | SL area / Euler area (mean, best-of-8) | **rl4 area / Euler area** (mean, best-of-8) | mean tracks SL / rl4 |
|---|---|---|---|---|---|---|---|---|
| val / test_id | in distribution | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 1.102, 1.007 / 1.086, 1.011 | 1.056, 1.013 / 1.033, 1.009 | 2.5 / 2.7 |
| ood6-8 dev | 5 -> 6-8 inputs | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 1.08-1.09, 1.00-1.01 | 1.05, 1.02-1.03 | 2.4-2.8 / 2.3-2.8 |
| ood9 / ood10 | 5 -> 9 / 10 inputs | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 1.088, 1.014 / 1.069, 1.010 | 1.059, 1.029 / 1.038, 1.019 | 2.4, 1.8 / 2.4, 1.7 |
| oodstack5 dev | stack 4 -> 5 | 77 / 100 | 80 / 97 | 100 / 100 | 100 / 100 | 1.143, 1.030 | 1.086, 1.027 | 3.4 / 3.4 |
| oodstack6 | stack 4 -> 6 | 83 / 99 | 88 / 98 | 100 / 100 | 100 / 100 | 1.125, 1.026 | 1.084, 1.028 | 3.3 / 3.2 |
| ooddev10 dev | 9 -> 10 per side | 56 / 98 | 63 / 84 | 100 / 100 | 100 / 100 | 1.178, 1.038 | 1.128, 1.069 | 3.7 / 3.7 |
| ooddev11 dev | 9 -> 11 per side | 32 / 84 | 38 / 63 | 100 / 100 | 99.3 / 100 | 1.188, 1.058 | 1.176, 1.084 | 4.0 / 4.0 |
| ooddev12 | 9 -> 12 per side | 21 / 87 | 43 / 80 | 100 / 100 | 100 / 100 | 1.207, 1.053 | 1.129, 1.047 | 4.3 / 4.0 |

Probe sets (k = 4, no Euler reference, so pass and tracks only):

| probe (cells) | trained up to -> tested | SL, fixed 3 tracks | rl4, fixed 3 tracks | SL, free tracks | rl4, free tracks | mean tracks SL / rl4 |
|---|---|---|---|---|---|---|
| i8d14 (20) | 5 & 9 -> 8 inputs & 14 per side | 29 / 70 | 35 / 55 | 100 / 100 | 100 / 100 | 4.0 / 4.0 |
| i10d16 (20) | -> 10 inputs & 16 per side | 18 / 50 | 44 / 60 | 100 / 100 | 100 / 100 | 4.4 / 4.1 |
| i12d20 (20) | -> 12 inputs & 20 per side, stack 5 | 9 / 30 | 6 / 15 | 100 / 100 | 100 / 100 | 5.0 / 4.8 |
| i16d24 (10) | -> 16 inputs & 24 per side, stack 5 | 5 / 20 | 0 / 0 | 100 / 100 | 92.5 / 100 | 4.9 / 4.8 |
| stage2 (30) | 1 -> 2 stages | 83 / 93 | 83 / 90 | 100 / 100 | 100 / 100 | 2.9 / 3.0 |
| stage3 (30) | 1 -> 3 stages, 1-2 outputs | 36 / 67 | 43 / 67 | 99.2 / 100 | 100 / 100 | 3.9 / 3.9 |

- **The routing wall was the template.** With free tracks the SL-only policy passes 100% per sample on every
  set (one stage3 sample aside), out to 16 inputs / 24 per side and to 3-stage, 2-output cells, none of which
  it trained on. Composition (gates driven by internal nets, several outputs) works zero-shot. At a fixed
  3 tracks the same placements pass 5-29% at 14-24 per side; the Euler placer itself routed 28/30 stage2 and
  19/30 stage3 cells at 3 tracks.
- **The cost of size is area, not failure.** Against Euler with the same freedom, SL samples are 7-10% larger
  in the training range and at 6-10 inputs, 13-14% at stack 5-6, 18-21% at 10-12 per side. Best-of-8 stays
  within 1-6%. The policy's placements need 3.7-4.3 tracks at 10-12 per side, where the Euler references
  use 3-5.
- **Area RL on in-distribution cells transfers to every unseen set, on the mean.** In-distribution val area
  ratio 1.145 -> 1.081 over 80 steps (monotone, not flat at the end); per-sample area on the unseen sets
  1.07-1.09 -> 1.04-1.06 (6-10 inputs), 1.13-1.14 -> 1.08-1.09 (stack 5-6), 1.18 / 1.19 / 1.21 ->
  1.13 / 1.18 / 1.13 (10 / 11 / 12 per side). The reward is dense (no zeros once every placement routes), which
  is likely why it moves faster than rl2's width reward (val reward 0.855 -> 0.919 here vs 0.731 -> 0.744).
- **The gain is width, not tracks.** Mean tracks did not move (val 3.28 -> 3.33 in training; unchanged on OOD
  sets). The policy did not learn to place for fewer tracks.
- **RL collapsed sampling diversity, and best-of-8 got worse.** Distinct placements per cell (of 8) fell from
  5.4-7.7 to 2.2-4.1. Best-of-8 area is flat or worse on most sets (ood8 1.011 -> 1.027, ooddev10 1.038 ->
  1.069, ooddev11 1.058 -> 1.084), and at fixed 3 tracks pass@8 at 10 / 11 per side dropped 98 -> 84 and
  84 -> 63 even though per-sample pass rose. A system that takes the best of k is not better after this RL;
  a system that takes one sample is.
- **New failure after RL: placements that never finish.** rl4 has 5 `cap` episodes (1 each on ood7 and
  ooddev11, 3 of 40 on i16d24); SL has none.

### Not done / caveats

- A second RL run (rl5: 150 more steps from rl4) was lost at step ~50 when the pod's host dropped; rl4's
  weights went with it (no volume). Its log to that point showed the per-batch area ratio still falling
  (about 1.06-1.09) with 8-13 of 32 groups already flat, i.e. more collapse.
- Probe sets have no Euler area reference, so probe quality is tracks only. i12d20 and i16d24 are at stack 5
  (20+ per side cannot exist at stack 4), so they mix size with stack.
- The corpus reference is the narrowest 3-track Euler plan re-routed, not the minimum-area Euler placement,
  which is slightly generous to the model.
- `cellgym/synth/tokens.py` (old tiny-model tokenizer) and `cellgym/llm/route_env.py` are not track-aware.
- Cost: A40 about 4.6 h at $0.59/h, about $2.7 of the $7 cap (billing showed $2.10 when checked; it may lag).
  An RTX 5090 listed at $0.69/h billed $1.39/h and was deleted within seconds.

### Next

1. Keep diversity through RL: an entropy term over the legal-move distribution (or a stronger KL anchor /
   lower lr), and select the checkpoint on best-of-k, not the mean. Save checkpoints off the pod.
2. Make tracks learnable: the policy never sees the channel, so it cannot trade a column for a track. Show the
   open channel nets in the step state, now that the area reward gives the contrast.
3. Euler area references for the probe sets (cheap for stage2/3; needs a time-limited search at 16-24 per side).

## Channel-aware state and entropy-regularized area RL (2026-11-20, A40, run10: step05v2 + rl6 + rl7)

**Trained on** (all models): <= 5 inputs, <= 9 devices per side, stack <= 4, 1 stage, 1 output. step05v2 = SL
as step05s1 (seed 1) with the v2 step state. rl6 = step05v2 + 60 RL steps (32 cells x 8, lr 4e-6, beta 0.02,
area reward, in-distribution TRAIN topologies only) with an entropy bonus; rl7 = 60 more steps from rl6.
**Tested on** the same never-seen sets as run7, plus Euler area references for the 2- and 3-stage probes.

### What changed

- **v2 step state** (`CELLGYM_STATE=v2`, default; v1 = the old state): one extra line, `open: A n3 OUT`, the
  non-rail nets that already appear in a placed column and still appear in an unplaced device, i.e. the nets
  crossing the frontier. Its length is the channel load there, not the cell size (+12% characters).
- **Entropy bonus over the legal-move distribution** with an adaptive coefficient (`--ent-target 0.7`:
  hold 70% of the first batch's per-step entropy; `--ent-coef 0.02`, `--ent-lr 0.3`, clipped at 1.0). Groups
  with equal rewards now stay in the update so the bonus acts on collapsed cells.
- **Checkpoint selection** on (mean area + best-of-k area) / 2 (`--select both`).
- **Probe references** (`arearef --search`): Euler area references for stage2 (30/30) and stage3 (26/30).
  The Euler search did not finish on any 14-24-per-side probe cell (run at a 20 s limit, 2 GB RAM free), so
  those sets still have no reference; a heuristic one (best of 2-17 routed random placements) is kept in
  `out/arearef/<set>.heur.json`, off by default.

### Results (free tracks, k = 8, probes k = 4; area / Euler area as mean, best-of-k)

| set | trained up to -> tested | SL v1 (run7) | SL v2 | rl4 (run7: 80 steps, no entropy, v1) | **rl6 (60 steps, entropy, v2)** |
|---|---|---|---|---|---|
| val / test_id | in distribution | 1.102, 1.007 / 1.086, 1.011 | 1.085, 1.002 / 1.069, 1.002 | 1.056, 1.013 / 1.033, 1.009 | 1.053, 0.996 / 1.034, 0.998 |
| ood6-8 dev | 5 -> 6-8 inputs | 1.08-1.09, 1.00-1.01 | 1.08-1.09, 1.00-1.01 | 1.05, 1.02-1.03 | 1.05-1.06, 1.00 |
| ood9 / ood10 | 5 -> 9 / 10 inputs | 1.088, 1.014 / 1.069, 1.010 | 1.083, 1.007 / 1.064, 1.004 | 1.059, 1.029 / 1.038, 1.019 | 1.057, 1.002 / 1.046, 1.004 |
| oodstack5 dev | stack 4 -> 5 | 1.143, 1.030 | 1.130, 1.029 | 1.086, 1.027 | 1.084, 1.017 |
| oodstack6 | stack 4 -> 6 | 1.125, 1.026 | 1.124, 1.034 | 1.084, 1.028 | 1.096, 1.016 |
| ooddev10 dev | 9 -> 10 per side | 1.178, 1.038 | 1.166, 1.031 | 1.128, 1.069 | 1.104, 1.010 |
| ooddev11 dev | 9 -> 11 per side | 1.188, 1.058 | 1.173, 1.040 | 1.176, 1.084 | 1.142, 1.043 |
| ooddev12 | 9 -> 12 per side | 1.207, 1.053 | 1.203, 1.054 | 1.129, 1.047 | 1.152, 1.041 |
| stage2 (30) | 1 -> 2 stages | 1.226, 1.106 | 1.214, 1.106 | 1.163, 1.100 | 1.173, 1.077 |
| stage3 (26 of 30 with a reference) | 1 -> 3 stages, 1-2 outputs | 1.314, 1.190 | 1.341, 1.201 | 1.259, 1.160 | 1.291, 1.144 |

Per-sample pass with free tracks is 100% for SL v2 and rl6 on every set, except stage3 (98.3% / 99.2%).
rl6 has no `cap` episodes (rl4 had 5). Fixed 3 tracks, per-sample / pass@8, SL v2 -> rl6: stack 5
76 / 99 -> 80 / 99, stack 6 84 / 100 -> 85 / 99, 10 per side 60 / 97 -> 61 / 95, 11 per side 30 / 84 ->
41 / 74, 12 per side 21 / 93 -> 28 / 67 (the last two sets are 19 and 15 cells).

- **The v2 state alone changes nothing in SL**: pass, area, tracks and diversity match SL v1 within about a
  point. Expected, since the SL labels are the same Euler placements.
- **With the entropy bonus, RL improves the mean and best-of-8 together, on every unseen set.** Best-of-8
  area is at Euler up to 10 inputs (1.00) and within 1-4% at stack 5-6 and 10-12 per side; rl4 had made
  best-of-8 worse on most of those sets. The mean is about as good as rl4's with 20 fewer steps.
  Distinct placements per 8: SL 5.2-7.7, rl4 2.2-4.1, rl6 4.0-6.7.
- **Composition is the weakest axis on quality.** With a true Euler reference, SL samples are 21-23% larger
  on 2-stage cells and 31-34% on 3-stage cells (best-of-4 11% and 19-20%); rl6 brings best-of-4 to 8% and
  14%. Every size axis is at most 20% (mean) and 5% (best-of-8).
- **Tracks still do not move** (mean tracks unchanged on every set, in training 3.30 -> 3.35), even with
  the channel visible. RL still improves width only.
- **The entropy controller oscillated, and the second half bought nothing.** Entropy fell to 43-47% of its
  start from step ~30 (target 70%), the coefficient climbed from 0.014 to its 1.0 limit, entropy then
  recovered to 63-69% and the coefficient dropped back to ~0. During the overshoot in-distribution val area
  went 1.087 -> 1.178 (step 80) -> 1.092 (step 120), never beating step 60, so rl7/best is rl7's starting
  point (= rl6/last) and its samples are a second draw from the step-60 policy, not a new model.
- **Confound:** rl4 used the v1 state, so rl4 -> rl6 mixes the entropy bonus with the state change. The state
  did nothing in SL and tracks did not move, so the bonus is the likelier cause; this run does not prove it.

### Cost and an operations failure

- run10 pod: 13.6 h, **$7.51**, of which the job was 4.9 h (~$2.9). The run finished at 10:50 UTC and the
  results were pulled automatically, but the pod was not terminated until 19:31 UTC: the turn that handled the
  completion notification was cut off before its terminate call, and nothing else could stop the pod. About
  8.6 h (~$5) was idle billing.
- Session total **$10.20** against a $7 cap (run7 pod $2.68, run10 pod $7.51).
- No weights were kept: a time-capped download of rl6/best got 91% of the file before the pod was terminated,
  and the partial file was deleted. step05v2, rl6 and rl7 exist only as samples and logs.

### Next

1. Unattended runs need a terminate that does not depend on a chat turn (a local `runpodctl` in the watcher
   script, or a pod-side self-terminate). Until then every unattended run carries uncapped idle risk.
2. Entropy control: a fixed small coefficient or a lower cap (0.05-0.1) and a slower rate; the adaptive
   rule at lr 0.3 with a 1.0 cap overshoots. 60 steps is where the gain was; more steps need a stable bonus.
3. Tracks: the reward already pays for fewer tracks and the state shows the channel, yet the policy does not
   trade a column for a track. Check whether such trades exist at all (does any placement of a cell reach a
   smaller area with one more column and one less track?) before spending more RL on it.
4. Composition: stage outputs are named `x1`, `OUT1`, tokens the policy never saw; test whether naming them
   like internal nets closes part of the 2-3 stage gap, at zero training cost.

### Second draw from the step-60 policy (rl7/best samples, scored 2026-11-20)

rl7 never beat its starting point, so its selected checkpoint is the step-60 policy and its samples are a second
draw of 8 per cell (believed to be the same weights as rl6/best; not verified, the weights are gone). Area /
Euler area, mean and best-of-8, rl6 samples -> second draw: in distribution 1.053, 0.996 / 1.034, 0.998 ->
1.045, 0.991 / 1.031, 0.992; ood6-10 1.05-1.06, 1.00 -> 1.05-1.06, 1.00; stack 5 / 6 1.084, 1.017 /
1.096, 1.016 -> 1.081, 1.014 / 1.093, 1.015; 10 / 11 / 12 per side 1.104, 1.010 / 1.142, 1.043 /
1.152, 1.041 -> 1.103, 1.010 / 1.124, 1.018 / 1.153, 1.031. Per-sample pass 100% on every set in both. Means
agree within 0.01 except 11 per side (0.02, 19 cells). This bounds sampling noise only: it is one training
run and one seed.

### Operations: unattended pod runs (2026-11-21)

- `cellgym/llm/pod_watch.sh POD IP PORT DONE_MARKER LOCAL_DIR [MAX_HOURS]` runs on the PC, copies result
  bundles every poll and terminates the pod with `runpodctl pod delete` when the done marker appears, when the
  host stops answering for ~10 minutes, or at the deadline. It refuses to start if runpodctl cannot read the
  pod. Launch every unattended run under it; a terminate that waits for a chat turn is what cost 8.6 idle
  hours on run10.
- runpodctl 3.1.1 is installed locally. The watcher uses its own key from the environment variable
  `RUNPOD_API_KEY_LAYOUTGEN` (pods read/write, billing read), mapped to `RUNPOD_API_KEY` for its own calls only;
  the machine's general `RUNPOD_API_KEY` is for something else and is left alone. Pod read and billing read
  were verified; the delete call has not been exercised on a real pod yet, so watch the first run that uses it.
- The watcher still needs the PC awake. Download from a pod to the PC ran at ~4 MB/s (a 1.26 GB model takes
  ~5 minutes): pull weights worth keeping as soon as they exist.

## Is area a width problem or a track problem? (2026-11-21, CPU only, `cellgym/llm/tradeoff.py`)

Question: area RL has never moved the mean track count (rl4, rl6), even with the channel visible in the state.
Is there anything to learn there? **Policies were trained on** <= 5 inputs, <= 9 per side, stack <= 4; the
analysis covers val / test_id (in distribution) and the 6-10 input, stack 5-6 and 10-12-per-side sets
(never trained on).

Method: per cell, pool the distinct placements among 4 x 8 saved samples (step05v2, rl6, step05s1, rl4) plus
the stored Euler placement (712 cells, 14,211 placements, about 20 per cell; the five 6-10-input sets cut to 60
cells each), and route every placement at every T = 1..8. Full report: `out/tradeoff/report.md`
(`python -m cellgym.llm.tradeoff --report`).

- **Area is a width problem, and width is a column-count problem.** Within a cell, area follows the column
  count (r = 0.93) and width (r = 0.89), not tracks (r = 0.04). Within-cell variance of log area is 116%
  width, 22% height, -38% covariance: narrower placements tend to need more tracks (r = -0.38).
- **Column-for-track trades are rare and small.** A placement with more than the fewest columns is the
  smallest in 20 of 712 cells (2.8%; 14 of them by fewer tracks), worth 0.08% of area averaged over all cells
  (2.8% where it exists, max 9.5%). One track is 5.7% of the height and one column 4.7% of the width, so a
  break buys a track at about break-even. On 21 small in-distribution cells with ~300 random legal placements
  each the answer is the same (1 cell, 0.10% mean).
- **The policy's gap to the best known placement is columns.** SL 12.3% = width 12.0% x height 0.3%; rl6
  9.0% = width 8.6% x height 0.4%. On the stack and 10-12-per-side sets the height part is 1.3% and comes with
  1.3 extra columns (breaks), so removing breaks recovers it; it is not a trade.
- **Ordering at a fixed column count is a real lever**, about 4% area range among a cell's fewest-column
  placements, mostly through width.
- **The best known placement is about 1.5% under the Euler reference** (below it in 45% of cells), so Euler
  is a slightly soft target.
- **Column count is a free, instant proxy for area.** Picking, among a cell's 8 samples, those with the
  fewest columns (no routing; ties averaged) against the routed best-of-8, area / Euler area:

| set | trained up to -> tested | SL mean of 8 | SL fewest-column pick | SL routed best-of-8 | rl6 mean of 8 | rl6 fewest-column pick | rl6 routed best-of-8 |
|---|---|---|---|---|---|---|---|
| val / test_id | in distribution | 1.085 / 1.069 | 1.015 / 1.015 | 1.002 / 1.002 | 1.053 / 1.034 | 1.005 / 1.005 | 0.996 / 0.998 |
| ood6-10 (60 cells each) | 5 -> 6-10 inputs | 1.07-1.09 | 1.010-1.016 | 1.00-1.01 | 1.05-1.06 | 1.006-1.013 | 0.99-1.01 |
| oodstack5 / 6 | stack 4 -> 5 / 6 | 1.130 / 1.124 | 1.045 / 1.051 | 1.029 / 1.034 | 1.084 / 1.096 | 1.032 / 1.029 | 1.017 / 1.016 |
| ooddev10 / 11 / 12 | 9 -> 10 / 11 / 12 per side | 1.166 / 1.173 / 1.203 | 1.044 / 1.051 / 1.069 | 1.031 / 1.040 / 1.054 | 1.104 / 1.142 / 1.152 | 1.024 / 1.056 / 1.049 | 1.010 / 1.043 / 1.041 |

  The fewest-column samples include the routed best in 89-100% of cells and sit 1.0-1.7% above it on average.

What this changes:

1. The "make tracks learnable" line is closed: there is no meaningful track lever in this template. The
   `open:` state line is not needed for it (it did nothing in SL either).
2. Selection among samples does not need the router. Sample many placements, keep the fewest-column ones,
   route only those: a larger k costs LM time only.
3. Breaks are known step by step, so RL can use a dense per-step reward (the plan's potential-based shaping)
   and search (beam over columns with the policy as prior) can run without routing.

Caveats: the pool is a sample from policies that imitate Euler placements, so "no trade found" is evidence,
not proof; 27.7% of placements have a T curve truncated at a capped search (this can hide a win by more
tracks, never by fewer); the multi-stage probes are not in this analysis.

## Router-free selection against k (2026-11-24, CPU only, `cellgym/llm/selectk.py`)

HANDOFF step 1, first half. **Policies were trained on** <= 5 inputs, <= 9 per side, stack <= 4 (SL = step05v2,
RL = rl6); the cells are val / test_id (in distribution) and the never-trained 6-10 input, stack 5-6 and
10-12-per-side sets (710 cells with an Euler area reference and all 8 samples in the trade-off cache).
Nothing is routed here: the routed curves come from `out/tradeoff/placements.jsonl`. Per k, averaged over
random k-subsets of a cell's 8 samples: `fewcol` = the fewest-column samples with ties averaged (no routing),
`fewroute` = the best of the fewest-column samples (route only those), `best` = routed best-of-k. Area / Euler
area; full table for k = 1..8: `out/tradeoff/selectk.md` (`python -m cellgym.llm.selectk`).

| group | trained up to -> tested | policy | k=1 | k=2 fewroute / best | k=4 fewroute / best | k=8 fewcol | k=8 fewroute / best | placements routed at k=8 (of distinct) |
|---|---|---|---|---|---|---|---|---|
| val / test_id (40) | in distribution | SL | 1.076 | 1.038 / 1.038 | 1.015 / 1.015 | 1.015 | 1.002 / 1.002 | 2.4 of 5.4 |
| | | rl6 | 1.041 | 1.016 / 1.016 | 1.004 / 1.004 | 1.005 | 0.997 / 0.997 | 2.4 of 4.2 |
| ood6-10 (300) | 5 -> 6-10 inputs | SL | 1.079 | 1.040 / 1.040 | 1.017 / 1.017 | 1.012 | 1.004 / 1.003 | 2.4 of 6.5 |
| | | rl6 | 1.053 | 1.028 / 1.027 | 1.012 / 1.012 | 1.011 | 1.002 / 1.002 | 2.4 of 5.3 |
| stack 5 / 6 (249) | stack 4 -> 5 / 6 | SL | 1.127 | 1.081 / 1.080 | 1.052 / 1.051 | 1.049 | 1.033 / 1.032 | 2.5 of 7.1 |
| | | rl6 | 1.092 | 1.058 / 1.058 | 1.034 / 1.034 | 1.030 | 1.017 / 1.016 | 2.3 of 5.8 |
| 10-12 per side (121) | 9 -> 10-12 per side | SL | 1.172 | 1.111 / 1.110 | 1.068 / 1.067 | 1.049 | 1.037 / 1.035 | 1.8 of 7.4 |
| | | rl6 | 1.117 | 1.069 / 1.068 | 1.039 / 1.038 | 1.032 | 1.021 / 1.019 | 2.1 of 6.2 |

- **Routing only the fewest-column samples loses 0.000-0.002 of area against routing all of them**, at every k
  and on every set, while routing about a third of the distinct placements (1.8-2.5 of 4-7 at k = 8).
- **Not routing at all** (average of the fewest-column ties) costs 0.8-1.7% against the routed best at k = 8.
- **Best-of-k has not levelled off at k = 8.** The step from k = 4 to k = 8 is still 1.3-3.2% for SL and
  0.7-1.9% for rl6 (largest at stack 5-6 and 10-12 per side). The number routed grows much slower than k
  (1.0 -> 2.4 from k = 1 to 8), so a larger k costs LM time and little routing.
- SL at k = 8 (1.002-1.037) is ahead of rl6 at k = 1 (1.041-1.117) on every group and close to rl6 at k = 4:
  RL and sampling more are partly substitutes; rl6 at k = 8 is still the best point everywhere.

Not shown here: k > 8 (no saved samples; needs the weights, which run11 below keeps) and the 2-3 stage probes.

## Local run11: fixed entropy coefficient and per-step break shaping (2026-11-24, RTX 3090, `run11_2026_11_24.sh`)

**Trained on** (all models): <= 5 inputs, <= 9 devices per side, stack <= 4, 1 stage, 1 output, v2 state.
step05v2 = the run10 SL recipe retrained locally (seed 1, 1053 optimizer steps, 76 min). rl8 = step05v2 + 60 RL
steps as rl6 (32 cells x 8, lr 4e-6, beta 0.02, area reward, `--select both`) but with a **fixed** entropy
coefficient 0.05 instead of the adaptive controller. rl9 = rl8's recipe plus `--break-shape 1` (per-step
advantage bonus for the steps taken before each diffusion break). Each RL run took about 1.6 h.
**Tested on** the same never-seen sets as run10. Weights are kept under `out/llm/{step05v2,rl8,rl9}` on the 3090 PC.

Area / Euler area as mean, best-of-k (free tracks, k = 8, stage probes k = 4):

| set | trained up to -> tested | SL v2 local | rl8 (fixed 0.05) | rl9 (+ break shaping) | rl6 (pod, adaptive) |
|---|---|---|---|---|---|
| val / test_id | in distribution | 1.098, 1.002 / 1.079, 0.994 | 1.044, 1.013 / 1.036, 1.000 | 1.057, 1.011 / 1.023, 0.998 | 1.053, 0.996 / 1.034, 0.998 |
| ood6 / ood7 / ood8 dev | 5 -> 6 / 7 / 8 inputs | 1.085, 0.999 / 1.077, 1.002 / 1.073, 1.008 | 1.057, 1.013 / 1.053, 1.011 / 1.054, 1.019 | 1.053, 1.013 / 1.049, 1.014 / 1.055, 1.023 | 1.05-1.06, 1.00 |
| ood9 / ood10 | 5 -> 9 / 10 inputs | 1.078, 1.007 / 1.064, 1.009 | 1.053, 1.021 / 1.042, 1.012 | 1.052, 1.023 / 1.040, 1.016 | 1.057, 1.002 / 1.046, 1.004 |
| oodstack5 dev | stack 4 -> 5 | 1.128, 1.017 | 1.079, 1.025 | 1.076, 1.026 | 1.084, 1.017 |
| oodstack6 | stack 4 -> 6 | 1.116, 1.018 | 1.069, 1.018 | 1.075, 1.024 | 1.096, 1.016 |
| ooddev10 dev | 9 -> 10 per side | 1.163, 1.028 | 1.094, 1.033 | 1.107, 1.031 | 1.104, 1.010 |
| ooddev11 dev (19) | 9 -> 11 per side | 1.179, 1.045 | 1.136, 1.068 | 1.132, 1.066 | 1.142, 1.043 |
| ooddev12 (15) | 9 -> 12 per side | 1.208, 1.063 | 1.158, 1.047 | 1.136, 1.046 | 1.152, 1.041 |
| stage2 (30) | 1 -> 2 stages | 1.227, 1.101 | 1.127, 1.076 | 1.118, 1.073 | 1.173, 1.077 |
| stage3 (with a reference) | 1 -> 3 stages | 1.346, 1.173 | 1.239, 1.152 | 1.239, 1.150 | 1.291, 1.144 |

Per-sample pass is 100% everywhere except i12d20 (SL 98.8, rl9 98.8) and stage3 (SL 99.2, rl8 98.3, rl9 95.0:
six `route:no-routing` samples); pass@k is 100% on every set for all three.

- **The local SL model reproduces the pod's SL v2** (run10 table) within about a point, mean and best-of-8.
- **A fixed coefficient of 0.05 does not hold diversity.** rl8 improves the mean on every set (as much as rl6,
  more on the stage probes) but best-of-8 gets about 1% worse than SL on the in-distribution and 6-10 input
  sets, the rl4 pattern. Training entropy fell from 0.29 to 0.13 (43%) and distinct placements per 8 from 7.3 to
  4.9; on the 96 eval cells distinct went 3.76 -> 2.58.
- **rl6's advantage came from its controller's late push, not from a small bonus.** rl6's log follows rl8's to
  step 40 (entropy 0.13, eval distinct 2.35); its coefficient then averaged 0.21 over steps 40-50 and 0.74 over
  steps 50-60, and eval distinct went back to 3.33 with best-of-8 1.039 -> 1.026. So the useful coefficient is
  in the 0.2-1.0 range; "fixed small" (the run10 recommendation) was the wrong reading of that run.
- **Break shaping changes nothing measurable.** rl9 against rl8 differs by at most 0.01 on every set with 75+
  cells, in both directions; training curves (area, entropy, distinct) coincide. Its only visible effect is
  more failed samples on stage3. Not worth keeping on.
- Tracks again did not move (training mean 3.33 -> 3.37).

Next (run13): the same 60 steps with a fixed coefficient of 0.3 and 0.6, no break shaping.

## Larger k with fewest-column selection (2026-11-24, local, `run12_2026_11_24.sh`)

**Trained on**: as run11 (<= 5 inputs, <= 9 devices per side, stack <= 4, 1 stage). **Tested on** the never-seen
sets below. One fresh draw of 32 placements per cell from each model; only the fewest-column distinct placements
among the first k are routed (`score --select fewcol --k`), and the best routed area is reported as area / Euler
area. Sampling 32 per cell over all 13 sets took 35-70 min per model on the 3090, scoring 17-30 min for all three k.

| set | trained up to -> tested | SL v2: k=8 / 16 / 32 | rl8: k=8 / 16 / 32 | rl9: k=8 / 16 / 32 |
|---|---|---|---|---|
| val (15) | in distribution | 1.020 / 0.998 / 0.997 | 0.997 / 0.995 / 0.995 | 1.016 / 1.016 / 1.008 |
| test_id (25) | in distribution | 1.010 / 0.994 / 0.988 | 1.005 / 0.994 / 0.994 | 1.004 / 1.002 / 1.000 |
| ood6 dev | 5 -> 6 inputs | 1.002 / 0.992 / 0.987 | 1.011 / 1.005 / 1.002 | 1.019 / 1.012 / 1.008 |
| ood7 dev | 5 -> 7 inputs | 1.004 / 0.994 / 0.987 | 1.007 / 1.003 / 0.997 | 1.015 / 1.010 / 1.004 |
| ood8 dev | 5 -> 8 inputs | 1.005 / 1.000 / 0.992 | 1.015 / 1.010 / 1.005 | 1.018 / 1.012 / 1.004 |
| ood9 | 5 -> 9 inputs | 1.006 / 0.998 / 0.991 | 1.019 / 1.013 / 1.007 | 1.022 / 1.018 / 1.013 |
| ood10 | 5 -> 10 inputs | 1.009 / 1.002 / 0.998 | 1.013 / 1.007 / 1.004 | 1.014 / 1.011 / 1.009 |
| oodstack5 dev | stack 4 -> 5 | 1.016 / 0.996 / 0.987 | 1.021 / 1.014 / 1.006 | 1.027 / 1.014 / 1.009 |
| oodstack6 | stack 4 -> 6 | 1.021 / 1.007 / 0.996 | 1.019 / 1.009 / 1.002 | 1.025 / 1.015 / 1.008 |
| ooddev10 dev | 9 -> 10 per side | 1.024 / 1.006 / 0.998 | 1.029 / 1.026 / 1.017 | 1.034 / 1.024 / 1.016 |
| ooddev11 dev (19) | 9 -> 11 per side | 1.029 / 1.017 / 1.001 | 1.081 / 1.063 / 1.039 | 1.061 / 1.049 / 1.043 |
| ooddev12 (15) | 9 -> 12 per side | 1.069 / 1.046 / 1.018 | 1.059 / 1.038 / 1.020 | 1.024 / 1.021 / 1.017 |

Placements routed per cell at k = 8 / 16 / 32: SL 1.6-2.5 / 2.3-3.3 / 3.0-4.8, rl8 1.5-2.2 / 1.5-2.5 / 1.8-2.9,
rl9 1.3-2.2 / 1.5-2.7 / 1.8-3.2. Pass is 100% on every set at every k. The stage probes were not sampled at k = 32.

- **Plain SL at k = 32 beats both RL models at k = 32 on every unseen set except 12 per side (level, 15
  cells), and matches or beats Euler on all the others** (0.987-1.001; 1.018 at 12 per side). SL keeps gaining 0.5-1.5% per doubling of k; the RL
  models gain about half that, because their extra samples are repeats (routed per cell barely grows).
- **The cost is small.** 32 samples per cell route only 3-5 placements with fewest-column selection, about what
  routing all of k = 8 costs, and sampling is the cheap side.
- **So this RL recipe is not worth its diversity loss at a fixed inference budget above k = 8.** rl8/rl9 are
  better than SL only at k = 1 (the mean) and roughly level at k = 8. An RL policy would have to keep SL's
  diversity while raising the mean to be worth deploying over "SL, sample more"; that is what the run13
  coefficient sweep tests.
- The first 8 of this draw differ from run11's k = 8 draw by up to 0.02 on the 15-25 cell sets (val SL 1.020
  here against 1.002 there), which is the draw-to-draw noise at that size; the 75+ cell sets agree within 0.005.
