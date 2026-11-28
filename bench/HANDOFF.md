# Handoff: continuing on another machine (written 2026-11-24, updated 2026-11-27 after the fourth local session)

Branch `m4-benchmark`. Everything needed to pick the work up is in this repo; nothing below depends on
the machine it was written on. Full results and reasoning: [`llm_runs.md`](llm_runs.md) (read the last
six sections first). The project plan and open-item list live in the private notebook doc on claude.ai,
not in this repo.

## Where the work stands (updated 2026-11-28, 03:00, fifth local 3090 session)

- **Scale path (2026-11-27/28, `llm_runs.md` last three sections).** A search-free randomised greedy placer
  (`cellgym/llm/classical.py`) matches or beats the release model on columns on every set and keeps working to
  64 per side (within 1.6-3.3% of the column floor). `step05big` (run22: step05v3s_e2 + 1 epoch with 8,000
  cells at 10-32 per side and 3-4 stages, labelled by that placer, `cellgym/llm/bigcorpus.py`) equals the
  placer's best of 1,000 with 32 samples at 14-24 per side (66 of 70 cells), is level with it at equal sample
  count there, and is behind from 32 per side up; it is ahead on 1 of 110 cells and far slower. It is not a
  release candidate: standard test sets, routing and area were not run on it. Open choices, none started:
  (a) labels better than the heuristic (the model's own best samples or a search seeded by them, then retrain);
  (b) use the model inside the heuristic (as the tie-break) instead of against it; (c) stop the scale path and
  release step05v3s_e2 with the classical placer stated as the stronger column method. Keep CPU jobs at 8 and
  cap GPU memory in training (run22 filled the card; the user wants headroom).
- **Release candidate: `step05v3s_e2`** (state v3, 2 epochs; second seed `step05v3s_e2_s2`). It replaces
  step05v2s: level at best-of-32, better at best-of-8 on the far sets, stage probes with selection measured
  (0.995 / 1.014 on 2 / 3 stages). Both seeds agree to 0.006 at best-of-32 on every set with 75 cells or more.
  **Needs `CELLGYM_STATE=v3` on every infer/score/distill command.**
- **Ground truth now runs on the 3090 PC** (WSL distro `Ubuntu-22.04`, see `env/LOCAL.md`). All 4,137 layouts deployed by step05v2s and both seeds of step05v3s_e2 (single-stage sets, stage probes, large probes) are clean under KLayout DRC, Magic DRC and Netgen LVS; the fast checker agreed on every one. On the 1,234 single-stage test cells an exact solver finishes, step05v3s_e2 is at the minimum column count on 1,177.
  Export with `python -m cellgym.llm.gtexport`, audit with `python -m cellgym.check.audit`; summaries in
  `bench/results/gt_audit_2026_11_26/`.
- **Large probes** (8-16 inputs, 14-24 per side; `run21_2026_11_26.sh`): at k = 32 every cell passes and the
  column count is within 2.4-4.5% of a search-free lower bound (`regret.break_lower_bound`,
  `python -m cellgym.llm.bound`). The Euler search and the exact solver do not finish there.
- `cellgym/llm/reproduce.sh [SEED]` chains corpora, training, evaluation and layout export for the release
  model. Run end to end from a clean clone on 2026-11-27 (6 h 45 min after corpora): data byte-identical, results
  within 0.004 of seed 1 on single-stage sets, one 3-stage cell of 30 without a passing layout (the fewest-column
  rule has no fallback to the next column count; worth adding).
- `README.md` has a "Learned placement model" section with the release tables and the stated limits.
- Open for discussion with the owner: the headline is parity with the Euler baseline (0.98-1.01), not a win;
  the argument for the model is that it keeps working where the search stops. Weights are not published
  (publishing needs the owner's go-ahead).

- 2026-11-26 additions (last four sections of `llm_runs.md`): `cellgym/llm/regret.py` gives the exact per-step
  break regret of a placement; state v3 (`CELLGYM_STATE=v3`, an `odd:` line) and two models on it,
  `step05v3s` (1 epoch) and `step05v3s_e2` (2 epochs), script `run20_2026_11_26.sh`. **v3 models need
  `CELLGYM_STATE=v3` on every infer/score/distill command.** step05v3s_e2 is level with step05v2s at
  best-of-32, better at best-of-8 on the far sets, and its stage probes with selection are unmeasured, so
  step05v2s stays the current model until run14 is run on it.
- 2026-11-25 additions: a second seed of step05v2s (`step05v2s_s2`, run16) reproduces the 3-stage result;
  the single-shot mean was studied (run17 temperature, run18 distillation). `step05v2s_d1` / `_d2` are the
  distilled models (better single-shot mean, d1 same best-of-32).

- Previous model: `step05v2s` (SL only, Qwen2.5-0.5B step policy, v2 state; weights in `out/llm/step05v2s/model`
  on the 3090 PC, not in the repo). **Trained on:** <= 5 inputs, <= 9 devices per side, stack <= 4, 1-2 stages,
  1 output. **Tested on:** 6-10 inputs, stack 5-6, 10-12 per side, probes up to 16 inputs / 24 per side,
  3 stages with 1-2 outputs. The top rung of every axis stays unseen.
- **Deployment recipe: sample k = 32, route only the fewest-column placements** (`score --select fewcol --k 32`;
  3-5 routed per cell). With it step05v2s is at or below Euler area on 6-10 inputs, stack 5-6 and 10 per
  side, 1.011 at 11 and 1.003 at 12 per side, 1.005 on 2-stage and 1.017 on unseen 3-stage cells (one draw;
  level with step05v2 on the single-stage sets).
- Everything runs locally now (`env/LOCAL.md`); no pod is needed. Scripts `cellgym/llm/run11..run16_2026_11_24.sh`, `run17..run18_2026_11_25.sh`, `run19..run20_2026_11_26.sh`.
- Findings of this session (details in `llm_runs.md`, the 2026-11-24 sections):
  - RL with an entropy bonus is not worth it at k > 8: fixed coefficient 0.05 collapses diversity (rl8), 0.3 and
    0.6 diverge (rl10, rl11), per-step break shaping is neutral (rl9), and "SL, sample more" beats every RL
    model at k = 32. rl6's good result came from its controller's late push, not from a small bonus.
  - The composition gap was diffusion breaks at stage boundaries. More samples and a familiar spelling of the
    stage nets help a little; putting 2-stage cells in the corpus (`cellgym/llm/stagecorpus.py`) fixes most of
    it, carries to 3 stages, and costs nothing on single-stage cells.
- Area is a column-count problem (r = 0.93); there is no track lever.

## Next steps (in order)

1. The remaining gap is the **mean** (k = 1): 6-19% above Euler on unseen sets at T = 0.7, and it is
   already present on the training cells (a fitting / lookahead problem, not generalisation). Measured so far,
   greedy label-match on the 5,000 cells of `out/distill1` (7 min per model, the metric to move first):
   step05v2s 42%, + state v3 50%, + second epoch 62%; distillation rounds on v2 42% -> 54% -> 59%. Every one of
   these improves the fit to training cells much more than held-out greedy area, which moves by 0.00-0.02 and
   gets worse on 3 stages for both the 2-epoch and the twice-distilled model. Per-step regret says the loss is
   now spread over all steps (start 10%, gate/sd 2% each, restarts 15% wrong), no single target left. Next, in
   order: (a) done 2026-11-26 (run14 and run17 on `step05v3s_e2`: it replaces step05v2s); (b) distill from step05v3s_e2 (run18 with
   `CELLGYM_STATE=v3`, `BASE=step05v3s_e2`, data built in v3) since distillation gave the best held-out greedy;
   (c) the 3-stage regression under better fitting: more 2-stage cells or 3-stage cells in the corpus;
   (d) a larger base (Qwen2.5-1.5B). More epochs on state v2 alone was not measured (`run19` exists, ~7 h for 3).
2. Area references for the 14-24-per-side probes (needs a compact search). Columns now have a lower bound
   (`cellgym.llm.bound`); area there is still unreferenced.
3. Second seed for step05v2 (single-stage training only; still one seed). step05v2s has two
   (`out/llm/step05v2s_s2`, run16): the 3-stage result reproduces (best-of-32 1.014 / 1.007), seed spread is
   <= 0.002 at fewest-column best-of-32 on all single-stage sets but 12 per side (0.011), and up to 0.025 on
   the plain k = 8 mean.
4. Multi-output training cells (the 3-stage probe's `OUT1 OUT2` are still unseen tokens), then a 4-stage probe
   as the new top rung.
5. Release packaging: decide where the weights go, write a model card from the README section, add a
   fallback to the next column count in fewest-column selection and re-measure the 3-stage pass rate.

## Restore the local state

`out/` is git-ignored. No model weights from run7 or run10 exist anywhere (the pods are gone), so the
saved samples are the only record of those policies. They are bundled in [`results/`](results/):

```bash
mkdir -p out/llm/run7 out/llm/run10/a out/llm/run10/b out/llm/run10/c
tar xzf bench/results/results_run7.tgz    -C out/llm/run7
tar xzf bench/results/results_run10_a.tgz -C out/llm/run10/a   # step05v2 (SL, v2 state)
tar xzf bench/results/results_run10_b.tgz -C out/llm/run10/b   # rl6
tar xzf bench/results/results_run10_c.tgz -C out/llm/run10/c   # rl7
tar xzf bench/results/local_state_2026_11_24.tgz               # out/arearef, out/tradeoff, out/probe
tar xzf bench/results/eval_sets_2026_11_24.tgz                 # out/ood6..10, oodstack5/6, ooddev10..12, adder
```

The last bundle holds the unseen-axis test sets as built (each `out/<set>_build.txt` records the
generator parameters). `cellgym/llm/infer.py` `SETS` maps set names to these directories.

Rebuild the rest (all free, deterministic):

```bash
python -m cellgym.netlists.generate --out out/cells
python -m cellgym.baselines.euler --out out/plans
python -m cellgym.llm.build --out out/llm/data_v2 --steps-per-plan 0   # SL data, v2 state
python -m cellgym.llm.tradeoff --report                                # rewrites out/tradeoff/report.md
python -m pytest tests                                                 # last full run: 302 passed, 11 skipped
```

A pod run needs the repo plus `out/arearef`, `out/probe` and the test-set directories uploaded to the pod
(the run scripts expect them shipped, not built there). Ground truth
(KLayout, Magic, Netgen) needs the WSL2 environment in [`../env/SETUP.md`](../env/SETUP.md); on the 3090 PC
it is the `Ubuntu-22.04` distro (`../env/LOCAL.md`).

## Machine setup for pod runs

- `runpodctl` on the PATH, and an API key in the environment variable `RUNPOD_API_KEY_LAYOUTGEN`
  (Pods read/write + Billing read). `pod_watch.sh` maps it to `RUNPOD_API_KEY` for its own calls only.
  Do not reuse or overwrite any other `RUNPOD_API_KEY` on the machine.
- An SSH key registered with RunPod. Pass the public key as `env.PUBLIC_KEY` when creating the pod;
  `startSsh` alone injected the string "null".
- Image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, then
  `pip install --break-system-packages transformers==4.57.1`.
- A40: billed about $0.59/h (the listed price can be about half the billed rate; check billing right
  after creation). An RTX 5090 billed $1.39/h against a listed $0.69.
- No volume was used: a dropped host loses everything on the pod (run8 was lost this way). Pull bundles
  as they are produced; download from the pod runs at about 4 MB/s. Score locally, not on the pod.
- Watcher: `bash cellgym/llm/pod_watch.sh POD_ID IP PORT DONE_MARKER LOCAL_DIR [MAX_HOURS] [BUNDLE_GLOB]`.
  It pulls bundles on every poll and deletes the pod on the done marker, a dead host, or the deadline.
  Its delete path has not yet been exercised on a real pod, so confirm the pod is gone after the first use.
- Run scripts to copy from: `cellgym/llm/run10_2026_11_19.sh` (latest), `run7_2026_11_18.sh`.

## Working rules carried over

- Never leave a billed pod idle. Terminating it is the first action when a run finishes (an idle pod cost
  about $5 on 2026-11-20; spend that session was $10.20 against a $7 cap).
- Report every result as trained-on against tested-on, per axis.
- Environment switches: `CELLGYM_STATE=v1|v2|v3` (step state, default v2; the release model is v3), `CELLGYM_AREAREF=heur|euler+heur`.
