# Handoff: continuing on another machine (written 2026-11-24)

Branch `m4-benchmark`. Everything needed to pick the work up is in this repo; nothing below depends on
the machine it was written on. Full results and reasoning: [`llm_runs.md`](llm_runs.md) (read the last
three sections first). The project plan and open-item list live in the private notebook doc on claude.ai,
not in this repo.

## Where the work stands

- **Trained on:** <= 5 inputs, <= 9 devices per side, stack <= 4, 1 stage, 1 output.
  **Tested on:** 6-10 inputs, stack 5-6, 10-12 per side, probes up to 16 inputs / 24 per side, 2-3 stages.
  The top rung of every axis stays unseen; walls are fixed by representation, never by training on them.
- The routing wall was the fixed 3-track template. With channel tracks as a plan parameter (1-8) every
  set passes, and the objective is area (width x height at the area-optimal track count).
- Step-wise Qwen2.5-0.5B placement policy + classical router. Entropy-regularized area RL (rl6) improves
  mean and best-of-8 on every set, including the unseen ones. Composition (2-3 stages) is the weakest axis.
- Area is a column-count problem (r = 0.93); there is no track lever. The fewest-column sample of 8, picked
  without routing, lands 1.0-1.7% above the routed best-of-8.

## Next steps (in order)

1. **Free, CPU only, about an hour.** Router-free selection (sample, keep the fewest-column placements,
   route only those) and a per-step break reward in `cellgym/llm/rl.py` (potential-based shaping; breaks
   are known as each column is placed). Measure best-of-k against k = 1..8 on the saved samples; the routes
   are cached in `out/tradeoff/placements.jsonl` (restored below).
2. **Paid, about 5 h on an A40, about $3. Needs a new budget.** SL + 60 RL steps with a *fixed small*
   entropy coefficient (the adaptive controller overshot in rl7), the per-step reward, larger k, and the
   weights pulled to the PC this time. Launch under `cellgym/llm/pod_watch.sh`.
3. Open: Euler area references for the 14-24-per-side probes (needs a compact search); rerun the trade-off
   analysis on the 2-3 stage probes; second seed; rl4 -> rl6 confounds the entropy term with the state change.

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
(KLayout, Magic, Netgen) needs the WSL2 environment in [`../env/SETUP.md`](../env/SETUP.md); nothing in
the next steps requires it.

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
- Environment switches: `CELLGYM_STATE=v1|v2` (step state, default v2), `CELLGYM_AREAREF=heur|euler+heur`.
