# Running locally (RTX 3090 PC, 2026-11-23)

Machine: RTX 3090 24 GB, 16 threads, WSL2 Ubuntu 26.04 with 28 GB RAM. Everything
below runs in WSL from the Windows checkout (`/mnt/c/Users/<you>/LayoutGen`);
`out/` is gitignored and has to be rebuilt (or copied) per machine.

## What runs here

| Work | Local? | Notes |
|---|---|---|
| Netlists, DSL, renderer, fast checker, baselines, tests | yes | stdlib; 233 tests pass in WSL |
| Synthetic corpus + dev/probe sets (CPU) | yes | synth5 takes 20 min at 15 jobs; 11-12-per-side Euler labels are the slow part |
| Tiny GPT pretrain / GRPO (`cellgym.synth`, `cellgym.rl`) | yes | was a 4 GB 3050; a 4090 pod did 4000 steps in ~7 min |
| Qwen2.5-0.5B SL (`cellgym.llm.train`) | yes | 24 GB, same as the 4090 pod; measured 69 min against the pod's 32 |
| Step-policy RL (`cellgym.llm.rl`) | yes | pod run was 4.9 h on an A40 (roughly 3090 speed): an overnight job |
| Bigger bases (1.5B-3B full fine-tune, 7B LoRA) | yes, not yet tried | needs `--grad-ckpt` or smaller `--micro-tokens` |
| Ground truth (KLayout + Magic + Netgen) | not yet | `SETUP.md` targets Ubuntu 22.04; this WSL is 26.04, where the KLayout `.deb` deps (Qt5, libruby3.0, libgit2-1.1) are gone. Needs an Ubuntu-22.04 WSL distro or a ported setup. Only audits need it; training uses the fast checker |

No pod is needed for anything on the current roadmap. A pod only buys wall-clock
(parallel runs) from here on.

## Setup (no sudo)

```bash
uv venv --python 3.12 ~/lg-env
uv pip install -p ~/lg-env/bin/python torch --index-url https://download.pytorch.org/whl/cu128
uv pip install -p ~/lg-env/bin/python "transformers>=4.46,<5" accelerate numpy pytest safetensors
source ~/lg-env/bin/activate && cd /mnt/c/Users/<you>/LayoutGen && export PYTHONPATH=.
python -m cellgym.netlists.generate --out out/cells      # 5 s
python -m cellgym.baselines.euler --out out/plans        # 40 s
python -m pytest tests -q                                # 2.5 min; rl tests need out/cells
```

Installed here: torch 2.11.0+cu128, transformers 4.57.6. Run from Git Bash with
`MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash /mnt/c/...script.sh` (Git Bash rewrites
`/mnt/...` paths otherwise).

## Rebuilding `out/`

```bash
python -m cellgym.synth.generate --out out/synth5 --cache out/synth5/topologies.json --max-inputs 5 \
  --max-stack 4 --extra-devices 4 --max-devices 9 --renames 6 --jobs 14
#   -> 31184 plans, 10079 cells, 1934 topologies (about 20 min); matches run2's corpus exactly.
#   No --placements: synth5 used the default (4). With --placements 3 you get 24,616 plans / 1,932 topologies.
python -m cellgym.llm.build --out out/llm/data --steps-per-plan 0
bash cellgym/llm/run_local.sh step05_local 0             # SL + sample + score on val/test_id
```

Reproduction check (2026-11-24, `step05_local`, seed 0): 539,063 step examples and 944
optimizer steps, identical to the pod run; training 69 min (4090 pod: 32 min), sampling
34 s for val + test_id at k=8.

| set | per-sample / pass@8 | W/W_euler mean / best-of-8 | pod step05 |
|---|---|---|---|
| val (15) | 100 / 100 | 1.108 / 0.999 | 100 / 100, 1.108 / 0.994 |
| test_id (25) | 100 / 100 | 1.094 / 1.003 | 100 / 100, 1.092 / 0.999 |

`out/ood6` rebuilt with default arguments (1 min): 968 cells, 801 pool / 167 dev, as in
`bench/pretrain_runs.md`.

The other eval sets, the probes, the area references and the trade-off cache come from the bundles
in `bench/results/` (see `bench/HANDOFF.md`, "Restore the local state"); the bundled `out/ood6` is
identical to the local rebuild. Each `out/<set>_build.txt` records that set's generator arguments.

Local runs: `cellgym/llm/run11_2026_11_24.sh` (SL + two 60-step RL runs, weights kept under
`out/llm`). Keep the PC from sleeping while it runs.

Later the same day: `run12` (k = 32 sampling, fewest-column scoring), `run13` (entropy coefficient sweep),
`run14` (stage probes at k = 32 under three spellings) and `run15` (SL on synth5 + the 2-stage corpus,
`out/llm/step05v2s`, 91 min). Rebuild the 2-stage corpus (6 min at 8 jobs):

```bash
python -m cellgym.llm.stagecorpus --out out/synth2s --stages 2 --count 2500 --per-side 3:9 --seed 9100   --exclude out/probe/stage2 out/probe/stage3 --jobs 8      # 7,975 plans, 2,413 cells
```

Do not edit a `run*.sh` while it is running: bash reads the file as it goes and the run breaks.
