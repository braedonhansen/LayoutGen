#!/bin/bash
# Reproduce the release model and its evaluation from a clean checkout (one 24 GB GPU, about 14 h on an RTX 3090).
# Python env: env/LOCAL.md ("Setup"). Ground truth (step 6) needs env/SETUP.md and is run separately.
#   bash cellgym/llm/reproduce.sh [SEED=1]         progress in out/llm/progress
# Trained on: <= 5 inputs, <= 9 devices per side, stack depth <= 4, 1-2 stages, 1 output.
set -e
cd "$(dirname "$0")/../.."
source "${LG_ENV:-$HOME/lg-env}/bin/activate"
export PYTHONPATH=. CELLGYM_STATE=v3
S=${1:-1}; T=step05v3s_e2; [ "$S" = 1 ] || T=${T}_s$S
J=${JOBS:-8}

# 1. Benchmark cells, Euler plans, and the never-trained-on test sets / probes / area references (bundled).
[ -d out/cells ] || python -m cellgym.netlists.generate --out out/cells
[ -d out/plans ] || python -m cellgym.baselines.euler --out out/plans
[ -d out/ood6 ] || tar xzf bench/results/eval_sets_2026_11_24.tgz
[ -d out/arearef ] || tar xzf bench/results/local_state_2026_11_24.tgz

# 2. Training corpora (CPU, about 30 min): single-stage synthetic cells and 2-stage cells.
[ -f out/synth5/plans.jsonl ] || python -m cellgym.synth.generate --out out/synth5 --cache out/synth5/topologies.json \
  --max-inputs 5 --max-stack 4 --extra-devices 4 --max-devices 9 --renames 6 --jobs $J
[ -f out/synth2s/plans.jsonl ] || python -m cellgym.llm.stagecorpus --out out/synth2s --stages 2 --count 2500 --per-side 3:9 \
  --seed 9100 --exclude out/probe/stage2 out/probe/stage3 --jobs $J

# 3. Step examples in state v3, then 2 epochs of supervised learning from Qwen2.5-0.5B (about 3.5 h), greedy
#    label-match on training cells, greedy + T = 0.7 means, and k = 32 with fewest-column selection.
TAG=$T EPOCHS=2 SEED=$S bash cellgym/llm/run20_2026_11_26.sh

# 4. 2- and 3-stage probes at k = 32.
VARIANTS=base bash cellgym/llm/run14_2026_11_24.sh out/llm/$T/model $T

# 5. 14-24-per-side probes at k = 32 against the column lower bound.
bash cellgym/llm/run21_2026_11_26.sh $T=out/llm/$T/model

# 6. Layouts for the ground-truth audit (then, in the env/SETUP.md environment:
#    python -m cellgym.check.audit out/gtaudit/<dir> --cells out/gtaudit/<dir> --workdir ~/gt/<dir> --md <dir>.audit.md)
X="i8d14=out/probe/i8d14:all i10d16=out/probe/i10d16:all i12d20=out/probe/i12d20:all i16d24=out/probe/i16d24:all"
python -m cellgym.llm.gtexport out/llm/k32/$T/samples.jsonl --out out/gtaudit/$T --k 32 --jobs $J
python -m cellgym.llm.gtexport out/llm/stage/$T/base.jsonl --out out/gtaudit/${T}_stage --k 32 --jobs $J \
  --extra stage2=out/probe/stage2:all stage3=out/probe/stage3:all
python -m cellgym.llm.gtexport out/llm/probe32/$T/samples.jsonl --out out/gtaudit/${T}_probe --k 32 --jobs $J --extra $X
echo "results: out/llm/k32/$T/score_fewcol_k32.txt  out/llm/stage/$T/base_fewcol_k32.txt  out/llm/probe32/$T/bound_k32.txt"
