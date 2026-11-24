#!/bin/bash
# Local reproduction of step05 (step-wise placement SL, Qwen2.5-0.5B) on one 24 GB GPU.
# Run from WSL after env/LOCAL.md:   bash cellgym/llm/run_local.sh [NAME] [SEED] [SETS...]
# Needs out/cells, out/synth5/plans.jsonl and out/llm/data (see env/LOCAL.md). Progress -> out/llm/progress
cd "$(dirname "$0")/../.." || exit 1
source "${LG_ENV:-$HOME/lg-env}/bin/activate"
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NAME=${1:-step05_local}; SEED=${2:-0}; shift 2 2>/dev/null
SETS=${*:-val test_id}                       # add ood sets once they exist under out/
J=${JOBS:-14}; D=out/llm; P=$D/progress
echo "$NAME start $(date)" >> $P
python -m cellgym.llm.train --model Qwen/Qwen2.5-0.5B --data $D/data/step_train.jsonl --val $D/data/step_val.jsonl \
  --out $D/$NAME --epochs 1 --micro-tokens 4096 --tokens-per-step 65536 --lr 5e-5 --seed $SEED > $D/${NAME}_train.log 2>&1 \
  || { echo "$NAME TRAIN FAILED $(date)" >> $P; exit 1; }
echo "$NAME trained $(date)" >> $P
python -m cellgym.llm.infer --model $D/$NAME/model --mode step --k 8 --sets $SETS --out $D/$NAME/samples.jsonl > $D/${NAME}_infer.log 2>&1 \
  || { echo "$NAME INFER FAILED $(date)" >> $P; exit 1; }
python -m cellgym.llm.score $D/$NAME/samples.jsonl --jobs $J > $D/${NAME}_score.txt 2>&1
echo "$NAME DONE rc=$? $(date)" >> $P
