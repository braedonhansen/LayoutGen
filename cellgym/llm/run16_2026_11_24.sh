#!/bin/bash
# Second training seed for step05v2s (local): same base, data (out/llm/data_v2s) and recipe as run15, --seed 2.
# Every step05v2 / step05v2s comparison so far rests on one draw per model; this gives the seed-to-seed spread.
#   step05v2s_s2: trained on <= 5 inputs, <= 9 per side, stack <= 4, 1-2 stages, 1 output
# Tested on the stage probes at k=32 (run14, base spelling) and on the single-stage sets at k=32 (run12:
# fewest-column k=8/16/32), plus the plain mean / best-of-8 over the first 8 of those samples.
#   bash cellgym/llm/run16_2026_11_24.sh [SEED]       last progress line RUN16_DONE
cd "$(dirname "$0")/../.." || exit 1
source "${LG_ENV:-$HOME/lg-env}/bin/activate"
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CELLGYM_STATE=v2
S=${1:-2}; D=out/llm; P=$D/progress; N=step05v2s_s$S; O=$D/$N
say() { echo "$* $(date)" >> $P; }
run() { local name=$1 log=$2; shift 2; "$@" > $log 2>&1; local rc=$?
        if [ $rc -eq 0 ]; then say "$name ok"; else say "$name FAILED rc=$rc"; fi; return $rc; }
say "run16 start seed $S"
[ -d $O/model ] || run "$N trained" $D/${N}_train.log python -m cellgym.llm.train --model Qwen/Qwen2.5-0.5B \
  --data $D/data_v2s/step_train.jsonl --val $D/data_v2s/step_val.jsonl \
  --out $O --epochs 1 --micro-tokens 4096 --tokens-per-step 65536 --lr 5e-5 --seed $S || exit 1
VARIANTS=base bash cellgym/llm/run14_2026_11_24.sh $O/model $N
bash cellgym/llm/run12_2026_11_24.sh $N=$O/model
run "$N k8 scored area" $D/k32/$N/score_all_k8.txt python -m cellgym.llm.score $D/k32/$N/samples.jsonl \
  --jobs 8 --tracks area --k 8 --tag all_k8
echo "RUN16_DONE $(date)" >> $P
