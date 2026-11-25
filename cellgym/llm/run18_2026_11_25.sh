#!/bin/bash
# Mean (single-shot) gap, second lever: one round of self-distillation (local). run17 showed greedy decoding
# removes only part of the gap, so the policy's most likely placement is itself above Euler.
#   1. sample k=8 (T=0.7) from step05v2s on 5,000 cells of its own training corpora (synth5 + synth2s)
#   2. keep per cell up to 3 distinct fewest-column placements, if no wider than the Euler label (no router)
#   3. fine-tune step05v2s on those placements only (1 epoch, lr 2e-5) -> step05v2s_d1
# Trained axes do not widen: <= 5 inputs, <= 9 per side, stack <= 4, 1-2 stages, 1 output.
# Tested: greedy and T=0.7 k=8 on all single-stage sets and the stage probes (run17), then k=32 (run12).
#   bash cellgym/llm/run18_2026_11_25.sh        last progress line RUN18_DONE
cd "$(dirname "$0")/../.." || exit 1
source "${LG_ENV:-$HOME/lg-env}/bin/activate"
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CELLGYM_STATE=v2
D=out/llm; P=$D/progress; B=${BASE:-step05v2s}; N=${NAME:-${B}_d1}; O=$D/$N; C=${CORPUS:-out/distill1}
say() { echo "$* $(date)" >> $P; }
run() { local name=$1 log=$2; shift 2; "$@" > $log 2>&1; local rc=$?
        if [ $rc -eq 0 ]; then say "$name ok"; else say "$name FAILED rc=$rc"; fi; return $rc; }
say "run18 start base=$B corpus=$C"
mkdir -p $C
[ -f $C/samples.jsonl ] || run "distill sampled" $C/sample.log python -m cellgym.llm.distill sample --model $D/$B/model \
  --plans out/synth5/plans.jsonl out/synth2s/plans.jsonl --cells ${CELLS:-5000} --k 8 --seed 0 --out $C || exit 1
run "distill selected" $C/select.log python -m cellgym.llm.distill select --out $C --keep 3 || exit 1
[ -f $D/data_$N/step_train.jsonl ] || run "data_$N built" $D/build_$N.log python -m cellgym.llm.build \
  --plans $C/plans.jsonl --out $D/data_$N --steps-per-plan 0 || exit 1
[ -d $O/model ] || run "$N trained" $D/${N}_train.log python -m cellgym.llm.train --model $D/$B/model \
  --data $D/data_$N/step_train.jsonl --val $D/data_v2s/step_val.jsonl \
  --out $O --epochs 1 --micro-tokens 4096 --tokens-per-step 65536 --lr ${LR:-2e-5} --seed 1 || exit 1
TEMPS="0 0.7" bash cellgym/llm/run17_2026_11_25.sh $N=$O/model
[ -n "$SKIP_K32" ] || bash cellgym/llm/run12_2026_11_24.sh $N=$O/model
echo "RUN18_DONE $(date)" >> $P
