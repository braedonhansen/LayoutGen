#!/bin/bash
# Composition as a trained axis (local). SL from the same base and recipe as step05v2 (seed 1, v2 state), on
# synth5 plus the 2-stage corpus out/synth2s (cellgym.llm.stagecorpus; stage probes excluded by cell id):
#   step05v2s: trained on <= 5 inputs, <= 9 per side, stack <= 4, 1-2 stages, 1 output
# Tested on every set as run11 (k=8, probes k=4) and on the stage probes at k=32 (run14, base spelling):
# stage2 = unseen cells of a trained axis (and 10-11 per side unseen), stage3 = 2 -> 3 stages, 1-2 outputs.
#   bash cellgym/llm/run15_2026_11_24.sh        last progress line RUN15_DONE
cd "$(dirname "$0")/../.." || exit 1
source "${LG_ENV:-$HOME/lg-env}/bin/activate"
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CELLGYM_STATE=v2
D=out/llm; P=$D/progress; N=step05v2s; O=$D/$N
say() { echo "$* $(date)" >> $P; }
run() { local name=$1 log=$2; shift 2; "$@" > $log 2>&1; local rc=$?
        if [ $rc -eq 0 ]; then say "$name ok"; else say "$name FAILED rc=$rc"; fi; return $rc; }
say "run15 start"
[ -f $D/data_v2s/step_train.jsonl ] || run "data_v2s built" $D/build_v2s.log python -m cellgym.llm.build \
  --plans out/synth5/plans.jsonl out/synth2s/plans.jsonl --out $D/data_v2s --steps-per-plan 0 || exit 1
[ -d $O/model ] || run "$N trained" $D/${N}_train.log python -m cellgym.llm.train --model Qwen/Qwen2.5-0.5B \
  --data $D/data_v2s/step_train.jsonl --val $D/data_v2s/step_val.jsonl \
  --out $O --epochs 1 --micro-tokens 4096 --tokens-per-step 65536 --lr 5e-5 --seed 1 || exit 1
VARIANTS=base bash cellgym/llm/run14_2026_11_24.sh $O/model $N
X=""; for s in i8d14 i10d16 i12d20 i16d24; do X="$X $s=out/probe/$s:all"; done
[ -f $O/samples.jsonl ] || run "$N sampled" $D/${N}_infer.log python -m cellgym.llm.infer --model $O/model --mode step --k 8 --out $O/samples.jsonl
[ -f $O/probe_samples.jsonl ] || run "$N probes sampled" $D/${N}_probe_infer.log python -m cellgym.llm.infer --model $O/model --mode step --k 4 --extra $X --out $O/probe_samples.jsonl
run "$N scored area" $D/${N}_score_area.txt python -m cellgym.llm.score $O/samples.jsonl --jobs 8 --tracks area --tag area &
run "$N probes scored area" $D/${N}_probe_score_area.txt python -m cellgym.llm.score $O/probe_samples.jsonl --jobs 5 --tracks area --tag area --extra $X &
wait
echo "RUN15_DONE $(date)" >> $P
