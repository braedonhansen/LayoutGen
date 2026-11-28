#!/bin/bash
# Scale: train past the reach of the Euler search, on labels from the classical placer, and test the model
# against that placer at equal budget (bench/llm_runs.md, "A search-free classical placer ...").
# Corpus out/synthbig: 5,000 single-stage cells at 10-32 devices per side (stack 4-6) and 1,500 each of
# 3- and 4-stage cells (12-28 per side, stack <= 4), labelled best-of-1,000 by cellgym.llm.classical
# (heuristic labels, not minimal). Every probe / test-set cell id is excluded. The model continues from
# step05v3s_e2 for one epoch on synth5 + synth2s + synthbig, state v3.
# Trained on after this run: <= 16 inputs, <= 32 per side, stack <= 6, 1-4 stages, 1-2 outputs.
# Tested on: unseen cells at 14-32 per side and 4 stages (inside the new limits), 48 and 64 per side (outside).
#   [TAG=step05big BASE=out/llm/step05v3s_e2/model JOBS=8 BUILD_ONLY=] bash cellgym/llm/run22_2026_11_27.sh
#   last progress line RUN22_DONE. JOBS stays at 8 of 16 threads on purpose (shared machine).
cd "$(dirname "$0")/../.." || exit 1
source "${LG_ENV:-$HOME/lg-env}/bin/activate"
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CELLGYM_STATE=v3
D=out/llm; P=$D/progress; N=${TAG:-step05big}; O=$D/$N; J=${JOBS:-8}
say() { echo "$* $(date)" >> $P; }
run() { local name=$1 log=$2; shift 2; "$@" > $log 2>&1; local rc=$?
        if [ $rc -eq 0 ]; then say "$name ok"; else say "$name FAILED rc=$rc"; fi; return $rc; }
say "run22 start $N"
EX=""; for d in out/probe/i8d14 out/probe/i10d16 out/probe/i12d20 out/probe/i16d24 out/probe/stage2 out/probe/stage3 \
  out/probe2/i16d32 out/probe2/i20d48 out/probe2/i24d64 out/probe2/stage4 \
  out/ood6 out/ood7 out/ood8 out/ood9 out/ood10 out/oodstack5 out/oodstack6 out/ooddev10 out/ooddev11 out/ooddev12 out/adder; do EX="$EX $d"; done
[ -f out/synthbig/plans.jsonl ] || run "synthbig built" out/synthbig.log python -m cellgym.llm.bigcorpus --out out/synthbig \
  --single 5000 --per-side 10:32 --multi 3:1500 4:1500 --tries 1000 --placements 2 --exclude $EX --jobs $J || exit 1
[ -f $D/data_big/step_train.jsonl ] || run "data_big built" $D/build_big.log python -m cellgym.llm.build \
  --plans out/synth5/plans.jsonl out/synth2s/plans.jsonl out/synthbig/plans.jsonl --out $D/data_big --steps-per-plan 0 || exit 1
[ -n "$BUILD_ONLY" ] && exit 0
[ -d $O/model ] || run "$N trained" $D/${N}_train.log python -m cellgym.llm.train --model ${BASE:-$D/step05v3s_e2/model} \
  --data $D/data_big/step_train.jsonl --val $D/data_big/step_val.jsonl \
  --out $O --epochs ${EPOCHS:-1} --micro-tokens 4096 --tokens-per-step 65536 --lr 5e-5 --seed ${SEED:-1} || exit 1
# Evaluation: greedy and k = 32 at T = 0.7 on the probes, against the classical placer.
X=""; for s in i8d14 i10d16 i12d20 i16d24; do X="$X $s=out/probe/$s:all"; done
for s in i16d32 i20d48 i24d64 stage4; do X="$X $s=out/probe2/$s:all"; done
E=$D/scale/$N; mkdir -p $E
[ -f $E/t0.jsonl ] || run "$N probes greedy" $E/t0_infer.log python -m cellgym.llm.infer --model $O/model --mode step --k 1 \
  --temperature 0 --max-tokens 4096 --batch 96 --extra $X --out $E/t0.jsonl
[ -f $E/k32.jsonl ] || run "$N probes k32" $E/k32_infer.log python -m cellgym.llm.infer --model $O/model --mode step --k 32 \
  --max-tokens 4096 --batch 96 --extra $X --out $E/k32.jsonl
python -m cellgym.llm.classical $E/t0.jsonl --k 1 --extra $X --tries 1 32 1000 --jobs $J --out $E/classical_t0.json > $E/classical_t0.txt 2>&1
python -m cellgym.llm.classical $E/k32.jsonl --k 32 --extra $X --tries 1 32 1000 --jobs $J --out $E/classical_k32.json > $E/classical_k32.txt 2>&1
python -m cellgym.llm.bound $E/k32.jsonl --k 32 --extra $X --out $E/bound_k32.json > $E/bound_k32.txt 2>&1
say "$N probes compared"
echo "RUN22_DONE $(date)" >> $P
