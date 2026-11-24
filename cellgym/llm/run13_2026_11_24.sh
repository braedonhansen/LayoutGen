#!/bin/bash
# Local follow-up to run11: the fixed entropy coefficient 0.05 (rl8, rl9) let diversity collapse as in rl4;
# rl6's step-60 policy had been pushed by a coefficient of 0.2-1.0 over its last 20 steps. Sweep the
# fixed coefficient upward from the same SL model:
#   rl10 = 60 steps, --ent-coef 0.3      rl11 = 60 steps, --ent-coef 0.6
# Extra rl.py flags for both runs via RL_FLAGS (e.g. RL_FLAGS="--break-shape 1").
#   bash cellgym/llm/run13_2026_11_24.sh [NAME=COEF ...]     progress: out/llm/progress, last line RUN13_DONE
cd "$(dirname "$0")/../.." || exit 1
source "${LG_ENV:-$HOME/lg-env}/bin/activate"
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CELLGYM_STATE=v2
D=out/llm; P=$D/progress
say() { echo "$* $(date)" >> $P; }
run() { local name=$1 log=$2; shift 2; "$@" > $log 2>&1; local rc=$?
        if [ $rc -eq 0 ]; then say "$name ok"; else say "$name FAILED rc=$rc"; fi; return $rc; }
X=""; for s in i8d14 i10d16 i12d20 i16d24 stage2 stage3; do
  [ -f out/probe/$s/index.json ] && X="$X $s=out/probe/$s:all"; done
R=${*:-rl10=0.3 rl11=0.6}
say "run13 start: $R flags=${RL_FLAGS:-none}"
for x in $R; do
  name=${x%%=*}; ec=${x#*=}; o=$D/$name
  [ -d $o/last ] || run "$name trained" $D/$name.log python -m cellgym.llm.rl --init $D/step05v2/model --out $o \
    --steps ${RL_STEPS:-60} --cells 32 --group 8 --lr 4e-6 --beta 0.02 --reward area \
    --ent-coef $ec --select both --keep-ckpts --eval-every 20 --eval-cells 96 --jobs 8 $RL_FLAGS || continue
  [ -f $o/samples.jsonl ] || run "$name sampled" $D/${name}_infer.log python -m cellgym.llm.infer --model $o/best --mode step --k 8 --out $o/samples.jsonl
  if [ ! -f $o/probe_samples.jsonl ]; then
    for p in $X; do
      s=${p%%=*}
      run "$name probe $s sampled" $D/${name}_probe_${s}_infer.log timeout 2400 \
        python -m cellgym.llm.infer --model $o/best --mode step --k 4 --extra $p --out $o/probe_$s.jsonl \
        && cat $o/probe_$s.jsonl >> $o/probe_samples.tmp
    done
    [ -f $o/probe_samples.tmp ] && mv $o/probe_samples.tmp $o/probe_samples.jsonl
  fi
  ( run "$name scored area" $D/${name}_score_area.txt python -m cellgym.llm.score $o/samples.jsonl --jobs 6 --tracks area --tag area
    [ -f $o/probe_samples.jsonl ] && run "$name probes scored area" $D/${name}_probe_score_area.txt \
      python -m cellgym.llm.score $o/probe_samples.jsonl --jobs 6 --tracks area --tag area --extra $X ) &
done
wait
echo "RUN13_DONE $(date)" >> $P
