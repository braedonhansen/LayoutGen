#!/bin/bash
# RL on cell AREA with a free channel-track count: SL step policy (seed 1, as step05s1) -> RL (rl4) with
# reward = max(0.05, 1 + (A_ref - A) / A_ref), A = width x height at the smallest-area track count
# (route_min_area), A_ref = the cell's Euler placement routed the same way (cellgym.llm.arearef).
# In-distribution cells only. Then every set (+ the probe sets shipped in out/probe) is sampled from
# both models and scored twice: fixed 3 tracks (old columns) and --tracks area.
# Progress -> out/llm/progress (last line RUN7_DONE). A failed stage is logged and later stages that
# do not need it still run; every samples file that exists is scored.
# Needs on the pod: cellgym/, bench/{splits_v0,euler_widths_v0}.json, out/synth5/plans.jsonl (or
# .xz/.gz, unpacked here), out/cells, out/plans, the eval set dirs (cells, plans, index.json),
# out/probe, optionally out/arearef (then step b is a no-op).
cd /workspace/lg
export HF_HOME=/root/hf PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=out/llm; P=$D/progress; mkdir -p $D
J=${JOBS:-32}
say() { echo "$* $(date)" >> $P; }
run() {   # run NAME LOGFILE CMD...: run a stage, record its exit code
  local name=$1 log=$2; shift 2
  "$@" > $log 2>&1; local rc=$?
  if [ $rc -eq 0 ]; then say "$name ok"; else say "$name FAILED rc=$rc"; fi
  return $rc
}
say "start"

# a. data
S=out/synth5/plans.jsonl
[ -f $S ] || { [ -f $S.xz ] && xz -dk $S.xz; }
[ -f $S ] || { [ -f $S.gz ] && gunzip -k $S.gz; }
run "data built" $D/build.log python -m cellgym.llm.build --out $D/data --steps-per-plan 0

# b. Euler area references (corpus + every eval set), in the background while SL trains
(python -m cellgym.llm.arearef --all --jobs ${AREF_JOBS:-24} > $D/arearef.log 2>&1
 echo "arearef rc=$? $(date)" >> $P) &
AREF=$!

# c. SL, exactly run6's step05s1 (seed 1)
run "step05s1 trained" $D/step05s1_train.log python -m cellgym.llm.train --model Qwen/Qwen2.5-0.5B \
  --data $D/data/step_train.jsonl --val $D/data/step_val.jsonl \
  --out $D/step05s1 --epochs 1 --micro-tokens 4096 --tokens-per-step 65536 --lr 5e-5 --seed 1

# probe sets present (shipped, not built here)
X=""; for s in i8d14 i10d16 i12d20 i16d24 stage2 stage3; do
  [ -f out/probe/$s/index.json ] && X="$X $s=out/probe/$s:all"; done
say "probes:${X:- none}"

sample() {   # sample MODEL OUTDIR TAG: all sets k=8, then each probe set k=4 (one call per set)
  local m=$1 o=$2 tag=$3
  [ -d $m ] || { say "$tag sampling skipped (no model $m)"; return 1; }
  run "$tag sampled" $D/${tag}_infer.log python -m cellgym.llm.infer --model $m --mode step --k 8 --out $o/samples.jsonl
  rm -f $o/probe_samples.jsonl
  for x in $X; do
    local s=${x%%=*}
    run "$tag probe $s sampled" $D/${tag}_probe_${s}_infer.log timeout ${PROBE_TIMEOUT:-2400} \
      python -m cellgym.llm.infer --model $m --mode step --k 4 --extra $x --out $o/probe_$s.jsonl \
      && cat $o/probe_$s.jsonl >> $o/probe_samples.jsonl
  done
}

# d. sample the SL model
sample $D/step05s1/model $D/step05s1 step05s1

# e. RL on area (needs the corpus references; rl computes them itself if step b failed)
wait $AREF
if [ -d $D/step05s1/model ]; then
  run "rl4 trained" $D/rl4.log python -m cellgym.llm.rl --init $D/step05s1/model --out $D/rl4 \
    --steps ${RL_STEPS:-100} --cells 32 --group 8 --lr ${RL_LR:-4e-6} --beta 0.02 --reward area \
    --eval-every 20 --eval-cells 96 --jobs $J
else
  say "rl4 skipped (no SL model)"
fi

# f. sample the RL model
sample $D/rl4/best $D/rl4 rl4

# g. score every samples file twice (SKIP_SCORE=1: score locally instead, CPU only)
for tag in $( [ -n "$SKIP_SCORE" ] || echo step05s1 rl4 ); do
  f=$D/$tag/samples.jsonl
  if [ -s $f ]; then
    run "$tag scored t3" $D/${tag}_score.txt timeout ${SCORE_TIMEOUT:-5400} python -m cellgym.llm.score $f --jobs $J
    run "$tag scored area" $D/${tag}_score_area.txt timeout ${SCORE_TIMEOUT:-5400} python -m cellgym.llm.score $f --jobs $J --tracks area --tag area
  else
    say "$tag: no samples to score"
  fi
  f=$D/$tag/probe_samples.jsonl
  if [ -s $f ]; then
    # only the probe sets that were actually sampled
    XS=""; for x in $X; do [ -s $D/$tag/probe_${x%%=*}.jsonl ] && XS="$XS $x"; done
    run "$tag probes scored t3" $D/${tag}_probe_score.txt timeout ${SCORE_TIMEOUT:-5400} python -m cellgym.llm.score $f --jobs $J --extra $XS
    run "$tag probes scored area" $D/${tag}_probe_score_area.txt timeout ${SCORE_TIMEOUT:-5400} python -m cellgym.llm.score $f --jobs $J --extra $XS --tracks area --tag area
  fi
done

# h. results (no model weights)
tar czf $D/results_run7.tgz $D/*.txt $D/*.log $D/progress $D/data/manifest.json out/arearef/*.json \
  $D/step05s1/*.jsonl $D/step05s1/*.summary.json $D/rl4/*.jsonl $D/rl4/*.summary.json 2>/dev/null
say "results $(ls -la $D/results_run7.tgz 2>/dev/null | awk '{print $5}') bytes"
echo "RUN7_DONE $(date)" >> $P
