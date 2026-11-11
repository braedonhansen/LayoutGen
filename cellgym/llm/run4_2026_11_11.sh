#!/bin/bash
# second pod (A40): scratch control + step05 seed-1 replicate + probes. Progress -> out/llm/progress
cd /workspace/lg
export HF_HOME=/root/hf PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=out/llm; P=$D/progress; mkdir -p $D
echo "start $(date)" >> $P
python -m cellgym.llm.build --out $D/data --steps-per-plan 0 > $D/build.log 2>&1
echo "data built $(date)" >> $P
for s in 8:14 10:16 8:18; do (timeout 3600 python -m cellgym.llm.probe --out out/probe --spec $s --count 20 > $D/probe_${s/:/_}.log 2>&1 &); done
# A: control, random init, same step data
python -m cellgym.llm.train --model Qwen/Qwen2.5-0.5B --scratch 8x512 --data $D/data/step_train.jsonl --val $D/data/step_val.jsonl \
  --out $D/stepscratch --epochs 1 --micro-tokens 8192 --tokens-per-step 65536 --lr 6e-4 --warmup 50 > $D/stepscratch_train.log 2>&1
echo "stepscratch trained $(date)" >> $P
python -m cellgym.llm.infer --model $D/stepscratch/model --mode step --k 8 --out $D/stepscratch/samples.jsonl > $D/stepscratch_infer.log 2>&1
(python -m cellgym.llm.score $D/stepscratch/samples.jsonl --jobs 6 > $D/stepscratch_score.txt 2>&1; echo "stepscratch scored $(date)" >> $P) &
# B: step05 replicate, seed 1
python -m cellgym.llm.train --model Qwen/Qwen2.5-0.5B --data $D/data/step_train.jsonl --val $D/data/step_val.jsonl \
  --out $D/step05s1 --epochs 1 --micro-tokens 4096 --tokens-per-step 65536 --lr 5e-5 --seed 1 > $D/step05s1_train.log 2>&1
echo "step05s1 trained $(date)" >> $P
python -m cellgym.llm.infer --model $D/step05s1/model --mode step --k 8 --out $D/step05s1/samples.jsonl > $D/step05s1_infer.log 2>&1
(python -m cellgym.llm.score $D/step05s1/samples.jsonl --jobs 6 > $D/step05s1_score.txt 2>&1; echo "step05s1 scored $(date)" >> $P) &
# C: probes with whatever finished
X=""; for s in i8d14 i10d16 i8d18; do [ -f out/probe/$s/index.json ] && X="$X $s=out/probe/$s:all"; done
echo "probes:$X $(date)" >> $P
if [ -n "$X" ]; then
  for m in step05s1 stepscratch; do
    python -m cellgym.llm.infer --model $D/$m/model --mode step --k 4 --extra $X --out $D/$m/probe_samples.jsonl > $D/${m}_probe_infer.log 2>&1
    python -m cellgym.llm.score $D/$m/probe_samples.jsonl --jobs 8 --extra $X > $D/${m}_probe_score.txt 2>&1
  done
fi
wait
tar czf $D/results.tgz $D/*.txt $D/*.log $D/progress $D/*/samples.summary.json $D/*/probe_samples.summary.json $D/*/log.jsonl $D/*/done.json 2>/dev/null
echo "ALL_DONE $(date)" >> $P
