#!/bin/bash
# control: same step format, small transformer from random init (no pretraining); then rescore step05 with best-of-k widths
cd /workspace/lg
export HF_HOME=/root/hf PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=out/llm; P=$D/progress
while ! grep -q ALL_DONE $P; do sleep 30; done
python -m cellgym.llm.score $D/step05/samples.jsonl --jobs 8 > $D/step05_score.txt 2>&1
python -m cellgym.llm.score $D/random_samples.jsonl --jobs 8 > $D/random_score.txt 2>&1
python -m cellgym.llm.train --model Qwen/Qwen2.5-0.5B --scratch 8x512 --data $D/data/step_train.jsonl --val $D/data/step_val.jsonl \
  --out $D/stepscratch --epochs 1 --micro-tokens 8192 --tokens-per-step 65536 --lr 6e-4 --warmup 50 > $D/stepscratch_train.log 2>&1
echo "stepscratch trained $(date)" >> $P
python -m cellgym.llm.infer --model $D/stepscratch/model --mode step --k 8 --out $D/stepscratch/samples.jsonl > $D/stepscratch_infer.log 2>&1
python -m cellgym.llm.score $D/stepscratch/samples.jsonl --jobs 14 > $D/stepscratch_score.txt 2>&1
echo "CONTROL_DONE $(date)" >> $P
