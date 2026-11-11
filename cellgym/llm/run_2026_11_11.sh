#!/bin/bash
# LLM experiment, 2026-11-11. Every line of progress -> out/llm/progress
cd /workspace/lg
export HF_HOME=/root/hf PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=out/llm; P=$D/progress
echo "start $(date)" >> $P
# random-legal baseline on CPU while the GPU trains
(python -m cellgym.llm.infer --mode random --k 8 --out $D/random_samples.jsonl > $D/random_infer.log 2>&1 \
  && python -m cellgym.llm.score $D/random_samples.jsonl --jobs 6 > $D/random_score.txt 2>&1; echo "random done $(date)" >> $P) &
# A: step-wise, 0.5B, 1 epoch over every step of every corpus plan
python -m cellgym.llm.train --model Qwen/Qwen2.5-0.5B --data $D/data/step_train.jsonl --val $D/data/step_val.jsonl \
  --out $D/step05 --epochs 1 --micro-tokens 4096 --tokens-per-step 65536 --lr 5e-5 > $D/step05_train.log 2>&1
echo "step05 trained $(date)" >> $P
python -m cellgym.llm.infer --model $D/step05/model --mode step --k 8 --out $D/step05/samples.jsonl > $D/step05_infer.log 2>&1
echo "step05 sampled $(date)" >> $P
(python -m cellgym.llm.score $D/step05/samples.jsonl --jobs 8 > $D/step05_score.txt 2>&1; echo "step05 scored $(date)" >> $P) &
# B: one-shot, 0.5B, 2 epochs
python -m cellgym.llm.train --model Qwen/Qwen2.5-0.5B --data $D/data/oneshot_train.jsonl --val $D/data/oneshot_val.jsonl \
  --out $D/oneshot05 --epochs 2 --micro-tokens 4096 --tokens-per-step 65536 --lr 5e-5 > $D/oneshot05_train.log 2>&1
echo "oneshot05 trained $(date)" >> $P
python -m cellgym.llm.infer --model $D/oneshot05/model --mode oneshot --k 4 --temperature 0.7 --max-tokens 2048 \
  --out $D/oneshot05/samples.jsonl > $D/oneshot05_infer.log 2>&1
echo "oneshot05 sampled $(date)" >> $P
python -m cellgym.llm.score $D/oneshot05/samples.jsonl --jobs 14 > $D/oneshot05_score.txt 2>&1
wait
echo "ALL_DONE $(date)" >> $P
