#!/usr/bin/env bash
# Bring up Granite-4.1-30B with TP=2 across both 3090s.
# Yields ~216 tok/s aggregate at C=8 through http://localhost:8500.
# Single instance only — TP=2 wins over the 2-replica + LB pattern at this size
# because the 15.5 GB INT4 weights don't leave enough KV headroom on a single card
# for max-num-seqs > 4. See blog/granite4.1-on-rtx3090.html for the analysis.
#
# Requires:
#   - vLLM nightly image already pulled
#   - drawais/Granite-4.1-30B-AWQ-INT4 in HF cache (sym INT4 g=128)
#   - DO NOT use cyankiwi/granite-4.1-30b-AWQ-INT4 — its asymmetric INT4 g=32
#     hits a Marlin kernel correctness bug on Ampere SM 8.6.
set -e

NIGHTLY_TAG=nightly-07351e0883470724dd5a7e9730ed10e01fc99d08
HF_CACHE=/home/ver/.cache/huggingface
MODEL_ID=drawais/Granite-4.1-30B-AWQ-INT4

docker rm -f granite-30b 2>/dev/null || true

docker run -d --name granite-30b --gpus all \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -p 8500:8000 --ipc=host --shm-size=16gb \
  -e VLLM_NO_USAGE_STATS=1 -e HF_HUB_OFFLINE=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e NCCL_CUMEM_ENABLE=0 -e NCCL_P2P_DISABLE=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 -e OMP_NUM_THREADS=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512 \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  "vllm/vllm-openai:$NIGHTLY_TAG" \
  --model "$MODEL_ID" --served-model-name granite-30b \
  --tensor-parallel-size 2 --max-model-len 16384 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 8 --max-num-batched-tokens 4096 \
  --enable-prefix-caching --enable-chunked-prefill \
  --host 0.0.0.0 --port 8000

echo "Wait ~60s for TP=2 init + profile_run, then:"
echo "  curl http://localhost:8500/v1/models"
