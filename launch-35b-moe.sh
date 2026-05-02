#!/usr/bin/env bash
# Bring up Qwen3.6-35B-A3B (MoE) on dual RTX 3090s with TP=2 + expert-parallel.
# Yields ~282 tok/s aggregate at C=4 on http://localhost:8500.
# Single instance only — the model is too big to fit on one 3090, so the
# 2-replica + LB pattern from the 27B stack does not apply here.
#
# Requires:
#   - driver >= 575 (we used 580)
#   - vLLM nightly image already pulled
#   - Genesis patches at $STACK/genesis
#   - QuantTrio/Qwen3.6-35B-A3B-AWQ in HF cache
set -e

STACK=/home/ver/qwen3.6/overnight-stack
NIGHTLY_TAG=nightly-07351e0883470724dd5a7e9730ed10e01fc99d08
MODEL_CACHE=/home/ver/.cache/huggingface/hub/models--QuantTrio--Qwen3.6-35B-A3B-AWQ
# Hardcoded snapshot SHA. The dynamic `ls` lookup we previously had only
# worked when the launcher ran on the host — inside the qwen-control
# container /home/ver/.cache isn't mounted, so the lookup returned empty.
# Note: this is the bare SHA (no "snapshots/" prefix); the path below
# already prefixes "/model/snapshots/".
SNAP_REL="119886a1072372348f73ef0df2d801cdcc0f455b"

docker rm -f qwen36-moe 2>/dev/null || true

# Settings rationale:
#   --max-num-seqs 4 = peak; bumping to 8 hits a vLLM modular-kernel
#     workspace-lock bug under load and crashes the engine.
#   --enable-expert-parallel = shards experts across both GPUs,
#     less NCCL traffic than dense TP=2.
#   No --speculative-config = MTP causes GDN profile_run OOM on Ampere+A3B,
#     and published benchmarks show no spec-decode benefit on this MoE anyway.
docker run -d --name qwen36-moe --gpus all \
  -v "$MODEL_CACHE":/model:ro \
  -v "$STACK/genesis/vllm/_genesis":/usr/local/lib/python3.12/dist-packages/vllm/_genesis:ro \
  -v "$STACK/repo/patches/patch_tolist_cudagraph.py":/patches/patch_tolist_cudagraph.py:ro \
  -p 8500:8000 --ipc=host --shm-size=16gb \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e NCCL_CUMEM_ENABLE=0 -e NCCL_P2P_DISABLE=1 \
  -e VLLM_NO_USAGE_STATS=1 -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512 \
  -e VLLM_FLOAT32_MATMUL_PRECISION=high -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e OMP_NUM_THREADS=1 -e CUDA_DEVICE_MAX_CONNECTIONS=8 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL=1 \
  -e GENESIS_ENABLE_P82=1 \
  -e GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT=1 \
  --entrypoint /bin/bash \
  "vllm/vllm-openai:$NIGHTLY_TAG" \
  -c "set -e
      pip install xxhash pandas scipy -q
      python3 -m vllm._genesis.patches.apply_all
      python3 /patches/patch_tolist_cudagraph.py
      exec vllm serve /model/snapshots/$SNAP_REL \
        --served-model-name qwen36-35b-moe \
        --quantization awq_marlin --dtype float16 --tensor-parallel-size 2 \
        --enable-expert-parallel \
        --max-model-len 32000 --gpu-memory-utilization 0.92 \
        --max-num-seqs 4 --max-num-batched-tokens 4096 \
        --kv-cache-dtype fp8_e5m2 \
        --trust-remote-code --reasoning-parser qwen3 \
        --enable-prefix-caching --enable-chunked-prefill \
        --host 0.0.0.0 --port 8000"

echo "Wait ~90s for Genesis + weight load + profile_run, then:"
echo "  curl http://localhost:8500/v1/models"
echo "  curl http://localhost:8500/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"qwen36-35b-moe\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'"
