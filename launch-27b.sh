#!/usr/bin/env bash
# Bring back the dual-3090 Qwen3.6-27B 2-replica + nginx LB stack.
# Yields ~225 tok/s aggregate at C=4 through http://localhost:8400.
# Requires:
#   - driver >= 575 (we used 580)
#   - vLLM nightly image already pulled
#   - Genesis patches at $STACK/genesis
#   - Lorbus AutoRound model in HF cache
#   - nginx.conf at $STACK/nginx.conf
set -e

STACK=/home/ver/qwen3.6/overnight-stack
NIGHTLY_TAG=nightly-07351e0883470724dd5a7e9730ed10e01fc99d08
SNAP_REL="snapshots/c3aea2d531678621989e5e2db034e32b22536e79/"
MODEL_CACHE=/home/ver/.cache/huggingface/hub/models--Lorbus--Qwen3.6-27B-int4-AutoRound

docker rm -f qwen36-vllm-1 qwen36-vllm-2 qwen36-lb 2>/dev/null || true

for IDX in 0 1; do
  PORT=$((8500 + IDX))
  docker run -d --name "qwen36-vllm-$((IDX+1))" --gpus '"device='"$IDX"'"' \
    -v "$MODEL_CACHE":/model:ro \
    -v "$STACK/genesis/vllm/_genesis":/usr/local/lib/python3.12/dist-packages/vllm/_genesis:ro \
    -v "$STACK/repo/patches/patch_tolist_cudagraph.py":/patches/patch_tolist_cudagraph.py:ro \
    -p "$PORT":8000 --ipc=host --shm-size=16gb \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e NCCL_CUMEM_ENABLE=0 -e NCCL_P2P_DISABLE=1 \
    -e VLLM_NO_USAGE_STATS=1 -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512 \
    -e VLLM_FLOAT32_MATMUL_PRECISION=high -e VLLM_USE_FLASHINFER_SAMPLER=1 \
    -e OMP_NUM_THREADS=1 -e CUDA_DEVICE_MAX_CONNECTIONS=8 \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
    -e GENESIS_ENABLE_P64_QWEN3CODER_MTP_STREAMING=1 \
    -e GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL=1 \
    -e GENESIS_ENABLE_P82=1 \
    -e GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT=1 \
    --entrypoint /bin/bash \
    "vllm/vllm-openai:$NIGHTLY_TAG" \
    -c "set -e
        pip install xxhash pandas scipy -q
        python3 -m vllm._genesis.patches.apply_all
        python3 /patches/patch_tolist_cudagraph.py
        exec vllm serve /model/$SNAP_REL \
          --served-model-name qwen36-27b \
          --quantization auto_round --dtype float16 --tensor-parallel-size 1 \
          --max-model-len 16000 --gpu-memory-utilization 0.92 \
          --max-num-seqs 2 --max-num-batched-tokens 2048 \
          --kv-cache-dtype fp8_e5m2 \
          --trust-remote-code --reasoning-parser qwen3 \
          --enable-prefix-caching --enable-chunked-prefill \
          --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}' \
          --host 0.0.0.0 --port 8000"
done

docker run -d --name qwen36-lb --network host \
  -v "$STACK/nginx.conf":/etc/nginx/nginx.conf:ro \
  nginx:alpine

echo "Wait ~90s for both replicas to apply Genesis + load weights, then:"
echo "  curl http://localhost:8400/v1/models"
