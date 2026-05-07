#!/usr/bin/env bash
# Bring up the Granite-4.1-8B 2-replica + nginx LB stack.
# Yields ~1222 tok/s aggregate at C=64, ~307 tok/s at C=4 through http://localhost:8400.
# Requires:
#   - vLLM nightly image already pulled
#   - cyankiwi/granite-4.1-8b-AWQ-INT4 in HF cache
set -e

NIGHTLY_TAG=nightly-07351e0883470724dd5a7e9730ed10e01fc99d08
HF_CACHE=/home/ver/.cache/huggingface
MODEL_ID=cyankiwi/granite-4.1-8b-AWQ-INT4
NGCONF=/home/ver/qwen3.6/blog/control/nginx-granite-8b.conf

docker rm -f granite-8b-1 granite-8b-2 granite-lb-8b 2>/dev/null || true

start_replica() {
  local NAME=$1 GPU=$2 PORT=$3
  docker run -d --name "$NAME" --gpus "\"device=$GPU\"" \
    -v "$HF_CACHE":/root/.cache/huggingface \
    -p "$PORT":8000 --ipc=host --shm-size=16gb \
    -e VLLM_NO_USAGE_STATS=1 -e HF_HUB_OFFLINE=1 \
    -e VLLM_USE_FLASHINFER_SAMPLER=1 -e OMP_NUM_THREADS=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512 \
    "vllm/vllm-openai:$NIGHTLY_TAG" \
    --model "$MODEL_ID" --served-model-name granite-8b \
    --tensor-parallel-size 1 --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 32 --max-num-batched-tokens 4096 \
    --enable-prefix-caching --enable-chunked-prefill \
    --host 0.0.0.0 --port 8000
}
start_replica granite-8b-1 0 8600
start_replica granite-8b-2 1 8601

cat > "$NGCONF" <<'EOF'
events { worker_connections 4096; }
http {
    upstream vllm_pool {
        least_conn;
        server 127.0.0.1:8600 max_fails=3 fail_timeout=10s;
        server 127.0.0.1:8601 max_fails=3 fail_timeout=10s;
    }
    proxy_read_timeout 900s;
    proxy_send_timeout 900s;
    proxy_buffering off;
    proxy_request_buffering off;
    server {
        listen 8400;
        location / {
            proxy_pass http://vllm_pool;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
        }
    }
}
EOF

docker run -d --name granite-lb-8b --network host \
  -v "$NGCONF":/etc/nginx/nginx.conf:ro nginx:alpine

echo "Replicas: 8600 (GPU 0), 8601 (GPU 1) ; LB: 8400"
echo "Wait ~30s for the engines to load, then:"
echo "  curl http://localhost:8400/v1/models"
