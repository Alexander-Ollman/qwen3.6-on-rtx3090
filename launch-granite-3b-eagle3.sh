#!/usr/bin/env bash
# Bring up Granite-4.1-3B BF16 with our trained EAGLE-3 head as the speculative draft.
# Yields ~105 tok/s C=1 (vs 92.6 baseline = +13.5%) through http://localhost:8500.
# Single instance, single GPU.
#
# Requires:
#   - vllm-eagle3-granite:v2 image (vLLM nightly + EAGLE3 whitelist + EagleModelMixin patch)
#   - ibm-granite/granite-4.1-3b in HF cache
#   - Trained EAGLE-3 head dir at $EAGLE_HEAD with config.json + model.safetensors + tokenizer
#
# Train the head with granite-stack/eagle-3/ first (see blog/granite4.1-on-rtx3090.html).
set -e

EAGLE_HEAD="${EAGLE_HEAD:-/home/ver/qwen3.6/granite-stack/eagle-3/outputs/granite-3b-eagle3-1gpu-5k/epoch_0_step_5000}"
NSPEC="${NSPEC:-5}"
HF_CACHE=/home/ver/.cache/huggingface

if [ ! -d "$EAGLE_HEAD" ] || [ ! -f "$EAGLE_HEAD/config.json" ]; then
  echo "EAGLE-3 head not found at $EAGLE_HEAD"
  echo "Train one first with granite-stack/eagle-3/ scripts."
  exit 1
fi

docker rm -f granite-3b-eagle 2>/dev/null || true

docker run -d --name granite-3b-eagle --gpus '"device=0"' \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -v "$EAGLE_HEAD":/eagle-head:ro \
  -p 8500:8000 --ipc=host --shm-size=16gb \
  -e VLLM_NO_USAGE_STATS=1 -e HF_HUB_OFFLINE=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 -e OMP_NUM_THREADS=1 \
  vllm-eagle3-granite:v2 \
  --model ibm-granite/granite-4.1-3b --served-model-name granite-3b-eagle \
  --tensor-parallel-size 1 --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 16 --max-num-batched-tokens 4096 \
  --enable-prefix-caching --enable-chunked-prefill \
  --speculative-config "{\"method\":\"eagle3\",\"model\":\"/eagle-head\",\"num_speculative_tokens\":$NSPEC}" \
  --host 0.0.0.0 --port 8000

echo "EAGLE-3 head: $EAGLE_HEAD (n=$NSPEC)"
echo "Wait ~30s, then: curl http://localhost:8500/v1/models"
