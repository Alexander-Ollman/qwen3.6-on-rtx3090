# qwen3.6-on-rtx3090

Reproducible recipes and a full investigative blog post for serving the **Qwen3.6** family on consumer NVIDIA RTX hardware (3090/4090/5090) with vLLM, Sandermage's Genesis patches, and an nginx load balancer.

**Headline numbers (dual RTX 3090):**
- **Qwen3.6-27B (dense)** — 100 tok/s single-stream, **225 tok/s aggregate** at C=4 through one endpoint
- **Qwen3.6-35B-A3B (MoE)** — **283 tok/s aggregate** at C=4 through one endpoint
- 9× over a stock `vllm serve` baseline (25 tok/s)

## What's in this repo

| File / dir | Purpose |
|---|---|
| [`qwen3.6-on-rtx3090.md`](qwen3.6-on-rtx3090.md) | Full blog post in Markdown — the investigative journey, both rounds |
| [`qwen3.6-on-rtx3090.html`](qwen3.6-on-rtx3090.html) | Self-contained HTML version with animated SVG charts (drag-and-drop into a browser) |
| [`launch-27b.sh`](launch-27b.sh) | One-shot script: bring up the 27B-dense 2-replica + nginx LB stack |
| [`launch-35b-moe.sh`](launch-35b-moe.sh) | One-shot script: bring up the 35B-A3B MoE single-instance TP=2+EP stack |
| [`control/`](control/) | **Web UI + OpenAI-compatible proxy** — switch models from a browser, single endpoint at `:9000/v1/`, bearer auth, tailnet-only. See [control/README.md](control/README.md). |
| `*-chart.svg` | Performance charts referenced from the blog |

## System requirements

We tested on the configuration below. Other versions likely work but are unverified.

### Operating system

- **Ubuntu 22.04.5 LTS (jammy)** or **Ubuntu 24.04 LTS (noble)**
- Kernel **6.8+** (HWE on 22.04 ships this; default on 24.04)
- **Secure Boot disabled** (or you'll need to MOK-sign every NVIDIA module after each kernel update — painful and out of scope here)

### NVIDIA stack

| Component | Tested version | Why this minimum |
|---|---|---|
| Driver | **`nvidia-driver-580` (580.159.03)** | Required for CUDA 12.9 / 13.0. Driver 570 caps at CUDA 12.8 — vLLM nightly images post April 2026 won't load. Driver 575 also works. |
| CUDA toolkit | 13.0 (host) — runtime is bundled in the vLLM container | Userland matches what the container expects |
| DKMS | ≥ **3.1.8** | Older DKMS (jammy default 2.8.7) fails to build the 580 modules |
| Docker | 24+ with NVIDIA Container Toolkit (`nvidia-container-toolkit`) | `--gpus all` and `--gpus '"device=N"'` flags |

### Hardware

| GPU | Verified | Notes |
|---|---|---|
| **2× RTX 3090** (24 GB each) | ✅ Both 27B and 35B-MoE | The full setup the blog post covers. |
| **1× RTX 3090** (24 GB) | ✅ 27B only (peak 100 tok/s) | 35B-MoE is too big for a single 3090. |
| **1× / 2× RTX 4090** (24 GB) | ⚠️ Untested | Same recipe should run; expect higher tok/s due to higher SM throughput / memory bandwidth. |
| **1× / 2× RTX 5090** (32 GB) | ⚠️ Untested | The 8 GB extra VRAM lets a 5090 fit the 35B MoE on **one card**, unlocking the 2-replica + LB pattern that's blocked on 24 GB cards. May need driver ≥ 590 for newer Blackwell fixes. |

### Disk

- **~50 GB free** for image + model weights + Genesis patches:
  - vLLM nightly Docker image: ~22 GB
  - `Lorbus/Qwen3.6-27B-int4-AutoRound`: ~19 GB
  - `QuantTrio/Qwen3.6-35B-A3B-AWQ`: ~24 GB (only if you want the MoE)
  - Genesis patches + tooling: <100 MB

## Software dependencies (mounted into the container — not host-installed)

The container does the work. Host needs only Docker + driver + Genesis patches as a checked-out git tree.

| Dependency | Source | Pin / version |
|---|---|---|
| **vLLM** | Docker image `vllm/vllm-openai` | **`nightly-07351e0883470724dd5a7e9730ed10e01fc99d08`** (vLLM 0.19.2 / dev205, late April 2026). Newer nightlies likely also work; pinned for reproducibility. |
| **Sandermage Genesis patches** | [`Sandermage/genesis-vllm-patches`](https://github.com/Sandermage/genesis-vllm-patches) | v7.14+ (modular `vllm/_genesis/` package). Mounted into the container at runtime. |
| **noonghunna stack helpers** | [`noonghunna/qwen36-27b-single-3090`](https://github.com/noonghunna/qwen36-27b-single-3090) | HEAD. We use only `patches/patch_tolist_cudagraph.py` from this repo. |
| **Lorbus AutoRound INT4 (27B)** | [`Lorbus/Qwen3.6-27B-int4-AutoRound`](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound) | HEAD. AutoRound INT4 with BF16 MTP head preserved — critical for clean spec-decode drafts. |
| **QuantTrio AWQ (35B-MoE)** | [`QuantTrio/Qwen3.6-35B-A3B-AWQ`](https://huggingface.co/QuantTrio/Qwen3.6-35B-A3B-AWQ) | HEAD. Community AWQ INT4 quantization. |
| **nginx** | Docker image `nginx:alpine` | HEAD (used only as a tiny load balancer, ~30 lines of config). |
| **`hf` CLI** (host-side) | `pip install huggingface_hub` | Any 1.x. Used to download model weights into the HF cache. |

## Quick start

### 1. Install the driver (once, host)

```bash
sudo apt update
sudo apt install -y nvidia-driver-580
sudo reboot
nvidia-smi   # confirm driver 580.x
```

If you hit file-overwrite conflicts with an older `nvidia-driver-NNN` (common when going from 570 → 580 on Ubuntu 22.04), use:
```bash
sudo apt-get -o Dpkg::Options::="--force-overwrite" install -y -f libnvidia-gl-580
sudo apt autoremove -y --purge
```
Full troubleshooting in the [blog post](qwen3.6-on-rtx3090.md#stage-4-the-wall--and-why-we-had-to-rebuild-from-the-foundation).

### 2. Pull the vLLM image

```bash
docker pull vllm/vllm-openai:nightly-07351e0883470724dd5a7e9730ed10e01fc99d08
```

### 3. Clone the patches and download the model

```bash
mkdir -p ~/qwen3.6-stack && cd ~/qwen3.6-stack
git clone https://github.com/noonghunna/qwen36-27b-single-3090.git repo
git clone https://github.com/Sandermage/genesis-vllm-patches.git genesis

# 27B dense
hf download Lorbus/Qwen3.6-27B-int4-AutoRound

# Or for the 35B MoE
hf download QuantTrio/Qwen3.6-35B-A3B-AWQ
```

### 4. Launch

```bash
# 27B dense — 2 replicas + nginx LB at http://localhost:8400
bash launch-27b.sh

# OR: 35B MoE — single instance TP=2+EP at http://localhost:8500
bash launch-35b-moe.sh
```

Wait ~90 s after launch for Genesis patches to apply and weights to load, then:
```bash
curl http://localhost:8400/v1/models   # 27B endpoint
curl http://localhost:8500/v1/models   # MoE endpoint
```

### 5. Benchmark (optional)

The vLLM image ships `vllm bench serve`:
```bash
docker run --rm --network host --entrypoint vllm \
  vllm/vllm-openai:nightly-07351e0883470724dd5a7e9730ed10e01fc99d08 \
  bench serve --backend openai --base-url http://localhost:8400 \
  --model qwen36-27b --tokenizer Lorbus/Qwen3.6-27B-int4-AutoRound \
  --dataset-name random --random-input-len 1024 --random-output-len 1024 \
  --num-prompts 16 --max-concurrency 4 --trust-remote-code
```
Send a warm-up request first — the first generation incurs cudagraph-recompile latency that skews 5-prompt averages.

## Reading the blog post

The post is the *why* — every config decision and what we measured along the way:

- [Markdown version](qwen3.6-on-rtx3090.md) — renders well on GitHub
- [HTML version](qwen3.6-on-rtx3090.html) — self-contained, animated charts, dark theme

## Known limitations

- **Single-GPU 35B-MoE doesn't work** on 24 GB cards. The model is 24 GB on disk and vLLM needs more headroom. Use TP=2+EP or wait for a 32 GB+ card.
- **MTP speculative decoding regresses on 35B-A3B MoE.** Confirmed independently by [thc1006](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090) on llama.cpp. Don't enable `--speculative-config` for the MoE.
- **`--max-num-seqs > 4` crashes the MoE engine** with a vLLM modular-kernel workspace-lock bug under load. Stay at 4.
- **GDN linear-attention OOM cliff** is independent of GPU class — happens on any prompt batched at high `--max-num-seqs` with large batched-token budgets. Pre-allocated workspace can't grow under real load. Stay at the values in the launch scripts until upstream lands the fix.
- **TP=2 for the 27B dense is *worse* than 2 replicas + LB** on consumer 3090s without NVLink. NCCL all-reduces over PCIe Gen4 are the bottleneck. The MoE's `--enable-expert-parallel` shards differently and is genuinely faster than dense TP.

## Credits

- **Sandermage** — the [Genesis patches](https://github.com/Sandermage/genesis-vllm-patches), single most important component
- **noonghunna** — the [reproducible compose stack](https://github.com/noonghunna/qwen36-27b-single-3090) we built on
- **fzbcwvv** — the [overnight-stack Medium write-up](https://medium.com/@fzbcwvv/an-overnight-stack-for-qwen3-6-27b-85-tps-125k-context-vision-on-one-rtx-3090-0d95c6291914) that pointed us at the recipe
- **Lorbus**, **QuantTrio** — the quantized checkpoints
- **thc1006** — the spec-decode benchmark that called the MoE result before we re-measured
- **vLLM project** — for shipping nightly cu129 images and merging TurboQuant in time
- **Chris Dzombak** — [original dual-3090 vLLM compose](https://www.dzombak.com/blog/2026/04/a-vllm-docker-compose-recipe-for-running-qwen-3-6-27b-on-dual-rtx-3090s-opencode-configuration/) that set our baseline expectations

## License

Code (the `.sh` scripts, nginx config, SVG charts) — MIT.
Blog post text and HTML — CC BY 4.0.
