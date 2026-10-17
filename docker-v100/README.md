# docker-v100

Docker files for running this fork of vLLM on NVIDIA Tesla V100 (Volta, sm_70).
日本語版: [README.jp.md](./README.jp.md)

| File | Purpose |
|---|---|
| `Dockerfile.v100` | Multi-stage from-source build of this fork for V100 (main image) |
| `Dockerfile.v100.dockerignore` | Build-context exclusions for `Dockerfile.v100` |
| `compose.yaml` | Runs the image as an OpenAI-compatible server on GPU 0 |
| `Dockerfile.gemma3` | Experimental, unrelated to the V100 build (see below) |

## Prerequisites

- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (only the host driver is needed; CUDA 12.6 libraries ship inside the image)
- A Tesla V100 (other CC < 8.0 GPUs go through the same fallback path)
- Disk: the final image is about 18 GB; the build needs extra space for the build stage

## Build the image

Run from the **repository root** (the build context is the whole repo):

```bash
docker build -f docker-v100/Dockerfile.v100 -t vllm-v100:latest .
```

Or through compose (same image, same tag):

```bash
docker compose -f docker-v100/compose.yaml build
```

How `Dockerfile.v100` works:

1. **build stage** (`nvidia/cuda:12.6.3-devel`): creates `/opt/venv` with `uv`, installs
   `torch==2.11.0+cu126` (the only combination that has both sm_70 kernels and the
   stable-ABI headers the V100 adapter needs), and compiles this repo with
   `TORCH_CUDA_ARCH_LIST="7.0"`. This builds the V100 C++ adapter
   (`_C_stable_libtorch`, `_moe_C_stable_libtorch`).
2. **runtime stage** (`nvidia/cuda:12.6.3-runtime`): copies only `/opt/venv`.
   `gcc` and `python3.12-dev` are kept because Triton JIT-compiles a small helper on
   the first inference request.

Build args:

```bash
docker build -f docker-v100/Dockerfile.v100 --build-arg MAX_JOBS=8 -t vllm-v100:latest .
```

- `MAX_JOBS` (default `4`): parallel compile jobs. Raise it if you have the RAM.
- `CUDA_VERSION` (default `12.6.3`), `PYTHON_VERSION` (default `3.12`): base image / Python.
  Keep CUDA at 12.6.x to match the torch wheel.

The version string inside the image is fixed to `0.0.0+v100` because `.git` is not
part of the build context.

## Run with compose

```bash
export HF_TOKEN=hf_xxx            # only needed for gated models
docker compose -f docker-v100/compose.yaml up -d
curl http://localhost:8000/v1/models
docker compose -f docker-v100/compose.yaml logs -f
docker compose -f docker-v100/compose.yaml down
```

What `compose.yaml` sets:

- `build:` builds `Dockerfile.v100` with the repo root as context.
- `ports: 8000:8000`: OpenAI-compatible API.
- `CUDA_VISIBLE_DEVICES=0`: uses GPU 0. Change it to pick another GPU.
- `volumes`: mounts your Hugging Face cache (`${HF_HOME}` if set, else `~/.cache/huggingface`) so models are not re-downloaded.
- `deploy.resources.reservations.devices`: requests one NVIDIA GPU.
- `command`: the `vllm serve ...` line. The default is `Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4`.

### Changing only the model

Edit the `command:` block in `compose.yaml`. The flags that matter on V100:

- `--dtype half`: V100 has no BF16, so always use FP16.
- `--enforce-eager`: the configuration verified on V100.
- `--max-model-len` / `--gpu-memory-utilization`: size these to the 16 GB of VRAM.
- `--quantization gptq`: only for GPTQ models. Remove it for unquantized models.

Example, an unquantized small model:

```yaml
    command: >
      vllm serve Qwen/Qwen2.5-0.5B-Instruct
        --dtype half
        --max-model-len 2048
        --gpu-memory-utilization 0.85
        --enforce-eager
```

Example, Gemma 4 E2B:

```yaml
    command: >
      vllm serve google/gemma-4-e2b-it
        --dtype half
        --max-model-len 2048
        --gpu-memory-utilization 0.85
        --enforce-eager
```

To try a model once without editing the file:

```bash
docker compose -f docker-v100/compose.yaml run --rm --service-ports vllm \
  vllm serve Qwen/Qwen2.5-0.5B-Instruct --dtype half --enforce-eager --max-model-len 2048
```

Or without compose:

```bash
docker run --rm --gpus '"device=0"' -p 8000:8000 --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm-v100:latest \
  vllm serve Qwen/Qwen2.5-0.5B-Instruct --dtype float16 --enforce-eager --max-model-len 2048
```

Note: another process holding GPU memory (for example `llama-server`) makes startup fail with
`Free memory on device ... is less than desired GPU memory utilization`. Stop it or lower
`--gpu-memory-utilization`.

## What is `Dockerfile.gemma3`?

It is a separate, older experiment: it starts from the upstream `vllm/vllm-openai:v0.7.2`
image and upgrades `transformers` from GitHub master to get Gemma 3 support. It does **not**
contain this fork's V100 adapter and is not used by `compose.yaml`. For Gemma on V100, use the
main image instead (Gemma 4 E2B was verified with it). Build it only if you specifically need
the upstream v0.7.2 environment:

```bash
docker build -f docker-v100/Dockerfile.gemma3 -t vllm-gemma3:latest docker-v100
```

## Verified

Built from a clean checkout and run on a Tesla V100-SXM2-16GB with `Qwen/Qwen2.5-0.5B-Instruct`
and `google/gemma-4-e2b-it` (FP16, `--enforce-eager`, `max_model_len=2048`), both generating
correct output. The compose file itself (`up`) and `Dockerfile.gemma3` were not run.
