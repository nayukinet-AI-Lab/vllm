# docker-v100

このフォークの vLLM を NVIDIA Tesla V100（Volta, sm_70）で動かすための Docker ファイル群です。
English: [README.md](./README.md)

| ファイル | 用途 |
|---|---|
| `Dockerfile.v100` | V100 向けにこのフォークをソースからビルドする multi-stage Dockerfile（メインのイメージ） |
| `Dockerfile.v100.dockerignore` | `Dockerfile.v100` 用のビルドコンテキスト除外設定 |
| `compose.yaml` | イメージを GPU 0 で OpenAI 互換サーバーとして起動する |
| `Dockerfile.gemma3` | V100 ビルドとは無関係の実験用（後述） |

## 前提条件

- Docker と [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)（ホスト側はドライバのみで可。CUDA 12.6 のライブラリはイメージ内にあります）
- Tesla V100（CC < 8.0 の他の GPU も同じフォールバック経路を通ります）
- ディスク: 完成イメージは約 18 GB。ビルド中は build ステージ分の空きも必要です

## イメージのビルド

**リポジトリのルート**で実行します（ビルドコンテキストはリポジトリ全体です）。

```bash
docker build -f docker-v100/Dockerfile.v100 -t vllm-v100:latest .
```

compose 経由でも同じイメージ・同じタグでビルドできます。

```bash
docker compose -f docker-v100/compose.yaml build
```

`Dockerfile.v100` の仕組み:

1. **build ステージ**（`nvidia/cuda:12.6.3-devel`）: `uv` で `/opt/venv` を作り、
   `torch==2.11.0+cu126`（sm_70 のカーネルと、V100 アダプタに必要な stable-ABI ヘッダの両方を
   満たす唯一の組み合わせ）を入れ、`TORCH_CUDA_ARCH_LIST="7.0"` でこのリポジトリをコンパイルします。
   V100 用 C++ アダプタ（`_C_stable_libtorch`、`_moe_C_stable_libtorch`）もここでビルドされます。
2. **runtime ステージ**（`nvidia/cuda:12.6.3-runtime`）: `/opt/venv` だけをコピーします。
   `gcc` と `python3.12-dev` は、Triton が最初の推論リクエスト時に小さなヘルパーを JIT コンパイルするため残しています。

ビルド引数:

```bash
docker build -f docker-v100/Dockerfile.v100 --build-arg MAX_JOBS=8 -t vllm-v100:latest .
```

- `MAX_JOBS`（既定 `4`）: コンパイルの並列数。メモリに余裕があれば増やせます。
- `CUDA_VERSION`（既定 `12.6.3`）、`PYTHON_VERSION`（既定 `3.12`）: ベースイメージ / Python。
  CUDA は torch の wheel に合わせて 12.6.x のままにしてください。

ビルドコンテキストに `.git` が含まれないため、イメージ内のバージョン表記は `0.0.0+v100` 固定です。

## compose で起動する

```bash
export HF_TOKEN=hf_xxx            # gated モデルを使う場合のみ必要
docker compose -f docker-v100/compose.yaml up -d
curl http://localhost:8000/v1/models
docker compose -f docker-v100/compose.yaml logs -f
docker compose -f docker-v100/compose.yaml down
```

`compose.yaml` の設定内容:

- `build:` リポジトリルートをコンテキストに `Dockerfile.v100` をビルドします。
- `ports: 8000:8000`: OpenAI 互換 API。
- `CUDA_VISIBLE_DEVICES=0`: GPU 0 を使います。別の GPU を使うならここを変えます。
- `volumes`: Hugging Face キャッシュ（`HF_HOME` があればそれ、なければ `~/.cache/huggingface`）をマウントし、モデルの再ダウンロードを避けます。
- `deploy.resources.reservations.devices`: NVIDIA GPU を 1 枚確保します。
- `command`: `vllm serve ...` の行。既定は `Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4` です。

### モデルだけ変える場合

`compose.yaml` の `command:` を編集します。V100 で重要なオプション:

- `--dtype half`: V100 は BF16 非対応なので、常に FP16 にします。
- `--enforce-eager`: V100 で動作確認した構成です。
- `--max-model-len` / `--gpu-memory-utilization`: VRAM 16 GB に合わせて調整します。
- `--quantization gptq`: GPTQ モデルのときだけ指定します。量子化なしのモデルでは削除してください。

例: 量子化なしの小さなモデル

```yaml
    command: >
      vllm serve Qwen/Qwen2.5-0.5B-Instruct
        --dtype half
        --max-model-len 2048
        --gpu-memory-utilization 0.85
        --enforce-eager
```

例: Gemma 4 E2B

```yaml
    command: >
      vllm serve google/gemma-4-e2b-it
        --dtype half
        --max-model-len 2048
        --gpu-memory-utilization 0.85
        --enforce-eager
```

ファイルを編集せずに一度だけ試すには:

```bash
docker compose -f docker-v100/compose.yaml run --rm --service-ports vllm \
  vllm serve Qwen/Qwen2.5-0.5B-Instruct --dtype half --enforce-eager --max-model-len 2048
```

compose を使わない場合:

```bash
docker run --rm --gpus '"device=0"' -p 8000:8000 --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm-v100:latest \
  vllm serve Qwen/Qwen2.5-0.5B-Instruct --dtype float16 --enforce-eager --max-model-len 2048
```

注意: 他のプロセス（例: `llama-server`）が GPU メモリを使っていると、
`Free memory on device ... is less than desired GPU memory utilization` で起動に失敗します。
そのプロセスを止めるか、`--gpu-memory-utilization` を下げてください。

## `Dockerfile.gemma3` とは？

別系統の古い実験用ファイルです。upstream の `vllm/vllm-openai:v0.7.2` イメージを土台に、Gemma 3 対応のため
`transformers` を GitHub の master から更新しています。このフォークの V100 アダプタは**含まれておらず**、
`compose.yaml` からも使われません。V100 で Gemma を動かすにはメインのイメージを使ってください
（Gemma 4 E2B で動作確認済み）。upstream v0.7.2 の環境がどうしても必要な場合のみ、次のようにビルドします。

```bash
docker build -f docker-v100/Dockerfile.gemma3 -t vllm-gemma3:latest docker-v100
```

## 動作確認済みの範囲

クリーンな checkout からビルドし、Tesla V100-SXM2-16GB 上で `Qwen/Qwen2.5-0.5B-Instruct` と
`google/gemma-4-e2b-it`（FP16, `--enforce-eager`, `max_model_len=2048`）の両方で正しい出力が得られました。
compose 自体の `up` と `Dockerfile.gemma3` は実行していません。
