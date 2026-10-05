# V100 (sm_70) C++ アダプタ層 実装ドキュメント

本ドキュメントは、NVIDIA Tesla V100 (Volta / Compute Capability 7.0 / sm_70) 上で
vLLM を動作させるための **C++ アダプタ層**の実装と、ビルドシステムへの組み込み内容を
まとめたものです。

**状態 (2026-10-05 更新)**: アダプタは実際の `_C_stable_libtorch` /
`_moe_C_stable_libtorch` 拡張としてビルド・ロードされ、実機 V100 上で
`Qwen/Qwen2.5-0.5B-Instruct`、`Qwen/Qwen3.5-0.8B`、`google/gemma-4-e2b-it` の
3 アーキテクチャ族での生成まで含めてエンドツーエンドで動作確認済みです(§6, §7 参照)。
それまでは `torch 2.6` 環境では stable 拡張自体が常にスキップされており、アダプタは
一度も実ビルドに組み込まれたことがありませんでした。

関連ドキュメント:

- 設計思想: `docs/v100_fallback_archtecture/v100_adapter_archtecture.md`
- オプ互換性マトリクス: `docs/v100_fallback_archtecture/sm70_op_compatibility_matrix.md`
- 運用ルール: `.claude/rules/v100-csrc-adapter.md`
- フォールバック追加手順: `.claude/skills/v100-add-fallback-op/SKILL.md`

---

## 1. 背景と課題

V100 (sm_70) には、Ampere 以降 (CC >= 8.0) 固有のハードウェア機能
(FlashAttention-2/3、FP8/FP4 Tensor Core、sm_80+ PTX 命令、ハードウェア BF16 等) が
存在しません。このため vLLM のネイティブ C++ 拡張のうち一部カスタムオプが sm_70 で
動作せず、モデル実行時に以下のようなエラーが発生します。

```
AttributeError: '_OpNamespace' '_C' object has no attribute '<op_name>'
```

これを Python 側のモデル定義の改変 (monkey patching) で回避するのは保守性が悪いため、
本フォークでは **Invasive-Free C++ Adapter Pattern** を採用します。すなわち:

- Python モデル層 (`vllm/model_executor/models/**`) は upstream 準拠のまま一切変更しない。
- sm_70 で欠落・非対応のオプは、すべて C++ レイヤ (`csrc/v100_adapter/`) で
  `STABLE_TORCH_LIBRARY_IMPL`(libtorch stable ABI、§2.2 参照)により捕獲し、
  ATen ネイティブ実装へフォールバックさせる。

---

## 2. 実施した設計判断

### 2.1 二重登録問題とビルド時ゲート方式

対象オプ (`rms_norm`, `silu_and_mul` 等) は、実際には
`csrc/libtorch_stable/torch_bindings.cpp` の
`STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops)` 内で**既に CUDA ディスパッチ登録済み**です
(かつネイティブ `layernorm_kernels.cu` 等に sm_80 固有依存は無く、sm_70 でもコンパイル可能)。

このため、アダプタ側で同一オプ・同一ディスパッチキー (`_C::rms_norm` / `CUDA`) に
素朴に追加登録すると、PyTorch ディスパッチャが**二重登録エラー**となり拡張のロードに
失敗します (sm_70 だけでなく全 GPU で破損)。

これを解決するため **ビルド時ゲート方式** (`VLLM_V100_ADAPTER` マクロ) を採用しました。

- CMake が、ターゲット CUDA アーキが**すべて CC < 8.0** の場合にのみ
  `VLLM_V100_ADAPTER` を定義する。
- アダプタ側の登録は `#ifdef VLLM_V100_ADAPTER` で囲む。
- ネイティブ側の該当登録は `#ifndef VLLM_V100_ADAPTER` で囲む。

これにより、sm_70-only ビルドでは「ネイティブ登録を外し、ATen フォールバックを登録」、
CC >= 8.0 ビルドでは「ネイティブ登録のまま、アダプタは一切コンパイルされない」となり、
**各オプにつき CUDA カーネルが常に 1 つだけ**登録される状態を保証します。

### 2.2 libtorch stable ABI での実装(2026-10-05 書き直し)

`csrc/v100_adapter/v100_fallback_ops.cpp` は `_C_stable_libtorch` ターゲットの一部として
コンパイルされます。このターゲットは `-DPy_LIMITED_API=3 -DTORCH_TARGET_VERSION=...` 付きで
ビルドされるため、レガシー ATen ヘッダ (`torch/all.h`, `torch/library.h`, `at::Tensor`) は
`#error` で弾かれ使用できません。当初の実装はレガシー API のまま書かれていましたが、
`torch 2.6` では stable 拡張自体が常にスキップされていたため、この非互換性は実際に
`_C_stable_libtorch` をビルドするまで一度も検出されていませんでした。

実装は `torch::stable::Tensor` / `torch/csrc/stable/library.h`
(`STABLE_TORCH_LIBRARY_IMPL`, `TORCH_BOX`) ベースに全面書き直しました。ATen オプの呼び出しは
優先度順に 3 通りの方法を使い分けています。

1. `torch::stable::ops.h` の既製ラッパー (`narrow`, `copy_`, `to`, `full`, `sum` 等)。
2. ディスパッチャスタック直接呼び出し (`torch_call_dispatcher("aten::op", ...)`) —
   Scalar 型引数を持たないオプ用 (`silu`, `sigmoid`, `rsqrt`, `gelu` など)。
3. コード生成済み C シム関数 (`torch/csrc/inductor/aoti_torch/generated/c_shim_cuda.h` の
   `aoti_torch_cuda_<op>`) — `Scalar` 型引数を持つオプ用。`Scalar` は `StableIValue` スタック
   経由では汎用的にボックス化できないため(`torch::stable::ops.h` の `fill_`/`full` の
   コメント参照)、`double` を直接取るこれらのシム関数を使う。`clamp`/`clamp_max` には
   シムが無いため、`torch::stable::full(...)` で定数テンソルを作り
   `aten::minimum`/`aten::maximum` (Tensor のみ、Scalar 不要) で代替。

さらに、手作りの pow/mean/rsqrt 合成ではなく、既存の複合オプ
(`aoti_torch_cuda__fused_rms_norm`)を 1 回呼ぶだけで `rms_norm`/`fused_add_rms_norm` を
実装できる場合はそちらを優先しました(手動 fp32 式と数値的に完全一致することを確認済み)。

---

## 3. 変更ファイル一覧

| ファイル | 変更種別 | 内容 |
| :-- | :-- | :-- |
| `csrc/v100_adapter/v100_fallback_ops.cpp` | 新規→書き直し | stable ABI (`torch::stable::Tensor`) での ATen フォールバック実装 + `STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops)` 登録 |
| `csrc/libtorch_stable/torch_bindings.cpp` | 変更 | 対象 6 オプのネイティブ登録を `#ifndef VLLM_V100_ADAPTER` でゲート。加えて DeepGEMM 専用の `silu_and_mul_quant` / `persistent_masked_m_silu_mul_quant` も同様にゲート(§5.3) |
| `CMakeLists.txt` | 変更 | `VLLM_V100_ADAPTER` ゲート、アダプタソース追加、2 つの stable 拡張ターゲット再有効化、`VLLM_SM70_EXCLUDED_SRCS` に実際の除外ファイルを登録 |
| `csrc/libtorch_stable/cuda_vec_utils.cuh` | 変更 | BF16 パックド変換 3 関数に `__CUDA_ARCH__ >= 800` ガード追加(§5.3） |
| `csrc/libtorch_stable/moe/topk_softmax_kernels.cu`<br>`csrc/libtorch_stable/moe/topk_softplus_sqrt_kernels.cu` | 変更 | MoE ルーティングカーネルの BF16 パックド変換呼び出し箇所に同様のガード追加 |
| `csrc/libtorch_stable/quantization/activation_kernels.cu` | 変更 | `constexpr` な BF16 生成が `__CUDA_ARCH__ < 800` でコンパイル不可だった箇所を修正。ファイル全体は `VLLM_SM70_EXCLUDED_SRCS` で除外(§5.3) |
| `pyproject.toml` / `requirements/cuda.txt` / `requirements/build/cuda.txt` | 変更 | torch ピンを `2.11.0+cu126` に更新(§5.0) |
| `test_v100/conftest.py` | 変更 | 実ビルド済み `vllm._C_stable_libtorch` を検出する前に import していなかったバグを修正 |
| `docs/v100_fallback_archtecture/sm70_op_compatibility_matrix.md` | 変更 | 実装済みオプ・除外オプのステータス更新 |
| `test_v100/` | 既存 | CI 向け数値検証スイート(実ビルドで 8/8 パス確認済み) |

---

## 4. アダプタ実装 (`csrc/v100_adapter/v100_fallback_ops.cpp`)

互換性マトリクスで **Missing** とされる以下 6 オプの ATen フォールバックを実装し、
実機で動作確認済みです(§6)。数値はネイティブカーネルに合わせ **fp32 中間演算**で
計算し、出力 dtype へキャストします。

| オプ | フォールバック実装 |
| :-- | :-- |
| `rms_norm` | `aten::_fused_rms_norm` を直接呼び出し(= `input * rsqrt(mean(input^2, -1) + eps) [* weight]` と数値一致を確認済み) |
| `fused_add_rms_norm` | `residual := input + residual`(`aten::add.Tensor` 経由); `input := rms_norm(residual) [* weight]`(上記 `rms_norm` 実装を再利用、in-place) |
| `silu_and_mul` | `silu(gate) * up` |
| `silu_and_mul_with_clamp` | `(gate.clamp_max(limit) * sigmoid(alpha*gate)) * (up.clamp(±limit) + beta)`(clamp は `full()` + `minimum`/`maximum` で代替、§2.2) |
| `gelu_and_mul` | `gelu(gate, 'none') * up` |
| `gelu_tanh_and_mul` | `gelu(gate, 'tanh') * up` |

- gated activation の入力は `[..., 2*d]` レイアウトで、前半 `gate` / 後半 `up` に分割。
- シグネチャは `_C` の各オプスキーマに厳密に一致させ、Python 呼び出し側は無変更。
- 登録は `STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops)` を `#ifdef VLLM_V100_ADAPTER` で囲んで実施。

未実装 (今後の課題): `rotary_embedding`, `get_cuda_view_from_cpu_tensor`,
`topk_topp_sampler` — マトリクスに「⏳ Planned」として記載。
`paged_attention_v1/v2`, `reshape_and_cache` は sm_70 ネイティブ対応済みのため**変更禁止**。

---

## 5. ビルドシステムへの組み込み (`CMakeLists.txt` / `setup.py`)

### 5.0 必要な torch バージョン: `2.11.0+cu126`

`_C_stable_libtorch` / `_moe_C_stable_libtorch` を実際にビルドするには、以下 2 条件を
**両方**満たす torch が必要です。どちらも独立にチェックする必要があります。

1. **sm_70 (Volta) がバンドルされていること**: cu126 系ホイールは torch 2.14.1 現在まで
   sm_70 を含む。**cu128/cu129 系は torch 2.11 で sm_70 を打ち切っている**(cuDNN
   更新のため)。cu13x 系はそもそも sm_70 非対応。
2. **アダプタが使う stable ABI ヘッダが揃っていること**: CMake のゲートは
   `torch/csrc/stable/library.h` の存在のみを見るが(torch 2.7+ で存在)、実際のコードは
   `torch/csrc/stable/ops.h` と `torch/headeronly/util/Exception.h` も必要とする。
   これらは **torch 2.8 にはまだ無く**、torch 2.9 以降で揃う。

この 2 条件の交差点が `torch==2.11.0+cu126` でした。`pyproject.toml`
(`[build-system].requires` と `[tool.uv].find-links`)、`requirements/cuda.txt`、
`requirements/build/cuda.txt` の 3 箇所のピンをすべて更新する必要があります
(`[tool.uv]` の `no-build-isolation-package = ["torch"]` と組み合わさっているため、
ピンの不整合があると `uv` のビルド分離回避が壊れて解決に失敗します)。

今後さらに torch をアップグレードする場合も、上記 2 条件をそれぞれ独立に
再確認してください(`torch.cuda.get_arch_list()` と
`find .venv/.../torch/include/torch/csrc/stable -name ops.h`)。

### 5.1 `VLLM_V100_ADAPTER` ゲート

`VLLM_STABLE_EXT_SRC` 定義直後に、`CUDA_ARCHS` が全て CC < 8.0 のときのみ
`VLLM_V100_ADAPTER_ENABLE` を ON にする判定を追加。ON の場合:

- アダプタソースを `VLLM_STABLE_EXT_SRC` に追加。
- `set_source_files_properties(...)` で、アダプタと `torch_bindings.cpp` の両方に
  `COMPILE_DEFINITIONS "VLLM_V100_ADAPTER"` を付与。

判定は接尾辞付きアーキ (`9.0a`, `10.0f`, `12.0f` 等) も正しく扱います
(`7.0`/`7.5` のみ → ON、8.0 以上を含む → OFF)。

### 5.2 stable 拡張ターゲットの再有効化

試行錯誤時にビルドエラー回避のためコメントアウトされていた
`define_extension_target(_C_stable_libtorch ...)` および
`define_extension_target(_moe_C_stable_libtorch ...)`(各 `target_compile_definitions` /
ROCm リンクブロック含む)を復元し、`setup.py` の `ext_modules` にも両拡張を復元済みです
(これが無いと `build_ext` が `--target` を発行せず、CMake 側を直してもビルドされません)。

**2026-10-05 に §5.0 の torch 2.11.0+cu126 で実際にビルド・リンク・ロードまで確認**
しました。生成される `.so` は `vllm/_C_stable_libtorch.abi3.so` /
`vllm/_moe_C_stable_libtorch.abi3.so`で、`import vllm._C_stable_libtorch` 後に
`torch.ops._C.rms_norm` 等が解決できることを確認済みです。

Volta 向けソースビルド:

```bash
# ninja 実行ファイルが PATH に無い場合は shim を用意してから prepend する
printf '#!/bin/bash\nexec ".venv/bin/python" -m ninja "$@"\n' > bin/ninja && chmod +x bin/ninja
export PATH="$(pwd)/bin:$PATH" CUDA_HOME=/usr
TORCH_CUDA_ARCH_LIST="7.0" MAX_JOBS=4 .venv/bin/python setup.py develop
```

torch バージョンを変更した直後は `build/temp.linux-x86_64-cpython-312` を削除してから
再ビルドしてください(古い `TORCH_INCLUDE_DIRS` が CMake キャッシュに残り、ヘッダ不整合の
原因になります)。

### 5.3 sm_80+ ソースの自動除外(無関係カーネルのガード)

sm_70-only ビルドでも sm_80+ 専用カーネルは既に下記の仕組みで除外されます:

- **重量級ファミリー** (machete, cutlass scaled_mm/moe, marlin bf16/fp8, fp4, w4a8):
  既存の `cuda_archs_loose_intersection` によりアーキ交差が空となり自動除外。
  各 `.cu` は自身でオプ登録するため、除外してもリンクエラーにならない。
- **sm_80+ 外部プロジェクト** (flash-attn, deepgemm, fmha_sm100, flashmla, flashkda,
  qutlass, tml_fa4): `_HAS_AMPERE_OR_NEWER` ガードにより sm_70 ビルドから除外。

上記で漏れた「無条件リストされる基本ソース」が sm_70 でコンパイル失敗する場合に備え、
拡張可能な除外フックを用意しています:

```cmake
set(VLLM_SM70_EXCLUDED_SRCS "...")        # _C_stable_libtorch 用
set(VLLM_SM70_MOE_EXCLUDED_SRCS "")       # _moe_C_stable_libtorch 用
```

実際に初回の `TORCH_CUDA_ARCH_LIST="7.0"` ビルドで 1 ファイルがこのパターンに該当しました:

- `csrc/libtorch_stable/quantization/activation_kernels.cu`(DeepGEMM 向け
  `silu_mul_fp8_quant_deep_gemm_kernel` 他): BF16/FP8 ハードウェア組み込み関数
  (`make_bfloat162` 等)を多数無条件使用しており、sm_70 では個別パッチが現実的でないため
  ファイル全体を `VLLM_SM70_EXCLUDED_SRCS` に登録して除外。登録していたオプ
  (`silu_and_mul_quant`, `persistent_masked_m_silu_mul_quant`)は
  `torch_bindings.cpp` 側で `#ifndef VLLM_V100_ADAPTER` ガード追加(未定義シンボル回避)。
  これらのオプの唯一の Python 呼び出し元 (`vllm/model_executor/layers/fusion/fused_act_quant.py`)
  が既に `has_device_capability(90)` でゲートされているため、sm_70 では元々到達不能であり
  安全に除外できる。

**除外するほどではない単発の呼び出し箇所**には、もう一段軽いパターンも使いました:
`__CUDA_ARCH__ >= 800` の `#if` ガードで分岐し、`#else` 側は `__trap()`
(V100 では BF16 テンソルが `vllm/platforms/cuda.py` の BF16→FP16 フォールバックにより
そもそも到達しないため、到達不能コードとして安全)。適用箇所:
`csrc/libtorch_stable/cuda_vec_utils.cuh` の BF16 パックド変換 3 関数、
`csrc/libtorch_stable/moe/topk_softmax_kernels.cu` /
`topk_softplus_sqrt_kernels.cu` の BF16 ベクトルロード経路。
これは upstream の `csrc/libtorch_stable/type_convert.cuh` に既にある
`#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800) ...` という既存の慣習と同じパターンです。

**注意**: 除外するソースのオプが `torch_bindings.cpp` に中央登録されている場合は、
その `ops.impl(...)` も `#ifndef VLLM_V100_ADAPTER` でガードしないと
未定義シンボルのリンクエラーになります(アダプタと同じパターン)。

---

## 6. 検証 (`test_v100/`)

upstream の `tests/` とは分離した `test_v100/` に、フォーク専用の数値検証スイートを
配置しました(取り込み時のマージ衝突回避と、独立実行のため)。

- CC < 8.0 の GPU(V100 優先)を選択。無い環境では自動 skip。
- 既にビルド済みの `vllm._C_stable_libtorch` があればそれを直接使用(installed モード、
  §5 の実ビルドでこのパスを使用)。無ければ実アダプタ `.cpp` を
  `torch.utils.cpp_extension` で `-DVLLM_V100_ADAPTER` 付き **JIT コンパイル**して
  フル vLLM ビルド無しで検証できる想定だったが、アダプタが stable ABI 専用コードに
  なったため(§2.2)この JIT パスは現状未検証/要見直し。
  `test_v100/conftest.py` は、installed モードの検出前に
  `import vllm._C_stable_libtorch` を一度も行っておらず常に JIT パスに落ちるバグが
  あったため、2026-10-05 に修正済み。
- 各 `torch.ops._C.<op>` を呼び出し、fp32 参照と比較(fp16 許容誤差)。

実行:

```bash
.venv/bin/python -m pytest test_v100/ -v
```

### 実機確認結果 (Tesla V100-SXM2-16GB, sm_70, 2026-10-05, 実ビルド経由)

全 8 ケース PASS。

```
test_rms_norm[True] / [False]            PASSED
test_fused_add_rms_norm                  PASSED
test_silu_and_mul                        PASSED
test_gelu_and_mul                        PASSED
test_gelu_tanh_and_mul                   PASSED
test_silu_and_mul_with_clamp[defaults] / [alpha_beta]  PASSED
```

さらに、実モデルでの end-to-end 動作も確認しました:

```python
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", dtype="float16",
          gpu_memory_utilization=0.5, max_model_len=512, enforce_eager=True)
out = llm.generate(["The capital of France is", "2+2="],
                    SamplingParams(temperature=0.0, max_tokens=32))
```

- `TRITON_ATTN` バックエンドが選択される(`FLASH_ATTN`/`FLASHINFER` は CC < 8.0 で
  除外、設計通り)。
- モデルの `bfloat16` 重みが `float16` へキャストされる(BF16→FP16 フォールバック)。
- `rms_norm` / `silu_and_mul` 等がアダプタ経由(`vllm_c` IR優先度)で呼ばれ、正しい
  生成結果("The capital of France is" → " Paris. ..." 等)を得た。

### 他アーキテクチャでの確認 (Qwen3.5 / Gemma4, 2026-10-05)

アダプタが Qwen2 系だけでなく、本フォークに実装されている新しいアーキテクチャ族でも
汎用的に機能することを、以下 2 モデルで追加確認しました(同じ `dtype=float16`,
`enforce_eager=True` レシピ)。

| モデル | アーキテクチャ | 確認内容 |
| :-- | :-- | :-- |
| `Qwen/Qwen3.5-0.8B` | `Qwen3_5ForConditionalGeneration`(Mamba+Attention ハイブリッド、Gated DeltaNet 線形アテンション) | ロード成功。GDN の融合 CUDA デコードカーネルは「CC >= 8.0 必須」と判定され Triton/FLA 実装へ正しくフォールバック。FA2 も明示的に拒否され `TRITON_ATTN` を選択。生成結果は正しい("The capital of France is" → " Paris." 等、0.8B モデルなりの反復はあるが数値的には正しい)。 |
| `google/gemma-4-e2b-it` | `Gemma4ForConditionalGeneration`(マルチモーダル、不均一アテンションヘッド次元 `{sliding_attention: 256, full_attention: 512}`) | 約 9.85 GiB の重みをロードし V100 の VRAM に収まることを確認。FA4 非対応と判定され `TRITON_ATTN` を選択。**チャットテンプレート経由 (`llm.chat(...)`)** では `"The capital of France is **Paris**."` / `"2 + 2 = **4**"` と完全に正しい出力。生テキスト補完 (`llm.generate()`) では反復ループが出たが、これは instruct モデルに素のプロンプトを与えた場合の一般的な挙動であり、V100/アダプタ固有のバグではないことを同一条件のチャット形式比較で確認済み。 |

両モデルとも `rms_norm`/`fused_add_rms_norm`/`silu_and_mul` 系の V100 アダプタ経路を
通過しており、アダプタが特定のモデルファミリーに限定されず機能することの追加証拠と
なっています。

---

## 7. 今後の課題

1. ~~`TORCH_CUDA_ARCH_LIST="7.0"` でのフルソースビルド実行と、初回失敗ソースの
   `VLLM_SM70_EXCLUDED_SRCS` への登録~~ → **完了** (§5.3)。
2. `rotary_embedding`, `get_cuda_view_from_cpu_tensor`, `topk_topp_sampler`
   など、互換性マトリクスで「⏳ Planned」のままの残りオプのフォールバック実装
   (現状は該当オプが呼ばれる経路を踏まない限り未発覚のまま)。
3. ~~ビルド成功後、実モデル(Qwen2.5 / Gemma 等)での end-to-end 動作・精度評価。~~
   → **完了**(Qwen2.5-0.5B-Instruct, Qwen3.5-0.8B, Gemma4-E2B-it で確認済み、上記参照)。
   より大きいモデル(7B+)・長い生成・バッチ処理・MoE アーキテクチャでの追加検証は
   引き続き今後の課題。
4. `test_v100/` の JIT コンパイルパス(installed モードでない場合のフォールバック)が
   stable ABI 化後も動作するかの検証・修正。
5. 将来 torch をさらにアップグレードする際は、§5.0 の 2 条件(sm_70 の有無 / stable ABI
   ヘッダの有無)を都度独立に再確認すること。
