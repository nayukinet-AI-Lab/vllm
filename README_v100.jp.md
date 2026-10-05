# V100 (sm_70) C++ アダプタ層 実装ドキュメント

本ドキュメントは、NVIDIA Tesla V100 (Volta / Compute Capability 7.0 / sm_70) 上で
vLLM を動作させるための **C++ アダプタ層**の初期実装と、ビルドシステムへの組み込み内容を
まとめたものです。

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
  `TORCH_LIBRARY_IMPL` により捕獲し、ATen ネイティブ実装へフォールバックさせる。

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

---

## 3. 変更ファイル一覧

| ファイル | 変更種別 | 内容 |
| :-- | :-- | :-- |
| `csrc/v100_adapter/v100_fallback_ops.cpp` | 新規 | ATen ネイティブ・フォールバック実装 + `TORCH_LIBRARY_IMPL(_C, CUDA)` 登録 |
| `csrc/libtorch_stable/torch_bindings.cpp` | 変更 | 対象 6 オプのネイティブ登録を `#ifndef VLLM_V100_ADAPTER` でゲート |
| `CMakeLists.txt` | 変更 | `VLLM_V100_ADAPTER` ゲート、アダプタソース追加、2 つの stable 拡張ターゲット再有効化、sm_70 ソース除外フック |
| `setup.py` | 変更 | `_C_stable_libtorch` / `_moe_C_stable_libtorch` を `ext_modules` に復元 |
| `docs/v100_fallback_archtecture/sm70_op_compatibility_matrix.md` | 変更 | 実装済みオプのステータス更新 |
| `test_v100/` | 新規 | CI 向け数値検証スイート |

---

## 4. アダプタ実装 (`csrc/v100_adapter/v100_fallback_ops.cpp`)

初期実装として、互換性マトリクスで **Missing** とされる以下 6 オプの ATen フォールバックを
実装しました。数値はネイティブカーネルに合わせ **fp32 中間演算**で計算し、出力 dtype へ
キャストします。

| オプ | フォールバック実装 |
| :-- | :-- |
| `rms_norm` | `input * rsqrt(mean(input^2, -1) + eps) [* weight]` |
| `fused_add_rms_norm` | `residual := input + residual`; `input := rms_norm(residual) [* weight]`(in-place) |
| `silu_and_mul` | `silu(gate) * up` |
| `silu_and_mul_with_clamp` | `(gate.clamp_max(limit) * sigmoid(alpha*gate)) * (up.clamp(±limit) + beta)` |
| `gelu_and_mul` | `gelu(gate, 'none') * up` |
| `gelu_tanh_and_mul` | `gelu(gate, 'tanh') * up` |

- gated activation の入力は `[..., 2*d]` レイアウトで、前半 `gate` / 後半 `up` に分割。
- シグネチャは `_C` の各オプスキーマに厳密に一致させ、Python 呼び出し側は無変更。
- 登録は `TORCH_LIBRARY_IMPL(_C, CUDA, m)` を `#ifdef VLLM_V100_ADAPTER` で囲んで実施。

未実装 (今後の課題): `rotary_embedding`, `get_cuda_view_from_cpu_tensor`,
`topk_topp_sampler` — マトリクスに「⏳ Planned」として記載。
`paged_attention_v1/v2`, `reshape_and_cache` は sm_70 ネイティブ対応済みのため**変更禁止**。

---

## 5. ビルドシステムへの組み込み (`CMakeLists.txt` / `setup.py`)

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
ROCm リンクブロック含む)を復元しました。あわせて `setup.py` の `ext_modules` にも
両拡張を復元しています(これが無いと `build_ext` が `--target` を発行せず、
CMake 側を直してもビルドされません)。

Volta 向けソースビルド:

```bash
TORCH_CUDA_ARCH_LIST="7.0" MAX_JOBS=4 python setup.py develop
```

### 5.3 sm_80+ ソースの自動除外(無関係カーネルのガード)

sm_70-only ビルドでも sm_80+ 専用カーネルは既に下記の仕組みで除外されます:

- **重量級ファミリー** (machete, cutlass scaled_mm/moe, marlin bf16/fp8, fp4, w4a8):
  既存の `cuda_archs_loose_intersection` によりアーキ交差が空となり自動除外。
  各 `.cu` は自身でオプ登録するため、除外してもリンクエラーにならない。
- **sm_80+ 外部プロジェクト** (flash-attn, deepgemm, fmha_sm100, flashmla, flashkda,
  qutlass, tml_fa4): `_HAS_AMPERE_OR_NEWER` ガードにより sm_70 ビルドから除外。

上記で漏れた「無条件リストされる基本ソース」が万一 sm_70 でコンパイル失敗する場合に
備え、拡張可能な除外フックを用意しました:

```cmake
set(VLLM_SM70_EXCLUDED_SRCS "")        # _C_stable_libtorch 用
set(VLLM_SM70_MOE_EXCLUDED_SRCS "")    # _moe_C_stable_libtorch 用
```

初回の `TORCH_CUDA_ARCH_LIST="7.0"` ビルドで失敗したソースがあれば、そのパスを
上記変数に追記するだけで `list(REMOVE_ITEM ...)` により除外されます。
**注意**: 除外するソースのオプが `torch_bindings.cpp` に中央登録されている場合は、
その `ops.impl(...)` も `#ifndef VLLM_V100_ADAPTER` でガードしないと
未定義シンボルのリンクエラーになります(アダプタと同じパターン)。

---

## 6. 検証 (`test_v100/`)

upstream の `tests/` とは分離した `test_v100/` に、フォーク専用の数値検証スイートを
配置しました(取り込み時のマージ衝突回避と、独立実行のため)。

- CC < 8.0 の GPU(V100 優先)を選択。無い環境では自動 skip。
- 実アダプタ `.cpp` を `torch.utils.cpp_extension` で `-DVLLM_V100_ADAPTER` 付き
  **JIT コンパイル**(`_C` スキーマ定義を同梱)。**フル vLLM ビルド不要**。
  (アダプタは純 C++ で ATen を呼ぶだけであり、CUDA カーネルは導入済み libtorch が提供)
- 各 `torch.ops._C.<op>` を呼び出し、fp32 参照と比較(fp16 許容誤差)。
- 既にビルド済みの `vllm._C` があればそれを直接使用(installed モード)。

実行:

```bash
.venv/bin/python -m pytest test_v100/ -v
```

### 実機確認結果 (Tesla V100-SXM2-16GB, sm_70)

全 8 ケース PASS(max_abs_err 〜 1e-3、fp16 許容範囲内)。

```
test_rms_norm[True] / [False]            PASSED
test_fused_add_rms_norm                  PASSED
test_silu_and_mul                        PASSED
test_gelu_and_mul                        PASSED
test_gelu_tanh_and_mul                   PASSED
test_silu_and_mul_with_clamp[defaults] / [alpha_beta]  PASSED
```

---

## 7. 今後の課題

1. `TORCH_CUDA_ARCH_LIST="7.0"` でのフルソースビルド実行と、初回失敗ソースの
   `VLLM_SM70_EXCLUDED_SRCS` への登録(端数対応)。
2. `rotary_embedding` 等、残りの Missing オプのフォールバック実装。
3. ビルド成功後、実モデル(Qwen2.5 / Gemma 等)での end-to-end 動作・精度評価。
