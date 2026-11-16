# 当前可运行主线

主线并列管理 STR-01（Matched Baseline）、Frozen WSSL-STR（Retained Model）和 AI-WSSL-STR（Experimental Candidate）。AI 通过新增统一注册 API 和独立配置/训练/推理入口调用；原 STR/WSSL 底层工厂、实现及历史入口保持兼容，不导入 MFAM 等其他论文复现。正式模型指针仍为 Frozen WSSL-STR。

## 路径

- 模型：`src/models/pure_deep.py`、`wssl.py`、`bilateral.py`、`activity_fusion.py`
- 统一配置：`configs/str01_seed42.yaml`、`str01_seed43.yaml`、`str01_seed44.yaml`
- PADS：`../data/raw/pads/` 与 `../data/processed/pads_multi_activity/v2_l1_full_length/`
- 冻结特征：`../data/ssl/harnet10_embeddings.npz`（与原缓存 SHA 相同）
- 现有权重：`../results/str01/`、`../results/frozen_wssl/`；均指向 deep_final 内部归档
- HarNet10 原官方源码/资产已在 `../artifacts/phase3b_external_ssl_20260928/vendor/ssl-wearables/` 留存，不下载、不改权重、不做重提取或微调。

使用服务器已有 `/home/zyt/envs/mfam_analysis/bin/python`；本次没有安装依赖。模型推理使用处理后六通道 Acc+Gyro；Frozen WSSL 额外按 subject/activity/wrist 从冻结 cache 取 1024-D 表征。保留 train-fold normalization、mask、true length、双腕融合、activity order、structured readout 和正式 recipe。

## 开发入口

```bash
cd /home/zyt/deep_final/mainline
/home/zyt/envs/mfam_analysis/bin/python -B scripts/run_development.py --help
```

该入口只接受 `str01` 或 `frozen_wssl`、三个固定 seeds 和一个显式 development context/inner；test subjects 被置为空，只在训练/验证 subject 子集上加载信号。现有输出存在时拒绝覆盖，不执行 outer-final、不搜索参数。本次整理没有调用训练入口。未来训练前仍需有具体实验授权和协议。

`scripts/verify_mainline.py` 是本次迁移验证工具，执行开发集推理与 norm refit，不训练；已有验证输出时拒绝覆盖。结果已保存于 `../conclusions/organization_20261111/`。

配置中的绝对数据路径已改到 deep_final。正式归档 checkpoint/config/source 不改路径，以保留旧 hash；验证时用统一配置构建相同模型并 strict load 原权重，不宣称迁移源码 hash 等于训练时历史整棵源码 hash。


## Independent model registry

[Project model index](../conclusions/MODEL_INDEX.md) · [Registry](models/registry.json) · [AI-WSSL-STR architecture/config/provenance](docs/models/AI_WSSL_STR.md).

```python
from src.models.registry import build_registered_model, list_models
from src.models.registered_checkpoint import load_registered_checkpoint
model = build_registered_model("AI-WSSL-STR", seed=42)
model, provenance = load_registered_checkpoint(
    "AI-WSSL-STR",
    checkpoint_id="AI-WSSL-STR:certified-l1:seed42:context0:inner0",
    device="cuda",
)
# No training: named checkpoint forward on matching cached bilateral tokens.
result = model.forward_tokens(bilateral_tokens, activity_mask)
```

`src.models.registry.build_registered_model` accepts all three model names. The original `src.models.build_model` remains the unchanged low-level STR-backbone factory used by historical STR/WSSL code; it is not silently repurposed. AI owns `src/models/ai_wssl_str.py`, three independent `configs/ai_wssl_str_seed*.yaml`, `scripts/train_ai_wssl_str.py`, the named `scripts/model_cli.py`, and a48-checkpoint catalog. New save format records model ID/status/config; historical checkpoints strict-load unchanged.

```bash
/home/zyt/envs/mfam_analysis/bin/python -B scripts/model_cli.py --list
/home/zyt/envs/mfam_analysis/bin/python -B scripts/model_cli.py --model AI-WSSL-STR --checkpoint-id AI-WSSL-STR:certified-l1:seed42:context0:inner0 --device cuda
/home/zyt/envs/mfam_analysis/bin/python -B scripts/train_ai_wssl_str.py --help
```

AI training remains the original Phase8B token recipe (batch8/workers0/train-balanced CE/AdamW/cosine50/patience12); only the added67,522 parameters train. The training CLI offers `--validate-only`, which was used for registration tests. No new training, L1 processing, normalization fit, HarNet extraction, threshold fitting or full15-split comparison was run. The candidate's current Certified pilot has DD Recall tradeoffs and is **not Retained**.
