# 当前可运行主线

这里只保留 PADS PD vs DD 的 STR-01 与 retained Frozen WSSL-STR，以及所需的数据、训练、指标和 provenance 工具。模型工厂拒绝 MFAM/subject-MFAM 等其他模型；没有导入旧复现的 s4_final、r2_stabilization、targeted_ablation、H1 或其他数据集项目。原 STR 算子源码直接复制；WSSL 使用归档的相同 forward 和 checkpoint state keys。

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
