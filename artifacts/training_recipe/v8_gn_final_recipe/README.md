# V8-GN 训练超参数确认报告

日期：2026-09-19  
范围：V8-GN 架构完全冻结；只使用固定 5×3 inner-development folds；未访问 outer-test。

## 1. 目的和协议

本轮不是网络搜索，而是在固定 NR-01 / V8-GroupNorm backbone 上，对 learning rate、
batch size、AdamW weight decay、temporal dropout 和 scheduler 做少量、分阶段确认。
seed 42 的完整 15 folds 仅作为方向筛选；任何最终 recipe 必须使用 seed 42/43/44
确认。单 seed 最高值不得作为最终选择依据。

原 recipe：AdamW，LR 2e-4，batch 8，WD 1e-4，encoder/activity/classifier dropout
0.1/0.1/0.2，cosine scheduler（minimum LR 1e-6），50 epochs，early-stopping patience
12，gradient clip 5.0，FP32。

方向筛选门槛：相对当前阶段 recipe，BA 至少 +0.005、至少 9/15 folds 获胜，且
AUROC/Macro-F1 不发生明显回退。最终候选还需三 seed 检验 seed 方差和类别召回。

## 2. 实验演化表

| 实验 | 单一改动 | 证据范围 | BA | AUROC | Macro-F1 | PD/DD recall | 对照 BA 差值 | 结论 |
|---|---|---|---:|---:|---:|---|---:|---|
| TR-BASE | 原 V8-GN recipe | 3 seeds | 0.6707±0.0088 | 0.6800±0.0103 | 0.6631±0.0137 | 0.7782/0.5631 | — | 冻结起点 |
| TR-LR01 | LR 2e-4→1e-4 | seed 42 | 0.6840 | 0.6937 | 0.6836 | 0.8134/0.5546 | +0.0061 | 进入最终确认 |
| TR-LR02 | LR 2e-4→3e-4 | seed 42 | 0.6790 | 0.6935 | 0.6753 | 0.8035/0.5545 | +0.0012 | 拒绝 |
| TR-BS01 | batch 8→16，LR=1e-4 | seed 42 | 0.6821 | 0.6905 | 0.6833 | 0.8225/0.5417 | -0.0018 vs LR01 | 拒绝 |
| TR-WD01 | WD 1e-4→1e-3，LR=1e-4 | seed 42 | 0.6863 | 0.6967 | 0.6838 | 0.8043/0.5683 | +0.0023 vs LR01 | 拒绝 |
| TR-DO01 | temporal dropout 0.1→0.2 | seed 42 | 0.6831 | 0.6962 | 0.6811 | 0.8088/0.5574 | -0.0009 vs LR01 | 拒绝 |
| TR-SC01 | cosine→constant LR | seed 42 | 0.6814 | 0.6943 | 0.6774 | 0.8050/0.5579 | -0.0025 vs LR01 | 拒绝 |
| TR-FINAL | LR=1e-4，其余原 recipe | 3 seeds | 0.6749±0.0133 | 0.6821±0.0100 | 0.6718±0.0128 | 0.8028/0.5470 | +0.0043 | 拒绝；回退原 recipe |

## 3. 分项结论

### Learning rate

1e-4 在 seed 42 达到方向筛选门槛：BA +0.0061、9/15 folds 胜、Macro-F1 +0.0085。
3e-4 只有 BA +0.0012、5/15 folds 胜，且 NLL 明显恶化，因此高侧候选被拒绝。

但 1e-4 的三 seed 确认显示：BA 0.6749±0.0133，相对原 recipe 仅 +0.0043；尽管
11/15 folds 胜、Macro-F1 +0.0087、Accuracy +0.0127、NLL -0.0231，但 BA 未达到
预登记 +0.005，BA seed SD 从 0.0088 增至 0.0133，DD recall 下降 0.0160。因此
不能用 seed 42 的改善替换原 recipe。

### Batch size

在 LR=1e-4 下，batch 16 相对 batch 8：BA -0.0018、AUROC -0.0032、DD recall
-0.0128。更大 batch 同时减少了每 epoch 的更新次数，未形成稳定收益，保留 batch 8。

### Weight decay

WD 1e-3 相对 1e-4：BA +0.0023，但仅 3 folds 改善、10 folds 完全持平；F1 几乎
不变。变化主要来自少量阈值临界样本，不能视为稳定正则化收益，保留 1e-4。

### Dropout

只提高 temporal encoder dropout 至 0.2 后，BA -0.0009、Macro-F1 -0.0025；虽然
NLL/Brier 改善，但主要判别指标没有改善，保留 0.1。

### Scheduler

恒定 1e-4 相对 cosine：BA -0.0025、Macro-F1 -0.0062、NLL/Brier 回退。部分 folds
出现较晚峰值，但整体不稳定，保留 cosine。

### Optimizer

未增加优化器对照。AdamW 下唯一进入多 seed 确认的候选仍未达到最终替换门槛，
没有明确证据表明优化器本身是主要瓶颈。此时加入新优化器会扩大同一开发集上的
搜索自由度，不符合小规模受控确认原则。

## 4. 最终冻结 recipe

```yaml
optimizer: adamw
learning_rate: 2.0e-4
batch_size: 8
weight_decay: 1.0e-4
adam_betas: [0.9, 0.999]
temporal_encoder_dropout: 0.1
activity_attention_dropout: 0.1
classifier_dropout: 0.2
scheduler:
  type: cosine
  minimum_lr: 1.0e-6
epochs: 50
early_stopping:
  metric: balanced_accuracy
  patience: 12
gradient_clip_norm: 5.0
mixed_precision: false
```

冻结对象为 **V8-GN 架构 + 原 training recipe**。本轮结果表明附近超参数区域没有
稳定且有意义的全面优势，因此停止调参，进入跨个体泛化与不变表征研究。

## 5. 解释边界

- 所有结论只来自 inner-development；outer-test 未访问。
- 1e-4 的多数 fold BA/F1 改善是真实实验事实，但跨 seed 方差和 DD recall 同时恶化；
  因此“更优”结论不成立。
- 1e-4 下若干 folds 训练到 50-epoch 上限，提示 epochs 可能与 LR 耦合；本轮为保持
  单因素控制没有改变 epochs，也不再追加搜索。
- 最终冻结是保守模型选择决定，不意味着原 recipe 在所有未来队列上必然最优。

## 6. 产物位置

- 工作目录：`/home/zyt/deep_final/foundation_validation`
- 本轮完整运行：`outputs/training_recipe/`
- 三 seed 汇总：`outputs/training_recipe/tr_final_three_seed_analysis.json`
- 冻结索引：`/home/zyt/deep_final/artifacts/training_recipe/v8_gn_final_recipe`

