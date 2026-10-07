# WSSL Phase B：Activity Scaling 运行前固定协议（2026-10-08）

仅一个候选：11个 unconstrained scalar，配置 activity 顺序、同活动双腕共享、g=1、完整 scratch training。乘在 Linear(LayerNorm(SSL)) 完整输出上（含 bias），原 activity/wrist mask 随后应用。原模型源不改。既有独立接口采用显式 SSL 输入/实例本地 loader bridge；新实例构造及 load_state_dict，不 deepcopy。所有原 classifier 参数训练；HarNet 冻结且只读缓存。

15固定 development splits × seeds42/43/44。使用原引擎与原 train-only normalization。AdamW LR2e-4/WD1e-4/betas(.9,.999)，batch8，FP32，clip5，cosine50/floor1e-6，max50，balanced train-fold weighted CE，patience12/strict BA first-best，阈值.5/argmax/tiePD。g 进入同一 AdamW param group，继承 WD；不新增排除或超参数。初始普通参数/RNG需逐seed相同。优化器多11参数，梯度clip作用于所有 trainable 参数，这是新增参数的正常影响。

Baseline重复训练不做：复用正式45checkpoint与已完成EMA实验中的普通模型45次精确复现（仅 ordinary 部分），正式训练前重新核验普通/正式全部状态、log metrics、预测、norm、epoch和source provenance。已有Phase A全validation复现与15norm refit也核验SHA。45ckpt各至少两个真实validation batches做g=1 logits严格相等、reload测试。归档没有logits，只声称与重算baseline logits一致。

正式前：11参数/配置index、mask/activity/wrist/padding isolation、finite nonzero g gradient、原STR与projection update、缓存不变、optimizer/model reload精确、原引擎smoke PASS。Smoke只检查实现，不能选择设计。原始checkpoint/cache/source SHA需全程不变。每run不得resume未完成stage或覆盖；只允许完整已审计stage复用。

统计：各split先平均3seed指标，以15split配对。固定bootstrap10000、RNG20261008、同一配对resample；mean/median delta、正/零/负split、SD、95%CI、dz，Wilcoxon/BH六指标仅exploratory，不额外加入未要求的显著性闸门。seed SD=15split内样本SD的均值；seed-mean SD=3个seed各自15split均值的样本SD；split SD=15个seed-first均值的样本SD。报告同seed方法agreement、模型内部三seed agreement、PD/DD error recovery/harm。Subjects/seed-runs/pairs不作独立样本。反复复用划分与顺序筛选带来选择乐观；CI仅重复development描述。

固定 RETAIN 联合规则：ΔBA>0、≥10/15正、BA CI lower>0；ΔAUROC≥0；ΔMacroF1≥0；PD和DD mean delta均≥−.01；BA/AUROC within-split seedSD≤baseline的1.25倍（沿用此前波动guard，设计选择不是已知最优）。AUROC正split数另报告，不额外强制10。若BA mean≤0或recall guard失败，则REJECT；否则若联合保留规则不全通过，则INCONCLUSIVE/DO NOT RETAIN。不能事后调整。

完整45runs结果才分析g：每run/seed/activity；seed-first15split均值、overall均值/SD/range、3seed means。Phase A BA/AUC贡献与g的Spearman以及TouchNose/TouchIndex scale与split PD/DD变化关系仅posthoc描述、不作选择、不临床因果归因。不以g排序替代性能证据。

REJECT后停止activity/wrist/22scalar/MLP/attention及救援搜索。只开展独立fixed-denominator weighted CE numerical audit（真实training batches、loss/gradient/class/batch composition/AdamW update；不保存更新到正式权重），核查历史相同candidate，不自动训练。INCONCLUSIVE则停止，不自动开启下一候选。RETAIN也只报告后续资格，不自动开额外训练。

Boundary: outer data/performance=NO；threshold changed=NO；HarNet adapted=NO；training recipe changed=NO；additional hyperparameter search=NO。没有 Frequency/EMA/adaptation/auxiliary/encoder/sampler/augmentation/LR等组合。数据/权重/cache/逐subject输出不Git公开，代码报告小型汇总发布并标记。
