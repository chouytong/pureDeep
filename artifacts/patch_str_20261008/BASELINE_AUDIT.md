# Patch-STR：Repository / P0 审查（2026-10-08）

## 已确认

原STR75524参数：6-channel normalized Acc+Gyro→shared64D temporal stem(kernel15,stride2)→kernel7 dilation[1,2,4] depthwise residual、GroupNorm8、dropout.1；temporal learned feature attention/mean/std；独立norm-free learned-moment32D(kernels1/15/63)→融合64D wrist。双腕Full融合left/right/mean/absolute difference/mask为258D；activity-ID attention保留11活动身份→258D subject；原classifier与ordered11×16 structured residual logits相加。moment/bilateral/ordered STR路径不改。

输入是已有L1处理Acc与Gyro、固定trim48、976/2000真长度、train-fold [11,2,6,1] mean/std。实际dataset保存activity_lengths[B,11]，不是另一套per-wrist length；原manifest loader强制左右shape一致，否则报错。因此活动真长度对两个valid wrists同时成立；原subject forward按真长度分组并slice，且只有valid wrists送入encoder。新wrapper沿用该逻辑，不重写或猜测长度。Missing activity0length被mask排除。原批容器padding不作真实采样。100Hz为标称时间尺度，不宣称已恢复精确clock或生理适宜窗口。

正式P0完整45checkpoint全validation重算PASS：IDs/labels/decision/两列prob/6metric一致，最大概率差1.11e-16。15train-only norm refit对45checkpoint逐位一致。已有Phase2 baseline45次retrain状态、预测、norm/epoch严格匹配，checkpoint provenance当前source fae112dcf6d05de5f94bdbe0c7e2b3748d79d343200f441762f4dd8db82f6617。早期whole-tree4fd74c与当前不同，不能称整棵历史源码相同；current-source matched45资产及本轮全forward证明数值匹配。本次不重复P0训练。

P0 seed-first正式均值：Acc.730718227877/BA.697189405719/AUROC.717630721370/F1.686067804389/PD.777864165535/DD.616514645903，与用户近似值一致。类0PD/1DD，固定argmax/.5，tiePD。AUROC沿用两概率列的tie-aware class AUC均值，不替换为单DD sklearn AUC。AdamW2e-4/WD1e-4/betas.9,.999/batch8/FP32/clip5/cosine50floor1e-6/max50/strict BA first-best/patience12/train-fold balanced CE不变。

## 历史区别与停止边界

- LR-01改变原主干dilation为[1,4,16,64,128]，BA−.0167、3/15胜，被拒绝；本轮保留原[1,2,4]并显式提取局部patch，不重复long-dilation。
- V8-GN gyro-only log-rFFT/128bins+concat被拒绝，BA/AUC−.0146/−.0163；最新固定五频带STR residual也REJECT，不重开FFT/filter/frequency搜索。
- Phase2 jitter/scaling/mild displacement/mixup、loss/sampling等16单因素无联合保留者；本轮不修改它们。
- HarNet front/back窗口特征等历史WSSL实验是外部SSL缓存路径；本轮STR normalized6channel直接学习200sample patches+9/19 patch aggregation，不是相同条件。
- WSSL Activity Scaling最新INCONCLUSIVE，不保留；总体retained best仍originalFrozenWSSL(BA.719644/AUC.759553)。本轮P0按用户明确为STR，不使用或叠加WSSL。
- 检索development报告/config/source未发现完全相同的200/100、共享6→32→64 patch encoder、zero-init64→16→64、bag/ordered depthwise+attention三条件实验。检索范围不构成对不可见历史的绝对证明。

## 解释限制

“activity内信息过早压缩”仅假设，不把先前PAG/其他机制诊断停止结论改成已证明loss。P2−P1改变sequence conv及attention pooling，额外321参数，不是唯一order因素或严格capacity/算子匹配；即便正结果只能支持当前ordered aggregation组合，不能隔离所有pooling/capacity原因。inference shuffle可补充顺序依赖，但仍非临床因果。

PaAno为异常检测(ICLR2026正式论文)，本轮只借鉴patch/时序关系思想，不复制triplet、InstanceNorm、memory bank/kNN。来源：https://proceedings.iclr.cc/paper_files/paper/2026/hash/5991785875cf5eee9c29e2ed7d26d1c0-Abstract-Conference.html 。

本阶段新增审查脚本/common与聚合审核JSON/CSV，无新候选训练，无outer signal/预测/performance，无recipe更改。下一步条件满足：固定patch extraction实现及测试。
