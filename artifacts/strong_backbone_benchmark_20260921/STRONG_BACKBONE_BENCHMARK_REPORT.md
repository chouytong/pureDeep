# Strong Backbone Benchmark Report

日期：2026-09-21  
任务：PADS PD-vs-DD，双腕、11 activities、subject-level aggregation  
开发协议：固定5×3 subject-level inner-development splits；seeds 42/43/44；train-only normalization  
证据边界：未创建、读取或预测outer-final/test数据；outer信息未参与模型选择、调参或统计推断

## 结论摘要

V8-GN是本轮最强且最稳定的pure-deep backbone，并正式冻结为后续研究基座。标准ResNet1D
显著方向性退化；InceptionTime仅与V8-GN持平，同时参数量约增至6.8倍且PD/DD recall的
seed波动更大。唯一的Gyro-only learnable FFT分支降低BA、AUROC、Macro-F1和DD recall，
并显著恶化seed稳定性，因此拒绝且停止整个frequency-branch搜索路线。

本轮也确认一个重要限制：development-only H1 handcrafted/statistical diagnostic
（BA 0.6918、AUROC 0.7515）仍优于最强pure-deep V8-GN（0.6707、0.6800）。不能通过继续
增加深度模块掩盖这一事实。

## 固定设计与公平性

所有模型保持相同输入、双腕full fusion、11-activity attention aggregation、balanced CE、
AdamW（lr 2e-4，weight decay 1e-4）、batch size 8、cosine schedule、50 epochs、BA early
stopping、无AMP及相同固定splits。候选没有专属超参数搜索。三seed均完整执行15个split；
统计时先在每个split聚合三个seed，再以15个独立split为单位比较。

模型规模：V8-GN 71,026；ResNet1D 279,137；InceptionTime 484,449；V8-GN + Gyro FFT
78,450。新增实现的模型/fusion契约测试18/18通过。

## Phase A：Pure-deep backbone benchmark

| Backbone | Params | Accuracy | BA | Macro-F1 | AUROC | PD Recall | DD Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| V8-GN | 71,026 | 0.7154±0.0177 | 0.6707±0.0088 | 0.6631±0.0137 | 0.6800±0.0103 | 0.7782±0.0303 | 0.5631±0.0134 |
| ResNet1D | 279,137 | 0.7093±0.0072 | 0.6470±0.0094 | 0.6380±0.0135 | 0.6540±0.0049 | 0.7968±0.0073 | 0.4972±0.0162 |
| InceptionTime | 484,449 | 0.7078±0.0160 | 0.6692±0.0093 | 0.6579±0.0083 | 0.6819±0.0082 | 0.7625±0.0441 | 0.5760±0.0564 |

数值为三seed均值±sample SD。

相对matched V8-GN，ResNet1D的BA/AUROC/Macro-F1/DD recall分别变化-0.0237/-0.0260/
-0.0251/-0.0659，仅4/3/4/4个split改善。对应paired rank-biserial为-0.750/-0.617/
-0.517/-0.567。其BA未校正p=0.0084，但对Phase A/B 12项比较BH校正后q=0.1003；其他
q也≥0.1329。虽然多重校正后未显著，效应方向、幅度与低胜率一致，足以按预注册性能门槛拒绝。

InceptionTime相对V8-GN变化-0.0014/+0.0019/-0.0052/+0.0129，改善split为8/9/6/9；
paired rank-biserial绝对值≤0.142，四项q均0.8040。它未达到BA +0.01、AUROC +0.015，
Macro-F1反而下降，且PD/DD recall的seed SD明显增大。因此仅为统计持平，不替换V8-GN。

Phase A回答：当前结果不支持V8-GN存在明显的标准CNN结构能力不足。严格边界是，在固定recipe
和本次有限成熟候选下，增加深度/多尺度容量没有带来实质跨受试者收益；不能由此证明任务性能
已达到上限。

## Phase B：唯一的frequency-aware单因素实验

唯一改动是在V8-GN wrist encoder后加入Gyro XYZ专用的可微log-rFFT分支：0–12 Hz、128
bins、轻量Conv1D hidden 24、16维spectral embedding，再与64维temporal embedding简单
融合。没有handcrafted频带/峰值特征，也没有比较STFT、wavelet或其他频域模型。

| Model | Params | Accuracy | BA | Macro-F1 | AUROC | PD Recall | DD Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| V8-GN | 71,026 | 0.7154 | 0.6707 | 0.6631 | 0.6800 | 0.7782 | 0.5631 |
| V8-GN + Gyro FFT | 78,450 | 0.7041 | 0.6561 | 0.6480 | 0.6637 | 0.7718 | 0.5404 |

频域模型BA seeds为0.6753/0.6553/0.6377（SD 0.0188），AUROC为0.6827/0.6653/
0.6429（SD 0.0199），均比V8-GN的seed SD 0.0088/0.0103更不稳定。相对V8-GN，
BA/AUROC/Macro-F1/DD recall变化-0.0146/-0.0163/-0.0151/-0.0227，改善split为
5/5/4/4；rank-biserial为-0.600/-0.483/-0.583/-0.390，BH q为0.1329/0.1834/
0.1329/0.2970。

Phase B回答：频域分支没有提供独立、稳定增益。SEPA-01发现的稳定频谱结构是真实的表型证据，
但本实验表明其不能通过该轻量learnable branch转化为额外泛化性能；可能已被时域主干部分利用，
或主要反映PD/DD标签空间重叠。按预注册规则停止frequency branch，不再搜索频率范围、FFT/
STFT/wavelet或更大频域网络。

## 五个最终问题

1. 最强且最稳定的pure-deep backbone：V8-GN。
2. V8-GN是否明显结构能力不足：否；ResNet1D更差，InceptionTime仅持平且成本/波动更高。
3. frequency-aware branch是否有独立稳定增益：否；所有关键指标下降，split胜率低，seed方差增大。
4. 正式冻结模型：V8-GN，71,026参数，Full bilateral fusion与现有training recipe不变。
5. 是否仍弱于handcrafted/statistical baseline：是。现有development-only H1为BA 0.6918、
   AUROC 0.7515，高于V8-GN 0.6707/0.6800；尤其AUROC高0.0715。H1并非本轮选择信号，
   也未使用outer信息，但该差距必须如实保留。

## 可复现产物与完整性

- Phase A：`artifacts/strong_backbone_benchmark_20260921/phase_a/`
- Phase B：`artifacts/strong_backbone_benchmark_20260921/phase_b/`
- 统一配对推断：`artifacts/strong_backbone_benchmark_20260921/paired_inference.json`
- 训练输出：`foundation_validation/outputs/strong_backbone_benchmark/`
- 实现：`foundation_validation/src/models/pure_deep.py`
- 配置：`foundation_validation/configs/strong_*`与`v8gn_gyro_fft_seed*.yaml`

9个新增seed运行均为15 folds；所有`development_summary.json`中的
`outer_test_loader_created`、`outer_test_signal_accessed`、`outer_test_predictions_accessed`
均为false。失败候选及其完整结果均保留。
