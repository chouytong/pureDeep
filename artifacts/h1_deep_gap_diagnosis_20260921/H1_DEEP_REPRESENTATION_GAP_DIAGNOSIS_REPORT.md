# H1–Deep Representation Gap Diagnosis Report

日期：2026-09-21  
任务：PADS PD-vs-DD，固定 subject-level inner-development protocol  
证据边界：只使用15个development splits；V8-GN seeds 42/43/44先在相同split内聚合；未访问outer-final/test；未训练或修改backbone；未搜索H1或probe超参数

## 结论摘要

H1约0.0715的AUROC优势不能由单一feature family完整解释，但有两个稳定主贡献：
`time_location_scale`与`band_fraction`。前者包含真实实现中的mean/std/variance/RMS/
median/MAD/IQR/min/max/peak-to-peak/signal-energy，后者是0.5–3、3–7、7–12、12–20 Hz
相对频带功率。leave-one-family-out时，移除time/location/scale令BA/AUROC下降
0.0174/0.0181；移除band fraction令其下降0.0159/0.0130。两者AUROC下降在BH校正后
均显著。没有任何family-only模型接近full H1，说明优势来自跨family、跨activity、双腕的
联合统计结构，而非一个孤立特征。

V8-GN并非完全没有编码这些信息。wrist/activity-context表示可在unseen subjects上恢复局部
time/location/scale（R²约0.38–0.39）和band fraction（R²约0.17–0.21）；所有15个split
均为正。但从最终subject embedding恢复完整family结构时，R²分别为-0.689和-1.413。
train-only PCA 16维family representation的公平性检查仍为负，且简单mean-pooled subject
representations同样失败。因此缺口不是“原始时频模式完全没进入网络”，而是activity/wrist
specific统计结构没有以可线性恢复的形式保留到单一subject representation中。

H1只纠正了V8错误中的一部分。在1560个unseen-validation appearances中，H1-only correct为
194，V8-only correct为171，both-wrong为244。对V8 stable-error appearances，H1纠正率仅
30.0%；54名V8 stable-error subjects中，仅14名表现为persistent H1 rescue，32名仍是
persistent both-wrong，另8名mixed。H1更容易纠正靠近边界的V8错误，而不是最深的
phenotype-overlap：DD persistent rescue的V8 PD-like centroid margin为+3.00，both-wrong为
+4.66；PD为-1.31 vs -2.45。

下一版pure-deep研究最应针对的具体gap是：在不依赖handcrafted输入的前提下，保留并向
subject decision暴露activity-conditioned、wrist-structured的幅值/离散度统计与相对频谱分配，
尤其time/location/scale与band-fraction-like信息。当前诊断不授权立即设计或训练模块，也不
支持重启通用FFT或backbone搜索。

## A. H1 feature-family decomposition

### 真实feature schema

H1共4928维：11 activities × 2 wrists × 8 signals（Acc/Gyro XYZ及各自vector magnitude）×
28个真实统计特征。family grouping严格依据现有实现：

- `time_location_scale`：11类位置、尺度、范围和能量统计；
- `time_shape`：skewness、excess kurtosis、centered zero-crossing rate；
- `derivative`：derivative RMS/std；
- `spectral_global`：dominant frequency/power、spectral entropy、total power；
- `band_absolute`：四个绝对band powers；
- `band_fraction`：对应四个relative band fractions。

固定H1 pipeline为inner-train median imputation、zero-variance removal、train-only
StandardScaler及LogisticRegression(C=1, liblinear)，无超参数搜索。Full H1精确复现：

| Variant | Features | BA | AUROC | Macro-F1 | PD recall | DD recall |
|---|---:|---:|---:|---:|---:|---:|
| Full H1 | 4928 | 0.6918 | 0.7515 | 0.6858 | 0.7936 | 0.5900 |
| Time-domain all | 2816 | 0.6786 | 0.7341 | 0.6720 | 0.7801 | 0.5771 |
| Spectral all | 2112 | 0.6570 | 0.7161 | 0.6512 | 0.7699 | 0.5442 |
| time/location/scale only | 1936 | 0.6523 | 0.7034 | 0.6513 | 0.7909 | 0.5136 |
| band fraction only | 704 | 0.6505 | 0.6985 | 0.6439 | 0.7608 | 0.5402 |

没有family-only变体复现full H1。相对full H1的LOFO主结果：

| Removed family | ΔBA | BA wins/15 | BA rank-biserial | BA BH q | ΔAUROC | AUROC wins/15 | AUROC rank-biserial | AUROC BH q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| time/location/scale | -0.0174 | 4 | -0.667 | 0.0404 | -0.0181 | 1 | -0.950 | 0.0020 |
| band fraction | -0.0159 | 4 | -0.517 | 0.1338 | -0.0130 | 2 | -0.933 | 0.0023 |
| time shape | -0.0010 | 6 | -0.048 | 0.9377 | -0.0040 | 6 | -0.300 | 0.4308 |
| derivative | -0.0020 | 4 | -0.178 | 0.7933 | +0.0007 | 8 | +0.219 | 0.6270 |
| spectral global | +0.0063 | 8 | +0.341 | 0.4052 | +0.0004 | 6 | +0.067 | 0.9330 |
| band absolute | +0.0084 | 9 | +0.604 | 0.0969 | +0.0047 | 9 | +0.410 | 0.2835 |

BH校正覆盖family-only/LOFO × 6 families × 5指标共60项。time/location/scale是BA与
AUROC都稳定有价值的family；band fraction的AUROC贡献稳定、BA方向一致但校正后不显著。
absolute band power与global spectral statistics不是优势来源；去除后点估计反而改善。

## B. Deep recoverability

冻结V8-GN仅做前向推理，提取45个seed-fold的inner-train与unseen-validation wrist 64-D、
bilateral activity 258-D、activity-context 258-D及subject 258-D表示。固定Ridge α=1；X和
H1 targets均仅由inner-train拟合缩放。以下为三seed先按split聚合后的均值：

| Family | Wrist R² | Activity-context R² | Activity-context Spearman | Subject R² | Subject Spearman | Subject normalized MAE |
|---|---:|---:|---:|---:|---:|---:|
| time/location/scale | 0.382 | 0.394 | 0.697 | -0.689 | 0.232 | 1.007 |
| band fraction | 0.173 | 0.212 | 0.467 | -1.413 | 0.144 | 1.198 |
| time shape | 0.178 | 0.219 | 0.512 | -1.383 | 0.115 | 1.229 |
| derivative | 0.578 | 0.581 | 0.704 | -0.735 | 0.281 | 0.936 |
| spectral global | 0.189 | 0.193 | 0.467 | -0.592 | 0.146 | 1.125 |
| band absolute | 0.135 | 0.130 | 0.432 | -0.297 | 0.164 | 1.051 |

所有wrist、bilateral和activity-context family R²在15/15 splits为正，BH q=6.1e-5；所有
subject embedding R²在15/15为负。Activity-context相对subject的R²优势对六族均为15/15。

公平性检查将完整subject family target用inner-train PCA压缩到16维，并用相同subject数和
相同target比较wrist/activity简单均值与learned subject embedding。所有collapsed
representations的PCA-score R²仍为负；time/location/scale learned-subject R²=-1.263，band
fraction=-2.079。activity-context mean相对learned subject只在spectral-global上稳定更好
（BH q=0.0138），其他family没有稳定差异。因此不能把问题归因于某个attention权重；更稳妥的
结论是：activity/wrist-specific结构在压缩成单一subject vector后整体难以线性恢复。

高分类价值与低subject recoverability的交集明确指向：

1. `time_location_scale`：H1价值最高；局部表示已有中等编码，但subject readout不保留完整结构。
2. `band_fraction`：H1 AUROC价值稳定；局部编码更弱，subject recoverability最差。

## C. H1–V8 error complementarity

V8概率先在相同split内聚合seeds 42/43/44，H1按同一inner-train/validation subjects重拟合。

| Correctness group | PD rows | DD rows | Total |
|---|---:|---:|---:|
| Both correct | 769 | 182 | 951 |
| H1 only correct | 107 | 87 | 194 |
| V8 only correct | 123 | 48 | 171 |
| Both wrong | 105 | 139 | 244 |

H1纠正194/438（44.3%）个V8-wrong appearances，但纠正率强烈依赖V8错误稳定性：

| V8 stability | PD correction | DD correction |
|---|---:|---:|
| stable-correct subjects的偶发错误 | 26/36 = 72.2% | 9/11 = 81.8% |
| unstable | 62/106 = 58.5% | 37/85 = 43.5% |
| stable-error | 19/70 = 27.1% | 41/130 = 31.5% |

在54名V8 stable-error subjects中，persistent H1-only rescue仅14名（DD 10、PD 4），
persistent both-wrong为32名（DD 20、PD 12），其余8名mixed。只有5/54名在H1的四个
validation contexts中全部正确，27/54名四次均错。因此H1没有真正解决大多数V8稳定错误。

H1 rescue更常发生在较轻的opposite-class geometry：DD rescue与both-wrong的V8 PD-like
centroid margin为+3.00 vs +4.66，PD为-1.31 vs -2.45。DD rescue在CrossArms、DrinkGlas
的PD-like activity score也低于both-wrong（1.65 vs 2.82；0.40 vs 1.28）。这表明H1主要
纠正较靠近边界的V8错误，深度phenotype-overlap subjects仍共同失败。

H1 true-label方向贡献显示rescue不是由单一高风险activity驱动。H1-only DD中较大正贡献来自
TouchNose、RelaxedTask、HoldWeight和DrinkGlas；H1-only PD中来自RelaxedTask、DrinkGlas、
TouchNose和Entrainment，CrossArms贡献接近零或略负。H1利用的是跨activity联合统计轮廓。

对persistent H1-rescue与persistent both-wrong的高风险原始信号进行528项预设比较后，没有
任何结果通过BH q<0.10。若干PD LiftHold幅值/jerk差异未校正p较小，但最低q=0.262，不能作为
稳定解释。这个阴性结果反对用单一raw time/frequency phenotype强行解释H1 rescue。

## D. 最终五项回答

1. **H1主要优势来源：** time/location/scale与relative band fractions的联合使用；前者BA和
   AUROC均稳定，后者主要贡献AUROC。优势是跨family/activity/wrist组合，非单族模型。
2. **V8已编码的信息：** wrist与activity-context层已中等编码derivative、time/location/
   scale，并部分编码band fractions、time shape和global spectral statistics。
3. **明显缺失的信息：** 完整activity/wrist-specific family结构在subject vector中不可恢复；
   高价值缺口尤其是time/location/scale与relative band allocation的结构化保留。
4. **H1是否纠正稳定错误：** 仅纠正少数。stable-error appearance纠正率约30%，54名中只有
   14名persistent rescue、32名persistent both-wrong。H1主要修复较靠近边界的错误。
5. **下一版应针对的gap：** 不再寻找更大的通用backbone或独立FFT分支，而应针对“如何让
   activity-conditioned、wrist-structured的幅值/离散度与相对频谱分配在subject decision前
   保持可访问”。本阶段只确定representation target，不设计或训练新模块。

## 产物

- `feature_family_decomposition/summary.csv`与`fold_metrics.csv`
- `analysis/family_paired_inference.csv`
- `analysis/recoverability_45_seed_folds.csv`、`recoverability_15_splits.csv`及summary/inference
- `analysis/h1_v8_validation_records.csv`与subject/error/phenotype/geometry表
- `representation_extract/embeddings/`：45份冻结前向表示
- `analysis/protocol.json`

所有正负结果保留；所有统计推断以15个split为单位，V8三个seed先聚合。
