# Stage 3：Activity-conditioned disease prototype alignment

日期：2026-09-19  
唯一baseline：V8-GN + AdamW + LR=2e-4 + batch=8 + WD=1e-4 + cosine  
范围：inner-development only；seed 42/43/44；15个matched inner splits  
outer-test：未访问

## 1. 假设

当前activity embeddings同时含疾病、activity和subject-specific information。实验检验：
按activity和疾病建立train-fold prototypes，压缩同activity、同疾病的受试者表示，是否
能在保留PD/DD判别力的同时降低subject信息或unseen-subject shift。

## 2. 实现

- V8-GN架构、71,026个推理参数和frozen recipe完全不变。
- 每个epoch开始，用当前inner-train fold的全部训练受试者计算11×2个
  activity×disease prototypes；使用独立、非shuffle loader，validation不参与。
- 对Activity Attention聚合前的258维enriched activity embeddings进行L2归一化。
- alignment由正确疾病prototype的cosine compactness，加上正负疾病prototype之间
  margin=0.2的ranking separation组成。
- 训练总损失：`CE + λ × (compactness + separation)`；validation只计算CE。
- λ在训练前限定为0.01和0.05，不追加候选。
- 三seed使用相同15个subject splits。比较时先在同一split内聚合三个seed，再以15个
  matched splits报告均值、胜负和Wilcoxon配对检验。

## 3. 分类结果

| 模型 | Accuracy | BA | Macro-F1 | AUROC | PD recall | DD recall |
|---|---:|---:|---:|---:|---:|---:|
| V8-GN baseline | 0.7154±0.0177 | 0.6707±0.0088 | 0.6631±0.0137 | 0.6800±0.0103 | 0.7782±0.0303 | 0.5631±0.0134 |
| PA-01 λ=.01 | 0.7057±0.0174 | 0.6692±0.0092 | 0.6574±0.0137 | 0.6783±0.0071 | 0.7570±0.0288 | 0.5814±0.0104 |
| PA-02 λ=.05 | 0.7170±0.0151 | 0.6715±0.0040 | 0.6641±0.0088 | 0.6789±0.0072 | 0.7811±0.0352 | 0.5618±0.0356 |

PA-01相对baseline：BA -0.0015、AUROC -0.0017、Macro-F1 -0.0057、Accuracy
-0.0097。PA-02：BA +0.0008（8/15 splits提高，Wilcoxon p=.679）、AUROC -0.0011
（8/15提高，p=.978）、Macro-F1 +0.0010。PA-02分类能力基本持平，但不构成性能改善。

## 4. 表示诊断

| 指标 | Baseline | PA-01 | PA-02 | PA-02 Δ |
|---|---:|---:|---:|---:|
| Disease probe BA | 0.6126 | 0.6133 | 0.6207 | +0.0082 |
| Disease probe AUROC | 0.6599 | 0.6621 | 0.6692 | +0.0093 |
| Disease cosine silhouette | 0.0360 | 0.0344 | 0.0359 | -0.0000 |
| Subject retrieval Top-1 | 0.0632 | 0.0635 | 0.0645 | +0.0013 |
| Subject retrieval Top-5 | 0.1792 | 0.1791 | 0.1785 | -0.0006 |
| Same-subject similarity | 0.1752 | 0.1798 | 0.1782 | +0.0030 |
| Subject similarity gap | 0.1509 | 0.1545 | 0.1527 | +0.0018 |
| Disease-controlled domain AUC | 0.6334 | 0.6194 | 0.6255 | -0.0079 |
| Normalized mean shift | 0.1143 | 0.1135 | 0.1116 | -0.0027 |
| CORAL covariance shift | 0.4727 | 0.4638 | 0.4732 | +0.0005 |

PA-02 disease probe AUROC在12/15 splits提高，配对p=.0125，说明疾病类别原型确实让
线性疾病信息更易读出。但subject retrieval Top-1没有下降；same-subject similarity在
12/15 splits上升。它没有实现预期的subject-information suppression。

PA-02 domain AUC在12/15 splits下降，平均-0.0079，但双侧配对p=.073；mean shift只在
9/15 splits下降（p=.421），CORAL不降。因此shift减少仅是单一诊断上的趋势，尚不能
称为跨指标稳定改善。PA-01的domain AUC、mean shift和CORAL均值虽下降，但都只有
9/15 splits下降，p分别为.252/.421/.303，同样不稳定。

## 5. 结论

两个候选均拒绝，不替换baseline：

- PA-01同时损失Accuracy、Macro-F1和PD recall，且subject信息未下降。
- PA-02保持分类性能并提高disease probe，但没有降低subject-specific information；
  domain AUC的下降趋势没有得到mean shift和CORAL支持。
- 因此，“压缩activity×disease类中心自然产生跨个体不变性”这一假设未被实验支持。
- 不继续增加λ，也不叠加DANN、MMD或其他模块补救。唯一baseline仍为冻结V8-GN。

## 6. 已确认与尚未确认

**实验确认：** activity-conditioned prototypes可提高线性疾病可读出性；λ=.05没有明显
损伤最终分类；但subject身份相关相似性没有降低。

**方向性证据：** λ=.05可能降低disease-controlled train/validation domain AUC。

**未确认：** prototype alignment可改善真正unseen-subject泛化；domain AUC趋势是否
可复现；个体信息具体来自年龄、性别、动作质量、病程或其他因素。

## 7. 产物

- 六个完整训练目录：PA-01/PA-02 × seed 42/43/44。
- 两套45-fold representation diagnostics及15-split聚合统计。
- `prototype_alignment_comparison_20260919.csv`：baseline、PA-01、PA-02统一比较。
- 修改后的loss、prototype统计代码、配置、测试和完整provenance均随artifact保存。
