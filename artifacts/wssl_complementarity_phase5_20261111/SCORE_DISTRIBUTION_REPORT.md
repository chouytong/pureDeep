# Phase5A：分数分布和排序

| label | score | mean | sd | q05 | q25 | q50 | q75 | q95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DD | baseline_pdd | 0.577047 | 0.286580 | 0.138221 | 0.330626 | 0.616068 | 0.828722 | 0.928723 |
| DD | d_dd | 0.303034 | 0.039995 | 0.242681 | 0.277533 | 0.301089 | 0.329177 | 0.365712 |
| DD | d_pd | 0.320826 | 0.057685 | 0.240450 | 0.284442 | 0.315549 | 0.353505 | 0.416786 |
| DD | dd_aligned | -0.303034 | 0.039995 | -0.365712 | -0.329177 | -0.301089 | -0.277533 | -0.242681 |
| DD | relative | 0.017792 | 0.036118 | -0.029985 | -0.006210 | 0.013802 | 0.038635 | 0.078192 |
| DD | wssl_logit | 0.417361 | 1.939862 | -2.622225 | -0.911614 | 0.487397 | 1.753769 | 3.278683 |
| PD | baseline_pdd | 0.306748 | 0.236216 | 0.067678 | 0.127709 | 0.227620 | 0.426527 | 0.787184 |
| PD | d_dd | 0.303638 | 0.037976 | 0.248883 | 0.275431 | 0.300744 | 0.325876 | 0.369042 |
| PD | d_pd | 0.290269 | 0.045630 | 0.229357 | 0.257082 | 0.283858 | 0.315102 | 0.375778 |
| PD | dd_aligned | -0.303638 | 0.037976 | -0.369042 | -0.325876 | -0.300744 | -0.275431 | -0.248883 |
| PD | relative | -0.013369 | 0.033609 | -0.064517 | -0.035700 | -0.014696 | 0.006477 | 0.044494 |
| PD | wssl_logit | -1.410561 | 1.688665 | -3.883085 | -2.610284 | -1.494817 | -0.331225 | 1.474285 |
| all | baseline_pdd | 0.385742 | 0.281212 | 0.072560 | 0.148460 | 0.293308 | 0.607191 | 0.890503 |
| all | d_dd | 0.303458 | 0.038533 | 0.246275 | 0.276012 | 0.300475 | 0.327333 | 0.369094 |
| all | d_pd | 0.299194 | 0.051402 | 0.230552 | 0.261731 | 0.293515 | 0.326965 | 0.393490 |
| all | dd_aligned | -0.303458 | 0.038533 | -0.369094 | -0.327333 | -0.300475 | -0.276012 | -0.246275 |
| all | relative | -0.004264 | 0.037224 | -0.058779 | -0.029008 | -0.007216 | 0.017307 | 0.060452 |
| all | wssl_logit | -0.876249 | 1.957364 | -3.767308 | -2.320997 | -1.033858 | 0.453742 | 2.459960 |


核心score Spearman（先split内seed均值，再15split）：

| label | left | right | correlation |
| --- | --- | --- | --- |
| DD | d_pd | dd_aligned | -0.718442 |
| DD | d_pd | relative | 0.690422 |
| DD | dd_aligned | relative | -0.105619 |
| DD | wssl_logit | d_pd | 0.503598 |
| DD | wssl_logit | dd_aligned | -0.164558 |
| DD | wssl_logit | relative | 0.611911 |
| PD | d_pd | dd_aligned | -0.660745 |
| PD | d_pd | relative | 0.522475 |
| PD | dd_aligned | relative | 0.209987 |
| PD | wssl_logit | d_pd | 0.269452 |
| PD | wssl_logit | dd_aligned | 0.191685 |
| PD | wssl_logit | relative | 0.599875 |
| all | d_pd | dd_aligned | -0.660649 |
| all | d_pd | relative | 0.611330 |
| all | dd_aligned | relative | 0.110272 |
| all | wssl_logit | d_pd | 0.405635 |
| all | wssl_logit | dd_aligned | 0.084010 |
| all | wssl_logit | relative | 0.664238 |


WSSL融合输入为真实保存的 z_DD−z_PD，绝非raw z_DD；raw DD logit单独统计。PD-reference distance dPD为DD方向，DD-reference原距离dDD方向相反，相关分析用−dDD。relative=dPD−dDD，固定界0。分位数表是各unit分位数的seed-first平均，不是把重复appearance当独立人拟合总体分布。不同context的完整同口径结果见score_context_summary.csv；未跨监督encoder坐标拼embedding。

WSSL与relative总体Spearman .664238，PD .599875/DD .611911；相关但不相同不等于有独立增量。DD内dPD与−dDD高度反向（−.718442），两个单参考不是互相独立证据源。36项相关检验属于A_correlations探索族，全部p/q保留，不能选最有利一项论创新。

OOF/full均值和SD：

| score | label | OOF_mean | OOF_sd | full_validation_mean | full_validation_sd | standardized_mean_shift |
| --- | --- | --- | --- | --- | --- | --- |
| relative | all | -0.006932 | 0.044909 | -0.004264 | 0.037224 | 0.061573 |
| wssl_logit | DD | 0.060103 | 1.616457 | 0.417361 | 1.939862 | 0.228543 |
| wssl_logit | PD | -1.122893 | 1.433827 | -1.410561 | 1.688665 | -0.235292 |
| wssl_logit | all | -0.777134 | 1.587501 | -0.876249 | 1.957364 | -0.088738 |


WSSL OOF/DD均值.060103、full validation .417361；PD为−1.122893→−1.410561。总体尺度变化与类条件位置变化存在，但并非所有context同向。reference只有OOF总体均值/SD，类条件与联合分布缺失，不补做Scaler/PCA或距离重算。same_subject_OOF_full_train_shift另比较同一训练人的OOF/full in-sample分数，只用于区分数据角色，不能当泛化验证。


本轮属于同一390人、15个重叠development splits的重复开发。正式指标先同split平均三个seed的指标；三seed概率均值仅用于错误分组。split bootstrap、seed SD、subject-cluster CI均不能充当独立外部泛化证据。cluster bootstrap按PD276/DD114分层，10,000次，同一人的全部appearance/seed共享采样次数，仅固定已拟合模型条件下的不确定性；不覆盖重训练和多轮开发选择偏差。不同条件/估计对象的CI是否跨0，不构成它们之间差异的检验。
