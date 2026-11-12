# Certified L1 Activity Token Interaction 三seed小规模复核

**CORRECTNESS=PASS；PILOT_STATUS=COMPLETED；PERFORMANCE_PATTERN=PD/DD权衡仍存在。**

实际完成context0/inner0 ×42/43/44共3个新增模块run；没有其余42次，没有WSSL/L1/norm/HarNet重训重算。

ΔBA均值=0.004795，其中PD Recall贡献=.5ΔPD=0.058559，DD贡献=.5ΔDD=-0.053763。DD平均变化=-0.107527；超过.01下降的seed为[42, 43, 44]。AUROC平均变化=-0.004795，MacroF1=0.023579。各seed方向见配对表，不能只挑最佳seed。

| seed | condition | accuracy | ba | auroc | macro_f1 | pd_recall | dd_recall | nll | brier | ece | PD_as_PD | PD_as_DD | DD_as_PD | DD_as_DD | best_epoch | training_epochs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | WSSL-CERTIFIED-L1 | 0.685714 | 0.673932 | 0.704882 | 0.653535 | 0.702703 | 0.645161 | 0.928165 | 0.503116 | 0.192052 | 52 | 22 | 11 | 20 | 15 | 27.000000 |
| 42 | AI-WSSL-CERTIFIED-L1 | 0.723810 | 0.682214 | 0.698779 | 0.676923 | 0.783784 | 0.580645 | 1.572124 | 0.502447 | 0.220915 | 58 | 16 | 13 | 18 | 17 | 29.000000 |
| 43 | WSSL-CERTIFIED-L1 | 0.733333 | 0.688971 | 0.718396 | 0.685360 | 0.797297 | 0.580645 | 0.579838 | 0.392787 | 0.062079 | 59 | 15 | 13 | 18 | 5 | 17.000000 |
| 43 | AI-WSSL-CERTIFIED-L1 | 0.780952 | 0.713383 | 0.726678 | 0.723082 | 0.878378 | 0.548387 | 0.559058 | 0.370146 | 0.107735 | 65 | 9 | 14 | 17 | 11 | 23.000000 |
| 44 | WSSL-CERTIFIED-L1 | 0.666667 | 0.651046 | 0.712293 | 0.632537 | 0.689189 | 0.612903 | 0.803465 | 0.501211 | 0.197322 | 51 | 23 | 12 | 19 | 14 | 26.000000 |
| 44 | AI-WSSL-CERTIFIED-L1 | 0.733333 | 0.632738 | 0.695728 | 0.642162 | 0.878378 | 0.387097 | 1.266643 | 0.512210 | 0.274295 | 65 | 9 | 19 | 12 | 9 | 21.000000 |


| seed | context | inner | accuracy | ba | auroc | macro_f1 | pd_recall | dd_recall | nll | brier | ece | DD_recall_drop_gt01 | BA_delta_PD_component | BA_delta_DD_component |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 0 | 0 | 0.038095 | 0.008282 | -0.006103 | 0.023388 | 0.081081 | -0.064516 | 0.643959 | -0.000669 | 0.028862 | True | 0.040541 | -0.032258 |
| 43 | 0 | 0 | 0.047619 | 0.024412 | 0.008282 | 0.037723 | 0.081081 | -0.032258 | -0.020780 | -0.022641 | 0.045656 | True | 0.040541 | -0.016129 |
| 44 | 0 | 0 | 0.066667 | -0.018309 | -0.016565 | 0.009625 | 0.189189 | -0.225806 | 0.463177 | 0.010998 | 0.076973 | True | 0.094595 | -0.112903 |


| condition | metric | mean | sample_SD | seeds | independent_patient_cohorts | better_direction |
| --- | --- | --- | --- | --- | --- | --- |
| AI-WSSL-CERTIFIED-L1 | accuracy | 0.746032 | 0.030615 | 3 | 1 | higher |
| AI-WSSL-CERTIFIED-L1 | ba | 0.676112 | 0.040667 | 3 | 1 | higher |
| AI-WSSL-CERTIFIED-L1 | auroc | 0.707062 | 0.017057 | 3 | 1 | higher |
| AI-WSSL-CERTIFIED-L1 | macro_f1 | 0.680722 | 0.040594 | 3 | 1 | higher |
| AI-WSSL-CERTIFIED-L1 | pd_recall | 0.846847 | 0.054614 | 3 | 1 | higher |
| AI-WSSL-CERTIFIED-L1 | dd_recall | 0.505376 | 0.103695 | 3 | 1 | higher |
| AI-WSSL-CERTIFIED-L1 | nll | 1.132608 | 0.519663 | 3 | 1 | lower |
| AI-WSSL-CERTIFIED-L1 | brier | 0.461601 | 0.079353 | 3 | 1 | lower |
| AI-WSSL-CERTIFIED-L1 | ece | 0.200982 | 0.085050 | 3 | 1 | lower |
| WSSL-CERTIFIED-L1 | accuracy | 0.695238 | 0.034339 | 3 | 1 | higher |
| WSSL-CERTIFIED-L1 | ba | 0.671316 | 0.019097 | 3 | 1 | higher |
| WSSL-CERTIFIED-L1 | auroc | 0.711857 | 0.006767 | 3 | 1 | higher |
| WSSL-CERTIFIED-L1 | macro_f1 | 0.657144 | 0.026596 | 3 | 1 | higher |
| WSSL-CERTIFIED-L1 | pd_recall | 0.729730 | 0.058904 | 3 | 1 | higher |
| WSSL-CERTIFIED-L1 | dd_recall | 0.612903 | 0.032258 | 3 | 1 | higher |
| WSSL-CERTIFIED-L1 | nll | 0.770490 | 0.176490 | 3 | 1 | lower |
| WSSL-CERTIFIED-L1 | brier | 0.465705 | 0.063156 | 3 | 1 | lower |
| WSSL-CERTIFIED-L1 | ece | 0.150485 | 0.076606 | 3 | 1 | lower |
| AI-minus-baseline | accuracy | 0.050794 | 0.014548 | 3 | 1 | higher |
| AI-minus-baseline | ba | 0.004795 | 0.021573 | 3 | 1 | higher |
| AI-minus-baseline | auroc | -0.004795 | 0.012475 | 3 | 1 | higher |
| AI-minus-baseline | macro_f1 | 0.023579 | 0.014050 | 3 | 1 | higher |
| AI-minus-baseline | pd_recall | 0.117117 | 0.062416 | 3 | 1 | higher |
| AI-minus-baseline | dd_recall | -0.107527 | 0.103695 | 3 | 1 | higher |
| AI-minus-baseline | nll | 0.362119 | 0.343699 | 3 | 1 | lower |
| AI-minus-baseline | brier | -0.004104 | 0.017081 | 3 | 1 | lower |
| AI-minus-baseline | ece | 0.050497 | 0.024418 | 3 | 1 | lower |


DD p>.5、tie PD；AUROC为两概率列macro；NLL为unweighted raw-logit proper log loss；Brier是两列平方和；ECE为10等宽topconfidence。双方统一原Phase8B的float64 softmax；归档原概率float32差异单列baseline_metric_precision_comparison.csv，实际导出归档logits与float32概率均逐位一致。

| metric | paired_seed_mean_delta | ci_low | ci_high | draws | finite_draws | unique_validation_subjects | PD_unique | DD_unique | scope | shared_weights_all_seeds_models |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| accuracy | 0.050794 | 0.012698 | 0.088889 | 10000 | 10000 | 105 | 74 | 31 | fixed trained models and this selected development split; descriptive sensitivity only | True |
| ba | 0.004795 | -0.038724 | 0.047443 | 10000 | 10000 | 105 | 74 | 31 | fixed trained models and this selected development split; descriptive sensitivity only | True |
| auroc | -0.004795 | -0.019762 | 0.011189 | 10000 | 10000 | 105 | 74 | 31 | fixed trained models and this selected development split; descriptive sensitivity only | True |
| macro_f1 | 0.023579 | -0.022998 | 0.067043 | 10000 | 10000 | 105 | 74 | 31 | fixed trained models and this selected development split; descriptive sensitivity only | True |
| pd_recall | 0.117117 | 0.076577 | 0.162162 | 10000 | 10000 | 105 | 74 | 31 | fixed trained models and this selected development split; descriptive sensitivity only | True |
| dd_recall | -0.107527 | -0.182796 | -0.032258 | 10000 | 10000 | 105 | 74 | 31 | fixed trained models and this selected development split; descriptive sensitivity only | True |
| nll | 0.362119 | 0.208399 | 0.518475 | 10000 | 10000 | 105 | 74 | 31 | fixed trained models and this selected development split; descriptive sensitivity only | True |
| brier | -0.004104 | -0.045601 | 0.035217 | 10000 | 10000 | 105 | 74 | 31 | fixed trained models and this selected development split; descriptive sensitivity only | True |
| ece | 0.050497 | -0.001716 | 0.077909 | 10000 | 10000 | 105 | 74 | 31 | fixed trained models and this selected development split; descriptive sensitivity only | True |


## 是否申请完整复核

研究判断：当前结果不足以支持直接申请相同固定方案的完整15×3；应先解释混合方向或DD代价。若后续另获批准，使用原15split且本3run在来源完全一致时可复用。 这不是正式RETAIN结论或临床泛化证据。

## 来源、成本与边界

直接import冻结Phase8 ActivityInteraction原类；仅67522新增参数训练，143172原分类侧参数冻结/始终eval，HarNet10457408冻结。唯一部署总参数10668102。每seed分别重新导出Certified WSSL bilateral tokens，原归档及零残差逐位复现。完整来源、测试和冻结哈希见CORRECTNESS_AND_PROVENANCE。

| seed | context | inner | best_epoch | training_epochs | optimizer_steps | training_validation_wall_seconds | total_classifier_parameters | new_trainable_parameters | original_frozen_classifier_parameters | frozen_HarNet_parameters | unique_deployment_parameters | base_state_unchanged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 0 | 0 | 17 | 29 | 754 | 4.048085 | 210694 | 67522 | 143172 | 10457408 | 10668102 | True |
| 43 | 0 | 0 | 11 | 23 | 598 | 2.988992 | 210694 | 67522 | 143172 | 10457408 | 10668102 | True |
| 44 | 0 | 0 | 9 | 21 | 546 | 2.702103 | 210694 | 67522 | 143172 | 10457408 | 10668102 | True |


| seed | condition | batch_size | warmup | repeats | batch_ms | amortized_ms_per_subject | subjects_per_second | peak_GPU_allocated_bytes | train_only | resident_tokens_float32 | not_end_to_end | not_standalone_patient_latency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | baseline_readout | 8 | 3 | 10 | 0.438345 | 0.054793 | 18250.447749 | 35761664 | True | True | True | True |
| 42 | baseline_plus_interaction | 8 | 3 | 10 | 0.991824 | 0.123978 | 8065.950296 | 35906560 | True | True | True | True |
| 43 | baseline_readout | 8 | 3 | 10 | 0.424673 | 0.053084 | 18838.030286 | 35761664 | True | True | True | True |
| 43 | baseline_plus_interaction | 8 | 3 | 10 | 0.996313 | 0.124539 | 8029.602436 | 35906560 | True | True | True | True |
| 44 | baseline_readout | 8 | 3 | 10 | 0.424711 | 0.053089 | 18836.344888 | 35761664 | True | True | True | True |
| 44 | baseline_plus_interaction | 8 | 3 | 10 | 0.989466 | 0.123683 | 8085.168621 | 35906560 | True | True | True | True |


计时是resident token→logits，排除L1、norm、原编码、HarNet与IO；batch ms和batch/8摊销明确，不是单患者独立端到端延迟。

固定context0/inner0，205名train、105名validation；3seed不是3个独立患者队列。基础WSSL及新增模块都在重复development框架用validation选epoch；train监督token为in-sample，不是无监督或end-to-end OOF。患者bootstrap条件于固定模型与本split，不能消除选checkpoint和多轮开发偏差。每seed配对的是同seed基础模型，三个实例同时含base和module seed变化。无seed显著性检验、无三模型ensemble、无10/15正式保留门槛。

旧Phase8旧L1的15split汇总只作背景，不是本轮单split直接对照，不据本轮均值声称超越它。当前正式模型指针及旧产物保持。发布状态以外部publication receipt与远端精确tree回读为准，完成后停止。
