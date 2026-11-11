# A0 weighted CE 数值审计

结论：AUDIT_COMPLETE / PRECISION_LIMITED；候选 NOT_RUN_GATE_FAIL，新增deep训练0。五项闸门a/c/d通过，b历史明确全局风险目标未确认，e实际TF32参数梯度严格精度门槛未通过。不能把未训练候选写成性能REJECT。

| seed | actual_precision_tolerance_pass | IEEE_FP32_shadow_relative_error | cudnn_allow_tf32 | logit_gradient_scale | parameter_gradient_relative_error | batch_gradient_norm | fixed_gradient_norm | batch_clipped_norm | fixed_clipped_norm | clipped_direction_cosine | optimizer_updates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | True | 0.000000 | True | 0.831459 | 0.000006 | 8.430612 | 7.009707 | 4.999999 | 4.999999 | 1.000000 | 0 |
| 43 | True | 0.000000 | True | 0.702055 | 0.000013 | 9.591042 | 6.733437 | 5.000000 | 4.999999 | 1.000000 | 0 |
| 44 | False | 0.000001 | True | 1.090267 | 0.000064 | 1.570606 | 1.712380 | 1.570606 | 1.712380 | 1.000000 | 0 |


实际PyTorch2.8 hard-index、无smoothing路径：L_batch=Σw_y CE/Σw_y；固定目标L_fixed=Σw_y CE/(B·mean_train(w_y))。同一batch两者梯度相差公共标量Σw_y/(B·mean_train(w_y))，类别间相对权重不变。默认weighted mean正确，不是PyTorch bug；有限自然随机batch的期望比值不必等于固定全局平衡经验风险。逆频率train_balanced本身不能证明历史明确声明了后一目标，更不能证明DD错误由前者导致。[安装版本官方定义](https://docs.pytorch.org/docs/2.8/generated/torch.nn.CrossEntropyLoss.html)。

RandomSampler自然采样，batch8、drop_last=False，seed+原loader offset；45配置各50epoch标签组成shadow，2250记录，不更新optimizer。实际类别比例、末batch、归一化系数和类别总系数见A0_sampler_summary.csv，逐epoch风险见A0_epoch_risk.csv。epoch实现以batch大小加权平均batch-ratio，通常不等于全集weighted-risk，也不是proper unweighted NLL。

Float64手算/自动微分覆盖单PD/DD、B1、balanced、末小batch、未加权/加权，容差1e-12通过；soft-label默认按样本数mean，不能套hard-label分母。真实三seed固定模型/输入的logit梯度、参数梯度及裁剪均记录。seed44实际TF32相对误差6.412217e-5超过预定2e-5；未改容差。只在独立审计进程关闭TF32的IEEE-FP32影子均通过，正式训练设置未变。数学恒等式不等于有限精度逐位恒等。

失败修订：v1暴露TF32参数梯度误差；v2另发现以torch FP32重算类权重与正式Python双精度除法再转FP32在42/45组相差1ULP；v3直接用正式权重，并逐位复核原路径。修订未消除实际TF32失败、未开启训练。早期collection status=PASS仅代表收集完成，以A0_precision_assessment.json及A0_gate.json为最终解释。
