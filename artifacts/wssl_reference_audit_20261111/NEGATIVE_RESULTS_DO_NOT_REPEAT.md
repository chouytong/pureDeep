# 本轮与历史负结果边界

本轮：固定CE candidate未运行（b/e闸门失败），不是性能负结果；local probe无稳定增量；两分数融合REJECT（BA/F1/DD/BA seed稳定性）；relative不是更佳主模型；temperature只诊断保留，Brier/ECE CI跨0。实际TF32参数梯度seed44未过，不隐藏在collection PASS后。

历史：STR后WSR、prototype loss、strong backbone/GyroFFT、Phase2各16配置、Phase3A、HarNet layer5/multi、output adapter/gate/domain-masked、HarNet5/BioPM受限export、window dynamics、EMA、LR1e-4、frequency/capacity残差、Patch P1/P2均未成为当前retained。Activity scaling是INCONCLUSIVE，不称有害。详见完整ledger与原来源hash。

不得更换名字/rank/宽度/seed/阈值重复以上固定失败配置。也不由单配置拒绝否定PEFT/SSL/Gyro/patch整个家族。Phase3C的135模型OOF阈值诊断已完成，不重跑；PhaseC loss当时未启动，本次补齐但不训练。Full HarNet、Patch P3、cycle预训练、PaAno triplet、XTransfer层搜索均NOT_RUN/UNAUTHORIZED，本轮结束。
