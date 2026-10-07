# Phase B 正式前测试 / 协议冻结

45checkpoint×2个真实8subject validation batches（90batch）g=1 baseline/reload logits逐位一致；正式概率≤1e-12一致。三seed scratch原参数/RNG相同；原STR及projection更新、11g有限非零梯度、activity/wrist/padding/index隔离、共享双腕scalar、实例隔离和model/optimizer reload数值与dtype一致PASS。原引擎oneepoch/onebatch smoke PASS，分数未选择设计。只在独立临时模型执行2step gradient test及smoke，原正式资产不改。

附加11 unconstrained scalar、uniform1、同原AdamW组WD，143172→143183trainable；HarNet10457408冻结。正式与既有45ordinary reproduction全部状态、metric histories（不含耗时）、最佳epoch、预测、norm/source provenance核对相同，复用reference不重复训练。

PROTOCOL.md及实现/统计/保留规则于正式结果前SHA冻结。下一步只执行45scratch candidate，不同时启动loss或其他candidate。outer/threshold/HarNet/recipe/search均NO。

测试修正记录：初次比较误含duration_seconds；optimizer step重载存在合法CPU/GPU设备位置差异，改为转CPU精确比较dtype及数值。两者是审查工具修正，正式前完整测试重跑通过，未修改候选forward/recipe或查看正式候选性能。
