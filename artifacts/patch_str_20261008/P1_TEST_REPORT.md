# P1 正确性阶段

1. 已完成：共享 patch encoder、masked mean 和零初始化 wrist residual；未改变原 STR 前向路径。
2. 新增文件：patch_encoder.py、patch_models.py、test_models.py；结果 p1_tests.json 和 45-checkpoint 检查表。
3. 正确性：45 checkpoint 的零初始化 logits 与重载逐值一致；种子42/43/44原参数、CPU/CUDA RNG 一致；两步梯度、缺失 wrist/activity、padded samples/patches 隔离通过。第一步 encoder 梯度0，第二步0.0001919089；projection 两步均非零。
4. 无正式训练，仅两步正确性检查；无 outer 访问。
5. recipe 无更改；P1 参数80,340（新增4,816）。
6. 无协议偏离。
7. 结果：PASS；这些测试不是性能证据。
8. 下一步：PROCEED，仅实现并测试 P2。
