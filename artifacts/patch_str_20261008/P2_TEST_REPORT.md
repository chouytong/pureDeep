# P2 正确性阶段

1. 完成：固定64-channel depthwise k3序列卷积、GELU、masked attention pooling；原 STR 路径不变。
2. 修改：patch_models.py、test_models.py；新增 P2 测试结果与45-checkpoint表。
3. PASS：45 checkpoint零初始化与重载逐值一致；三种子原 STR 参数/RNG 一致；P1/P2 encoder、bottleneck、projection初始化逐值一致；两步梯度；空patch行、缺失wrist/activity和sample/patch padding隔离；order operator非置换不变。P1扩展后复核PASS。
4. 无正式训练，仅两步梯度检查；无outer访问。
5. 原recipe不变；P2总80,661，新增5,137，比P1多321。
6. 无协议偏离。
7. 结果：PASS；正确性不表示性能有效。
8. 下一步：PROCEED，冻结正式protocol及统计实现，完成原引擎smoke后再启动90次训练。
