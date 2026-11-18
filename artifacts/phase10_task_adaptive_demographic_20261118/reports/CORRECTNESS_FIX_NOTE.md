# Pre-training test correction

首次来源测试把记录状态名称误设为COMPLETE；原生产manifest8580行实际状态均为PASS。该schema断言在任何本轮临床训练前失败，已改为精确匹配原始PASS，未改数据、证书、模型或训练配方。首版测试源码与失败日志保留在服务器私有/日志目录。第二次测试全量信号hash、BN/梯度/原模型初始化/保存恢复均通过，synthetic optimizer updates=2，clinical training=0。后续又单独验证当前fit/query每条信号hash和负IO拒绝。
