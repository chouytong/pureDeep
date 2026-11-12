# 生产求解器与锁定依据

保持P=.5||y−x||²+50||D2x||1，D2=[1,−2,1]索引差分、完整raw序列、float64。生产为唯一固定的dual log-barrier Newton box-QP，q从零开始，每轴独立μ和Newton收敛/证书，无其他患者数值、warm start或全batch停止。D2D2ᵀ为五对角矩阵，复用的唯一cache只含长度相关只读矩阵。μ递减、初始化、maxNewton600/barrier30/line-search60及fail closed详见configs/solver_production_v1.json。

优先检查了修复独立停止和证书的ADMM参考：低幅例30000迭代仍未过证书，gap=0.00893179928，门槛=3.47606607e-08，将该低幅压力例的每轴成本外推到25740轴约31058.58 CPU秒，超过21600秒；这是压力例外推，既不是全真实队列实测，也不是严格成本下界。该ADMM参考尚未达证书，未以增加maxiter当修复完成。改用同一凸目标的已有SciPy线性代数实现，依据精度/预算选择，任何分类结果前冻结；不是按BA选solver，无依赖安装。

sourceSHA `6219e341c6ff605e73d83171aac5bebb516887323bb73d987c296f3f1da61537`；configSHA `bbb63bb7e6112e26c8fc89f34de85c160f365502fa56c6c267567bc1ead58b8a`。修订前未缓存实现、首轮测试、源码和缓存修订记录保留在logs/audit；未覆盖历史主线。达到上限但证书失败时显式失败，无旧算法/仿射回退。仿射只用于E_aff尺度定义。
