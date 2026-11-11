# Phase5C：表征漂移缺口

状态：BLOCKED_ADAPTED_REPRESENTATION_MISSING。

Frozen的local、ssl_residual、bilateral、subject、structured包存在；对应adapted post-model特征未找到。既有stage4是pre-adaptation共享中间缓存，不能与Frozen最终特征冒充同层成对表示，也不能将不同fold embedding拼接。按用户限制，本轮不新提取HarNet、不重放adapted forward、不重新训练。

analysis/representation_drift.csv保留45×6=270条model/layer状态行；cosine、L2、Frozen/adapted norm为NA，绝不填0。每行附同输入预处理来源检查状态及可合法计算的预测变化摘要，但预测变化不是embedding drift。Adapted原始双logits也未保存，raw DD logit displacement无法由softmax反推；双方保存概率可计算量化后的log odds，名称明确为probability-implied，不能伪称原始双logits差。

若未来另获推理授权，最小补充是对同一split/seed、同一ID和预处理成对导出既有checkpoint的指定层，保存源hash/IDs/形状/拟合状态，不训练任何网络。之后只做匹配坐标内的描述，cosine/norm本身仍不是疾病因果或性能收益证据。当前不自动执行。
