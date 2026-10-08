# Pre-formal correctness / code freeze

1. 已完成P0审查、extraction/P1/P2及统计正确性、原engine P1/P2各one-batch one-epoch smoke。
2. 文件：PROTOCOL.md；run_condition.py、launch_matrix.py、analyze_results.py、conditional_diagnostics.py、freeze_study.py、test_analysis_contract.py；study_lock.json。
3. 全部PASS。P2第二次backward encoder6.36068e-5、aggregator1.27307e-4，第一步均0；projection两步非零。原引擎best checkpoint重载后输出validation predictions。P1/P2总参数80,340/80,661，三种子原STR参数与RNG一致。
4. 无正式候选训练；仅两步gradient及one-epoch correctness smoke；outer未访问。
5. 原STR recipe全项不变；scratch初始化，不warm-start旧trained模型。
6. 无协议偏离。F0辅助文件名只是P0审计历史命名，不执行Frequency。
7. study_lock SHA：123d260ad9c3dc1d2294432c7a9b12350ba308078d512f331ea3056bd22a25c0。统计/retention/conditional gates均在结果前固定。
8. PROCEED：本报告、源码与protocol推送并核验Git后，执行90正式candidate runs；smoke分数不作为筛选条件。
