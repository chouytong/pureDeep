# AI-WSSL-STR 独立模型登记修改报告

登记日期：2026-11-16（Asia/Shanghai）。**身份：Experimental Candidate；不是 Retained Model。** STR-01 仍为 Matched Baseline，Frozen WSSL-STR 仍为 Retained Model。本轮新增训练、优化器更新和性能实验均为0。

## 最终修改

新增统一注册API `src.models.registry.build_registered_model`，三个模型按名称并列构建。原 `src.models.build_model` 与 STR/WSSL 工厂/源码保持原字节，用于历史兼容，不改变已有调用路径。新增独立AI架构、3份seed配置、token训练入口、命名加载/推理CLI及48条checkpoint标识目录。原.pt只读；新格式保存显式model_id/status/config，支持历史state严格恢复。

| 内容 | 文件 |
|---|---|
| 模型注册清单 | mainline/models/registry.json |
| 名称调用API | mainline/src/models/registry.py |
| 独立AI架构 | mainline/src/models/ai_wssl_str.py |
| 配置 | mainline/configs/ai_wssl_str_seed42.yaml（及43/44） |
| 独立训练入口 | mainline/scripts/train_ai_wssl_str.py |
| 加载/推理入口 | mainline/scripts/model_cli.py |
| checkpoint身份与来源 | mainline/models/ai_wssl_str_checkpoints.json |
| 架构说明与Mermaid结构图 | mainline/docs/models/AI_WSSL_STR.md |
| 项目模型索引 | conclusions/MODEL_INDEX.md |

ROOT/mainline/conclusions的README已按本轮授权更新，实验报告索引只追加关联项，旧条目保持原字节前缀。修改前这4份活跃文档已快照到本轮private区。历史报告、模型身份、数据、normalization和权重均未改写。完整逐文件变更见analysis/CHANGE_MANIFEST.csv。

## 架构及配置保持

AI核心从Phase8 ActivityInteraction逐字移植，仅改类名，完整AST比对通过。其在Frozen WSSL-STR bilateral token上加入跨活动Self-Attention与residual logit correction，原subject/STR logits保留；新增residual经原classifier权重映射且不重复bias。没有增加独立分类头/gate/SupCon，没有网络、损失或训练超参数搜索。原WSSL始终eval且冻结，新增67522参数可训练；冻结classifier侧143172、外部HarNet10457408，唯一部署总参数10668102。

三份独立AI配置的model/loss/training块均保持已核验数值。独立配置将token workers设为原Phase8B的0；没有复制全网络训练的workers4。继承的STR描述性development/阈值/预处理出处元数据另行纠正为AI现有固定规则，变更前配置快照及三大数值块不变证明见audit/config_metadata_closure.json。这是描述元数据闭合，不改变原训练配方。

训练仍为batch8、AdamW2e−4/WD1e−4/betas.9,.999、train-balanced CE、cosine50min1e−6、clip5、max50/patience12、strictvalBA、原shuffle/seed、AMPfalse。仅执行训练入口--validate-only，未构造优化器。新增完整signal接口只把已归一化信号和显式冻结HarNet送入原编码器，再复用完全相同token读出，不执行L1/normfit/HarNet提取。

## checkpoint与历史证据

45条legacy L1身份关联Phase8旧15×3，来源4745549dabf34ccc89a740ad361a0410c6a93a4d，历史gate结论REJECT保留。3条Certified L1身份关联context0/inner0三seed pilot，来源50a319441e2391ce662646cbd3e690ea64d0e1aa。每条目录项包含seed/split/data_version/checkpoint SHA/base WSSL SHA/normalization SHA/training source SHA及报告。旧数据45次不能充当Certified完整15split复核。

Certified pilot的平均ΔBA=+.004795、ΔAUROC=−.004795、ΔPD Recall=+.117117、ΔDD Recall=−.107527，三个seed的DD Recall均下降超过.01。登记是工程管理动作，不是性能提升认可；没有改正式指针、没有追加其余42次或任何新训练。模型卡中分别关联旧完整结果和新小规模结果，未作跨scope直接比较。

## 实测验证

| seed | 加载/源实现logits | 重复/恢复 | 基础模型不变 | 优化器更新 |
| --- | --- | --- | --- | --- |
| 42 | bitwise PASS | bitwise PASS | PASS | 0 |
| 43 | bitwise PASS | bitwise PASS | PASS | 0 |
| 44 | bitwise PASS | bitwise PASS | PASS | 0 |


独立模型按名称加载3个Certified checkpoint，使用各seed前8名train的已有token（合计24个受试者实例）前向，与原实现逐位相等；全state严格恢复及save/load roundtrip逐位相等。恢复测试副本明确标registration_test_only，不计新训练、不替换原checkpoint。另加载1个legacy checkpoint作合成token前向，full signal/HarNet接口仅用2名合成输入验证padding和token路径一致。没有新增validation性能评估。

mask非有限值/全无效拒绝、错误模型/hash/seed/架构拒绝、原STR/WSSL工厂初始化回归、命名CLI和validate-only CLI均通过。109个必要历史保护文件哈希未变；CURRENT_MODEL.json SHA=`a30c1262c7e37793050a73bfd1c0ba6f8f68c0cc020a21cda1434a0f5f05a820`，保持Frozen WSSL-STR。详见analysis/model_load_forward_restore_tests.json与registration_integrity.json。

## 调用与限制

```python
from src.models.registered_checkpoint import load_registered_checkpoint
model, info = load_registered_checkpoint("AI-WSSL-STR",
    checkpoint_id="AI-WSSL-STR:certified-l1:seed42:context0:inner0", device="cuda")
logits = model.forward_tokens(bilateral_tokens, activity_mask)["logits"]
```

三seed仍是同一development split，base和AI都经validation选epoch，不是独立test/OOF/无监督表示。完整Certified15-split配对验证尚未完成，DD代价未消除。新注册训练代码本轮仅验证输入绑定与固定配方，没有执行优化；不能把登记说成新模型训练完成。完整部署端到端延迟未新增测试。

发布辅助脚本的一次静态语法替换问题在发布前被发现并修正，失败源码和说明存本轮private/audit；未改变模型代码、未重跑训练。最终对全部新增脚本的语法检查通过。

仅公开代码、配置、模型目录、说明和安全测试聚合；权重、逐人输出、输入token和文档/配置快照保留private。提交/发布状态以本轮外部publication receipt及远端精确tree回读为准。
