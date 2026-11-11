# 冻结模型部署审计

实际读取当前文件而非按旧对话推定状态。Git mirror HEAD及刚fetch的origin/main均为6de6480fce87920a899181fa7c8586c01ce616f9，worktree干净；主线45个文件逐项与该已发布快照匹配。主线Python全文读取/AST解析和源文件哈希见mainline_source_audit.csv，关键forward/数据预处理另行审阅。启动前重读19个必需历史/入口文档，MISSING0；全文由审计工具扫描，数值表保留原文件哈希，提供narrative审阅，outer相关数值3行不对研究结果重新展示。此阅读控制不抹除历史接触。

| asset | expected_sha256 | actual_sha256 | status |
| --- | --- | --- | --- |
| checkpoint | 60b6083bcef9fcf5e7a468c03ed01a383eed4368e0fa261d6a7a3b1a72582d73 | 60b6083bcef9fcf5e7a468c03ed01a383eed4368e0fa261d6a7a3b1a72582d73 | PASS |
| normalization_file | ad8e617bce3ea20ec01eddafd6e4f47cdde76fe47c17ea938a903485e5144187 | ad8e617bce3ea20ec01eddafd6e4f47cdde76fe47c17ea938a903485e5144187 | PASS |
| canonical_config | 95a027d075f7aa7c13d3efc49b3d73c3a3b959e3e2eed99f9da9c5bab5d62ca8 | 95a027d075f7aa7c13d3efc49b3d73c3a3b959e3e2eed99f9da9c5bab5d62ca8 | PASS |
| preprocessing_config | 1ced7052705192511da4d89d1a066589cede62b658bb852ff5769d20551bbe6c | 1ced7052705192511da4d89d1a066589cede62b658bb852ff5769d20551bbe6c | PASS |
| official_HarNet10_checkpoint | c64f9135d99e2dcdfc9ae7cc0672f2bcc438df9ceb8215665882f92cddd162a6 | c64f9135d99e2dcdfc9ae7cc0672f2bcc438df9ceb8215665882f92cddd162a6 | PASS |


Norm文件、checkpoint内部mean/std逐值一致；norm fit IDs与原split的205名train完全一致，query/excluded保持分离，精确身份私有保存。配置11活动/2腕/6通道固定；官方HarNet loader所有131 feature-extractor键和权重与原checkpoint逐项一致，官方随机classifier不用于特征。显式IndependentWSSL strict load原143172参数分类侧，HarNet10457408参数固定，不创建45模型ensemble或新全390人模型。

合成推理所有参数requires_grad=false、inference_mode、eval；全部state（包括BN）前后fingerprint一致。原两logits、原argmax、pDD>.5/tiePD、CalibratorNONE。没有使用mainline中会加载全局cache的FrozenEmbeddingCache，也没有触发任何训练入口。源码相同/哈希锁定和合成正确性不等于这一个checkpoint已经取得独立临床验证。

当前阶段只测试固定内存接口。实际外部文件解析、坐标/时钟/任务定义核验、授权与身份独立、临床评价runner的真实数据验收仍待完成。技术上能生成合成logits不等于可以给任何PD/HC或少任务数据报告PD/DD性能。历史outer性能文字接触、旧manifest整表/旧NPZ整数组读取与Phase6公开搜索结果的个体元数据摘要暴露永久保留。本轮宽检索再次意外返回公开PADS个体元数据摘要，未主动打开、保存个体行或用于实验。本轮未读取旧outer性能；历史文档逐行扫描时屏蔽3行可能涉及outer数值的段落，不用其作性能证据。保护哈希读取完整文件字节、旧私有分数表和fit/query身份元数据仍属应用层审计，不能称OS物理隔离或从未接触个体信息。15重叠development splits、seed和患者cluster区间不构成独立临床验证。
