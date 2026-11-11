# 独立PD/DD验证与类条件可靠性协议（尚无真实外部结果）

## 目的与不可变模型

本协议在新的真实外部模型结果出现前固定；本轮只做合成测试和准备。未来验证对象仅为既有Frozen WSSL-STR seed42/context0/inner0、训练205人的checkpoint，不是45模型ensemble，不重新训练全390人模型。历史15split平均BA.719644/AUROC.759553不能当作这个实例的外部性能或临床性能目标。

五项实际文件哈希已逐一匹配：

| 资产 | SHA256 |
| --- | --- |
| checkpoint |60b6083bcef9fcf5e7a468c03ed01a383eed4368e0fa261d6a7a3b1a72582d73|
| normalization文件 |ad8e617bce3ea20ec01eddafd6e4f47cdde76fe47c17ea938a903485e5144187|
| canonical config |95a027d075f7aa7c13d3efc49b3d73c3a3b959e3e2eed99f9da9c5bab5d62ca8|
| preprocessing config |1ced7052705192511da4d89d1a066589cede62b658bb852ff5769d20551bbe6c|
| 官方HarNet10 |c64f9135d99e2dcdfc9ae7cc0672f2bcc438df9ceb8215665882f92cddd162a6|

使用原train-only normalization；其数值hash a7bd58abaf6206fd326ac97cc2e8a1ba778f83ef8d82cd678a209717326b2f35。分类PD=0/DD=1，原argmax，pDD>.5判DD，tiePD，Calibrator=NONE。禁止根据外部标签拟合normalization、温度、阈值或分类器。原两logits用于proper NLL，不用训练weighted CE代替。

## 数据硬门槛

数据管理方必须确认研究授权/伦理及同意范围、可发布聚合结果的条件、临床PD与DD诊断定义及确定性、无受试者重叠、11活动语义和双腕六通道兼容。授权不得用网页“Open”、能下载或仓库不同名字代替。与全部390开发参与者、监督/参考fit来源的重叠由受信任管理方安全核查，不公开身份；官方SSL预训练成员无法逐一核对时额外披露残余不确定性。

ET-only或MSA/PSP-only队列可作为预先说明的狭义DD亚型研究，不能冒充PADS完整DD构成。HC永不改标DD，prodromal/SWEDD或其他疾病也不自动改标。临床诊断由独立专家保管，记录随访确认与诊断变更；混合/不确定诊断按预定规则保留描述或不纳入主要PD/DD比较，不看预测后改诊断。

连续入组和固定纳排标准在揭盲前登记；不能按模型错误、距离、概率或任务完成好坏选择最有利患者。主分析用每人首次满足预先质量规则的完整session，时间排序/选择与标签无关；重复session作次要纵向描述并按患者cluster。主流程缺活动/腕/通道不补造，报告所有入组、排除及原因和两类覆盖率；不把无法推理者算正确，不将完整病例结果代表所有临床来访者。

## 无标签工程与盲法

确认源单位、采样率、轴方向、安装位置、时钟/任务分段。确定性通道重命名、m/s²→g（9.80665）、deg/s→rad/s、毫秒→秒、polyphase到100Hz允许；滤波、重采样和不同安装位置仍可能改变输入分布，必须如实说明。手背/手指/腰部与腕部不因轴名相同就等价。当前adapter拒绝未知方向/非腕位、缺失记录、非有限值、不规则时间；不自行学习设备映射、填活动、复制手臂或追加滤波来美化性能。

原预处理：100Hz整段trim48；local六通道Acc减固定L1趋势（lambda50/rho40/max5000、原容差），Gyro不变，原train统计归一化。SSL分支使用同一trim后原Acc，clip±3g；不足1000真实样本按原规则edge-pad，否则首末1000；resample3/10到300，再固定HarNet。活动、左右腕与真实长度对应，batch右侧零padding只是容器，模型按真实长度切片。

未来临床batch固定：无标签稳定身份顺序，每人按活动/腕/首末窗排序，external encoder最多64窗（每人不混排，现协议每人最多44窗），classifier8人并处理末批；eval/no-grad，BN和全部状态不更新。当前合成测试classifier用2以验证2+1末批，不将此改为临床性能选择。此每人确定布局不依赖内部PADS的全队列窗口元数据；会与历史CUDA大batch布局存在浮点路径差异，不能宣称新外部输入逐位复现历史缓存。临床标签揭盲前须确认真实格式parser、时钟/坐标/设备映射与原始文件转换的独立QA，冻结adapter/source/config/hash及逐人预测。合成通过不能替代上述真实数据验收。

标签由独立评价者保管，预测和清单锁定后一次性揭盲。协议和分析实现先锁，不选择最好设备/子组结果。发现代码错误应保存失败结果与修正理由，重新验收并披露，不能改任务或筛样本。旧outer既有接触不能撤回，该旧outer不再作为此研究的独立新队列。

## 分类和概率评价

主要估计对象为这一个锁定模型在预定队列中的subject-level BA及95%CI；不在没有临床共同设定的最低价值/等效界限时虚构“临床成功”。同时报告Accuracy、AUROC、Macro-F1、PD Recall、DD Recall、unweighted NLL、two-class-sum Brier、10等宽top-confidence ECE。报告混淆计数、类别先验、总体/分类别结果和遗漏人数。

ECE用confidence=max(pPD,pDD)与预测正确率差，分箱区间[0,.1),…,[.9,1]；Classwise ECE对每类概率与该类one-vs-rest标签分别用同样10箱加权绝对差，并报两类平均。两类概率可靠性曲线逐箱记录n/平均预测/观察比例，空箱NA；不可只用“真实PD人群内部的pPD正确率”冒充校准。概率恰0/1时proper NLL从logits用log-sum-exp稳定计算，不任意剪概率掩盖极端错误。

分层按PD/DD进行10000次subject bootstrap，seed20261111，报告95%percentile CI；重复记录整组抽取。CI条件于固定权重和此队列抽样，不覆盖模型重新拟合和选择偏差。任一诊断类缺失时AUROC/BA比较NA；类内少于20人时相关CI和比较标NA/不足，点估计仅描述。ECE总N<100明确小样本不稳定；每箱n<10仅描述，不把曲线抖动当发现。Macro-F1类别分母0时NA，不静默设0。所有无效bootstrap draw计数；有效draw少于9500则CI为NA，不能改口径重抽到显著。

设备、医院/采集时间/环境、DD亚型、预定年龄组(<65/≥65)、性别、诊断确定性、药物状态和活动质量均先登记原始定义及unknown组；禁止事后合并最好亚组。活动分析指完整11活动session中独立记录的任务质量/时长关联，不遮掉10活动后把它称为同模型单任务外部性能。设备/环境等与诊断或中心完全混杂时不可作因果归因。

分类指标本身主要报告估计与CI，不对9指标逐项作“胜过开发集”的筛选。临床亚组差异以预先列出的设备、环境、年龄、性别、诊断确定性、DD亚型六个总体异质性检验组成探索BH族m=6；具体水平及可估计的检验算法须在新队列标签揭盲前随数据字典冻结，不能事后挑level。未能预先确定或样本不足的检验保持NA，BH中只为保守校正视p=1；不可另选最优划分。没有实现/运行的检验不宣称完成。

## 类条件覆盖独立复核：有条件执行

H1：PD/DD的距离—错误关系不同。H2：该交互在真正独立临床队列重复。H3：DD异质性、任务质量或参考几何可能解释差异；H3是机制探索，不由H1显著自动成立。

**当前不具备原reference外部评分能力。** 原始query分数存在；reference1024D Scaler、PCA完整状态、变换后bank向量和完整训练reference OOF分数缺失。45个fusion的2维Scaler并不补齐22个activity/wrist的参考空间。不执行重拟合；原方法的外部复核目前BLOCKED。

若未来找回与创建时hash、fit/query隔离匹配的完整旧状态，可按与锁定模型一致的context0/inner0 reference bank使用，不把15bank平均当单模型对象。冻结旧u=min(dPD,dDD)、原活动腕均值、原q25=.2609641859943962/q75=.3100661256539425（以本轮核验CSV为准）。这些cutoff来自过去PADS validation u的分位数，属于已接触的development定义，不是训练OOF阈值；独立队列不得重算其分位数或寻优。无法证明相同几何/尺度则这些数值禁止迁移。

主检验直接交互 I=(PD high−low error)−(DD high−low error)，双侧H0:I=0、预期正方向，以分层subject-cluster bootstrap给效应与CI；若需要p值，使用在标签揭盲前完成合成验收的null-centered bootstrap `2*min(P(I*−Ihat≥Ihat),P(I*−Ihat≤Ihat))`，每个Monte Carlo尾概率用(count+1)/(B+1)，结果截到1，并披露近似性。主要族m=1；四个class×high/low格任一<10人则推断NA，仍报告人数和描述。保持中间u人群在总体模型评价中，不为了交互去掉他们的总体指标。未来单一模型/bank的外部估计对象不同于Phase6的15split平均，只复核预定方向和效应，不要求数值等于.439291。

探索覆盖族m=3：true probability margin交互、absolute margin交互、连续u×class的秩关联差异（明确不同估计对象）；全部用同一subject resampling、固定方法和BH3，无效比较保留NA并以p=1计入调整。不得用“一类显著另一类不显著”代替直接交互。无旧参考状态时上述整个族为NOT_EVALUATED，不用替代方法填数。

若需新reference方法，只能另立新方法协议/版本，先授权，定义训练bank、保存所有状态与fit/query/excluded、训练侧cutoff与完整OOF，再独立验证；不能称其精确复现Phase5或混合新旧几何结果。本轮不登记为已经开展的新方法，不拟合任何Scaler/PCA/gate。

## 样本量与停止条件

目前没有真实候选队列的最终获批人数，不能做假power。仅作预算示例：单类recall最坏p=.5，正态近似95%半宽.10需ceil(1.96²*.25/.10²)=97人/类；这不是充分临床样本量或交互power。四格错误率差中差半宽.20的最坏近似需每格ceil((1.96/.20)²)=97人，high/low共388人；中间区和类/距离不均衡会增加所需总入组，不能拿这当招募保证。具体power/精度设计应由临床目标、固定距离cutoff下可行分布和伦理预算共同决定，不从新测试标签试算后挑显著样本。

授权、身份独立、原始输入语义或任一锁定hash失败即停止对应真实推理；缺reference状态停止距离复核，不阻塞合格数据上的冻结分类验证。无临床合格数据时所有外部指标与交互为null/NOT_EVALUATED。本轮训练/新拟合/新访问申请/数据下载0；完成发布后停止，不自动启动Phase8。
