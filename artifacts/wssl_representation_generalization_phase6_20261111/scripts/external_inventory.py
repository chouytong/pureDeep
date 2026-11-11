"""Persist the primary-source metadata review and deterministic validation model lock."""
from common import *

def main():
 guard();sources=[
 dict(dataset='PADS1.0.0',source_url='https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/',access='open subject to CC BY-NC-SA4.0; already authorized existing server cohort',overlap='same source as current390 PD/DD subjects, not independent',acquisition='Muenster2018-2021;11 active tasks',sensor='bilateral Apple Watch Series4; Acc g/Gyro rad/s;100Hz',activities='same11 tasks',labels='clinical PD/DD/HC; HC is not DD',DD_subtypes='ET/atypical/MS/other',compatibility='input compatible but cohort already used',missing='no independent patients established',ready_for_external_validation=False),
 dict(dataset='mPower',source_url='https://www.synapse.org/Synapse:syn4993293/wiki/247860',access='controlled research access; current individual grant/use conditions not verified or accepted',overlap='separate study design; person-level overlap not verified',acquisition='smartphone active tasks/longitudinal study',sensor='handheld iOS motion sensors, touchscreen and other modalities',activities='walking/standing/tapping etc; not the fixed bilateral11task protocol',labels='PD/control; clinical DD case definition not established',DD_subtypes='not provided for current task',compatibility='not direct PD/DD dual-wrist11task test',missing='authorization; exact device/protocol mapping; DD; overlap review',ready_for_external_validation=False),
 dict(dataset='BEAT-PD CIS-PD/REAL-PD',source_url='https://www.synapse.org/Synapse:syn20825169/wiki/600903',access='research challenge portal; current access terms/grant not accepted or verified',overlap='separate studies; person-level overlap not verified',acquisition='free-living symptom/medication reports',sensor='wrist/mobile Acc/Gyro; devices differ by component study',activities='passive free-living; not fixed11tasks',labels='PD symptom severity/medication; no compatible DD group verified',DD_subtypes='not applicable/not supplied',compatibility='supports other severity questions, not direct PD/DD test',missing='DD and task/device mapping; access approval; overlap',ready_for_external_validation=False),
 dict(dataset='WATCH-PD',source_url='https://www.nature.com/articles/s41531-024-00721-2',access='3DT Stage2 members; others may submit proposal to steering committee; no user approval obtained',overlap='independent multicenter recruitment described; individual overlap not verified',acquisition='early untreated PD/control longitudinal study',sensor='Apple Watch4/5, iPhone10/11 and in-clinic Opal wearable sensors',activities='clinical/at-home gait,tremor,psychomotor etc;11task mapping unverified',labels='PD/age-matched controls, not demonstrated clinical DD',DD_subtypes='not supplied',compatibility='no direct eligible PD/DD input protocol established',missing='data grant; DD; exact task mapping; overlap review',ready_for_external_validation=False),
 dict(dataset='new hospital cohort (not provided)',source_url='unknown',access='unknown; no authorization supplied',overlap='must be verified by source',acquisition='unknown',sensor='unknown',activities='unknown',labels='unknown',DD_subtypes='unknown',compatibility='NOT_READY',missing='all source/permission/protocol fields',ready_for_external_validation=False),
 dict(dataset='independent time/device cohort (not provided)',source_url='unknown',access='unknown; no authorization supplied',overlap='new time/device alone does not establish new participants',acquisition='unknown',sensor='unknown',activities='unknown',labels='unknown',DD_subtypes='unknown',compatibility='NOT_READY',missing='all source/permission/protocol fields',ready_for_external_validation=False)]
 pd.DataFrame(sources).to_csv(HERE/'analysis/external_data_inventory.csv',index=False)
 p=stage('frozen',42,0,0);sp=load(p/'split.json');n=load(p/'normalization.json')
 write_json(HERE/'analysis/external_validation_lock.json',dict(status='MODEL_AND_DESIGN_LOCKED_NO_ELIGIBLE_EXTERNAL_DATA',model='Frozen WSSL-STR',selector='lexicographic seed42/context0/inner0; no validation performance used',checkpoint_sha256=sha(p/'checkpoints/best.pt'),checkpoint_path=str(p/'checkpoints/best.pt'),training_subject_count=len(sp['train_subjects']),normalization_file_sha256=sha(p/'normalization.json'),normalization_values_sha256=n['normalization_sha256'],canonical_config_sha256=sha(MAIN/'configs/str01_seed42.yaml'),preprocessing_config_sha256=sha(ROOT/'data/processed/pads_multi_activity/v2_l1_full_length/processing_config.json'),official_HarNet10_checkpoint_sha256=sha(OFFICIAL),classifier_trainable_parameter_count=143172,external_encoder_parameter_count=10457408,activities=ACTS,wrist_order=['left','right'],channels=['AccX','AccY','AccZ','GyroX','GyroY','GyroZ'],threshold=.5,tie='PD',class_encoding={'PD':0,'DD':1},calibrator='NONE; existing T diagnostic not deployed',external_normalization_fit_allowed=False,external_label_adaptation_allowed=False,external_data_ready=False,external_metrics=None,external_data_downloaded=False,external_data_adapter_validation='pending actual authorized compatible data',protocol_sha256=sha(HERE/'INDEPENDENT_VALIDATION_PROTOCOL.md'),full_cohort_model_new_training=False))
 text='''# 独立数据兼容性审查（仅元数据，不是外部评估）

本轮没有确认可直接用于当前PD/DD模型的、已授权且输入兼容的独立队列。不能将“有公开PD数据”当作“有可用PD/DD外部test”。下表记录已核查来源和未知项，未下载任何外部sensor/个体表，未申请或接受新数据协议。

'''+table(pd.DataFrame(sources)[['dataset','access','overlap','compatibility','missing']])+'''

原始来源与证据边界：

- [PADS官方数据页](https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/)核验数据许可、设备/单位；同一PADS重打包不能成为独立队列。[原始论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10770131/)支持其单中心及PD/DD/HC定义。本文不借其不同评估数字作性能目标。
- [mPower官方门户](https://www.synapse.org/Synapse:syn4993293/wiki/247860)支持handheld与PD/control任务信息；静态门户没有完整呈现当前个体授权条款，故不称已获访问。原论文/应用研究也不能替用户签署现行数据协议。
- [BEAT-PD官方门户](https://www.synapse.org/Synapse:syn20825169/wiki/600903)及[原研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC10047543/)支持free-living严重度/用药任务；CIS/REAL-PD的设备/标签不同，不自动合并为PD/DD。当前数据授权未完成。
- [WATCH-PD原论文](https://www.nature.com/articles/s41531-024-00721-2)说明PD/control、设备与3DT/steering committee申请规则。申请途径存在不等于用户已经获准；没有联系作者或发邮件。

审查并未穷尽所有世界数据来源。没有DD或缺少固定11任务/双腕Gyro时，不能补零后计算原模型“外部准确率”。新医院/采集时间/设备队列仅为待准备项，不虚构医院、样本、传感器或授权。即使有不同研究名称，也须安全核查与本390人及相关训练来源的参与者重叠；外部SSL预训练成员无法核验时亦须披露。

浏览暴露记录：搜索服务曾返回PADS公开file_list个体元数据行摘要；未主动打开该个体文件、未下载或持久化其行、未用于任何本轮统计/拟合。不能把本次浏览过程声称为完全未接触任何个体信息。统计数据仅来自已授权服务器当前development产物，历史outer暴露永久保留。

当前准备结论：模型实例与评价协议已锁，数据/授权/匹配验证未就绪。外部全部metrics为NOT_EVALUATED，不提供伪造CI。具体入组、盲法、模型锁和停止条件见INDEPENDENT_VALIDATION_PROTOCOL.md。
'''
 (HERE/'DATASET_COMPATIBILITY_AUDIT.md').write_text(text)
 guard();print('INDEPENDENT_VALIDATION_MODEL_LOCK_AND_DATA_AUDIT_WRITTEN')

if __name__=='__main__':main()
