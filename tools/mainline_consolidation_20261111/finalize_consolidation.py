"""Finalize navigation and publication rules only after migration verification."""
from pathlib import Path
import ast
import hashlib
import json
import shutil

ROOT = Path('/home/zyt/deep_final')


def main():
    audit_dir = ROOT / 'conclusions/organization_20261111'
    verification = json.loads((audit_dir / 'VERIFICATION.json').read_text())
    assert verification['status'] == 'PASS' and verification['checkpoint_validation_forwards'] == 90
    assert not verification['training_started'] and not verification['old_MFAM_runtime_dependency']
    copy_audit_path = audit_dir / 'COPY_AUDIT.json'
    copy_audit = json.loads(copy_audit_path.read_text())
    copy_audit['copy_stage_status'] = copy_audit['status']
    copy_audit['status'] = 'COPIED_AND_RUNTIME_VERIFIED'
    copy_audit['runtime_verification'] = str(audit_dir / 'VERIFICATION.json')
    copy_audit_path.write_text(json.dumps(copy_audit, ensure_ascii=False, indent=2) + '\n')
    backup = ROOT / 'archives/layout_before_20261111'
    backup.mkdir(parents=True, exist_ok=True)
    for name in ['README.md', 'CONTEXT_HANDOFF.md', 'AGENTS.md', 'GIT_VERSION_CONTROL.md']:
        target = backup / (name.replace('.md', '.before_20261111.md'))
        assert not target.exists(), 'Preserve existing archive'
        shutil.copy2(ROOT / name, target)
    publication = ROOT / 'tools/git_publish/publish_to_github.py'
    old_publisher = publication.read_text()
    shutil.copy2(publication, backup / 'publish_to_github.before_20261111.py')
    needle = "for root, prefix in [(PRIMARY, Path()), (SECONDARY, Path('MFAM'))]:"
    assert old_publisher.count(needle) == 1
    updated = old_publisher.replace(needle, 'for root, prefix in [(PRIMARY, Path())]:')
    needle = '    return selected, skipped\n'
    assert updated.count(needle) == 1
    replacement = '''    # Retire MFAM as a source, preserving already-published history without deletion.
    # A retained legacy file must still equal the previously recorded snapshot.
    prior_manifest = UPLOAD / 'PUBLICATION_MANIFEST.json'
    prior = json.loads(prior_manifest.read_text())
    legacy = {x['path']: x for x in prior['files'] if x['path'].startswith('MFAM/')}
    tracked_legacy = {x for x in git('ls-files', '-z').stdout.split('\\0') if x.startswith('MFAM/')}
    if tracked_legacy != set(legacy):
        raise RuntimeError('Retired MFAM inventory changed; inspect without deleting')
    for name, record in sorted(legacy.items()):
        path = UPLOAD / name
        if not path.is_file() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise RuntimeError('Retired MFAM snapshot changed: ' + name)
        selected[name] = (path, record['sha256'], path.stat().st_size)
    skipped['retired_MFAM_files_preserved_without_resync'] = len(legacy)
    return selected, skipped
'''
    updated = updated.replace(needle, replacement)
    updated = updated.replace("'secondary_source': str(SECONDARY)", "'secondary_source': None, 'retired_legacy_prefix': 'MFAM/', 'active_code': str(PRIMARY / 'mainline')")
    assert updated != old_publisher
    ast.parse(updated)
    publication.write_text(updated)

    means = verification['seed_first_means']
    columns = ['accuracy', 'ba', 'auroc', 'macro_f1', 'pd_recall', 'dd_recall']
    table = '| 模型 | Accuracy | BA | AUROC | Macro-F1 | PD Recall | DD Recall |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for condition, label in [('str01', 'STR-01（基线）'), ('frozen_wssl', 'Frozen WSSL-STR（当前保留）')]:
        table += '| ' + label + ' | ' + ' | '.join(f'{means[condition][m]:.6f}' for m in columns) + ' |\n'
    readme = f'''# PADS PD vs DD：当前主线

**统一工作目录：`/home/zyt/deep_final`。** 2026-11-11 完成目录整理。当前保留模型为 **STR-01 + pretrained-frozen HarNet10 wrist residual（Frozen WSSL-STR）**；STR-01 是正式 matched 基线。此目录不是“V8 仍为最终模型”的旧状态。

从这里开始：

- [主线代码和运行说明](mainline/README.md)
- [当前结论与关键证据](conclusions/README.md)
- [完整报告索引](conclusions/EXPERIMENT_REPORT_INDEX.csv)
- [整理后 90 checkpoint 验证](conclusions/organization_20261111/VERIFICATION.json)
- [数据复制和源码来源审计](conclusions/organization_20261111/COPY_AUDIT.json)

```text
deep_final/
  mainline/       当前 STR/WSSL 代码、配置、固定划分、开发入口
  data/           PADS 原始数据、处理后信号、冻结 HarNet10 特征（仅服务器）
  conclusions/    当前结论、所有主线报告入口、整理验证证据
  results/        现有 STR/WSSL/Patch 产物入口；新训练输出位置（本次未训练）
  artifacts/      保留的各阶段正式协议、源码、报告、checkpoint 等原始证据
  archives/       整理前的说明和发布脚本备份
  tools/          整理工具和 GitHub 发布工具
```

{table}
指标是固定 15 个 development splits 内先平均 seeds 42/43/44，再平均 splits；共享受试者，不是独立外部验证。Patch-STR 和 Frequency-Prior 均 REJECT，activity scaling 为 INCONCLUSIVE / 不保留，当前不启动 P3、WSSL+Patch 或额外搜索。

新主线只从本目录的 `mainline/` 和 `data/` 运行，不依赖 `/home/zyt/MFAM`。MFAM 旧目录仍在原处保留，未删除、未导入当前模型工厂。现有历史 `foundation_validation/`、`source/`、`frozen_project/`、`configs/`、`models/`、`BUNDLE_MANIFEST.json`、`SHA256SUMS` 是原阶段归档，不能当作新的统一入口；其中旧路径和旧结论保留用于 provenance，不重写冻结记录。

代码、配置、报告、聚合结果可同步 GitHub；`data/`、private checkpoint、个体预测、临床个体表、缓存不上传。后续只从 `deep_final` 收集新内容；仓库既有 `MFAM/` 作为冻结历史保留，不再从旧目录同步。[同步规则](GIT_VERSION_CONTROL.md)
'''
    (ROOT / 'README.md').write_text(readme)
    (ROOT / 'CONTEXT_HANDOFF.md').write_text('''# 当前主线交接入口（2026-11-11）

唯一工作根目录：`/home/zyt/deep_final`。先读 `README.md`，再读 `mainline/README.md` 与 `conclusions/README.md`。

当前保留：Frozen WSSL-STR；基线：STR-01。主线代码在 `mainline/`，原始/处理后 PADS 及冻结 SSL cache 在 `data/`。不要继续在 MFAM 中开发或从其配置启动新训练。

整理已通过 90 个开发 checkpoint 的完整验证推理、15 次 train-only normalization 重拟合；模型和训练 recipe 不变。本次未训练，未访问 outer 信号或结果。完整证据在 `conclusions/organization_20261111/`。

所有历史负实验、条件停止规则和冻结边界继续有效。旧完整交接记录保存在 `archives/layout_before_20261111/CONTEXT_HANDOFF.before_20261111.md`；原正式实验产物不改名、不删除、不重写。
''')
    (ROOT / 'mainline/README.md').write_text('''# 当前可运行主线

这里只保留 PADS PD vs DD 的 STR-01 与 retained Frozen WSSL-STR，以及所需的数据、训练、指标和 provenance 工具。模型工厂拒绝 MFAM/subject-MFAM 等其他模型；没有导入旧复现的 s4_final、r2_stabilization、targeted_ablation、H1 或其他数据集项目。原 STR 算子源码直接复制；WSSL 使用归档的相同 forward 和 checkpoint state keys。

## 路径

- 模型：`src/models/pure_deep.py`、`wssl.py`、`bilateral.py`、`activity_fusion.py`
- 统一配置：`configs/str01_seed42.yaml`、`str01_seed43.yaml`、`str01_seed44.yaml`
- PADS：`../data/raw/pads/` 与 `../data/processed/pads_multi_activity/v2_l1_full_length/`
- 冻结特征：`../data/ssl/harnet10_embeddings.npz`（与原缓存 SHA 相同）
- 现有权重：`../results/str01/`、`../results/frozen_wssl/`；均指向 deep_final 内部归档
- HarNet10 原官方源码/资产已在 `../artifacts/phase3b_external_ssl_20260928/vendor/ssl-wearables/` 留存，不下载、不改权重、不做重提取或微调。

使用服务器已有 `/home/zyt/envs/mfam_analysis/bin/python`；本次没有安装依赖。模型推理使用处理后六通道 Acc+Gyro；Frozen WSSL 额外按 subject/activity/wrist 从冻结 cache 取 1024-D 表征。保留 train-fold normalization、mask、true length、双腕融合、activity order、structured readout 和正式 recipe。

## 开发入口

```bash
cd /home/zyt/deep_final/mainline
/home/zyt/envs/mfam_analysis/bin/python -B scripts/run_development.py --help
```

该入口只接受 `str01` 或 `frozen_wssl`、三个固定 seeds 和一个显式 development context/inner；test subjects 被置为空，只在训练/验证 subject 子集上加载信号。现有输出存在时拒绝覆盖，不执行 outer-final、不搜索参数。本次整理没有调用训练入口。未来训练前仍需有具体实验授权和协议。

`scripts/verify_mainline.py` 是本次迁移验证工具，执行开发集推理与 norm refit，不训练；已有验证输出时拒绝覆盖。结果已保存于 `../conclusions/organization_20261111/`。

配置中的绝对数据路径已改到 deep_final。正式归档 checkpoint/config/source 不改路径，以保留旧 hash；验证时用统一配置构建相同模型并 strict load 原权重，不宣称迁移源码 hash 等于训练时历史整棵源码 hash。
''')
    (ROOT / 'conclusions/README.md').write_text(f'''# 当前结论与证据入口

当前保留 **Frozen WSSL-STR**；主线问题是 PADS PD vs DD subject-level supervised classification，不是连续严重度回归。以下结论均来自 development，不能宣称独立外部或临床泛化。

{table}
| 实验 | 状态 | 正式报告 |
|---|---|---|
| STR-01 | retained matched baseline | [STR01_REPORT](../artifacts/str01_structured_token_residual_20260921/STR01_REPORT.md) |
| Frozen WSSL-STR | retained current single-model | [PHASE3B_REPORT](../artifacts/phase3b_external_ssl_20260928/PHASE3B_REPORT.md) |
| WSSL 贡献与复现 | descriptive evidence，未训练新候选 | [REPRODUCTION_REPORT](../artifacts/wssl_contribution_20261007/REPRODUCTION_REPORT.md) |
| WSSL classifier LR 对照 | REJECT，保留原 recipe | [FINAL_REPORT](../artifacts/wssl_review_20261004/FINAL_REPORT.md) |
| Activity scaling | INCONCLUSIVE，不保留 | [FINAL_REPORT](../artifacts/wssl_activity_scaling_20261008/FINAL_REPORT.md) |
| Frequency-Prior STR | REJECT | [FINAL_REPORT](../artifacts/frequency_prior_str_20261007/FINAL_REPORT.md) |
| 固定 200/100 Patch-STR | REJECT；停止 P3/WSSL+Patch | [PATCH_STR_REPORT](../artifacts/patch_str_20261008/PATCH_STR_REPORT.md) |
| Patch 2026-11-11 复核 | 与原归档一致，未重复正式训练 | [RECHECK_REPORT](../artifacts/patch_str_recheck_20261111/RECHECK_REPORT.md) |

[全部主线阶段报告索引](EXPERIMENT_REPORT_INDEX.csv) 统一列出原始正式文件位置，不复制或改写历史统计。索引不包含已封存的 outer 产物。历史归档保留原位，避免破坏冻结 hash、checkpoint 引用和报告图片相对路径。

本次目录整理校验了 PADS 原始/处理后数据副本、冻结 cache、90 个开发 checkpoint 的概率/预测/六指标、15 次训练集 normalization、原正式文件 hash。没有重新训练、模型选择、threshold tuning 或 outer 分析。详细数字见 [VERIFICATION](organization_20261111/VERIFICATION.json) 与 [COPY_AUDIT](organization_20261111/COPY_AUDIT.json)。
''')
    (ROOT / 'AGENTS.md').write_text('''# Server mainline instructions

## Canonical root (user instruction, 2026-11-11)
- Develop the PADS PD-vs-DD mainline only in /home/zyt/deep_final/mainline. Keep its data under /home/zyt/deep_final/data and current conclusions under /home/zyt/deep_final/conclusions.
- /home/zyt/MFAM is retired historical material. Do not develop there or resynchronize its contents. Do not import unrelated paper replications into the active mainline.
- Current retained model: frozen WSSL-STR; matched baseline: STR-01. Read README.md and conclusions/README.md first. Do not treat old V8 root manifests as current-model selection.
- Preserve original frozen source/config/checkpoint/result paths, provenance hashes, negative experiments and outer boundaries. Migration is not permission for new training/search/outer evaluation.
- No automatic deletion, bulk deletion or force push. Preserve old MFAM files and original data; cleanup is a separate user action.

## Git publication
- Preview and publish using /home/zyt/deep_final/tools/git_publish/publish_to_github.py; retain all filtering and audits. New source is deep_final only. Already-published MFAM/ files remain an immutable legacy snapshot without resync or deletion.
- Use /home/zyt/pureDeep-github-upload as the independent mirror for https://github.com/chouytong/pureDeep.git. Do not initialize Git over the source tree or replace the old MFAM remote.
- Verify code, commit meaningfully and publish only code/config/report/aggregate evidence; exclude data, weights/checkpoints, individual clinical/prediction tables, features/caches, logs and credentials.
- If server HTTPS authentication fails, the already-authorized GitHub connector may publish the identical audited tree; preserve an unpublished local commit before aligning mirror HEAD. Never force-update remote main. Verify local HEAD equals remote main and the worktree is clean.
- Inspect and reconcile remote changes before publishing. Preserve commits and report remaining work if no authorized authentication is available.
''')
    (ROOT / 'GIT_VERSION_CONTROL.md').write_text('''# GitHub publication after mainline consolidation

Canonical source is `/home/zyt/deep_final` only; active code is `mainline/`, private data is `data/`, conclusions are `conclusions/`. `/home/zyt/MFAM` is not a publishing source anymore.

The independent mirror is `/home/zyt/pureDeep-github-upload`; repository is https://github.com/chouytong/pureDeep, branch main. Use `tools/git_publish/publish_to_github.py` to preview, filter, checksum, audit and prepare/commit the snapshot. Never use force push or remove files automatically.

The publisher retains already-published `MFAM/` files directly from the mirror only if their hashes match the previous publication manifest. This preserves history without importing new old-project code or requiring deletion. It must not read MFAM as an active source. Historical source paths inside reports/configs remain provenance, not current runtime dependencies.

Datasets, private raw/processed signals, checkpoint/model weights, individual predictions/clinical tables, fitted normalization with subject membership, caches/features and logs stay on the server. Data and result symlinks are excluded. Code, configs, reports, aggregate metrics and hashes are published.

The server HTTPS token was rejected in the 2026-11-11 sync. Existing authorized GitHub connector access for chouytong can publish the exact audited Git tree. Preserve any unpublished server commit on a local archive ref, then align the mirror only after proving identical trees and remote main. Connected-account access does not authorize changing data, recipe or experiments.

Network access can use a temporary Mac SSH reverse SOCKS tunnel at 127.0.0.1:18743; do not terminate unrelated tunnels or change permanent proxy/credential configuration. Passwords/tokens never belong in source, URLs, reports or commits.
''')
    audit = dict(status='PASS', canonical_root=str(ROOT), active_code=str(ROOT / 'mainline'),
        active_data=str(ROOT / 'data'), conclusions=str(ROOT / 'conclusions'),
        old_MFAM_retired_not_deleted=True, frozen_paths_preserved=True,
        publisher_source='deep_final only', legacy_MFAM_Git_snapshot_preserved=True,
        verification=verification, backup_directory=str(backup), training=False, outer_analysis=False)
    (audit_dir / 'ORGANIZATION_REPORT.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(status='DOCUMENTATION_AND_PUBLISHER_UPDATED', verification='PASS',
                          active_root=str(ROOT), backup=str(backup))), flush=True)


if __name__ == '__main__':
    main()
