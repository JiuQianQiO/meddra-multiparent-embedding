# `_release` 发布内容审核清单（GitHub + Zenodo）

## 适合放入

- `01_code/`：主方法、GAT/noGNN、归纳式、seed 复核、诊断脚本、三个基线。
- `02_data_dictionary/`：字段说明 + **完全合成** demo，无 MedDRA 许可风险。
- `04_results/`：汇总结果表，便于与正文核对。
- `05_resource_ledger/`：22 个配置的资源账本。
- `06_split_verification/`：统一划分校验脚本与记录。
- `07_run_artifacts/`：`test_metrics_*.json`、`joint_subgroup_diag.json`、`RUN_META.json`、训练历史、外部基线结果。
  - 总体积约 **10.7 MB / 124 个非 .git 文件**，适合 GitHub 和 Zenodo。
- `.zenodo.json`、`CITATION.cff`、`LICENSE`、`README.md`、`requirements.txt`、`.gitignore`、`.gitattributes`。

## 不应放入 / 已被 ignore 或应排除

- MedDRA 原始 CSV：任何 `MEDDRA_DATA_*.csv`，不能公开。
- 模型权重：`*.pt / *.pth / *.ckpt`。
- 缓存：图缓存、PubMedBERT 特征缓存。
- 日志：`*.log`。
- 节点嵌入 CSV：可由 checkpoint 重新生成。
- 本地虚拟环境、`__pycache__`、`.idea/.vscode`、`.DS_Store`。
- 个人邮箱、账号、密钥、内部批注稿、中文内部审阅稿、Cover Letter 草稿。

## 需要你在 push/release 前决定

1. **Zenodo DOI**：`.zenodo.json` 与 README 中仍是占位，首次 release 后回填。
2. **作者元数据**：`.zenodo.json` 目前只有 Zhi Wang，且已注明 provisional；投稿作者确定后同步更新。
3. **代码 MD5 溯源**：`01_code/Meddra_full_trainer_SAGE.py` 是英文注释转录版，其 MD5 与官方 C1V1 运行副本不同；如审稿要求逐字一致，请替换为官方运行副本，或在 README 说明“functional equivalent；original script_md5 见 RUN_META”。
4. **重复文件**：`platform_resources.csv` 同时出现在 `05_resource_ledger/` 与 `07_run_artifacts/`，可保留但 README 已指出 05 是正确目录，07 仅作运行产物留档。
5. **GitHub Release / Zenodo 归档**：只打包仓库 tracked files；`.git` 本身不进入归档。

## 发布后动作

- 生成 GitHub Release tag，Zenodo 自动归档或手动上传 `.zip`。
- 回填 `[Zenodo DOI]` 到主文与 `_release/README.md`。
- 如果作者信息/代码版本有变化，做一次 patch release 并同步 `CITATION.cff` 与 `.zenodo.json`。