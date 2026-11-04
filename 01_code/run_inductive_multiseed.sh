#!/usr/bin/env bash
# -*- mode: shell-script -*-
# 归纳式多种子复算：IND-SAGE-MASK 与 IND-NOGNN-PUBMEDBERT 各跑 5 个训练种子（42–46）。
#
# 目的（审稿意见 M5）：当前表 5 的 IND-SAGE-MASK / IND-NOGNN-PUBMEDBERT 均为单次观测，
# 补多种子后可报告 mean ± sd，与 IND-SAGE-KEEP（已 5 种子）口径对齐。
#
# 前置条件（服务器上，按需修改路径）：
#   - 训练器：Meddra_inductive_SAGE.py / Meddra_inductive_noGNN.py（与本脚本同目录）
#   - MedDRA CSV：MEDDRA_CSV_PATH
#   - 图/特征缓存：MEDDRA_CACHE_DIR（含 *graph.pt 与 *pubmedbert.pt）
#   - 划分索引：MEDDRA_SPLIT_INDEX_FILE（可选，缺省用训练器内建划分）
#   - 文本模型目录：MEDDRA_TEXT_MODEL_DIR（PubMedBERT 权重）
#
# 用法：
#   MEDDRA_CSV_PATH=/mnt/.../MEDDRA_DATA_29_0.csv \
#   MEDDRA_CACHE_DIR=/mnt/.../cache_shared \
#   MEDDRA_RUN_ROOT=/mnt/.../jbi_runs/REVIEW \
#   bash run_inductive_multiseed.sh
set -euo pipefail

SEEDS="${SEEDS:-42 43 44 45 46}"
RUN_ROOT="${MEDDRA_RUN_ROOT:-./review_runs}"
SAGE_TRAINER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/Meddra_inductive_SAGE.py"
NOGNN_TRAINER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/Meddra_inductive_noGNN.py"

echo "============================================================"
echo "归纳式多种子复算：SEEDS=[$SEEDS]"
echo "RUN_ROOT=$RUN_ROOT"
echo "============================================================"

for seed in $SEEDS; do
  export MEDDRA_TRAIN_SEED="$seed"

  export MEDDRA_RUN_DIR="$RUN_ROOT/IND-SAGE-MASK/seed$seed"
  export MEDDRA_OUTPUT_ROOT="$MEDDRA_RUN_DIR"
  echo ""
  echo "===== IND-SAGE-MASK  seed=$seed  ->  $MEDDRA_RUN_DIR ====="
  python "$SAGE_TRAINER" IND-SAGE-MASK

  export MEDDRA_RUN_DIR="$RUN_ROOT/IND-NOGNN-PUBMEDBERT/seed$seed"
  export MEDDRA_OUTPUT_ROOT="$MEDDRA_RUN_DIR"
  echo ""
  echo "===== IND-NOGNN-PUBMEDBERT  seed=$seed  ->  $MEDDRA_RUN_DIR ====="
  python "$NOGNN_TRAINER" IND-NOGNN-PUBMEDBERT
done

echo ""
echo "全部完成。各种子产物位于 $RUN_ROOT 下，可用 compile_inductive_seeds.py 汇总。"
