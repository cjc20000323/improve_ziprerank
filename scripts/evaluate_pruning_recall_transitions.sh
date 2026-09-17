#!/usr/bin/env bash

set -euo pipefail

# AutoDL-oriented defaults. Override any value through the corresponding
# PRUNING_RECALL_* environment variable when the model or dataset is elsewhere.
PRUNING_RECALL_PYTHON_BIN="${PRUNING_RECALL_PYTHON_BIN:-python}"
PRUNING_RECALL_MODEL_PATH="${PRUNING_RECALL_MODEL_PATH:-/root/autodl-fs/models/ziprerank_stage2/final}"
PRUNING_RECALL_FIRST_STAGE_FILE="${PRUNING_RECALL_FIRST_STAGE_FILE:-/root/autodl-tmp/data/mmdocir/first_stage_page_top20_dse.pkl}"
PRUNING_RECALL_PAGES_PARQUET="${PRUNING_RECALL_PAGES_PARQUET:-/root/autodl-tmp/MMDocIR/dataset/MMDocIR_pages.parquet}"
PRUNING_RECALL_OUTPUT_DIR="${PRUNING_RECALL_OUTPUT_DIR:-/root/autodl-tmp/outputs/pruning_recall_analysis}"

# sample_size=0 evaluates every query. All ten pruning conditions (0%-90% in
# 10% increments) reuse the same seed and query-selection arguments, and the
# Python comparison fails fast if their query sets or first-stage candidates do
# not match exactly.
PRUNING_RECALL_SAMPLE_SIZE="${PRUNING_RECALL_SAMPLE_SIZE:-0}"
PRUNING_RECALL_SEED="${PRUNING_RECALL_SEED:-42}"
PRUNING_RECALL_WINDOW_SIZE="${PRUNING_RECALL_WINDOW_SIZE:-20}"
PRUNING_RECALL_STRIDE="${PRUNING_RECALL_STRIDE:-10}"
PRUNING_RECALL_TEMPERATURE="${PRUNING_RECALL_TEMPERATURE:-0.1}"

# Similarity artifacts retain the same structure as the previous analysis:
# all eligible queries contribute to aggregate correct/incorrect histograms,
# while five examples per group keep raw token scores and receive individual plots.
PRUNING_RECALL_SIMILARITY_EXAMPLES="${PRUNING_RECALL_SIMILARITY_EXAMPLES:-5}"
PRUNING_RECALL_SIMILARITY_SEED="${PRUNING_RECALL_SIMILARITY_SEED:-42}"
PRUNING_RECALL_SIMILARITY_BINS="${PRUNING_RECALL_SIMILARITY_BINS:-80}"
# 这两个变量只控制每种剪枝条件下的全查询候选均值：0 个 GT 表示使用
# Top-K 内全部 GT；错误候选默认按最终重排名次取前 3 个。
PRUNING_RECALL_SIMILARITY_COMPARISON_GT_CANDIDATES="${PRUNING_RECALL_SIMILARITY_COMPARISON_GT_CANDIDATES:-0}"
PRUNING_RECALL_SIMILARITY_COMPARISON_INCORRECT_CANDIDATES="${PRUNING_RECALL_SIMILARITY_COMPARISON_INCORRECT_CANDIDATES:-3}"

"${PRUNING_RECALL_PYTHON_BIN}" scripts/evaluate_pruning_recall_transitions.py \
  --model_path "${PRUNING_RECALL_MODEL_PATH}" \
  --first_stage_file "${PRUNING_RECALL_FIRST_STAGE_FILE}" \
  --pages_parquet "${PRUNING_RECALL_PAGES_PARQUET}" \
  --output_dir "${PRUNING_RECALL_OUTPUT_DIR}" \
  --sample_size "${PRUNING_RECALL_SAMPLE_SIZE}" \
  --seed "${PRUNING_RECALL_SEED}" \
  --window_size "${PRUNING_RECALL_WINDOW_SIZE}" \
  --stride "${PRUNING_RECALL_STRIDE}" \
  --qi_early_temperature "${PRUNING_RECALL_TEMPERATURE}" \
  --similarity_num_examples "${PRUNING_RECALL_SIMILARITY_EXAMPLES}" \
  --similarity_seed "${PRUNING_RECALL_SIMILARITY_SEED}" \
  --similarity_num_bins "${PRUNING_RECALL_SIMILARITY_BINS}" \
  --similarity_comparison_num_ground_truth_candidates "${PRUNING_RECALL_SIMILARITY_COMPARISON_GT_CANDIDATES}" \
  --similarity_comparison_num_incorrect_candidates "${PRUNING_RECALL_SIMILARITY_COMPARISON_INCORRECT_CANDIDATES}" \
  "$@"
