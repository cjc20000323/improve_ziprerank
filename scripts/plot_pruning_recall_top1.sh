#!/usr/bin/env bash

set -euo pipefail

# AutoDL-oriented defaults. Override these paths without editing this file.
PRUNING_RECALL_PLOT_PYTHON_BIN="${PRUNING_RECALL_PLOT_PYTHON_BIN:-python}"
PRUNING_RECALL_PLOT_ANALYSIS_JSON="${PRUNING_RECALL_PLOT_ANALYSIS_JSON:-/root/autodl-tmp/outputs/pruning_recall_analysis/pruning_recall_query_groups.json}"
PRUNING_RECALL_PLOT_OUTPUT_DIR="${PRUNING_RECALL_PLOT_OUTPUT_DIR:-/root/autodl-tmp/outputs/pruning_recall_analysis}"

# Generates three line charts and three grouped bar charts for Recall@1,
# Recall@3, and Recall@5. Extra command-line arguments are appended last, for
# example to override --output_dir.
"${PRUNING_RECALL_PLOT_PYTHON_BIN}" scripts/plot_pruning_recall_top1.py \
  --analysis_json "${PRUNING_RECALL_PLOT_ANALYSIS_JSON}" \
  --output_dir "${PRUNING_RECALL_PLOT_OUTPUT_DIR}" \
  "$@"
