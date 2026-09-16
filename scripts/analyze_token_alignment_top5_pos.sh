#!/usr/bin/env bash

set -euo pipefail

# 使用本脚本前，需要先运行 evaluate_token_alignment.sh，生成包含查询 token
# 对齐结果的 JSON 文件；建议在需要统计整个测试集时为前一步传入
# --all_eligible_queries。该选项会自动把全量记录及 Top1 错误组所需的前置错误
# 候选写入单独的 *_all_eligible.json，同时让原主 JSON 和图片仍只保留 20 条。
# 如果这个全量 JSON 已经存在，则无需重新推理，可直接运行本脚本完成词性统计。

# Run the independent POS post-processing step on an existing token-alignment
# JSON file. This script does not run model inference and never rewrites the
# input JSON; the statistics are always saved to a separate output file.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname -- "${SCRIPT_DIR}")"

# AutoDL-oriented defaults. Override these variables when the alignment file,
# Python environment, or installed spaCy model is located elsewhere.
TOKEN_ALIGNMENT_POS_PYTHON_BIN="${TOKEN_ALIGNMENT_POS_PYTHON_BIN:-python}"
TOKEN_ALIGNMENT_POS_INPUT_FILE="${TOKEN_ALIGNMENT_POS_INPUT_FILE:-/root/autodl-tmp/outputs/token_alignment_analysis/token_alignment_20_queries_all_eligible.json}"
TOKEN_ALIGNMENT_POS_OUTPUT_FILE="${TOKEN_ALIGNMENT_POS_OUTPUT_FILE:-${TOKEN_ALIGNMENT_POS_INPUT_FILE%.*}_top5_pos.json}"
TOKEN_ALIGNMENT_POS_SPACY_MODEL="${TOKEN_ALIGNMENT_POS_SPACY_MODEL:-en_core_web_sm}"

# By default, rank query tokens using the number of all aligned visual tokens.
# Set COUNT_SCOPE to "kept" or "pruned" to rank only by retained or removed
# visual tokens. TOP_K controls how many query tokens each candidate contributes.
TOKEN_ALIGNMENT_POS_TOP_K="${TOKEN_ALIGNMENT_POS_TOP_K:-5}"
TOKEN_ALIGNMENT_POS_COUNT_SCOPE="${TOKEN_ALIGNMENT_POS_COUNT_SCOPE:-all}"

# This optional value provides a strict completeness check. When set, the
# Python program refuses to write statistics unless the input contains exactly
# this many query records. Leave it empty when the eligible-query count varies.
TOKEN_ALIGNMENT_POS_EXPECTED_NUM_QUERIES="${TOKEN_ALIGNMENT_POS_EXPECTED_NUM_QUERIES:-}"

pos_args=(
  --input_file "${TOKEN_ALIGNMENT_POS_INPUT_FILE}"
  --output_file "${TOKEN_ALIGNMENT_POS_OUTPUT_FILE}"
  --spacy_model "${TOKEN_ALIGNMENT_POS_SPACY_MODEL}"
  --top_k "${TOKEN_ALIGNMENT_POS_TOP_K}"
  --count_scope "${TOKEN_ALIGNMENT_POS_COUNT_SCOPE}"
)

if [[ -n "${TOKEN_ALIGNMENT_POS_EXPECTED_NUM_QUERIES}" ]]; then
  pos_args+=(
    --expected_num_queries "${TOKEN_ALIGNMENT_POS_EXPECTED_NUM_QUERIES}"
  )
fi

# Extra command-line arguments are appended last, so one-off options can be
# supplied without editing this file.
"${TOKEN_ALIGNMENT_POS_PYTHON_BIN}" \
  "${PROJECT_ROOT}/scripts/analyze_token_alignment_top5_pos.py" \
  "${pos_args[@]}" \
  "$@"
