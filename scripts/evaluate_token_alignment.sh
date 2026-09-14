#!/usr/bin/env bash

set -euo pipefail

# AutoDL-oriented defaults. Every path can be overridden through an environment
# variable, so the script can also be used with another model or dataset layout.
TOKEN_ALIGNMENT_PYTHON_BIN="${TOKEN_ALIGNMENT_PYTHON_BIN:-python}"
TOKEN_ALIGNMENT_MODEL_PATH="${TOKEN_ALIGNMENT_MODEL_PATH:-/root/autodl-fs/models/ziprerank_stage2/final}"
TOKEN_ALIGNMENT_FIRST_STAGE_FILE="${TOKEN_ALIGNMENT_FIRST_STAGE_FILE:-/root/autodl-tmp/data/mmdocir/first_stage_page_top20_dse.pkl}"
TOKEN_ALIGNMENT_PAGES_PARQUET="${TOKEN_ALIGNMENT_PAGES_PARQUET:-/root/autodl-tmp/MMDocIR/dataset/MMDocIR_pages.parquet}"
TOKEN_ALIGNMENT_OUTPUT_FILE="${TOKEN_ALIGNMENT_OUTPUT_FILE:-/root/autodl-tmp/outputs/token_alignment_analysis/token_alignment_20_queries.json}"
TOKEN_ALIGNMENT_LLM_LOG_FILE="${TOKEN_ALIGNMENT_LLM_LOG_FILE:-/root/autodl-tmp/outputs/token_alignment_analysis/rerank.log}"

# Query keys are shuffled with a fixed seed, and collection stops after 20
# eligible queries by default. An eligible query must contain both a correct
# candidate and an incorrect candidate; the analysis keeps the highest-ranked
# candidate from each group.
TOKEN_ALIGNMENT_NUM_QUERIES="${TOKEN_ALIGNMENT_NUM_QUERIES:-20}"
TOKEN_ALIGNMENT_SEED="${TOKEN_ALIGNMENT_SEED:-42}"
TOKEN_ALIGNMENT_WINDOW_SIZE="${TOKEN_ALIGNMENT_WINDOW_SIZE:-20}"
TOKEN_ALIGNMENT_STRIDE="${TOKEN_ALIGNMENT_STRIDE:-10}"

# Run the same query-aware visual-token pruning used during logits reranking.
# A keep ratio of 0.5 records the visual tokens retained in the most similar
# half, while the diagnostic output also records every token's alignment data.
TOKEN_ALIGNMENT_KEEP_RATIO="${TOKEN_ALIGNMENT_KEEP_RATIO:-0.5}"
TOKEN_ALIGNMENT_TEMPERATURE="${TOKEN_ALIGNMENT_TEMPERATURE:-0.1}"

# Extra command-line arguments are appended last. For example, pass
# "--qi_early_text_mode query_only" to restrict the text tokens used for
# query-image similarity without editing this script.
"${TOKEN_ALIGNMENT_PYTHON_BIN}" scripts/evaluate_token_alignment.py \
  --model_path "${TOKEN_ALIGNMENT_MODEL_PATH}" \
  --first_stage_file "${TOKEN_ALIGNMENT_FIRST_STAGE_FILE}" \
  --pages_parquet "${TOKEN_ALIGNMENT_PAGES_PARQUET}" \
  --output_file "${TOKEN_ALIGNMENT_OUTPUT_FILE}" \
  --num_queries "${TOKEN_ALIGNMENT_NUM_QUERIES}" \
  --seed "${TOKEN_ALIGNMENT_SEED}" \
  --window_size "${TOKEN_ALIGNMENT_WINDOW_SIZE}" \
  --stride "${TOKEN_ALIGNMENT_STRIDE}" \
  --use_logits \
  --qi_early_keep_ratio "${TOKEN_ALIGNMENT_KEEP_RATIO}" \
  --qi_early_temperature "${TOKEN_ALIGNMENT_TEMPERATURE}" \
  --llm_log_file "${TOKEN_ALIGNMENT_LLM_LOG_FILE}" \
  "$@"
