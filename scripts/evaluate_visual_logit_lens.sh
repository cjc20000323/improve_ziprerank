#!/usr/bin/env bash

set -euo pipefail

# The defaults follow scripts/evaluation.sh so this diagnostic can run against
# the same stage-2 checkpoint and MMDocIR Top-20 retrieval results. Prefixing all
# variables with LOGIT_LENS avoids overwriting common shell or system variables.
LOGIT_LENS_PYTHON_BIN="${LOGIT_LENS_PYTHON_BIN:-python}"
LOGIT_LENS_MODEL_PATH="${LOGIT_LENS_MODEL_PATH:-/root/autodl-fs/models/ziprerank_stage2/final}"
LOGIT_LENS_FIRST_STAGE_FILE="${LOGIT_LENS_FIRST_STAGE_FILE:-/root/autodl-tmp/data/mmdocir/first_stage_page_top20_dse.pkl}"
LOGIT_LENS_PAGES_PARQUET="${LOGIT_LENS_PAGES_PARQUET:-/root/autodl-tmp/MMDocIR/dataset/MMDocIR_pages.parquet}"
LOGIT_LENS_OUTPUT_DIR="${LOGIT_LENS_OUTPUT_DIR:-/root/autodl-tmp/outputs/visual_logit_lens_keep50}"
LOGIT_LENS_LLM_LOG_FILE="${LOGIT_LENS_LLM_LOG_FILE:-${LOGIT_LENS_OUTPUT_DIR}/rerank.log}"

# Sampling stops as soon as 20 eligible queries have been found. Each query must
# contain one ground-truth page and at least three incorrect candidates; all Top-20
# candidates stay in one multimodal forward so captured states match final ranks.
LOGIT_LENS_NUM_QUERIES="${LOGIT_LENS_NUM_QUERIES:-20}"
LOGIT_LENS_SEED="${LOGIT_LENS_SEED:-42}"
LOGIT_LENS_WINDOW_SIZE="${LOGIT_LENS_WINDOW_SIZE:-20}"
LOGIT_LENS_STRIDE="${LOGIT_LENS_STRIDE:-10}"

# QI-Early retains the 50% highest-similarity visual tokens per image. Logit Lens
# observes four evenly spaced blocks from 60% model depth through the final block,
# and stores the five highest-logit decoded vocabulary words for each kept token.
LOGIT_LENS_KEEP_RATIO="${LOGIT_LENS_KEEP_RATIO:-0.5}"
LOGIT_LENS_TEMPERATURE="${LOGIT_LENS_TEMPERATURE:-0.1}"
LOGIT_LENS_NUM_LAYERS="${LOGIT_LENS_NUM_LAYERS:-4}"
LOGIT_LENS_START_RATIO="${LOGIT_LENS_START_RATIO:-0.6}"
LOGIT_LENS_TOP_K_VOCAB="${LOGIT_LENS_TOP_K_VOCAB:-5}"
LOGIT_LENS_PROJECTION_CHUNK_SIZE="${LOGIT_LENS_PROJECTION_CHUNK_SIZE:-128}"
LOGIT_LENS_OVERLAY_ALPHA="${LOGIT_LENS_OVERLAY_ALPHA:-72}"

# Additional arguments supplied to this shell script are appended last. This
# makes one-off overrides such as --qi_early_text_mode query_only possible without
# editing the script, while the environment variables cover the common settings.
"${LOGIT_LENS_PYTHON_BIN}" scripts/evaluate_visual_logit_lens.py \
    --model_path "${LOGIT_LENS_MODEL_PATH}" \
    --first_stage_file "${LOGIT_LENS_FIRST_STAGE_FILE}" \
    --pages_parquet "${LOGIT_LENS_PAGES_PARQUET}" \
    --output_dir "${LOGIT_LENS_OUTPUT_DIR}" \
    --num_queries "${LOGIT_LENS_NUM_QUERIES}" \
    --seed "${LOGIT_LENS_SEED}" \
    --window_size "${LOGIT_LENS_WINDOW_SIZE}" \
    --stride "${LOGIT_LENS_STRIDE}" \
    --use_logits \
    --qi_early_keep_ratio "${LOGIT_LENS_KEEP_RATIO}" \
    --qi_early_temperature "${LOGIT_LENS_TEMPERATURE}" \
    --num_logit_lens_layers "${LOGIT_LENS_NUM_LAYERS}" \
    --logit_lens_start_ratio "${LOGIT_LENS_START_RATIO}" \
    --top_k_vocab "${LOGIT_LENS_TOP_K_VOCAB}" \
    --projection_chunk_size "${LOGIT_LENS_PROJECTION_CHUNK_SIZE}" \
    --overlay_alpha "${LOGIT_LENS_OVERLAY_ALPHA}" \
    --llm_log_file "${LOGIT_LENS_LLM_LOG_FILE}" \
    "$@"
