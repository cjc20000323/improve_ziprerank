#!/bin/bash

python scripts/evaluate.py \
    --model_path /root/autodl-fs/models/ziprerank_stage2/final \
    --first_stage_file /root/autodl-tmp/data/mmdocir/first_stage_page_top20_dse.pkl \
    --pages_parquet /root/autodl-tmp/MMDocIR/dataset/MMDocIR_pages.parquet \
    --mode page \
    --window_size 20 \
    --sample_size 0 \
    --use_logits