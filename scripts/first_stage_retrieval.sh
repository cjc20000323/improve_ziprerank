#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python scripts/first_stage_retrieval.py \
    --model_name /root/autodl-tmp/MrLight-dse-qwen2-2b-mrl-v1 \
    --pages_parquet /root/autodl-tmp/MMDocIR-MMDocIR_Evaluation_Dataset/MMDocIR_pages.parquet \
    --layouts_parquet /root/autodl-tmp/MMDocIR-MMDocIR_Evaluation_Dataset/MMDocIR_layouts.parquet \
    --annotation_file /root/autodl-tmp/MMDocIR-MMDocIR_Evaluation_Dataset/MMDocIR_annotations.jsonl \
    --output_dir /root/autodl-tmp/data/mmdocir/first_stage_page_top20_dse.pkl \
    --top_k 20 \
    --single_gpu