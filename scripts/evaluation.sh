#!/bin/bash

# 对 MMDocIR Page 数据执行全量 Top-20 重排并收集 token 相似度。
# 候选数和 window_size 都是 20，所以每个 query 只经过一个窗口；这样每个候选
# 只有一套明确的 token 相似度，能够与最终排序位置一一对应。
# qi_early_keep_ratio=0.5：每张图保留 50% 的视觉 token，即剪枝 50%。
# similarity_num_examples=5：正确组和错误组分别均匀抽取 5 个样例；seed 保证可复现。
# 输出目录中会生成整体图、两张逐样例图、JSON 摘要和原始 .pt 分数。
python scripts/evaluate.py \
    --model_path /root/autodl-fs/models/ziprerank_stage2/final \
    --first_stage_file /root/autodl-tmp/data/mmdocir/first_stage_page_top20_dse.pkl \
    --pages_parquet /root/autodl-tmp/MMDocIR/dataset/MMDocIR_pages.parquet \
    --mode page \
    --window_size 20 \
    --sample_size 0 \
    --use_logits \
    --use_qi_early \
    --qi_early_keep_ratio 0.5 \
    --analyze_token_similarity \
    --similarity_num_examples 5 \
    --similarity_seed 42 \
    --similarity_output_dir /root/autodl-tmp/outputs/ziprerank_similarity_keep50
