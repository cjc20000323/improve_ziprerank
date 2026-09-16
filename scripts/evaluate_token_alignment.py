"""Inspect which QI text token maximizes similarity for each visual token.

This is a standalone diagnostic entry point.  It reuses stable loading and
reranking functions from ``scripts/evaluate.py`` but does not modify that file.
By default it randomly selects 20 page-mode queries whose first-stage Top-K
contains both a ground-truth and a non-ground-truth candidate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, cast

import pandas as pd
import torch
from transformers import AutoProcessor


# Make imports independent of the shell's current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.qwen3vl_with_qi_early_token_alignment import (  # noqa: E402
    Qwen3VLWithQIEarlyTokenAlignment,
)
from scripts import evaluate as base_evaluate  # noqa: E402
from utils.token_alignment_analysis import (  # noqa: E402
    TokenAlignmentCollector,
    find_dataset_query_input_positions,
    select_all_eligible_queries,
    select_base_diagnostic_records,
    select_eligible_query_subset,
)


def parse_args() -> argparse.Namespace:
    # 参数尽量与常规 evaluate.py 保持一致，使诊断运行可以复用相同模型、候选集
    # 和重排方式；这里只额外提供样本筛选数量与对齐 JSON 的输出位置。
    parser = argparse.ArgumentParser(
        description=(
            "Rerank 20 MMDocIR queries and export, for every visual token in "
            "the best GT and best non-GT candidate, its maximum-similarity "
            "QI text token."
        )
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--first_stage_file", required=True)
    parser.add_argument(
        "--pages_parquet",
        default="MMDocIR/dataset/MMDocIR_pages.parquet",
    )
    parser.add_argument(
        "--output_file",
        default="outputs/token_alignment_analysis/token_alignment_20_queries.json",
    )
    parser.add_argument(
        "--num_queries",
        type=int,
        default=20,
        help="Number of eligible queries to sample uniformly (default: 20).",
    )
    parser.add_argument(
        "--all_eligible_queries",
        action="store_true",
        help=(
            "Also analyze every eligible query and save those records to a "
            "separate JSON without images. --output_file and its images still "
            "contain only the seeded --num_queries sample."
        ),
    )
    parser.add_argument(
        "--all_eligible_output_file",
        default=None,
        help=(
            "JSON-only output for --all_eligible_queries. By default, append "
            "_all_eligible to the stem of --output_file."
        ),
    )
    parser.add_argument(
        "--collect_incorrect_before_correct",
        action="store_true",
        help=(
            "Additionally serialize every non-GT candidate ranked ahead of the "
            "highest-ranked GT. The original two-candidate output remains unchanged."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--window_size",
        type=int,
        default=20,
        help=(
            "Must be at least the first-stage candidate count so each query "
            "uses exactly one model forward window."
        ),
    )
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument(
        "--use_logits",
        action="store_true",
        help="Use the faster first-token-logit reranking path.",
    )
    parser.add_argument(
        "--qi_early_keep_ratio",
        type=float,
        default=0.5,
        help="Visual-token retention ratio (default: 0.5).",
    )
    parser.add_argument("--qi_early_temperature", type=float, default=0.1)
    parser.add_argument(
        "--overlay_alpha",
        type=int,
        default=72,
        help="Opacity of the green retained-patch overlay in [0, 255].",
    )
    parser.add_argument(
        "--qi_early_text_mode",
        choices=["query_only", "all_before_image"],
        default=None,
        help=(
            "Optional override. If omitted, use the mode stored in the model "
            "configuration. This controls pruning only; diagnostic token "
            "matching always uses the dataset query text alone."
        ),
    )
    parser.add_argument(
        "--llm_log_file",
        default=None,
        help="Optional path for the same per-query LLM log as normal evaluation.",
    )
    return parser.parse_args()


def _consume_qi_token_alignment_stats(
    model,
    candidate_original_indices: Sequence[int],
    image_grid_thw: Any,
) -> Sequence[Dict[str, Any]]:
    """Copy diagnostic GPU tensors to CPU immediately after one forward.

    ``base_evaluate.rerank_window_with_images`` already has one well-defined
    place where its standard similarity records are consumed.  The standalone
    script temporarily substitutes this richer consumer at runtime, leaving
    the source code and behavior of normal ``evaluate.py`` unchanged.
    """
    # noinspection PyProtectedMember
    pruner = base_evaluate._find_qi_early_pruner(model)

    # 这四份缓存来自同一次 forward：raw_stats 按候选图像顺序排列，后三项则是
    # 所有候选共享的查询文本坐标系。复制前先整体取出，避免清缓存时丢失引用。
    raw_stats = pruner.last_similarity_stats
    query_token_ids = getattr(pruner, "last_dataset_query_token_ids", None)
    query_input_positions = getattr(
        pruner, "last_dataset_query_input_positions", None
    )
    text_extraction_mode = getattr(pruner, "last_alignment_text_scope", None)

    # Drop model-owned GPU references early. Local references remain valid long
    # enough for the explicit CPU copy below and are released when this returns.
    pruner.last_similarity_stats = None
    pruner.last_query_token_ids = None
    pruner.last_query_input_positions = None
    pruner.last_text_extraction_mode = None
    clear_dataset_query_metadata = getattr(
        pruner, "clear_dataset_query_alignment_metadata", None
    )
    if clear_dataset_query_metadata is not None:
        clear_dataset_query_metadata()

    if raw_stats is None:
        raise RuntimeError("QI-Early did not expose token-alignment statistics")
    if query_token_ids is None or query_input_positions is None:
        raise RuntimeError("QI-Early did not expose the text-token/input mapping")
    if text_extraction_mode is None:
        raise RuntimeError("QI-Early did not expose its text extraction mode")
    if len(raw_stats) != len(candidate_original_indices):
        raise RuntimeError(
            "QI-Early alignment/image count mismatch: "
            f"stats={len(raw_stats)}, images={len(candidate_original_indices)}"
        )

    # Processor grids are captured from the exact prepared window before inference.
    # Their image order is the same order used by the pruner and candidate indices.
    image_grids = torch.as_tensor(image_grid_thw).detach().to(
        device="cpu", dtype=torch.long
    ).contiguous()
    if image_grids.ndim != 2 or image_grids.shape[1] != 3:
        raise ValueError("image_grid_thw must have shape (num_candidates, 3)")
    if image_grids.shape[0] != len(candidate_original_indices):
        raise ValueError(
            "Image-grid/candidate count mismatch: "
            f"grids={image_grids.shape[0]}, candidates={len(candidate_original_indices)}"
        )

    if not isinstance(query_token_ids, torch.Tensor):
        raise TypeError("last_query_token_ids must be a torch.Tensor")
    if not isinstance(query_input_positions, torch.Tensor):
        raise TypeError("last_query_input_positions must be a torch.Tensor")

    # 查询元数据只复制一次到 CPU，随后由各候选记录共享。contiguous() 让后续
    # 序列化和相等性检查不依赖原 tensor 的切片步长或模型所在设备。
    shared_query_token_ids = query_token_ids.detach().to(
        device="cpu", dtype=torch.long
    ).contiguous()
    shared_query_input_positions = query_input_positions.detach().to(
        device="cpu", dtype=torch.long
    ).contiguous()

    copied_stats = []

    # candidate_original_indices 保存当前窗口图像在一阶段候选列表中的位置。
    # zip 后，每份视觉统计就获得稳定的 candidate_pos，可在最终排序完成后回接
    # 到正确的页面；逐视觉 token 的三个向量仍保持完全相同的原始 patch 顺序。
    # 这里的三个诊断向量是 query-only 分数、保留掩码和 query-only argmax；另存的
    # scores 是模型实际用于剪枝的分数，便于解释保留结果而不混淆两套文本范围。
    for image_index, (candidate_pos, image_stats) in enumerate(
        zip(candidate_original_indices, raw_stats)
    ):
        if (
            "dataset_query_max_similarities" not in image_stats
            or "dataset_query_max_text_token_indices" not in image_stats
        ):
            raise RuntimeError(
                "Missing per-visual-token dataset-query alignment. Ensure the diagnostic "
                "Qwen3VLWithQIEarlyTokenAlignment class is loaded."
            )
        copied_stats.append({
            "candidate_pos": int(candidate_pos),
            "scores": image_stats["scores"].detach().to(
                device="cpu", dtype=torch.float32
            ).contiguous(),
            "kept_mask": image_stats["kept_mask"].detach().to(
                device="cpu", dtype=torch.bool
            ).contiguous(),
            "query_max_similarities": image_stats[
                "dataset_query_max_similarities"
            ].detach().to(device="cpu", dtype=torch.float32).contiguous(),
            "max_text_token_indices": image_stats[
                "dataset_query_max_text_token_indices"
            ].detach().to(device="cpu", dtype=torch.long).contiguous(),
            "threshold": float(
                image_stats["threshold"].detach().float().cpu().item()
            ),
            "num_original_tokens": int(image_stats["num_original_tokens"]),
            "num_kept_tokens": int(image_stats["num_kept_tokens"]),
            "image_grid_thw": image_grids[image_index].tolist(),
            # These tensors are shared by all candidates in the current query;
            # they are not cloned twenty times in CPU memory.
            "query_token_ids": shared_query_token_ids,
            "query_input_positions": shared_query_input_positions,
            "text_extraction_mode": str(text_extraction_mode),
        })
    return copied_stats


def _open_log(path: Optional[str], args: argparse.Namespace):
    if path is None:
        return None
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    handle.write("=" * 80 + "\n")
    handle.write("TOKEN ALIGNMENT DIAGNOSTIC - ZipRerank\n")
    handle.write("=" * 80 + "\n")
    handle.write(f"Model: {args.model_path}\n")
    query_scope = "all eligible" if args.all_eligible_queries else args.num_queries
    handle.write(f"Queries: {query_scope}\n")
    handle.write(f"Keep ratio: {args.qi_early_keep_ratio}\n")
    handle.flush()
    return handle


def _all_eligible_output_path(args: argparse.Namespace) -> Optional[Path]:
    """Resolve the separate JSON-only output used by corpus-level statistics."""
    if not args.all_eligible_queries:
        if args.all_eligible_output_file is not None:
            raise ValueError(
                "--all_eligible_output_file requires --all_eligible_queries"
            )
        return None

    base_path = Path(args.output_file)
    if args.all_eligible_output_file is not None:
        all_path = Path(args.all_eligible_output_file)
    else:
        suffix = base_path.suffix or ".json"
        all_path = base_path.with_name(
            f"{base_path.stem}_all_eligible{suffix}"
        )
    if base_path.resolve() == all_path.resolve():
        raise ValueError(
            "--all_eligible_output_file must differ from --output_file"
        )
    return all_path


def main() -> None:
    args = parse_args()
    if args.num_queries <= 0:
        raise ValueError("--num_queries must be positive")
    if not 0.0 < args.qi_early_keep_ratio <= 1.0:
        raise ValueError("--qi_early_keep_ratio must be in (0, 1]")
    if not 0 <= args.overlay_alpha <= 255:
        raise ValueError("--overlay_alpha must be in [0, 255]")
    all_eligible_output_path = _all_eligible_output_path(args)

    print("Loading MMDocIR page parquet...")
    parquet_df = pd.read_parquet(args.pages_parquet)
    print(f"Loaded {len(parquet_df)} pages")

    print("Loading first-stage retrieval results...")
    first_stage_results = base_evaluate.load_first_stage_results(
        args.first_stage_file
    )

    # 无论是否启用全量统计，都先按原逻辑确定主 JSON 的固定 seed 样本。全量
    # 模式只扩大模型实际处理的集合，之后再把这批样本抽回主文件和图片目录。
    sampled_results, sample_selection_metadata = select_eligible_query_subset(
        first_stage_results,
        num_queries=args.num_queries,
        seed=args.seed,
    )
    if args.all_eligible_queries:
        selected_results, selection_metadata = select_all_eligible_queries(
            first_stage_results
        )
    else:
        selected_results = sampled_results
        selection_metadata = sample_selection_metadata
    expected_inference_queries = len(selected_results)
    max_candidates = max(
        len(item["top_k_global_indices"]) for item in selected_results.values()
    )

    # 对齐缓存只代表最近一次模型 forward，因此一个 query 必须完整放入同一窗口。
    # 若发生滑窗，多个 forward 会覆盖 query 元数据，候选与 argmax 将无法可靠配对。
    if max_candidates > args.window_size:
        raise ValueError(
            "Token alignment requires one reranking window per query, but the "
            f"selected queries contain up to {max_candidates} candidates and "
            f"--window_size={args.window_size}. Increase --window_size."
        )
    selection_seed = selection_metadata.get("selection_seed")
    seed_note = f", seed={selection_seed}" if selection_seed is not None else ""
    print(
        f"Found {len(selected_results)} eligible queries after inspecting "
        f"{selection_metadata['queries_inspected_before_early_stop']} entries"
        f" ({selection_metadata['selection_method']}{seed_note})"
    )

    print(f"Loading diagnostic model from {args.model_path}...")

    # 诊断子类沿用原检查点的全部参数，仅把无参数的 QI-Early pruner 换成会额外
    # 保存 argmax 的版本，所以模型打分和视觉 token 选择规则与正常评估一致。
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    model = Qwen3VLWithQIEarlyTokenAlignment.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.set_qi_early_keep_ratio(args.qi_early_keep_ratio)
    model.qi_early_pruner.temperature = args.qi_early_temperature
    if args.qi_early_text_mode is not None:
        model.set_qi_early_text_mode(args.qi_early_text_mode)
    model.qi_early_pruner.set_collect_similarity_stats(True)
    model.eval()

    # TokenAlignmentCollector 实现了 evaluate_mmdocir 所需的 add_query 接口。
    # 初始化基础评估模块的两个全局统计对象后，可以直接复用完整的推理与排序链路。
    def load_image_binary(global_idx: int) -> bytes:
        return bytes(parquet_df.iloc[int(global_idx)]["image_binary"])

    spatial_merge_size = int(model.config.vision_config.spatial_merge_size)
    collector = TokenAlignmentCollector(
        tokenizer=processor.tokenizer,
        image_binary_loader=load_image_binary,
        spatial_merge_size=spatial_merge_size,
        overlay_alpha=args.overlay_alpha,
        collect_incorrect_before_correct=(
            args.collect_incorrect_before_correct or args.all_eligible_queries
        ),
    )
    base_evaluate._inference_timer = base_evaluate.InferenceTimer(model)
    base_evaluate._eval_stats = base_evaluate.EvalStats(
        use_logits=args.use_logits
    )

    log_file = _open_log(args.llm_log_file, args)
    # noinspection PyProtectedMember
    original_consumer = base_evaluate._consume_qi_similarity_stats
    original_input_preparer = base_evaluate.prepare_ranking_inputs
    pending_image_grid_thw: Optional[torch.Tensor] = None

    def prepare_inputs_with_dataset_query_alignment(
        prompt,
        passage_images,
        active_processor,
    ):
        nonlocal pending_image_grid_thw
        ranking_inputs = original_input_preparer(
            prompt,
            passage_images,
            active_processor,
        )

        # Derive the exact query positions from the same tokenized input that will
        # enter the model. The diagnostic model consumes these positions once, so
        # fixed instructions remain available to inference but never enter argmax.
        query_input_positions = find_dataset_query_input_positions(
            prompt=prompt,
            input_ids=ranking_inputs["input_ids"],
            tokenizer=active_processor.tokenizer,
        )
        model.set_token_alignment_query_input_positions(query_input_positions)
        pending_image_grid_thw = torch.as_tensor(
            ranking_inputs["image_grid_thw"]
        ).detach().to(device="cpu", dtype=torch.long).contiguous()
        return ranking_inputs

    def consume_alignment_stats_with_image_grids(
        active_model,
        candidate_original_indices,
    ):
        nonlocal pending_image_grid_thw
        if pending_image_grid_thw is None:
            raise RuntimeError(
                "No processor image grid is available for the current alignment window"
            )
        current_image_grids = pending_image_grid_thw
        pending_image_grid_thw = None
        return _consume_qi_token_alignment_stats(
            active_model,
            candidate_original_indices,
            current_image_grids,
        )

    try:
        # Runtime substitution is local to this process. No line in the normal
        # evaluation source file is edited, and the original function is
        # restored even when evaluation raises an exception.
        base_evaluate._consume_qi_similarity_stats = (
            consume_alignment_stats_with_image_grids
        )
        base_evaluate.prepare_ranking_inputs = (
            prepare_inputs_with_dataset_query_alignment
        )
        results = base_evaluate.evaluate_mmdocir(
            model=model,
            processor=processor,
            first_stage_results=selected_results,
            parquet_df=parquet_df,
            mode="page",
            window_size=args.window_size,
            stride=args.stride,
            use_logits=args.use_logits,
            num_queries=None,
            sample_size=0,
            seed=args.seed,
            log_file=log_file,
            similarity_collector=cast(Any, collector),
        )
    finally:
        base_evaluate._consume_qi_similarity_stats = original_consumer
        base_evaluate.prepare_ranking_inputs = original_input_preparer
        if log_file is not None:
            log_file.close()

    # 主文件沿用原 20 条采样元数据；全量文件记录完整扫描信息。两者共享模型
    # 配置，但候选扩展标记只进入全量文件，避免改变原诊断 JSON 的既有语义。
    common_run_metadata = {
        "model_path": args.model_path,
        "first_stage_file": args.first_stage_file,
        "pages_parquet": args.pages_parquet,
        "window_size": args.window_size,
        "reranking_mode": "first-token logits" if args.use_logits else "generation",
        "qi_early_keep_ratio": args.qi_early_keep_ratio,
        "qi_early_temperature": args.qi_early_temperature,
        "effective_qi_early_text_mode": model.qi_early_text_mode,
        "token_alignment_text_scope": "dataset_query_only",
        "spatial_merge_size": spatial_merge_size,
        "retained_patch_overlay_alpha": args.overlay_alpha,
    }
    sample_run_metadata = {
        **common_run_metadata,
        "requested_queries": len(sampled_results),
        "evaluated_queries": len(sampled_results),
        **sample_selection_metadata,
    }
    if args.collect_incorrect_before_correct and not args.all_eligible_queries:
        sample_run_metadata["collect_incorrect_before_correct"] = True

    if len(collector.query_records) != expected_inference_queries:
        partial_target = all_eligible_output_path or Path(args.output_file)
        partial_metadata = {
            **common_run_metadata,
            "requested_queries": expected_inference_queries,
            "evaluated_queries": len(results),
            **selection_metadata,
        }
        partial_path = collector.save(
            str(partial_target),
            run_metadata=partial_metadata,
            write_images=not args.all_eligible_queries,
        )
        raise RuntimeError(
            f"Expected {expected_inference_queries} exported queries, got "
            f"{len(collector.query_records)}; partial output was saved to "
            f"{partial_path}"
        )

    saved_all_eligible_path = None
    if args.all_eligible_queries:
        all_run_metadata = {
            **common_run_metadata,
            "requested_queries": expected_inference_queries,
            "evaluated_queries": len(results),
            "requested_all_eligible_queries": True,
            "collect_incorrect_before_correct": True,
            **selection_metadata,
        }
        # 先写不含图片路径的全量 JSON；随后对 20 条深拷贝生成图片，不会反向
        # 污染这里的全量记录，也不会为其余查询创建庞大的可视化目录。
        saved_all_eligible_path = collector.save(
            str(all_eligible_output_path),
            run_metadata=all_run_metadata,
            write_images=False,
            total_queries_seen=expected_inference_queries,
        )
        base_query_records = select_base_diagnostic_records(
            collector.query_records,
            sampled_results,
        )
    else:
        base_query_records = collector.query_records

    output_path = collector.save(
        args.output_file,
        run_metadata=sample_run_metadata,
        query_records=base_query_records,
        write_images=True,
        total_queries_seen=len(base_query_records),
        reset_image_output_directory=args.all_eligible_queries,
    )

    output_summary = {
        "output_file": str(output_path),
        "image_output_directory": str(
            output_path.parent / f"{output_path.stem}_images"
        ),
        "queries_exported": len(base_query_records),
        "skipped_queries": dict(collector.skipped_counts),
    }
    if saved_all_eligible_path is not None:
        output_summary.update({
            "all_eligible_output_file": str(saved_all_eligible_path),
            "all_eligible_queries_exported": len(collector.query_records),
            "all_eligible_images_generated": False,
        })
    print(json.dumps(output_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
