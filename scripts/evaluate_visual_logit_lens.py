"""Run late-layer Logit Lens analysis on retained visual tokens.

The script is intentionally separate from ``scripts/evaluate.py``.  It reuses
the established prompt construction, model inference and final ranking logic,
while temporarily substituting a richer diagnostic-statistics consumer within
this process.  No production evaluation function or model method is edited.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, cast

import pandas as pd
import torch
from transformers import AutoProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.qwen3vl_with_qi_early_logit_lens import (  # noqa: E402
    Qwen3VLWithQIEarlyLogitLens,
)
from scripts import evaluate as base_evaluate  # noqa: E402
from utils.visual_logit_lens_analysis import (  # noqa: E402
    LogitLensVocabularyProjector,
    VisualLogitLensCollector,
    select_visual_logit_lens_query_subset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample 20 eligible MMDocIR queries and inspect late-layer vocabulary "
            "projections for QI-Early-retained visual tokens."
        )
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--first_stage_file", required=True)
    parser.add_argument(
        "--pages_parquet",
        default="MMDocIR/dataset/MMDocIR_pages.parquet",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/visual_logit_lens_analysis",
    )
    parser.add_argument(
        "--num_queries",
        type=int,
        default=20,
        help="Stop as soon as this many eligible queries have been found.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--window_size",
        type=int,
        default=20,
        help="Must cover every first-stage candidate in one multimodal forward.",
    )
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument(
        "--use_logits",
        action="store_true",
        help="Use first-token candidate logits rather than full ranking generation.",
    )
    parser.add_argument(
        "--qi_early_keep_ratio",
        type=float,
        default=0.5,
        help="Visual-token retention ratio; the requested analysis default is 0.5.",
    )
    parser.add_argument("--qi_early_temperature", type=float, default=0.1)
    parser.add_argument(
        "--qi_early_text_mode",
        choices=["query_only", "all_before_image"],
        default=None,
    )
    parser.add_argument(
        "--num_logit_lens_layers",
        "--last_n_layers",
        dest="num_logit_lens_layers",
        type=int,
        default=4,
        help=(
            "Number of decoder blocks sampled at spaced intervals in the late "
            "model depth. --last_n_layers is retained as a compatibility alias."
        ),
    )
    parser.add_argument(
        "--logit_lens_start_ratio",
        type=float,
        default=0.6,
        help=(
            "Earliest normalized decoder depth eligible for spaced sampling; "
            "default 0.6 samples from the final 40 percent of the model."
        ),
    )
    parser.add_argument(
        "--top_k_vocab",
        type=int,
        default=5,
        help="Vocabulary tokens retained per visual token and layer.",
    )
    parser.add_argument(
        "--projection_chunk_size",
        type=int,
        default=128,
        help="Visual states projected at once; lower this if GPU memory is tight.",
    )
    parser.add_argument(
        "--summary_top_tokens",
        type=int,
        default=20,
        help="Most frequent top-1 vocabulary tokens summarized for each layer.",
    )
    parser.add_argument(
        "--overlay_alpha",
        type=int,
        default=72,
        help="Green retained-patch fill opacity in [0, 255].",
    )
    parser.add_argument("--llm_log_file", default=None)
    return parser.parse_args()


def _find_logit_lens_owner(model) -> Qwen3VLWithQIEarlyLogitLens:
    """Locate the diagnostic model through common inference wrappers."""
    pending = [model]
    visited = set()
    while pending:
        owner = pending.pop(0)
        if owner is None or id(owner) in visited:
            continue
        visited.add(id(owner))
        if hasattr(owner, "pop_last_logit_lens_capture"):
            return cast(Qwen3VLWithQIEarlyLogitLens, owner)
        pending.extend([
            getattr(owner, "module", None),
            getattr(owner, "base_model", None),
        ])
    raise RuntimeError("Qwen3VLWithQIEarlyLogitLens was not found")


def _consume_visual_logit_lens_stats(
    model,
    candidate_original_indices: Sequence[int],
) -> List[Dict[str, Any]]:
    """Join pruning scores and captured late-layer states, then move them to CPU."""
    pruner = base_evaluate._find_qi_early_pruner(model)
    raw_stats = pruner.last_similarity_stats
    pruner.last_similarity_stats = None

    lens_owner = _find_logit_lens_owner(model)
    capture = lens_owner.pop_last_logit_lens_capture()
    if raw_stats is None:
        raise RuntimeError("QI-Early did not expose visual-token pruning statistics")
    if capture is None or not capture.get("capture_complete", False):
        raise RuntimeError("The multimodal forward produced no complete Logit Lens capture")

    num_images = len(candidate_original_indices)
    if len(raw_stats) != num_images:
        raise RuntimeError(
            f"Pruning statistics contain {len(raw_stats)} images, expected {num_images}"
        )

    grids = torch.as_tensor(capture["image_grid_thw"]).detach().cpu().long()
    selected_by_image = capture["selected_indices_by_image"]
    positions_by_image = capture["visual_sequence_positions_by_image"]
    kept_counts = [int(value) for value in capture["pruned_tokens_per_image"]]
    if not (
        grids.shape[0]
        == len(selected_by_image)
        == len(positions_by_image)
        == len(kept_counts)
        == num_images
    ):
        raise RuntimeError("Logit Lens image metadata has inconsistent lengths")

    # Hidden states are captured as one concatenated visual sequence per layer.
    # Split every layer with the same per-image counts before attaching candidate
    # positions, preserving both image order and spatial order inside each image.
    hidden_splits_by_layer: Dict[int, Sequence[torch.Tensor]] = {}
    for layer_index, hidden_states in capture["layer_hidden_states"].items():
        hidden_states = torch.as_tensor(hidden_states).detach().cpu().contiguous()
        if hidden_states.shape[0] != sum(kept_counts):
            raise RuntimeError(
                f"Layer {layer_index} visual-state count does not match kept tokens"
            )
        hidden_splits_by_layer[int(layer_index)] = torch.split(
            hidden_states, kept_counts, dim=0
        )

    copied_stats = []
    for image_index, (candidate_pos, image_stats) in enumerate(
        zip(candidate_original_indices, raw_stats)
    ):
        scores = image_stats["scores"].detach().to(
            device="cpu", dtype=torch.float32
        ).contiguous()
        kept_mask = image_stats["kept_mask"].detach().to(
            device="cpu", dtype=torch.bool
        ).contiguous()
        selected_indices = torch.as_tensor(selected_by_image[image_index]).long()
        selected_indices = selected_indices.cpu().contiguous()
        if not torch.equal(selected_indices, torch.where(kept_mask)[0]):
            raise RuntimeError(
                f"Candidate {candidate_pos} pruning mask and captured indices differ"
            )

        copied_stats.append({
            "candidate_pos": int(candidate_pos),
            "scores": scores,
            "kept_mask": kept_mask,
            "selected_indices": selected_indices,
            "visual_sequence_positions": torch.as_tensor(
                positions_by_image[image_index]
            ).long().cpu().contiguous(),
            "threshold": float(
                image_stats["threshold"].detach().float().cpu().item()
            ),
            "num_original_tokens": int(image_stats["num_original_tokens"]),
            "num_kept_tokens": int(image_stats["num_kept_tokens"]),
            "image_grid_thw": grids[image_index].tolist(),
            "layer_hidden_states": {
                layer_index: layer_splits[image_index]
                for layer_index, layer_splits in hidden_splits_by_layer.items()
            },
        })

    return copied_stats


def _open_log(path: Optional[str], args: argparse.Namespace):
    if path is None:
        return None
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    handle.write("=" * 80 + "\n")
    handle.write("VISUAL TOKEN LOGIT LENS DIAGNOSTIC - ZipRerank\n")
    handle.write("=" * 80 + "\n")
    handle.write(f"Model: {args.model_path}\n")
    handle.write(f"Requested queries: {args.num_queries}\n")
    handle.write(f"Keep ratio: {args.qi_early_keep_ratio}\n")
    handle.write(f"Spaced late layers: {args.num_logit_lens_layers}\n")
    handle.write(f"Late-layer start ratio: {args.logit_lens_start_ratio}\n")
    handle.flush()
    return handle


def main() -> None:
    args = parse_args()
    if args.num_queries <= 0:
        raise ValueError("--num_queries must be positive")
    if not 0.0 < args.qi_early_keep_ratio <= 1.0:
        raise ValueError("--qi_early_keep_ratio must be in (0, 1]")
    if not 0 <= args.overlay_alpha <= 255:
        raise ValueError("--overlay_alpha must be in [0, 255]")
    if args.num_logit_lens_layers <= 0:
        raise ValueError("--num_logit_lens_layers must be positive")
    if not 0.5 <= args.logit_lens_start_ratio < 1.0:
        raise ValueError("--logit_lens_start_ratio must be in [0.5, 1.0)")

    print("Loading MMDocIR page parquet...")
    parquet_df = pd.read_parquet(args.pages_parquet)
    print(f"Loaded {len(parquet_df)} pages")

    print("Loading first-stage retrieval results...")
    first_stage_results = base_evaluate.load_first_stage_results(
        args.first_stage_file
    )
    selected_results, selection_metadata = (
        select_visual_logit_lens_query_subset(
            first_stage_results,
            num_queries=args.num_queries,
            seed=args.seed,
            num_incorrect_candidates=3,
        )
    )
    max_candidates = max(
        len(item["top_k_global_indices"]) for item in selected_results.values()
    )
    if max_candidates > args.window_size:
        raise ValueError(
            "Visual Logit Lens analysis requires one forward per query, but the "
            f"selected Top-K contains up to {max_candidates} candidates and "
            f"--window_size={args.window_size}"
        )
    print(
        f"Found {len(selected_results)} eligible queries after inspecting "
        f"{selection_metadata['queries_inspected_before_early_stop']} entries "
        f"(seed={args.seed})"
    )

    print(f"Loading diagnostic model from {args.model_path}...")
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    model = Qwen3VLWithQIEarlyLogitLens.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.set_qi_early_enabled(True)
    model.set_qi_early_keep_ratio(args.qi_early_keep_ratio)
    model.qi_early_pruner.temperature = args.qi_early_temperature
    if args.qi_early_text_mode is not None:
        model.set_qi_early_text_mode(args.qi_early_text_mode)
    layer_indices = model.configure_spaced_late_logit_lens(
        num_selected_layers=args.num_logit_lens_layers,
        start_ratio=args.logit_lens_start_ratio,
    )
    model.qi_early_pruner.set_collect_similarity_stats(True)
    model.eval()
    total_decoder_layers = len(model.model.language_model.layers)
    print(
        "Logit Lens decoder layers (1-based): "
        + ", ".join(str(layer_index + 1) for layer_index in layer_indices)
    )

    projector = LogitLensVocabularyProjector(
        model,
        top_k=args.top_k_vocab,
        chunk_size=args.projection_chunk_size,
    )

    # The loader keeps image ownership in the parquet DataFrame.  Candidate
    # images are decoded only for the four selected pages while building records
    # and again when the final source/overlay files are written.
    def load_image_binary(global_idx: int) -> bytes:
        return bytes(parquet_df.iloc[int(global_idx)]["image_binary"])

    spatial_merge_size = int(model.config.vision_config.spatial_merge_size)
    collector = VisualLogitLensCollector(
        tokenizer=processor.tokenizer,
        projector=projector,
        image_binary_loader=load_image_binary,
        spatial_merge_size=spatial_merge_size,
        summary_top_tokens=args.summary_top_tokens,
        overlay_alpha=args.overlay_alpha,
    )

    base_evaluate._inference_timer = base_evaluate.InferenceTimer(model)
    base_evaluate._eval_stats = base_evaluate.EvalStats(
        use_logits=args.use_logits
    )
    log_file = _open_log(args.llm_log_file, args)

    # Reuse the stable evaluation loop, but replace its post-forward consumer
    # only inside this process.  The original function is restored even if model
    # inference, vocabulary projection or image processing raises an exception.
    original_consumer = base_evaluate._consume_qi_similarity_stats
    try:
        base_evaluate._consume_qi_similarity_stats = (
            _consume_visual_logit_lens_stats
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
        model.remove_logit_lens_hooks()
        if log_file is not None:
            log_file.close()

    run_metadata: Mapping[str, Any] = {
        "model_path": args.model_path,
        "first_stage_file": args.first_stage_file,
        "pages_parquet": args.pages_parquet,
        "requested_queries": args.num_queries,
        "evaluated_queries": len(results),
        "window_size": args.window_size,
        "reranking_mode": (
            "first-token logits" if args.use_logits else "generation"
        ),
        "qi_early_keep_ratio": args.qi_early_keep_ratio,
        "qi_early_temperature": args.qi_early_temperature,
        "effective_qi_early_text_mode": model.qi_early_text_mode,
        "logit_lens_layer_indices_zero_based": list(layer_indices),
        "logit_lens_layer_numbers_one_based": [
            layer_index + 1 for layer_index in layer_indices
        ],
        "logit_lens_layer_depth_fractions": [
            (layer_index + 1) / total_decoder_layers
            for layer_index in layer_indices
        ],
        "logit_lens_layer_selection": (
            "evenly spaced from the configured late-depth start through the "
            "final decoder block"
        ),
        "num_logit_lens_layers": args.num_logit_lens_layers,
        "logit_lens_start_ratio": args.logit_lens_start_ratio,
        "total_decoder_layers": total_decoder_layers,
        "top_k_vocabulary_per_visual_token": args.top_k_vocab,
        "projection_chunk_size": args.projection_chunk_size,
        "spatial_merge_size": spatial_merge_size,
        **selection_metadata,
    }
    output_paths = collector.save(args.output_dir, run_metadata=run_metadata)

    if len(collector.query_records) != args.num_queries:
        raise RuntimeError(
            f"Expected {args.num_queries} exported queries, got "
            f"{len(collector.query_records)}; partial output is in {args.output_dir}"
        )

    print(json.dumps({
        **output_paths,
        "queries_exported": len(collector.query_records),
        "queries_inspected": selection_metadata[
            "queries_inspected_before_early_stop"
        ],
        "layer_indices_zero_based": list(layer_indices),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
