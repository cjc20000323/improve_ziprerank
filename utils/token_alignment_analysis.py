"""Export visual-token to text-token maximum-similarity alignments.

The normal reranker only needs the maximum similarity value for each visual
token.  This diagnostic module additionally consumes the *argmax text-token
index* recorded by the diagnostic QI-Early model and converts it into readable
tokenizer information.

The JSON intentionally contains both:

* ``vocab_token``: the exact token string stored in the model vocabulary;
* ``decoded_text``: the result of decoding that single token ID.

These are tokenizer subword units rather than necessarily complete natural
language words.  Keeping both representations makes whitespace markers and
special tokens visible while still providing a human-readable form.
"""

from __future__ import annotations

import colorsys
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, DefaultDict, Dict, List, Mapping, Sequence, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont

from .visual_logit_lens_analysis import (
    draw_kept_patch_overlay,
    load_rgb_image,
    merged_visual_grid_shape,
    visual_token_patch_geometry,
)


_DATASET_QUERY_PREFIX = "Search Query: "
_DATASET_QUERY_SUFFIX = (
    "\n\nRank the passages above based on their relevance to the search query."
)


def find_dataset_query_input_positions(
    prompt: str,
    input_ids: Any,
    tokenizer,
) -> torch.Tensor:
    """Map only the dataset query text in ``prompt`` to full input positions."""
    query_prefix_start = prompt.find(_DATASET_QUERY_PREFIX)
    query_suffix_start = prompt.rfind(_DATASET_QUERY_SUFFIX)
    if query_prefix_start < 0 or query_suffix_start < 0:
        raise ValueError(
            "The ranking prompt does not contain the expected dataset-query boundaries"
        )

    query_start = query_prefix_start + len(_DATASET_QUERY_PREFIX)
    query_end = query_suffix_start
    if query_end <= query_start or not prompt[query_start:query_end].strip():
        raise ValueError("The dataset query extracted from the ranking prompt is empty")

    # Offset mappings give a character-level boundary inside the original prompt.
    # This avoids treating fixed strings such as "Search Query:" or the ranking
    # instructions as query tokens merely because they precede the first image.
    try:
        prompt_encoding = tokenizer(
            prompt,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "The tokenizer must support return_offsets_mapping so dataset-query "
            "tokens can be isolated strictly"
        ) from exc

    if "input_ids" not in prompt_encoding or "offset_mapping" not in prompt_encoding:
        raise RuntimeError(
            "Tokenizer output is missing input_ids or offset_mapping for the prompt"
        )

    prompt_token_ids = torch.as_tensor(
        prompt_encoding["input_ids"], dtype=torch.long
    )
    prompt_offsets = torch.as_tensor(
        prompt_encoding["offset_mapping"], dtype=torch.long
    )
    if prompt_token_ids.ndim == 2 and prompt_token_ids.shape[0] == 1:
        prompt_token_ids = prompt_token_ids[0]
    if prompt_offsets.ndim == 3 and prompt_offsets.shape[0] == 1:
        prompt_offsets = prompt_offsets[0]
    if prompt_token_ids.ndim != 1:
        raise ValueError(
            "Prompt tokenizer input_ids must be one-dimensional after batch removal"
        )
    if prompt_offsets.ndim != 2 or prompt_offsets.shape[1] != 2:
        raise ValueError(
            "Prompt offset_mapping must have shape (num_prompt_tokens, 2)"
        )
    if prompt_offsets.shape[0] != prompt_token_ids.numel():
        raise ValueError("Prompt token IDs and offset mappings have different lengths")

    # Include every model token that overlaps at least one query character. A token
    # may also cover boundary whitespace, but it must never contain a non-whitespace
    # character from the hard-coded prefix or suffix.
    query_prompt_indices: List[int] = []
    for token_index, (offset_start, offset_end) in enumerate(prompt_offsets.tolist()):
        if offset_end <= offset_start:
            continue
        if offset_end <= query_start or offset_start >= query_end:
            continue

        outside_query = (
            prompt[offset_start:min(offset_end, query_start)]
            + prompt[max(offset_start, query_end):offset_end]
        )
        if outside_query.strip():
            raise RuntimeError(
                "A tokenizer token crosses the dataset-query boundary and also "
                "contains hard-coded prompt text; strict alignment is impossible"
            )
        query_prompt_indices.append(token_index)

    if not query_prompt_indices:
        raise RuntimeError("No tokenizer tokens overlap the dataset query text")

    # Match a local context window, rather than the whole prompt, against the final
    # multimodal input. This is robust to chat-template tokens before the prompt and
    # still anchors the query with fixed text on both sides so repeated query words
    # elsewhere cannot silently select the wrong occurrence.
    context_start = max(0, query_prompt_indices[0] - 8)
    context_end = min(
        prompt_token_ids.numel(),
        query_prompt_indices[-1] + 9,
    )
    context_ids = prompt_token_ids[context_start:context_end]
    full_input_ids = torch.as_tensor(input_ids, dtype=torch.long)
    if full_input_ids.ndim == 2 and full_input_ids.shape[0] == 1:
        full_input_ids = full_input_ids[0]
    if full_input_ids.ndim != 1:
        raise ValueError("Full multimodal input_ids must contain one sequence")
    if context_ids.numel() > full_input_ids.numel():
        raise RuntimeError("Prompt context is longer than the full multimodal input")

    context_matches: List[int] = []
    candidate_starts = torch.where(full_input_ids == context_ids[0])[0].tolist()
    for candidate_start in candidate_starts:
        candidate_end = candidate_start + context_ids.numel()
        if candidate_end > full_input_ids.numel():
            continue
        if torch.equal(full_input_ids[candidate_start:candidate_end], context_ids):
            context_matches.append(candidate_start)
    if len(context_matches) != 1:
        raise RuntimeError(
            "Could not locate one unique dataset-query context in the prepared "
            f"multimodal input; matches={len(context_matches)}"
        )

    prompt_to_input_offset = context_matches[0] - context_start
    query_input_positions = torch.tensor(
        [prompt_to_input_offset + index for index in query_prompt_indices],
        dtype=torch.long,
    )
    if (
        int(query_input_positions.min()) < 0
        or int(query_input_positions.max()) >= full_input_ids.numel()
    ):
        raise RuntimeError("A mapped dataset-query token falls outside the model input")
    if not torch.equal(
        full_input_ids[query_input_positions],
        prompt_token_ids[query_prompt_indices],
    ):
        raise RuntimeError("Mapped dataset-query token IDs do not match the prompt")
    return query_input_positions


def _query_token_color(query_sequence_index: int) -> Tuple[int, int, int]:
    """Return a stable, high-contrast color for one query-token index."""
    hue = (0.08 + query_sequence_index * 0.61803398875) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.68, 0.92)
    return round(red * 255), round(green * 255), round(blue * 255)


def _load_annotation_font(size: int):
    """Load a readable font on common Linux/Windows hosts with a safe fallback."""
    font_candidates = (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "DejaVuSans.ttf",
    )
    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _legend_token_text(token: Mapping[str, Any], max_characters: int = 72) -> str:
    decoded_text = json.dumps(
        str(token.get("decoded_text") or ""), ensure_ascii=False
    )
    vocab_token = json.dumps(
        str(token.get("vocab_token") or ""), ensure_ascii=False
    )
    text = f"text={decoded_text}  vocab={vocab_token}"
    if len(text) > max_characters:
        return text[: max_characters - 1] + "…"
    return text


def draw_query_token_patch_map(
    image: Image.Image,
    visual_tokens: Sequence[Mapping[str, Any]],
    query_tokens: Sequence[Mapping[str, Any]],
    image_grid_thw: Sequence[int],
    spatial_merge_size: int,
    fill_alpha: int = 72,
    show_similarity: bool = False,
) -> Image.Image:
    """Color retained patches by their matched query token and append a legend."""
    if not 0 <= fill_alpha <= 255:
        raise ValueError("fill_alpha must be in [0, 255]")

    grid_t, grid_h, grid_w = merged_visual_grid_shape(
        image_grid_thw,
        spatial_merge_size,
    )
    if grid_t != 1:
        raise ValueError("Static query-token patch maps require grid_t == 1")
    expected_token_count = grid_t * grid_h * grid_w
    visual_tokens_by_index = {
        int(token["visual_token_index"]): token for token in visual_tokens
    }
    if set(visual_tokens_by_index) != set(range(expected_token_count)):
        raise ValueError(
            "visual_tokens must cover every pre-pruning merged-grid position"
        )

    query_tokens_by_index = {
        int(token["query_sequence_index"]): token for token in query_tokens
    }
    retained_tokens = [
        visual_tokens_by_index[index]
        for index in range(expected_token_count)
        if bool(visual_tokens_by_index[index]["kept"])
    ]
    if not retained_tokens:
        raise ValueError("A query-token patch map requires at least one retained token")

    used_query_indices = sorted({
        int(token["matched_query_sequence_index"])
        for token in retained_tokens
    })
    missing_query_indices = [
        index for index in used_query_indices if index not in query_tokens_by_index
    ]
    if missing_query_indices:
        raise ValueError(
            "Retained patches reference missing query-token indices: "
            f"{missing_query_indices}"
        )

    # The detail view is enlarged until a typical cell can hold both Q-index and
    # similarity. A maximum dimension prevents pathological inputs from producing
    # an unmanageably large bitmap; the color and legend remain usable at that cap.
    source = image.convert("RGB")
    source_cell_size = min(source.width / grid_w, source.height / grid_h)
    target_cell_size = 52.0 if show_similarity else 30.0
    requested_scale = max(1.0, target_cell_size / source_cell_size)
    maximum_scale = min(4.0, 6000.0 / max(source.size))
    scale = max(1.0, min(requested_scale, maximum_scale))
    display_size = (
        max(1, round(source.width * scale)),
        max(1, round(source.height * scale)),
    )
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    display_image = source.resize(display_size, resample=resampling).convert("RGBA")
    overlay = Image.new("RGBA", display_size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay, "RGBA")

    cell_width = display_image.width / grid_w
    cell_height = display_image.height / grid_h
    minimum_cell_size = min(cell_width, cell_height)
    label_font_size = max(9, min(22, round(minimum_cell_size * 0.30)))
    label_font = _load_annotation_font(label_font_size)

    # Faint lines expose the complete merged-token topology. Retained cells are
    # then colored by query_sequence_index, so the same query token has the same
    # hue in every candidate image belonging to this query.
    for column in range(1, grid_w):
        x = round(column * display_image.width / grid_w)
        overlay_draw.line(
            [(x, 0), (x, display_image.height)],
            fill=(255, 255, 255, 55),
            width=1,
        )
    for row in range(1, grid_h):
        y = round(row * display_image.height / grid_h)
        overlay_draw.line(
            [(0, y), (display_image.width, y)],
            fill=(255, 255, 255, 55),
            width=1,
        )

    match_counts: Counter[int] = Counter()
    for token in retained_tokens:
        visual_token_index = int(token["visual_token_index"])
        query_token_index = int(token["matched_query_sequence_index"])
        match_counts[query_token_index] += 1
        red, green, blue = _query_token_color(query_token_index)
        geometry = visual_token_patch_geometry(
            visual_token_index,
            image_grid_thw,
            spatial_merge_size,
            display_image.size,
        )
        x0, y0, x1, y1 = geometry["source_image_pixel_box_xyxy"]
        overlay_draw.rectangle(
            [x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)],
            fill=(red, green, blue, fill_alpha),
            outline=(20, 20, 20, 230),
            width=max(1, round(minimum_cell_size * 0.04)),
        )

        label = f"Q{query_token_index}"
        if show_similarity:
            label += f"\n{float(token['max_similarity']):.2f}"
        label_box = overlay_draw.multiline_textbbox(
            (0, 0),
            label,
            font=label_font,
            align="center",
            spacing=0,
            stroke_width=1,
        )
        label_width = label_box[2] - label_box[0]
        label_height = label_box[3] - label_box[1]
        if label_width <= x1 - x0 - 2 and label_height <= y1 - y0 - 2:
            overlay_draw.multiline_text(
                ((x0 + x1) / 2, (y0 + y1) / 2),
                label,
                fill=(255, 255, 255, 255),
                font=label_font,
                anchor="mm",
                align="center",
                spacing=0,
                stroke_width=2,
                stroke_fill=(0, 0, 0, 255),
            )

    annotated = Image.alpha_composite(display_image, overlay).convert("RGB")

    # The legend lists only query tokens that actually won at least one retained
    # patch. Exact one-token decoding and vocabulary pieces remain visible, while
    # the compact Q-index keeps text inside image cells short and collision-free.
    legend_font_size = 16 if not show_similarity else 18
    legend_font = _load_annotation_font(legend_font_size)
    legend_header_font = _load_annotation_font(legend_font_size + 2)
    legend_entries = [
        (
            query_token_index,
            f"Q{query_token_index}  "
            f"({match_counts[query_token_index]} patches)  "
            f"{_legend_token_text(query_tokens_by_index[query_token_index])}",
        )
        for query_token_index in used_query_indices
    ]
    measurement_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    measured_widths = [
        measurement_draw.textbbox((0, 0), text, font=legend_font)[2]
        for _, text in legend_entries
    ]
    legend_padding = 18
    swatch_size = legend_font_size + 2
    legend_width = min(
        900,
        max(380, max(measured_widths, default=0) + swatch_size + 3 * legend_padding),
    )
    line_height = max(swatch_size, legend_font_size + 6) + 7
    legend_height = (
        2 * legend_padding + legend_font_size + 8 + len(legend_entries) * line_height
    )
    canvas = Image.new(
        "RGB",
        (annotated.width + legend_width, max(annotated.height, legend_height)),
        color=(247, 247, 247),
    )
    canvas.paste(annotated, (0, 0))
    legend_draw = ImageDraw.Draw(canvas)
    legend_draw.text(
        (annotated.width + legend_padding, legend_padding),
        "Matched dataset-query tokens",
        fill=(20, 20, 20),
        font=legend_header_font,
    )
    legend_y = legend_padding + legend_font_size + 12
    for query_token_index, legend_text in legend_entries:
        color = _query_token_color(query_token_index)
        swatch_x = annotated.width + legend_padding
        legend_draw.rectangle(
            [
                swatch_x,
                legend_y,
                swatch_x + swatch_size,
                legend_y + swatch_size,
            ],
            fill=color,
            outline=(20, 20, 20),
            width=1,
        )
        legend_draw.text(
            (swatch_x + swatch_size + 10, legend_y),
            legend_text,
            fill=(20, 20, 20),
            font=legend_font,
        )
        legend_y += line_height
    return canvas


def select_eligible_query_subset(
    first_stage_results: Mapping[Any, Mapping[str, Any]],
    num_queries: int = 20,
    seed: int = 42,
) -> Tuple[Dict[Any, Mapping[str, Any]], Dict[str, Any]]:
    """Randomly select queries whose Top-K contains both a GT and a non-GT.

    A correct/incorrect candidate pair cannot be constructed when first-stage
    retrieval misses every ground-truth page.  Filtering before model inference
    guarantees that the requested number of exported queries can each contain
    the two requested candidates.

    Query keys are shuffled with a local seeded generator. Eligibility is then
    checked in that order and stops immediately after ``num_queries`` matches,
    avoiding a complete eligibility scan when enough examples are found early.
    """
    if num_queries <= 0:
        raise ValueError(f"num_queries must be positive, got {num_queries}")

    eligible_keys: List[Any] = []
    skipped: DefaultDict[str, int] = defaultdict(int)
    candidate_keys = list(first_stage_results.keys())
    random.Random(seed).shuffle(candidate_keys)
    inspected_queries = 0

    # page_id 是文档内部的局部页号，而第一阶段结果保存的是数据集全局索引。
    # 先用 start_idx 转到同一坐标系，才能正确判断 Top-K 中是否同时存在正负例。
    # This randomized traversal is equivalent to uniform sampling without
    # replacement from the eligible set, but it can stop before scanning all data.
    for key in candidate_keys:
        inspected_queries += 1
        item = first_stage_results[key]
        gt_page_ids = {int(page_id) for page_id in item.get("page_id", [])}
        top_k_global_indices = list(item.get("top_k_global_indices", []))

        if not gt_page_ids:
            skipped["no_ground_truth_page"] += 1
            continue
        if not top_k_global_indices:
            skipped["empty_first_stage_candidates"] += 1
            continue

        start_idx = int(item["start_idx"])
        end_idx = int(item["end_idx"])
        local_page_ids = [int(global_idx) - start_idx for global_idx in top_k_global_indices]

        if any(not 0 <= page_id <= end_idx - start_idx for page_id in local_page_ids):
            skipped["candidate_outside_document_range"] += 1
            continue
        if not any(page_id in gt_page_ids for page_id in local_page_ids):
            skipped["correct_candidate_missing_from_topk"] += 1
            continue
        if not any(page_id not in gt_page_ids for page_id in local_page_ids):
            skipped["incorrect_candidate_missing_from_topk"] += 1
            continue

        eligible_keys.append(key)
        if len(eligible_keys) == num_queries:
            break

    # 在推理开始前一次性验证可用样本数，避免模型已经运行很久后才发现无法凑齐
    # 指定数量；局部 Random 实例不会改变调用方可能依赖的全局随机状态。
    if len(eligible_keys) < num_queries:
        raise ValueError(
            f"Requested {num_queries} analyzable queries, but only "
            f"{len(eligible_keys)} contain both GT and non-GT candidates"
        )

    selected = {key: first_stage_results[key] for key in eligible_keys}
    selection_metadata = {
        "selection_method": (
            "seeded random query permutation; first N eligible queries with early stop"
        ),
        "selection_seed": seed,
        "total_first_stage_queries": len(first_stage_results),
        "queries_inspected_before_early_stop": inspected_queries,
        "queries_not_inspected": len(first_stage_results) - inspected_queries,
        "eligible_queries_encountered": len(eligible_keys),
        "selected_queries": len(selected),
        "required_correct_candidates": 1,
        "required_incorrect_candidates": 1,
        "prefilter_skipped_queries": dict(sorted(skipped.items())),
        "eligibility_rule": (
            "First-stage Top-K must contain at least one ground-truth page and "
            "at least one non-ground-truth page."
        ),
    }
    return selected, selection_metadata


class TokenAlignmentCollector:
    """Collect two candidates per query and serialize token-level alignments."""

    def __init__(
        self,
        tokenizer,
        image_binary_loader: Callable[[int], bytes],
        spatial_merge_size: int,
        overlay_alpha: int = 72,
    ) -> None:
        if spatial_merge_size <= 0:
            raise ValueError("spatial_merge_size must be positive")
        if not 0 <= overlay_alpha <= 255:
            raise ValueError("overlay_alpha must be in [0, 255]")
        self.tokenizer = tokenizer
        self.image_binary_loader = image_binary_loader
        self.spatial_merge_size = int(spatial_merge_size)
        self.overlay_alpha = int(overlay_alpha)
        self.query_records: List[Dict[str, Any]] = []
        self.total_queries_seen = 0
        self.skipped_counts: DefaultDict[str, int] = defaultdict(int)
        self._special_token_ids = set(getattr(tokenizer, "all_special_ids", []))

    def _describe_text_token(
        self,
        query_sequence_index: int,
        input_sequence_position: int,
        token_id: int,
    ) -> Dict[str, Any]:
        """Return both the exact vocabulary piece and its one-token decoding."""

        # vocabulary piece 忠实保留分词器内部的子词和空格标记，decoded_text 则
        # 更适合人工阅读。二者同时输出，避免仅解码后无法区分特殊 token 或边界。
        vocab_token = self.tokenizer.convert_ids_to_tokens(token_id)
        decoded_text = self.tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return {
            "query_sequence_index": query_sequence_index,
            "input_sequence_position": input_sequence_position,
            "token_id": token_id,
            "vocab_token": None if vocab_token is None else str(vocab_token),
            "decoded_text": decoded_text,
            "is_special_token": token_id in self._special_token_ids,
        }

    @staticmethod
    def _as_cpu_1d_tensor(value: Any, dtype: torch.dtype, name: str) -> torch.Tensor:
        # collector 只做诊断汇总，不应继续持有计算图或 GPU 内存。这里也统一形状
        # 和 dtype，使后续长度、索引范围及序列化检查具有确定的行为。
        tensor = torch.as_tensor(value).detach().to(device="cpu", dtype=dtype).contiguous()
        if tensor.ndim != 1:
            raise ValueError(f"{name} must be 1-D, got shape={tuple(tensor.shape)}")
        return tensor

    def _validate_shared_query_metadata(
        self,
        candidate_stats: Sequence[Mapping[str, Any]],
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Verify every image in a window used the same QI text-token sequence."""

        # 一次多图 forward 只构造一份查询 embedding，因此首个候选的元数据可以
        # 作为基准；其余候选必须逐项一致，否则不能共用同一个 argmax 下标空间。
        first = candidate_stats[0]
        query_token_ids = self._as_cpu_1d_tensor(
            first["query_token_ids"], torch.long, "query_token_ids"
        )
        query_input_positions = self._as_cpu_1d_tensor(
            first["query_input_positions"], torch.long, "query_input_positions"
        )
        text_extraction_mode = str(first["text_extraction_mode"])

        if query_token_ids.numel() == 0:
            raise ValueError("QI-Early text-token sequence must not be empty")
        if query_token_ids.numel() != query_input_positions.numel():
            raise ValueError(
                "query_token_ids/query_input_positions length mismatch: "
                f"{query_token_ids.numel()} vs {query_input_positions.numel()}"
            )

        for stats in candidate_stats[1:]:
            other_ids = self._as_cpu_1d_tensor(
                stats["query_token_ids"], torch.long, "query_token_ids"
            )
            other_positions = self._as_cpu_1d_tensor(
                stats["query_input_positions"], torch.long, "query_input_positions"
            )
            if (
                str(stats["text_extraction_mode"]) != text_extraction_mode
                or not torch.equal(other_ids, query_token_ids)
                or not torch.equal(other_positions, query_input_positions)
            ):
                raise ValueError("Candidates in one query used different text-token sequences")

        # The collector rejects legacy broad-prompt metadata instead of silently
        # exporting it under a query label. This makes the scope guarantee explicit
        # even when the collector is called outside the standalone entry point.
        if text_extraction_mode != "dataset_query_only":
            raise ValueError(
                "Token alignment requires text_extraction_mode=dataset_query_only, "
                f"got {text_extraction_mode!r}"
            )

        return query_token_ids, query_input_positions, text_extraction_mode

    def _build_candidate_record(
        self,
        *,
        role: str,
        candidate_pos: int,
        final_rank: int,
        global_idx: int,
        local_page_id: int,
        stats: Mapping[str, Any],
        query_tokens: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        scores = self._as_cpu_1d_tensor(
            stats["query_max_similarities"],
            torch.float32,
            "query_max_similarities",
        )
        pruning_scores = self._as_cpu_1d_tensor(
            stats["scores"], torch.float32, "scores"
        )
        kept_mask = self._as_cpu_1d_tensor(
            stats["kept_mask"], torch.bool, "kept_mask"
        )
        max_text_indices = self._as_cpu_1d_tensor(
            stats["max_text_token_indices"],
            torch.long,
            "max_text_token_indices",
        )

        # image_grid_thw describes the vision encoder grid before Qwen's spatial
        # merger. Converting it here lets us validate that every pre-pruning token
        # has exactly one cell in the merged grid before any image is written.
        image_grid_thw = [int(value) for value in stats["image_grid_thw"]]
        grid_t, merged_grid_h, merged_grid_w = merged_visual_grid_shape(
            image_grid_thw,
            self.spatial_merge_size,
        )
        expected_visual_tokens = grid_t * merged_grid_h * merged_grid_w

        # 三个一维向量按剪枝前视觉 token 顺序严格对齐：scores[v] 是 token v 的
        # 最大相似度，kept_mask[v] 表示是否保留，max_text_indices[v] 是命中的
        # QI 文本序列下标。任一长度不一致都会破坏逐 token 记录的语义。
        if not (scores.numel() == kept_mask.numel() == max_text_indices.numel()):
            raise ValueError(
                "Visual-token statistics length mismatch: "
                f"scores={scores.numel()}, kept={kept_mask.numel()}, "
                f"argmax={max_text_indices.numel()}"
            )
        if pruning_scores.numel() != scores.numel():
            raise ValueError(
                "Pruning/query-only score length mismatch: "
                f"pruning={pruning_scores.numel()}, query_only={scores.numel()}"
            )
        if scores.numel() != expected_visual_tokens:
            raise ValueError(
                "Visual-token count does not match the merged image grid: "
                f"tokens={scores.numel()}, grid="
                f"{grid_t}x{merged_grid_h}x{merged_grid_w}"
            )
        if scores.numel() == 0:
            raise ValueError("Candidate contains no visual tokens")
        if not torch.isfinite(scores).all():
            raise ValueError("Non-finite token similarity encountered")
        if not torch.isfinite(pruning_scores).all():
            raise ValueError("Non-finite pruning similarity encountered")
        if int(max_text_indices.min()) < 0 or int(max_text_indices.max()) >= len(query_tokens):
            raise ValueError(
                "A max-text-token index falls outside the extracted QI text sequence"
            )

        visual_tokens = []
        all_match_counts: Counter[int] = Counter()
        kept_match_counts: Counter[int] = Counter()
        pruned_match_counts: Counter[int] = Counter()
        tokens_per_frame = merged_grid_h * merged_grid_w

        # 对每个视觉 token 只做一次 argmax 回查，同时累计文本 token 的吸附次数。
        # 详细列表用于逐 token 排查，三个 Counter 则用于构造更紧凑的聚合视图。
        for visual_token_index in range(scores.numel()):
            matched_index = int(max_text_indices[visual_token_index])
            matched_token = query_tokens[matched_index]
            is_kept = bool(kept_mask[visual_token_index])
            temporal_index = visual_token_index // tokens_per_frame
            spatial_index = visual_token_index % tokens_per_frame

            all_match_counts[matched_index] += 1
            if is_kept:
                kept_match_counts[matched_index] += 1
            else:
                pruned_match_counts[matched_index] += 1

            visual_tokens.append({
                "visual_token_index": visual_token_index,
                "max_similarity": float(scores[visual_token_index]),
                "pruning_similarity": float(pruning_scores[visual_token_index]),
                "kept": is_kept,
                "patch_grid_position_trc": [
                    temporal_index,
                    spatial_index // merged_grid_w,
                    spatial_index % merged_grid_w,
                ],
                "matched_query_sequence_index": matched_index,
                "matched_input_sequence_position": matched_token[
                    "input_sequence_position"
                ],
                "matched_token_id": matched_token["token_id"],
                "matched_vocab_token": matched_token["vocab_token"],
                "matched_decoded_text": matched_token["decoded_text"],
                "matched_is_special_token": matched_token["is_special_token"],
            })

        # This compact view makes it easy to see which text vocabulary pieces
        # attract many visual tokens without manually grouping the detailed rows.
        matching_summary = []
        for matched_index, count in sorted(
            all_match_counts.items(), key=lambda item: (-item[1], item[0])
        ):
            token = query_tokens[matched_index]
            matching_summary.append({
                **token,
                "matched_visual_token_count": count,
                "matched_kept_visual_token_count": kept_match_counts[matched_index],
                "matched_pruned_visual_token_count": pruned_match_counts[matched_index],
            })

        # kept/pruned 索引仍使用剪枝前的视觉 token 坐标，因此可以直接与
        # visual_tokens 列表下标对应，不会因 top-k 压缩后的重新编号而产生歧义。
        kept_indices = torch.where(kept_mask)[0].tolist()
        pruned_indices = torch.where(~kept_mask)[0].tolist()
        threshold = float(torch.as_tensor(stats["threshold"]).detach().float().cpu())

        return {
            "role": role,
            "candidate_pos": candidate_pos,
            "candidate_letter": chr(ord("A") + candidate_pos),
            "final_rank": final_rank,
            "global_idx": global_idx,
            "local_page_id": local_page_id,
            "is_ground_truth": role == "highest_ranked_correct",
            "image_grid_thw_before_spatial_merge": image_grid_thw,
            "spatial_merge_size": self.spatial_merge_size,
            "merged_visual_grid": {
                "temporal": grid_t,
                "rows": merged_grid_h,
                "columns": merged_grid_w,
            },
            "pruning_threshold_similarity": threshold,
            "num_original_visual_tokens": scores.numel(),
            "num_kept_visual_tokens": len(kept_indices),
            "num_pruned_visual_tokens": len(pruned_indices),
            "kept_visual_token_indices": kept_indices,
            "pruned_visual_token_indices": pruned_indices,
            "matched_query_token_summary": matching_summary,
            "visual_tokens": visual_tokens,
        }

    def add_query(
        self,
        result: Mapping[str, Any],
        candidate_stats: Sequence[Mapping[str, Any]],
    ) -> None:
        """Select the best-ranked GT/non-GT candidate after final reranking."""
        # ranked_indices 排列的是 candidate_pos，而标注 page_id 使用 local_page_id
        # top_k_global_indices。一阶段检索到topk文档的全局id，按照从大到小的顺序
        # ranked_indices 重排后的一阶段文档列表id，也可代表选项，例如如果第一个是2，那么代表一阶段列表中第3个是重排后第一个，以此类推
        # gt_page_ids 看起来是ground-truth，但是是以局部id的形式来存的
        self.total_queries_seen += 1
        top_k_global_indices = [int(value) for value in result["top_k_global_indices"]]
        ranked_indices = [int(value) for value in result["ranked_indices"]]
        gt_page_ids = {int(value) for value in result["page_id"]}

        if not gt_page_ids:
            self.skipped_counts["no_ground_truth_page"] += 1
            return
        if not ranked_indices:
            self.skipped_counts["empty_ranking"] += 1
            return
        if len(candidate_stats) != len(top_k_global_indices):
            raise ValueError(
                "Similarity/image count mismatch for "
                f"{result.get('qid', '<unknown>')}: "
                f"stats={len(candidate_stats)}, candidates={len(top_k_global_indices)}"
            )

        # 统计数据的产生顺序未必等于最终重排顺序，先按 candidate_pos 建索引，
        # 后面即可用 ranked_indices 中的位置稳定地取得对应图像的逐 token 信息。
        stats_by_position = {
            int(stats["candidate_pos"]): stats for stats in candidate_stats
        }
        expected_positions = set(range(len(top_k_global_indices)))
        if set(stats_by_position) != expected_positions:
            raise ValueError("Alignment statistics do not cover every candidate position")
        if len(ranked_indices) != len(expected_positions) or set(ranked_indices) != expected_positions:
            raise ValueError("Final ranking is not a permutation of candidate positions")

        start_idx = int(result["start_idx"])
        end_idx = int(result["end_idx"])
        local_page_ids = [global_idx - start_idx for global_idx in top_k_global_indices]
        if any(not 0 <= page_id <= end_idx - start_idx for page_id in local_page_ids):
            raise ValueError("Candidate global index falls outside the document range")

        # Filtering in final reranking order means the first element of each
        # list is exactly the highest-ranked candidate for that class.
        correct_positions = [
            pos for pos in ranked_indices if local_page_ids[pos] in gt_page_ids
        ]
        incorrect_positions = [
            pos for pos in ranked_indices if local_page_ids[pos] not in gt_page_ids
        ]
        if not correct_positions:
            self.skipped_counts["correct_candidate_missing_from_topk"] += 1
            return
        if not incorrect_positions:
            self.skipped_counts["incorrect_candidate_missing_from_topk"] += 1
            return

        query_token_ids, query_input_positions, text_extraction_mode = (
            self._validate_shared_query_metadata(candidate_stats)
        )

        # 查询 token 只解码一次并保留两套位置：query_sequence_index 对应相似度
        # 矩阵行号，input_sequence_position 对应完整多模态输入中的原始位置。
        query_tokens = [
            self._describe_text_token(
                query_sequence_index=index,
                input_sequence_position=int(query_input_positions[index]),
                token_id=int(query_token_ids[index]),
            )
            for index in range(query_token_ids.numel())
        ]

        # 下面的candidate_pos描述的是，在一阶段排序local_id有序列表中第几项

        # rank_by_position 把“候选列表下标”反查为从 1 开始的最终名次；chosen
        # 固定只保留排名最靠前的一个正例和一个负例，控制 JSON 体积并便于比较。
        rank_by_position = {
            candidate_pos: rank + 1
            for rank, candidate_pos in enumerate(ranked_indices)
        }
        chosen = [
            ("highest_ranked_correct", correct_positions[0]),
            ("highest_ranked_incorrect", incorrect_positions[0]),
        ]
        candidates = []
        for role, candidate_pos in chosen:
            candidates.append(self._build_candidate_record(
                role=role,
                candidate_pos=candidate_pos,
                final_rank=rank_by_position[candidate_pos],
                global_idx=top_k_global_indices[candidate_pos],
                local_page_id=local_page_ids[candidate_pos],
                stats=stats_by_position[candidate_pos],
                query_tokens=query_tokens,
            ))

        top1_pos = ranked_indices[0]
        self.query_records.append({
            "qid": result.get("qid", f"{result['doc_name']}_{result['q_idx']}"),
            "doc_name": result["doc_name"],
            "domain": result["domain"],
            "q_idx": int(result["q_idx"]),
            "query": result["query"],
            "ground_truth_page_ids": sorted(gt_page_ids),
            "recall_at_1_correct": local_page_ids[top1_pos] in gt_page_ids,
            "text_extraction_mode": text_extraction_mode,
            "alignment_text_scope": "dataset_query_only",
            "num_qi_text_tokens": len(query_tokens),
            "qi_text_token_sequence": query_tokens,
            "candidates": candidates,
        })

    @staticmethod
    def _safe_path_component(value: Any, fallback: str) -> str:
        cleaned = re.sub(r"[^\w.-]+", "_", str(value), flags=re.UNICODE).strip("._")
        return (cleaned or fallback)[:100]

    def _write_candidate_images(
        self,
        image_output_path: Path,
        json_parent: Path,
        query_index: int,
        query_record: Dict[str, Any],
        candidate_record: Dict[str, Any],
    ) -> None:
        qid_component = self._safe_path_component(
            query_record["qid"], f"query_{query_index:02d}"
        )
        query_dir = image_output_path / f"query_{query_index:02d}_{qid_component}"
        role_component = self._safe_path_component(
            candidate_record["role"], "candidate"
        )
        candidate_dir = query_dir / (
            f"{role_component}_{candidate_record['candidate_letter']}_"
            f"page_{candidate_record['local_page_id']}"
        )
        candidate_dir.mkdir(parents=True, exist_ok=True)

        # The parquet bytes are decoded without resizing. The shared overlay helper
        # projects the merged-token grid proportionally onto this original resolution,
        # fills retained cells in translucent green, and outlines them in yellow.
        source_image = load_rgb_image(
            self.image_binary_loader(int(candidate_record["global_idx"]))
        )
        original_path = candidate_dir / "original.png"
        overlay_path = candidate_dir / "kept_patches.png"
        token_map_path = candidate_dir / "kept_patch_token_map.png"
        token_details_path = candidate_dir / "kept_patch_token_details.png"
        source_image.save(original_path, format="PNG")
        annotated_image = draw_kept_patch_overlay(
            source_image,
            candidate_record["kept_visual_token_indices"],
            candidate_record["image_grid_thw_before_spatial_merge"],
            candidate_record["spatial_merge_size"],
            fill_alpha=self.overlay_alpha,
        )
        annotated_image.save(overlay_path, format="PNG")

        # The overview emphasizes token identity; the detail view uses a larger
        # target cell and adds the strict dataset-query similarity below each Q-index.
        token_map = draw_query_token_patch_map(
            image=source_image,
            visual_tokens=candidate_record["visual_tokens"],
            query_tokens=query_record["qi_text_token_sequence"],
            image_grid_thw=candidate_record[
                "image_grid_thw_before_spatial_merge"
            ],
            spatial_merge_size=candidate_record["spatial_merge_size"],
            fill_alpha=self.overlay_alpha,
            show_similarity=False,
        )
        token_map.save(token_map_path, format="PNG")
        token_details = draw_query_token_patch_map(
            image=source_image,
            visual_tokens=candidate_record["visual_tokens"],
            query_tokens=query_record["qi_text_token_sequence"],
            image_grid_thw=candidate_record[
                "image_grid_thw_before_spatial_merge"
            ],
            spatial_merge_size=candidate_record["spatial_merge_size"],
            fill_alpha=self.overlay_alpha,
            show_similarity=True,
        )
        token_details.save(token_details_path, format="PNG")

        # Paths are relative to the JSON directory, making the complete output tree
        # movable while preserving direct links from each candidate to its two images.
        candidate_record["image_files"] = {
            "original": original_path.relative_to(json_parent).as_posix(),
            "kept_patch_overlay": overlay_path.relative_to(json_parent).as_posix(),
            "kept_patch_token_map": token_map_path.relative_to(
                json_parent
            ).as_posix(),
            "kept_patch_token_details": token_details_path.relative_to(
                json_parent
            ).as_posix(),
        }

    def save(
        self,
        output_file: str,
        *,
        run_metadata: Mapping[str, Any],
    ) -> Path:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Keep all generated images beside the JSON in a deterministic sibling
        # directory. One query directory contains the selected GT and non-GT page,
        # and each candidate contains both the untouched source and its overlay.
        image_output_path = output_path.parent / f"{output_path.stem}_images"
        for query_index, query_record in enumerate(self.query_records, start=1):
            for candidate_record in query_record["candidates"]:
                self._write_candidate_images(
                    image_output_path=image_output_path,
                    json_parent=output_path.parent,
                    query_index=query_index,
                    query_record=query_record,
                    candidate_record=candidate_record,
                )

        # 顶层同时写入索引、字符串和并列 argmax 的解释，使 JSON 脱离代码后仍可
        # 独立理解；queries 中才是每个 query、候选和视觉 token 的实际诊断数据。
        payload = {
            "schema_version": 4,
            "description": (
                "For each visual token, records the dataset query token that "
                "achieved its maximum cosine similarity and whether the visual "
                "token survived the model's configured QI-Early pruning."
            ),
            "index_semantics": {
                "visual_token_index": "Zero-based index before visual-token pruning.",
                "query_sequence_index": (
                    "Zero-based index inside the dataset query-token sequence; "
                    "hard-coded prompt tokens are excluded."
                ),
                "input_sequence_position": (
                    "Zero-based position in the full tokenized multimodal model input."
                ),
                "argmax_tie_rule": (
                    "PyTorch max returns the first query_sequence_index when values tie."
                ),
            },
            "token_string_semantics": {
                "vocab_token": "Exact tokenizer vocabulary piece (possibly a subword).",
                "decoded_text": "That one token ID decoded without skipping special tokens.",
            },
            "similarity_semantics": {
                "max_similarity": (
                    "Maximum cosine similarity over dataset query tokens only."
                ),
                "pruning_similarity": (
                    "Score actually used for QI-Early top-k pruning; its text scope "
                    "is run.effective_qi_early_text_mode."
                ),
                "kept": "Whether the visual token survived actual QI-Early pruning.",
            },
            "patch_semantics": {
                "patch_grid_position_trc": (
                    "[temporal index, row, column] in the merged visual-token grid."
                ),
                "kept_patch_overlay": (
                    "Faint white lines show the full grid; translucent green cells "
                    "with yellow borders are the visual tokens retained by QI-Early."
                ),
                "kept_patch_token_map": (
                    "Retained cells are colored by matched_query_sequence_index and "
                    "labeled Q-index; the side legend maps Q-index to decoded_text "
                    "and vocab_token. Colors are stable across candidates in a query."
                ),
                "kept_patch_token_details": (
                    "An enlarged token map whose patch label adds max_similarity "
                    "below the Q-index. This similarity uses dataset query tokens only."
                ),
                "image_resolution": (
                    "Both original.png and kept_patches.png preserve the source "
                    "page resolution from the parquet image_binary field."
                ),
                "image_path_base": "Directory containing this JSON file.",
            },
            "run": dict(run_metadata),
            "total_queries_seen": self.total_queries_seen,
            "num_queries_exported": len(self.query_records),
            "skipped_queries": dict(sorted(self.skipped_counts.items())),
            "queries": self.query_records,
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_path
