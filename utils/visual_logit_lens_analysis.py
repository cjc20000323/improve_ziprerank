"""Collect and export late-layer vocabulary projections for visual tokens.

The utilities in this module are independent of the normal evaluation code.  A
diagnostic model supplies CPU copies of late-layer hidden states for the visual
tokens retained by QI-Early.  Once the final ranking is known, this collector:

* keeps the highest-ranked ground-truth candidate and three highest-ranked
  non-ground-truth candidates;
* applies the model's final RMSNorm and language-model head in bounded chunks;
* converts top vocabulary IDs into both exact tokenizer pieces and readable
  one-token decodings;
* maps retained visual-token indices back to the merged image-patch grid; and
* saves the untouched source image plus an annotated retained-patch overlay.

The vocabulary projection is a Logit Lens observation, not a causal explanation
or a guaranteed natural-language label for a patch.  In particular, visual-token
positions are not trained as ordinary next-token prediction positions.
"""

from __future__ import annotations

import colorsys
import io
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

import torch
from PIL import Image, ImageDraw, ImageFont


ImageBinaryLoader = Callable[[int], bytes]


class VocabularyProjector(Protocol):
    """Structural interface used by real and lightweight test projectors."""

    def project(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return top vocabulary IDs and corresponding logits per state."""


def select_visual_logit_lens_query_subset(
    first_stage_results: Mapping[Any, Mapping[str, Any]],
    num_queries: int = 20,
    seed: int = 42,
    num_incorrect_candidates: int = 3,
) -> Tuple[Dict[Any, Mapping[str, Any]], Dict[str, Any]]:
    """Uniformly sample queries that can provide the requested four candidates.

    This follows the seeded sampling design used by ``evaluate_token_alignment``
    but avoids a full eligibility scan: randomly permute the query keys, inspect
    them one by one, and stop as soon as enough valid queries have been found.
    The Top-K requirement is also stricter because this analysis needs three
    negative pages rather than merely one.
    """
    if num_queries <= 0:
        raise ValueError(f"num_queries must be positive, got {num_queries}")
    if num_incorrect_candidates <= 0:
        raise ValueError(
            "num_incorrect_candidates must be positive, got "
            f"{num_incorrect_candidates}"
        )

    eligible_keys: List[Any] = []
    skipped: DefaultDict[str, int] = defaultdict(int)
    candidate_keys = list(first_stage_results.keys())
    random.Random(seed).shuffle(candidate_keys)
    inspected_queries = 0

    # Ground-truth page IDs are document-local, whereas retrieval candidates are
    # stored as global parquet row indices.  Converting first prevents a query
    # from being accepted because two unrelated coordinate systems happened to
    # contain the same integer.
    for key in candidate_keys:
        inspected_queries += 1
        item = first_stage_results[key]
        gt_page_ids = {int(page_id) for page_id in item.get("page_id", [])}
        global_indices = [
            int(value) for value in item.get("top_k_global_indices", [])
        ]
        if not gt_page_ids:
            skipped["no_ground_truth_page"] += 1
            continue
        if not global_indices:
            skipped["empty_first_stage_candidates"] += 1
            continue

        start_idx = int(item["start_idx"])
        end_idx = int(item["end_idx"])
        local_page_ids = [value - start_idx for value in global_indices]
        if any(
            not 0 <= page_id <= end_idx - start_idx
            for page_id in local_page_ids
        ):
            skipped["candidate_outside_document_range"] += 1
            continue

        num_correct = sum(page_id in gt_page_ids for page_id in local_page_ids)
        num_incorrect = len(local_page_ids) - num_correct
        if num_correct == 0:
            skipped["correct_candidate_missing_from_topk"] += 1
            continue
        if num_incorrect < num_incorrect_candidates:
            skipped["fewer_than_three_incorrect_candidates"] += 1
            continue

        eligible_keys.append(key)
        if len(eligible_keys) == num_queries:
            break

    if len(eligible_keys) < num_queries:
        raise ValueError(
            f"Requested {num_queries} Logit Lens queries, but only "
            f"{len(eligible_keys)} contain a GT and at least "
            f"{num_incorrect_candidates} non-GT candidates"
        )

    selected = {key: first_stage_results[key] for key in eligible_keys}
    metadata = {
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
        "required_incorrect_candidates": num_incorrect_candidates,
        "prefilter_skipped_queries": dict(sorted(skipped.items())),
    }
    return selected, metadata


def merged_visual_grid_shape(
    image_grid_thw: Sequence[int],
    spatial_merge_size: int,
) -> Tuple[int, int, int]:
    """Return the temporal, row and column counts seen by the language model."""
    if len(image_grid_thw) != 3:
        raise ValueError(
            f"image_grid_thw must contain [t, h, w], got {tuple(image_grid_thw)}"
        )
    if spatial_merge_size <= 0:
        raise ValueError("spatial_merge_size must be positive")

    grid_t, patch_grid_h, patch_grid_w = (int(value) for value in image_grid_thw)
    if grid_t <= 0 or patch_grid_h <= 0 or patch_grid_w <= 0:
        raise ValueError(
            "image_grid_thw values must be positive: "
            f"{tuple(image_grid_thw)}"
        )
    if (
        patch_grid_h % spatial_merge_size != 0
        or patch_grid_w % spatial_merge_size != 0
    ):
        raise ValueError(
            "Vision grid dimensions must be divisible by spatial_merge_size: "
            f"grid={tuple(image_grid_thw)}, merge={spatial_merge_size}"
        )

    return (
        grid_t,
        patch_grid_h // spatial_merge_size,
        patch_grid_w // spatial_merge_size,
    )


def visual_token_patch_geometry(
    visual_token_index: int,
    image_grid_thw: Sequence[int],
    spatial_merge_size: int,
    image_size: Tuple[int, int],
) -> Dict[str, Any]:
    """Map one pre-pruning visual-token index to its source-image patch box.

    Qwen3-VL's spatial merger emits tokens in temporal, merged-row, merged-column
    order.  The processor resizes a page to the vision grid without changing the
    grid topology, so proportional cell boundaries map the merged grid back onto
    the untouched source image.  Pixel boxes use half-open ``[x0, y0, x1, y1]``
    bounds, matching common array-slicing conventions.
    """
    grid_t, grid_h, grid_w = merged_visual_grid_shape(
        image_grid_thw, spatial_merge_size
    )
    num_tokens = grid_t * grid_h * grid_w
    if not 0 <= visual_token_index < num_tokens:
        raise IndexError(
            f"visual_token_index={visual_token_index} outside [0, {num_tokens})"
        )

    width, height = (int(value) for value in image_size)
    if width <= 0 or height <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")

    tokens_per_frame = grid_h * grid_w
    temporal_index = visual_token_index // tokens_per_frame
    spatial_index = visual_token_index % tokens_per_frame
    row = spatial_index // grid_w
    column = spatial_index % grid_w

    x0 = math.floor(column * width / grid_w)
    x1 = math.ceil((column + 1) * width / grid_w)
    y0 = math.floor(row * height / grid_h)
    y1 = math.ceil((row + 1) * height / grid_h)

    return {
        "temporal_index": temporal_index,
        "merged_grid_row": row,
        "merged_grid_column": column,
        "normalized_box_xyxy": [
            column / grid_w,
            row / grid_h,
            (column + 1) / grid_w,
            (row + 1) / grid_h,
        ],
        "source_image_pixel_box_xyxy": [x0, y0, x1, y1],
    }


def load_rgb_image(image_binary: bytes) -> Image.Image:
    """Decode parquet image bytes into an independent RGB PIL image."""
    with Image.open(io.BytesIO(bytes(image_binary))) as opened:
        return opened.convert("RGB")


def draw_kept_patch_overlay(
    image: Image.Image,
    kept_visual_token_indices: Sequence[int],
    image_grid_thw: Sequence[int],
    spatial_merge_size: int,
    fill_alpha: int = 72,
) -> Image.Image:
    """Highlight retained merged patches while preserving source resolution."""
    if not 0 <= fill_alpha <= 255:
        raise ValueError(f"fill_alpha must be in [0, 255], got {fill_alpha}")

    grid_t, grid_h, grid_w = merged_visual_grid_shape(
        image_grid_thw, spatial_merge_size
    )
    if grid_t != 1:
        raise ValueError(
            "Static page visualization expects image_grid_thw[0] == 1, got "
            f"{grid_t}"
        )

    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    width, height = base.size

    # Faint grid lines show the complete merged-token tessellation.  Retained
    # cells then receive a translucent green fill and opaque yellow boundary,
    # keeping underlying document text visible while making selection explicit.
    for column in range(1, grid_w):
        x = round(column * width / grid_w)
        draw.line([(x, 0), (x, height)], fill=(255, 255, 255, 42), width=1)
    for row in range(1, grid_h):
        y = round(row * height / grid_h)
        draw.line([(0, y), (width, y)], fill=(255, 255, 255, 42), width=1)

    cell_width = width / grid_w
    cell_height = height / grid_h
    outline_width = max(1, round(min(cell_width, cell_height) * 0.06))
    for token_index in kept_visual_token_indices:
        geometry = visual_token_patch_geometry(
            int(token_index), image_grid_thw, spatial_merge_size, base.size
        )
        x0, y0, x1, y1 = geometry["source_image_pixel_box_xyxy"]
        draw.rectangle(
            [x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)],
            fill=(35, 210, 105, fill_alpha),
            outline=(255, 210, 0, 235),
            width=outline_width,
        )

    return Image.alpha_composite(base, overlay).convert("RGB")


def _logit_lens_word_color(word_number: int) -> Tuple[int, int, int]:
    """Return a deterministic, high-contrast color for one query-local word."""
    hue = (0.08 + int(word_number) * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 0.92)
    return round(red * 255), round(green * 255), round(blue * 255)


_CJK_UNICODE_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2FA1F),
)


def _required_cjk_codepoints(text: str) -> frozenset:
    """Return the Chinese/Japanese/Korean ideographs that need real glyphs."""
    return frozenset(
        ord(character)
        for character in str(text)
        if any(
            range_start <= ord(character) <= range_end
            for range_start, range_end in _CJK_UNICODE_RANGES
        )
    )


@lru_cache(maxsize=64)
def _font_face_indices(font_path: str) -> Tuple[int, ...]:
    """Return loadable face indices for a single font or a TTC collection."""
    maximum_faces = 32 if Path(font_path).suffix.lower() == ".ttc" else 1
    face_indices = []
    for face_index in range(maximum_faces):
        try:
            ImageFont.truetype(font_path, size=24, index=face_index)
        except OSError:
            break
        face_indices.append(face_index)
    return tuple(face_indices)


def _glyph_signature(font: Any, character: str) -> Tuple[Any, Any, bytes]:
    """Describe one rendered glyph so missing-glyph boxes can be detected."""
    mask = font.getmask(character, mode="L")
    return mask.size, font.getbbox(character), bytes(mask)


@lru_cache(maxsize=512)
def _font_face_supports_codepoints(
    font_path: str,
    face_index: int,
    required_codepoints: Tuple[int, ...],
) -> bool:
    """Reject faces that substitute FreeType's .notdef box for Chinese text."""
    try:
        font = ImageFont.truetype(font_path, size=32, index=face_index)
        missing_glyph = _glyph_signature(font, chr(0x10FFFF))
        return all(
            _glyph_signature(font, chr(codepoint)) != missing_glyph
            for codepoint in required_codepoints
        )
    except (OSError, UnicodeEncodeError, ValueError):
        return False


@lru_cache(maxsize=1)
def _discover_installed_cjk_font_paths() -> Tuple[str, ...]:
    """Find additional CJK fonts in common OS font directories once per run."""
    font_roots = (
        Path("C:/Windows/Fonts"),
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path("/System/Library/Fonts"),
        Path("/Library/Fonts"),
    )
    cjk_name_markers = (
        "notosanscjk",
        "notoserifcjk",
        "notosanssc",
        "notoserifsc",
        "sourcehansans",
        "sourcehanserif",
        "droidsansfallback",
        "wenquanyi",
        "wqy",
        "msyh",
        "simhei",
        "simsun",
        "pingfang",
        "heiti",
        "songti",
        "arphic",
    )
    discovered = []
    for font_root in font_roots:
        if not font_root.is_dir():
            continue
        try:
            for font_path in font_root.rglob("*"):
                normalized_name = font_path.name.lower().replace("-", "")
                if (
                    font_path.suffix.lower() in {".ttf", ".ttc", ".otf"}
                    and any(marker in normalized_name for marker in cjk_name_markers)
                ):
                    discovered.append(str(font_path))
        except OSError:
            continue
    return tuple(dict.fromkeys(discovered))


def _candidate_annotation_font_paths(
    preferred_font_path: Optional[str],
) -> Tuple[str, ...]:
    """Order explicit, bundled, common, and dynamically discovered fonts."""
    configured_font_path = preferred_font_path or os.environ.get(
        "ZIPRERANK_CJK_FONT"
    )
    if configured_font_path:
        return (str(Path(configured_font_path).expanduser()),)

    bundled_font_path = (
        Path(__file__).resolve().parent
        / "fonts"
        / "NotoSansCJKsc-Regular.otf"
    )
    paths = [
        str(bundled_font_path),
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/opentype/noto/NotoSansSC-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/usr/share/fonts/opentype/source-han-sans/SourceHanSansSC-Regular.otf",
        "/usr/share/fonts/adobe-source-han-sans/SourceHanSansSC-Regular.otf",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        *_discover_installed_cjk_font_paths(),
    ]
    return tuple(
        dict.fromkeys(
            str(Path(font_path).expanduser())
            for font_path in paths
            if font_path
        )
    )


def _resolve_annotation_font_face(
    required_text: str,
    preferred_font_path: Optional[str],
) -> Optional[Tuple[str, int]]:
    """Choose a font face that renders every required CJK character."""
    configured_font_path = preferred_font_path or os.environ.get(
        "ZIPRERANK_CJK_FONT"
    )
    if configured_font_path and not Path(configured_font_path).expanduser().is_file():
        raise FileNotFoundError(
            "Configured CJK annotation font does not exist: "
            f"{Path(configured_font_path).expanduser()}"
        )

    required_codepoints = _required_cjk_codepoints(required_text)
    for font_path in _candidate_annotation_font_paths(preferred_font_path):
        if not Path(font_path).is_file():
            continue
        for face_index in _font_face_indices(font_path):
            if not required_codepoints or _font_face_supports_codepoints(
                font_path,
                face_index,
                tuple(sorted(required_codepoints)),
            ):
                return font_path, face_index

    if required_codepoints:
        if configured_font_path:
            raise RuntimeError(
                "The configured annotation font does not contain all Chinese "
                f"characters required by this result: {configured_font_path}. "
                "Please select a complete CJK font such as Noto Sans CJK SC."
            )
        raise RuntimeError(
            "The Logit Lens legend contains Chinese text, but no installed font "
            "covers all required CJK glyphs. Install Noto Sans CJK SC, pass "
            "--annotation_font_path /path/to/NotoSansCJKsc-Regular.otf, or set "
            "ZIPRERANK_CJK_FONT to a compatible .ttf/.ttc/.otf file."
        )
    return None


@lru_cache(maxsize=64)
def _load_font_face(font_path: str, face_index: int, size: int) -> Any:
    """Cache PIL font objects because each query exports many layer maps."""
    return ImageFont.truetype(
        font_path,
        size=max(8, int(size)),
        index=int(face_index),
    )


def _load_patch_annotation_font(
    size: int,
    required_text: str = "",
    preferred_font_path: Optional[str] = None,
) -> Any:
    """Load a verified CJK-capable font instead of silently drawing tofu boxes."""
    resolved_face = _resolve_annotation_font_face(
        required_text, preferred_font_path
    )
    if resolved_face is not None:
        font_path, face_index = resolved_face
        return _load_font_face(font_path, face_index, max(8, int(size)))

    # This branch is reached only when the text has no CJK characters and the
    # machine has no matching system font, so Pillow's ASCII fallback is safe.
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=max(8, int(size)))
    except OSError:
        return ImageFont.load_default()


def _short_legend_text(value: Any, max_characters: int = 42) -> str:
    """Make whitespace and tokenizer markers visible without an oversized legend."""
    visible = json.dumps("" if value is None else str(value), ensure_ascii=False)
    if len(visible) <= max_characters:
        return visible
    return f"{visible[:max_characters - 3]}..."


def _fit_patch_label(
    draw: Any,
    label_options: Sequence[Tuple[str, str]],
    maximum_width: int,
    maximum_height: int,
    initial_font_size: int,
    annotation_font_path: Optional[str],
) -> Tuple[str, str, Any, Tuple[int, int, int, int], int, bool]:
    """Fit the most informative available label into one retained patch.

    Options are ordered by information content.  The detailed map therefore
    tries ``W# + logit`` first, then ``W#``, and finally the numeric portion of
    the W label.  Font size is reduced with a bounded binary search before any
    information is dropped.
    """
    minimum_font_size = 5
    maximum_width = max(1, int(maximum_width))
    maximum_height = max(1, int(maximum_height))

    for label_kind, label in label_options:
        lower = minimum_font_size
        upper = max(minimum_font_size, int(initial_font_size))
        best_result = None
        while lower <= upper:
            font_size = (lower + upper) // 2
            stroke_width = 2 if font_size >= 9 else 1
            font = _load_patch_annotation_font(
                font_size, preferred_font_path=annotation_font_path
            )
            text_box = draw.multiline_textbbox(
                (0, 0),
                label,
                font=font,
                spacing=1,
                stroke_width=stroke_width,
            )
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
            if text_width <= maximum_width and text_height <= maximum_height:
                best_result = (
                    label_kind,
                    label,
                    font,
                    text_box,
                    stroke_width,
                    True,
                )
                lower = font_size + 1
            else:
                upper = font_size - 1
        if best_result is not None:
            return best_result

    # Extremely small cells may not hold even a five-pixel numeric label. Draw
    # it centered anyway instead of silently making a retained patch look as if
    # it has no Logit Lens result. Such cases are counted in the image subtitle.
    label_kind, label = label_options[-1]
    font = _load_patch_annotation_font(
        minimum_font_size, preferred_font_path=annotation_font_path
    )
    text_box = draw.multiline_textbbox(
        (0, 0), label, font=font, spacing=1, stroke_width=1
    )
    return label_kind, label, font, text_box, 1, False


def draw_logit_lens_word_patch_map(
    image: Image.Image,
    layer_record: Mapping[str, Any],
    word_registry: Mapping[int, Mapping[str, Any]],
    image_grid_thw: Sequence[int],
    spatial_merge_size: int,
    fill_alpha: int = 72,
    show_top1_logit: bool = False,
    annotation_font_path: Optional[str] = None,
    patch_label_mode: str = "one_per_word",
) -> Image.Image:
    """Draw retained patches using query-consistent Top-1 vocabulary colors.

    ``word_registry`` is deliberately shared by every candidate and sampled
    layer belonging to one query.  Consequently, labels such as ``W3`` and
    their colors keep exactly the same meaning across all exported images for
    that query.  Pruned cells receive neither a color nor a word label.
    """
    if not 0 <= fill_alpha <= 255:
        raise ValueError(f"fill_alpha must be in [0, 255], got {fill_alpha}")
    if patch_label_mode not in {"one_per_word", "all"}:
        raise ValueError(
            "patch_label_mode must be 'one_per_word' or 'all', got "
            f"{patch_label_mode!r}"
        )

    grid_t, grid_h, grid_w = merged_visual_grid_shape(
        image_grid_thw, spatial_merge_size
    )
    if grid_t != 1:
        raise ValueError(
            "Static page visualization expects image_grid_thw[0] == 1, got "
            f"{grid_t}"
        )

    # Small source images are enlarged just enough for patch labels to remain
    # legible.  Normal document pages retain their original resolution.
    source = image.convert("RGB")
    cell_width = source.width / grid_w
    cell_height = source.height / grid_h
    target_cell_size = 42 if show_top1_logit else 30
    requested_scale = target_cell_size / max(1.0, min(cell_width, cell_height))
    maximum_scale = max(1.0, min(3.0, 4500.0 / max(source.size)))
    render_scale = min(maximum_scale, max(1.0, requested_scale))
    if render_scale > 1.0:
        resampling = getattr(
            getattr(Image, "Resampling", object()), "LANCZOS", 1
        )
        rendered_size = (
            round(source.width * render_scale),
            round(source.height * render_scale),
        )
        source = source.resize(rendered_size, resample=resampling)

    base = source.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    cell_width = base.width / grid_w
    cell_height = base.height / grid_h

    # The complete grid provides spatial context, while only retained tokens
    # receive the vocabulary-dependent fill, border, and compact W label.
    for column in range(1, grid_w):
        x = round(column * base.width / grid_w)
        draw.line([(x, 0), (x, base.height)], fill=(255, 255, 255, 48), width=1)
    for row in range(1, grid_h):
        y = round(row * base.height / grid_h)
        draw.line([(0, y), (base.width, y)], fill=(255, 255, 255, 48), width=1)

    label_font_size = max(
        9,
        round(min(cell_width, cell_height) * (0.25 if show_top1_logit else 0.32)),
    )
    outline_width = max(1, round(min(cell_width, cell_height) * 0.045))
    used_token_counts: Counter[int] = Counter()
    label_fallback_counts: Counter[str] = Counter()

    # Repeated Top-1 words share one color, so printing W# on every matching
    # patch adds clutter without adding vocabulary information. In the compact
    # default mode, the highest-logit patch is the single labeled representative
    # for that word; all matching patches remain colored and counted in legend.
    predictions = list(layer_record["token_predictions"])
    if patch_label_mode == "one_per_word":
        representative_by_token: Dict[int, Tuple[float, int, int]] = {}
        for prediction_index, prediction in enumerate(predictions):
            top1 = prediction["top_vocabulary"][0]
            token_id = int(top1["token_id"])
            ranking = (
                float(top1["logit"]),
                -int(prediction["original_visual_token_index"]),
                prediction_index,
            )
            if token_id not in representative_by_token or ranking > (
                representative_by_token[token_id]
            ):
                representative_by_token[token_id] = ranking
        labeled_prediction_indices = {
            ranking[2] for ranking in representative_by_token.values()
        }
    else:
        labeled_prediction_indices = set(range(len(predictions)))

    for prediction_index, prediction in enumerate(predictions):
        vocabulary = prediction.get("top_vocabulary", [])
        if not vocabulary:
            raise ValueError("A retained patch has no Logit Lens vocabulary result")
        top1 = vocabulary[0]
        token_id = int(top1["token_id"])
        if token_id not in word_registry:
            raise KeyError(f"Top-1 token ID {token_id} is missing from word_registry")

        word_info = word_registry[token_id]
        color = tuple(int(channel) for channel in word_info["color_rgb"])
        visual_token_index = int(prediction["original_visual_token_index"])
        geometry = visual_token_patch_geometry(
            visual_token_index,
            image_grid_thw,
            spatial_merge_size,
            base.size,
        )
        x0, y0, x1, y1 = geometry["source_image_pixel_box_xyxy"]
        draw.rectangle(
            [x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)],
            fill=(*color, fill_alpha),
            outline=(*color, 255),
            width=outline_width,
        )

        used_token_counts[token_id] += 1
        if prediction_index not in labeled_prediction_indices:
            continue

        word_label = str(word_info["word_label"])
        numeric_label = word_label[1:] if word_label.startswith("W") else word_label
        if show_top1_logit:
            label_options = (
                ("word_and_logit", f"{word_label}\n{float(top1['logit']):.2f}"),
                ("word_only", word_label),
                ("numeric_only", numeric_label),
            )
        else:
            label_options = (
                ("word_only", word_label),
                ("numeric_only", numeric_label),
            )
        (
            label_kind,
            patch_label,
            patch_font,
            text_box,
            text_stroke_width,
            label_fits,
        ) = _fit_patch_label(
            draw,
            label_options,
            maximum_width=(x1 - x0) - 2,
            maximum_height=(y1 - y0) - 2,
            initial_font_size=label_font_size,
            annotation_font_path=annotation_font_path,
        )
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        text_x = x0 + ((x1 - x0) - text_width) / 2
        text_y = y0 + ((y1 - y0) - text_height) / 2 - text_box[1]
        draw.multiline_text(
            (text_x, text_y),
            patch_label,
            font=patch_font,
            fill=(255, 255, 255, 255),
            spacing=1,
            align="center",
            stroke_width=text_stroke_width,
            stroke_fill=(0, 0, 0, 230),
        )
        if label_kind != label_options[0][0] or not label_fits:
            fallback_key = label_kind if label_fits else "forced_numeric"
            label_fallback_counts[fallback_key] += 1

    annotated = Image.alpha_composite(base, overlay).convert("RGB")

    # Append a legend instead of writing full vocabulary strings directly into
    # patches.  This keeps dense page regions readable and makes the same W label
    # easy to compare between the four candidates and four sampled layers.
    used_token_ids = sorted(
        used_token_counts,
        key=lambda registry_token_id: int(
            word_registry[registry_token_id]["word_number"]
        ),
    )
    legend_required_text = "".join(
        str(word_registry[token_id].get("decoded_text") or "")
        + str(word_registry[token_id].get("vocab_token") or "")
        for token_id in used_token_ids
    )
    legend_font = _load_patch_annotation_font(
        14,
        required_text=legend_required_text,
        preferred_font_path=annotation_font_path,
    )
    title_font = _load_patch_annotation_font(
        19, preferred_font_path=annotation_font_path
    )
    legend_line_height = 42
    legend_header_height = 72
    rows_that_fit_image = max(
        18,
        min(48, (annotated.height - legend_header_height - 24) // legend_line_height),
    )
    legend_columns = max(1, math.ceil(len(used_token_ids) / rows_that_fit_image))
    rows_per_column = max(1, math.ceil(len(used_token_ids) / legend_columns))
    legend_column_width = 440
    legend_width = legend_columns * legend_column_width + 32
    legend_height = legend_header_height + rows_per_column * legend_line_height + 24
    canvas = Image.new(
        "RGB",
        (annotated.width + legend_width, max(annotated.height, legend_height)),
        (248, 248, 248),
    )
    canvas.paste(annotated, (0, 0))
    legend_draw = ImageDraw.Draw(canvas)
    legend_x = annotated.width + 18
    layer_number = int(layer_record["layer_number_one_based"])
    legend_draw.text(
        (legend_x, 12),
        f"Layer {layer_number}: retained-patch Top-1 vocabulary",
        font=title_font,
        fill=(20, 20, 20),
    )
    if patch_label_mode == "one_per_word":
        subtitle = (
            "one representative W# + raw logit per word"
            if show_top1_logit
            else "one representative W# per word"
        )
    else:
        subtitle = (
            "patch label = W# + raw logit"
            if show_top1_logit
            else "patch label = W#"
        )
    fallback_total = sum(label_fallback_counts.values())
    if fallback_total:
        subtitle += f"; {fallback_total} tiny patches use a shorter label"
    legend_draw.text(
        (legend_x, 43), subtitle, font=legend_font, fill=(80, 80, 80)
    )

    for entry_index, token_id in enumerate(used_token_ids):
        column = entry_index // rows_per_column
        row = entry_index % rows_per_column
        x = legend_x + column * legend_column_width
        y = legend_header_height + row * legend_line_height
        word_info = word_registry[token_id]
        color = tuple(int(channel) for channel in word_info["color_rgb"])
        legend_draw.rectangle([x, y + 5, x + 18, y + 23], fill=color, outline=(0, 0, 0))
        first_line = (
            f"{word_info['word_label']} ({used_token_counts[token_id]} patches) "
            f"text={_short_legend_text(word_info.get('decoded_text'), 28)}"
        )
        second_line = (
            f"vocab={_short_legend_text(word_info.get('vocab_token'), 30)}"
        )
        legend_draw.text((x + 27, y + 1), first_line, font=legend_font, fill=(25, 25, 25))
        legend_draw.text((x + 27, y + 23), second_line, font=legend_font, fill=(70, 70, 70))

    return canvas


class LogitLensVocabularyProjector:
    """Apply the model's final norm and LM head to CPU hidden-state chunks."""

    def __init__(self, model, top_k: int = 5, chunk_size: int = 128) -> None:
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self.model = model
        self.top_k = top_k
        self.chunk_size = chunk_size

    @staticmethod
    def _parameter_device_dtype(module) -> Tuple[torch.device, Any]:
        try:
            parameter = next(module.parameters())
        except StopIteration as exc:
            raise RuntimeError("Logit Lens projection module has no parameters") from exc
        if parameter.device.type == "meta":
            raise RuntimeError(
                "Logit Lens projection does not support an LM head offloaded to "
                "the meta device"
            )
        return parameter.device, parameter.dtype

    def project(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return top vocabulary IDs and raw logits for each hidden-state row."""
        hidden_states = torch.as_tensor(hidden_states).detach().cpu().contiguous()
        if hidden_states.ndim != 2:
            raise ValueError(
                "hidden_states must have shape [visual_tokens, hidden_size], got "
                f"{tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[0] == 0:
            raise ValueError("Cannot project an empty visual-token sequence")

        final_norm = self.model.model.language_model.norm
        lm_head = self.model.lm_head
        norm_device, norm_dtype = self._parameter_device_dtype(final_norm)
        head_device, head_dtype = self._parameter_device_dtype(lm_head)
        vocab_size = int(getattr(lm_head, "out_features", lm_head.weight.shape[0]))
        actual_top_k = min(self.top_k, vocab_size)

        id_chunks = []
        logit_chunks = []
        with torch.inference_mode():
            for start in range(0, hidden_states.shape[0], self.chunk_size):
                # Moving only one chunk bounds peak vocabulary-logit memory at
                # roughly ``chunk_size * vocab_size`` rather than allocating the
                # full candidate matrix at once.
                chunk = hidden_states[start:start + self.chunk_size].to(
                    device=norm_device,
                    dtype=norm_dtype,
                )
                normalized = final_norm(chunk)
                normalized = normalized.to(device=head_device, dtype=head_dtype)
                logits = lm_head(normalized)
                top_logits, top_ids = torch.topk(
                    logits, k=actual_top_k, dim=-1, largest=True, sorted=True
                )
                id_chunks.append(top_ids.detach().to(device="cpu", dtype=torch.long))
                logit_chunks.append(
                    top_logits.detach().to(device="cpu", dtype=torch.float32)
                )

        top_ids = torch.cat(id_chunks, dim=0).contiguous()
        top_logits = torch.cat(logit_chunks, dim=0).contiguous()
        if not torch.isfinite(top_logits).all():
            raise ValueError("Non-finite Logit Lens vocabulary logits encountered")
        return top_ids, top_logits


class VisualLogitLensCollector:
    """Select four candidates per query and serialize their visual-token lenses."""

    def __init__(
        self,
        tokenizer,
        projector: VocabularyProjector,
        image_binary_loader: ImageBinaryLoader,
        spatial_merge_size: int,
        summary_top_tokens: int = 20,
        overlay_alpha: int = 72,
        annotation_font_path: Optional[str] = None,
        patch_label_mode: str = "one_per_word",
    ) -> None:
        if summary_top_tokens <= 0:
            raise ValueError("summary_top_tokens must be positive")
        if patch_label_mode not in {"one_per_word", "all"}:
            raise ValueError(
                "patch_label_mode must be 'one_per_word' or 'all', got "
                f"{patch_label_mode!r}"
            )
        self.tokenizer = tokenizer
        self.projector = projector
        self.image_binary_loader = image_binary_loader
        self.spatial_merge_size = int(spatial_merge_size)
        self.summary_top_tokens = summary_top_tokens
        self.overlay_alpha = overlay_alpha
        self.annotation_font_path = annotation_font_path
        self.patch_label_mode = patch_label_mode

        # Validate an explicitly configured path before model inference starts.
        # Actual CJK glyph coverage is checked later against each layer's words.
        configured_font_path = annotation_font_path or os.environ.get(
            "ZIPRERANK_CJK_FONT"
        )
        if configured_font_path and not Path(configured_font_path).expanduser().is_file():
            raise FileNotFoundError(
                "Configured CJK annotation font does not exist: "
                f"{Path(configured_font_path).expanduser()}"
            )

        self.query_records: List[Dict[str, Any]] = []
        self.total_queries_seen = 0
        self.skipped_counts: DefaultDict[str, int] = defaultdict(int)
        self._special_token_ids = set(getattr(tokenizer, "all_special_ids", []))
        self._token_description_cache: Dict[int, Dict[str, Any]] = {}

    @staticmethod
    def _cpu_1d_tensor(value: Any, dtype: Any, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(value).detach().to(device="cpu", dtype=dtype)
        tensor = tensor.contiguous()
        if tensor.ndim != 1:
            raise ValueError(f"{name} must be 1-D, got shape={tuple(tensor.shape)}")
        return tensor

    def _describe_token(self, token_id: int) -> Dict[str, Any]:
        token_id = int(token_id)
        if token_id not in self._token_description_cache:
            vocab_token = self.tokenizer.convert_ids_to_tokens(token_id)
            decoded_text = self.tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            self._token_description_cache[token_id] = {
                "token_id": token_id,
                "vocab_token": None if vocab_token is None else str(vocab_token),
                "decoded_text": decoded_text,
                "is_special_token": token_id in self._special_token_ids,
            }
        return dict(self._token_description_cache[token_id])

    def _layer_record(
        self,
        layer_index: int,
        hidden_states: torch.Tensor,
        selected_indices: torch.Tensor,
    ) -> Dict[str, Any]:
        top_ids, top_logits = self.projector.project(hidden_states)
        if top_ids.shape[0] != selected_indices.numel():
            raise RuntimeError(
                f"Layer {layer_index} projected {top_ids.shape[0]} visual tokens, "
                f"expected {selected_indices.numel()}"
            )

        token_predictions = []
        top1_counts: Counter[int] = Counter()
        top1_logit_sums: DefaultDict[int, float] = defaultdict(float)
        for kept_order, original_index in enumerate(selected_indices.tolist()):
            vocabulary = []
            for vocabulary_rank in range(top_ids.shape[1]):
                token_id = int(top_ids[kept_order, vocabulary_rank])
                vocabulary.append({
                    "vocabulary_rank": vocabulary_rank + 1,
                    **self._describe_token(token_id),
                    "logit": float(top_logits[kept_order, vocabulary_rank]),
                })

            top1_id = int(top_ids[kept_order, 0])
            top1_counts[top1_id] += 1
            top1_logit_sums[top1_id] += float(top_logits[kept_order, 0])
            token_predictions.append({
                "kept_token_order": kept_order,
                "original_visual_token_index": int(original_index),
                "top_vocabulary": vocabulary,
            })

        # A compact top-1 frequency table makes dominant vocabulary directions
        # visible without scanning every patch.  Detailed per-token top-k values
        # remain available below for spatial or trajectory analysis.
        sorted_top1 = sorted(
            top1_counts,
            key=lambda vocabulary_token_id: (
                -top1_counts[vocabulary_token_id],
                -(
                    top1_logit_sums[vocabulary_token_id]
                    / top1_counts[vocabulary_token_id]
                ),
                vocabulary_token_id,
            ),
        )[: self.summary_top_tokens]
        top1_summary = [
            {
                **self._describe_token(token_id),
                "visual_token_count": top1_counts[token_id],
                "visual_token_fraction": (
                    top1_counts[token_id] / selected_indices.numel()
                ),
                "mean_top1_logit": (
                    top1_logit_sums[token_id] / top1_counts[token_id]
                ),
            }
            for token_id in sorted_top1
        ]

        return {
            "layer_index_zero_based": int(layer_index),
            "layer_number_one_based": int(layer_index) + 1,
            "num_visual_tokens": selected_indices.numel(),
            "top1_vocabulary_summary": top1_summary,
            "token_predictions": token_predictions,
        }

    def _build_candidate_record(
        self,
        *,
        role: str,
        candidate_pos: int,
        final_rank: int,
        global_idx: int,
        local_page_id: int,
        stats: Mapping[str, Any],
    ) -> Dict[str, Any]:
        scores = self._cpu_1d_tensor(stats["scores"], torch.float32, "scores")
        kept_mask = self._cpu_1d_tensor(
            stats["kept_mask"], torch.bool, "kept_mask"
        )
        selected_indices = self._cpu_1d_tensor(
            stats["selected_indices"], torch.long, "selected_indices"
        )
        sequence_positions = self._cpu_1d_tensor(
            stats["visual_sequence_positions"],
            torch.long,
            "visual_sequence_positions",
        )

        if scores.numel() == 0 or scores.numel() != kept_mask.numel():
            raise ValueError("Candidate scores and kept_mask must be non-empty and aligned")
        expected_selected = torch.where(kept_mask)[0]
        if not torch.equal(selected_indices, expected_selected):
            raise ValueError(
                "selected_indices must equal the True positions of kept_mask in "
                "original visual-token order"
            )
        if sequence_positions.numel() != selected_indices.numel():
            raise ValueError(
                "Each retained visual token must have one pruned LLM sequence position"
            )

        grid_thw = [int(value) for value in stats["image_grid_thw"]]
        grid_t, merged_grid_h, merged_grid_w = merged_visual_grid_shape(
            grid_thw, self.spatial_merge_size
        )
        expected_original_tokens = grid_t * merged_grid_h * merged_grid_w
        if scores.numel() != expected_original_tokens:
            raise ValueError(
                f"Visual score count {scores.numel()} does not match merged grid "
                f"{grid_t}x{merged_grid_h}x{merged_grid_w}"
            )

        source_image = load_rgb_image(self.image_binary_loader(global_idx))
        image_size = source_image.size
        kept_visual_tokens = []
        for kept_order, (original_index, sequence_position) in enumerate(
            zip(selected_indices.tolist(), sequence_positions.tolist())
        ):
            kept_visual_tokens.append({
                "kept_token_order": kept_order,
                "original_visual_token_index": int(original_index),
                "pruned_llm_sequence_position": int(sequence_position),
                "query_image_max_similarity": float(scores[original_index]),
                **visual_token_patch_geometry(
                    int(original_index),
                    grid_thw,
                    self.spatial_merge_size,
                    image_size,
                ),
            })

        layer_hidden_states = stats["layer_hidden_states"]
        if not isinstance(layer_hidden_states, Mapping) or not layer_hidden_states:
            raise ValueError("Candidate has no late-layer visual hidden states")
        layer_records = [
            self._layer_record(
                int(layer_index),
                torch.as_tensor(hidden_states),
                selected_indices,
            )
            for layer_index, hidden_states in sorted(
                layer_hidden_states.items(), key=lambda item: int(item[0])
            )
        ]

        pruned_indices = torch.where(~kept_mask)[0].tolist()
        return {
            "role": role,
            "candidate_pos": candidate_pos,
            "candidate_letter": chr(ord("A") + candidate_pos),
            "final_rank": final_rank,
            "global_idx": global_idx,
            "local_page_id": local_page_id,
            "is_ground_truth": role == "highest_ranked_correct",
            "source_image_size": {
                "width": image_size[0],
                "height": image_size[1],
            },
            "image_grid_thw_before_spatial_merge": grid_thw,
            "spatial_merge_size": self.spatial_merge_size,
            "merged_visual_grid": {
                "temporal": grid_t,
                "rows": merged_grid_h,
                "columns": merged_grid_w,
            },
            "pruning_threshold_similarity": float(stats["threshold"]),
            "num_original_visual_tokens": scores.numel(),
            "num_kept_visual_tokens": selected_indices.numel(),
            "num_pruned_visual_tokens": len(pruned_indices),
            "actual_keep_fraction": selected_indices.numel() / scores.numel(),
            "kept_visual_token_indices": selected_indices.tolist(),
            "pruned_visual_token_indices": pruned_indices,
            "kept_visual_tokens": kept_visual_tokens,
            "logit_lens_layers": layer_records,
        }

    def add_query(
        self,
        result: Mapping[str, Any],
        candidate_stats: Sequence[Mapping[str, Any]],
    ) -> None:
        """Select one best-ranked GT and the three best-ranked non-GT pages."""
        self.total_queries_seen += 1
        global_indices = [int(value) for value in result["top_k_global_indices"]]
        ranked_indices = [int(value) for value in result["ranked_indices"]]
        gt_page_ids = {int(value) for value in result["page_id"]}

        if not gt_page_ids:
            self.skipped_counts["no_ground_truth_page"] += 1
            return
        if len(candidate_stats) != len(global_indices):
            raise ValueError(
                "Logit Lens statistics must cover every first-stage candidate"
            )

        expected_positions = set(range(len(global_indices)))
        if len(ranked_indices) != len(expected_positions) or set(
            ranked_indices
        ) != expected_positions:
            raise ValueError("Final ranking is not a permutation of candidate positions")
        stats_by_position = {
            int(stats["candidate_pos"]): stats for stats in candidate_stats
        }
        if set(stats_by_position) != expected_positions:
            raise ValueError("Candidate statistics have missing or duplicate positions")

        start_idx = int(result["start_idx"])
        end_idx = int(result["end_idx"])
        local_page_ids = [value - start_idx for value in global_indices]
        if any(
            not 0 <= page_id <= end_idx - start_idx
            for page_id in local_page_ids
        ):
            raise ValueError("Candidate global index falls outside its document range")

        # Filtering the final permutation preserves reranking order.  Therefore
        # the first correct position is the best GT page and the first three
        # incorrect positions are exactly the requested hard negatives.
        correct_positions = [
            pos for pos in ranked_indices if local_page_ids[pos] in gt_page_ids
        ]
        incorrect_positions = [
            pos for pos in ranked_indices if local_page_ids[pos] not in gt_page_ids
        ]
        if not correct_positions:
            self.skipped_counts["correct_candidate_missing_from_topk"] += 1
            return
        if len(incorrect_positions) < 3:
            self.skipped_counts["fewer_than_three_incorrect_candidates"] += 1
            return

        rank_by_position = {
            candidate_pos: final_rank + 1
            for final_rank, candidate_pos in enumerate(ranked_indices)
        }
        chosen = [("highest_ranked_correct", correct_positions[0])]
        chosen.extend(
            (f"highest_ranked_incorrect_{index + 1}", candidate_pos)
            for index, candidate_pos in enumerate(incorrect_positions[:3])
        )

        candidates = [
            self._build_candidate_record(
                role=role,
                candidate_pos=candidate_pos,
                final_rank=rank_by_position[candidate_pos],
                global_idx=global_indices[candidate_pos],
                local_page_id=local_page_ids[candidate_pos],
                stats=stats_by_position[candidate_pos],
            )
            for role, candidate_pos in chosen
        ]

        top1_position = ranked_indices[0]
        self.query_records.append({
            "qid": result.get("qid", f"{result['doc_name']}_{result['q_idx']}"),
            "doc_name": result["doc_name"],
            "domain": result["domain"],
            "q_idx": int(result["q_idx"]),
            "query": result["query"],
            "ground_truth_page_ids": sorted(gt_page_ids),
            "recall_at_1_correct": local_page_ids[top1_position] in gt_page_ids,
            "candidates": candidates,
        })

    @staticmethod
    def _safe_path_component(value: Any, fallback: str) -> str:
        cleaned = re.sub(r"[^\w.-]+", "_", str(value), flags=re.UNICODE).strip("._")
        return (cleaned or fallback)[:100]

    @staticmethod
    def _build_query_word_registry(
        query_record: Dict[str, Any],
    ) -> Dict[int, Dict[str, Any]]:
        """Assign query-wide W labels and colors to every observed Top-1 token."""
        word_registry: Dict[int, Dict[str, Any]] = {}

        # Traversal order is deterministic: candidates follow final-rank-based
        # selection order, layers are already sorted, and patches retain their
        # original visual-token order.  First occurrence therefore gives stable
        # compact W labels without exposing long vocabulary strings on patches.
        for candidate_record in query_record["candidates"]:
            for layer_record in candidate_record["logit_lens_layers"]:
                for prediction in layer_record["token_predictions"]:
                    vocabulary = prediction.get("top_vocabulary", [])
                    if not vocabulary:
                        raise ValueError(
                            "A retained patch has no Logit Lens vocabulary result"
                        )
                    top1 = vocabulary[0]
                    token_id = int(top1["token_id"])
                    if token_id in word_registry:
                        continue

                    word_number = len(word_registry)
                    word_registry[token_id] = {
                        "word_number": word_number,
                        "word_label": f"W{word_number}",
                        "token_id": token_id,
                        "vocab_token": top1.get("vocab_token"),
                        "decoded_text": top1.get("decoded_text", ""),
                        "is_special_token": bool(
                            top1.get("is_special_token", False)
                        ),
                        "color_rgb": list(_logit_lens_word_color(word_number)),
                    }

        query_record["logit_lens_word_legend"] = [
            dict(word_info) for word_info in word_registry.values()
        ]
        return word_registry

    def _write_candidate_images(
        self,
        output_path: Path,
        query_index: int,
        query_record: Dict[str, Any],
        candidate_record: Dict[str, Any],
        word_registry: Mapping[int, Mapping[str, Any]],
    ) -> None:
        qid_component = self._safe_path_component(
            query_record["qid"], f"query_{query_index:02d}"
        )
        query_dir = output_path / f"query_{query_index:02d}_{qid_component}"
        role_component = self._safe_path_component(
            candidate_record["role"], "candidate"
        )
        candidate_dir = query_dir / (
            f"{role_component}_{candidate_record['candidate_letter']}_"
            f"page_{candidate_record['local_page_id']}"
        )
        candidate_dir.mkdir(parents=True, exist_ok=True)

        source_image = load_rgb_image(
            self.image_binary_loader(int(candidate_record["global_idx"]))
        )
        original_path = candidate_dir / "original.png"
        overlay_path = candidate_dir / "kept_patches.png"
        source_image.save(original_path, format="PNG")

        annotated = draw_kept_patch_overlay(
            source_image,
            candidate_record["kept_visual_token_indices"],
            candidate_record["image_grid_thw_before_spatial_merge"],
            candidate_record["spatial_merge_size"],
            fill_alpha=self.overlay_alpha,
        )
        annotated.save(overlay_path, format="PNG")

        image_files: Dict[str, Any] = {
            "original": original_path.relative_to(output_path).as_posix(),
            "kept_patch_overlay": overlay_path.relative_to(output_path).as_posix(),
        }

        # Each sampled layer receives a compact Top-1 map and a second detailed
        # map whose patch label also includes the raw logit.  Both share the
        # query-wide registry, so W3 means the same vocabulary token everywhere.
        word_maps_by_layer: Dict[str, Dict[str, str]] = {}
        for layer_record in candidate_record["logit_lens_layers"]:
            layer_number = int(layer_record["layer_number_one_based"])
            layer_key = str(layer_number)
            filename_prefix = f"layer_{layer_number:02d}"
            top1_map_path = candidate_dir / f"{filename_prefix}_top1_word_map.png"
            detail_map_path = candidate_dir / f"{filename_prefix}_word_details.png"

            top1_map = draw_logit_lens_word_patch_map(
                source_image,
                layer_record,
                word_registry,
                candidate_record["image_grid_thw_before_spatial_merge"],
                candidate_record["spatial_merge_size"],
                fill_alpha=self.overlay_alpha,
                show_top1_logit=False,
                annotation_font_path=self.annotation_font_path,
                patch_label_mode=self.patch_label_mode,
            )
            top1_map.save(top1_map_path, format="PNG")

            detail_map = draw_logit_lens_word_patch_map(
                source_image,
                layer_record,
                word_registry,
                candidate_record["image_grid_thw_before_spatial_merge"],
                candidate_record["spatial_merge_size"],
                fill_alpha=self.overlay_alpha,
                show_top1_logit=True,
                annotation_font_path=self.annotation_font_path,
                patch_label_mode=self.patch_label_mode,
            )
            detail_map.save(detail_map_path, format="PNG")
            word_maps_by_layer[layer_key] = {
                "top1_word_map": top1_map_path.relative_to(output_path).as_posix(),
                "word_details": detail_map_path.relative_to(output_path).as_posix(),
            }

        image_files["logit_lens_word_maps_by_layer"] = word_maps_by_layer
        candidate_record["image_files"] = image_files

    @staticmethod
    def _compact_vocabulary_word(vocabulary_record: Mapping[str, Any]) -> str:
        """Return one readable word while hiding tokenizer bookkeeping fields."""
        decoded_text = str(vocabulary_record.get("decoded_text") or "")
        if decoded_text and not decoded_text.isspace():
            return decoded_text

        vocab_token = vocabulary_record.get("vocab_token")
        if vocab_token is not None and str(vocab_token):
            return str(vocab_token)
        return f"<token_id:{int(vocabulary_record['token_id'])}>"

    def _compact_visual_tokens(
        self,
        candidate_record: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        """Build one concise pruning-and-layer-word record per original token."""
        num_original_tokens = int(candidate_record["num_original_visual_tokens"])
        kept_indices = {
            int(index) for index in candidate_record["kept_visual_token_indices"]
        }
        words_by_token: Dict[int, Dict[str, List[str]]] = {
            index: {} for index in kept_indices
        }
        expected_layer_keys = []

        # The detailed in-memory layer representation is reduced to a one-based
        # layer-number key and an ordered list of decoded Top-K words. Rank, token
        # ID, special-token flags and raw logits are intentionally not serialized.
        for layer_record in candidate_record["logit_lens_layers"]:
            layer_key = str(int(layer_record["layer_number_one_based"]))
            expected_layer_keys.append(layer_key)
            for prediction in layer_record["token_predictions"]:
                visual_token_index = int(
                    prediction["original_visual_token_index"]
                )
                if visual_token_index not in kept_indices:
                    raise RuntimeError(
                        "A pruned visual token unexpectedly has a late-layer "
                        f"Logit Lens prediction: {visual_token_index}"
                    )
                words_by_token[visual_token_index][layer_key] = [
                    self._compact_vocabulary_word(vocabulary_record)
                    for vocabulary_record in prediction["top_vocabulary"]
                ]

        expected_layer_key_set = set(expected_layer_keys)
        for visual_token_index, layer_words in words_by_token.items():
            if set(layer_words) != expected_layer_key_set:
                raise RuntimeError(
                    f"Retained visual token {visual_token_index} does not have "
                    "Logit Lens words for every selected layer"
                )

        merged_grid = candidate_record["merged_visual_grid"]
        grid_t = int(merged_grid["temporal"])
        grid_rows = int(merged_grid["rows"])
        grid_columns = int(merged_grid["columns"])
        if grid_t * grid_rows * grid_columns != num_original_tokens:
            raise RuntimeError(
                "Merged visual grid does not match the original visual-token count"
            )

        visual_tokens = []
        tokens_per_frame = grid_rows * grid_columns
        for visual_token_index in range(num_original_tokens):
            temporal_index = visual_token_index // tokens_per_frame
            spatial_index = visual_token_index % tokens_per_frame
            visual_tokens.append({
                "visual_token_index": visual_token_index,
                "pruned": visual_token_index not in kept_indices,
                "patch_grid_position_trc": [
                    temporal_index,
                    spatial_index // grid_columns,
                    spatial_index % grid_columns,
                ],
                "logit_lens_words_by_layer": words_by_token.get(
                    visual_token_index, {}
                ),
            })
        return visual_tokens

    def _compact_candidate_record(
        self,
        candidate_record: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Keep candidate identity, pruning totals, images and compact token rows."""
        return {
            "role": candidate_record["role"],
            "candidate_letter": candidate_record["candidate_letter"],
            "final_rank": candidate_record["final_rank"],
            "local_page_id": candidate_record["local_page_id"],
            "is_ground_truth": candidate_record["is_ground_truth"],
            "pruning_threshold_similarity": candidate_record[
                "pruning_threshold_similarity"
            ],
            "num_original_visual_tokens": candidate_record[
                "num_original_visual_tokens"
            ],
            "num_kept_visual_tokens": candidate_record[
                "num_kept_visual_tokens"
            ],
            "num_pruned_visual_tokens": candidate_record[
                "num_pruned_visual_tokens"
            ],
            "actual_keep_fraction": candidate_record["actual_keep_fraction"],
            "merged_visual_grid": candidate_record["merged_visual_grid"],
            "visual_tokens": self._compact_visual_tokens(candidate_record),
            "image_files": candidate_record["image_files"],
        }

    def _compact_query_record(
        self,
        query_record: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Remove duplicated diagnostic details before writing a query to JSON."""
        return {
            "qid": query_record["qid"],
            "doc_name": query_record["doc_name"],
            "domain": query_record["domain"],
            "q_idx": query_record["q_idx"],
            "query": query_record["query"],
            "ground_truth_page_ids": query_record["ground_truth_page_ids"],
            "recall_at_1_correct": query_record["recall_at_1_correct"],
            "logit_lens_word_legend": [
                {
                    "word_label": word_info["word_label"],
                    "vocab_token": word_info["vocab_token"],
                    "decoded_text": word_info["decoded_text"],
                    "is_special_token": word_info["is_special_token"],
                    "color_rgb": word_info["color_rgb"],
                }
                for word_info in query_record["logit_lens_word_legend"]
            ],
            "candidates": [
                self._compact_candidate_record(candidate_record)
                for candidate_record in query_record["candidates"]
            ],
        }

    def save(
        self,
        output_dir: str,
        *,
        run_metadata: Mapping[str, Any],
    ) -> Dict[str, str]:
        """Write source images, retained-patch overlays and self-describing JSON."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Image paths and the query-wide word registry are inserted before JSON
        # emission.  Every candidate can therefore link directly to its source,
        # pruning overlay, and layer-specific vocabulary maps.
        for query_index, query_record in enumerate(self.query_records, start=1):
            word_registry = self._build_query_word_registry(query_record)
            for candidate_record in query_record["candidates"]:
                self._write_candidate_images(
                    output_path,
                    query_index,
                    query_record,
                    candidate_record,
                    word_registry,
                )

        json_path = output_path / "visual_logit_lens_analysis.json"
        payload = {
            "schema_version": 3,
            "description": (
                "Compact late-layer Logit Lens words and pruning status for every "
                "visual token from one highest-ranked GT and three highest-ranked "
                "non-GT candidates per query."
            ),
            "interpretation_warning": (
                "A vocabulary projection indicates directions readable by the final "
                "RMSNorm and LM head. It is not a causal attribution or guaranteed "
                "natural-language label for an image patch."
            ),
            "projection_semantics": {
                "operation": "final language-model RMSNorm followed by lm_head",
                "stored_result": (
                    "decoded Top-K words ordered by raw logit; token IDs and logit "
                    "values are omitted from JSON, while detail images display the "
                    "Top-1 raw logit"
                ),
                "layer_key": "one-based decoder block number stored as a JSON key",
                "token_position": (
                    "retained visual-token position in the pruned multimodal prefill"
                ),
            },
            "patch_semantics": {
                "visual_token_index": (
                    "zero-based index in the per-image merged visual grid before pruning"
                ),
                "patch_grid_position_trc": (
                    "[temporal index, row, column] in the merged visual grid"
                ),
                "overlay": (
                    "green/yellow cells are the actual QI-Early retained tokens; "
                    "faint white lines show the complete merged grid"
                ),
                "logit_lens_word_maps": (
                    "only retained cells are colored; W labels and colors use the "
                    "query-level logit_lens_word_legend and stay consistent across "
                    "all four candidates and sampled layers"
                ),
                "patch_label_mode": self.patch_label_mode,
                "word_details": (
                    "one_per_word labels only the highest-logit representative patch "
                    "for each Top-1 vocabulary word; all matching retained patches "
                    "keep the same color. Fonts shrink per labeled patch, and "
                    "exceptionally small cells prioritize W# or its numeric portion"
                ),
            },
            "run": dict(run_metadata),
            "total_queries_seen": self.total_queries_seen,
            "num_queries_exported": len(self.query_records),
            "skipped_queries": dict(sorted(self.skipped_counts.items())),
            "queries": [
                self._compact_query_record(query_record)
                for query_record in self.query_records
            ],
        }
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        return {
            "output_directory": str(output_path),
            "analysis_json": str(json_path),
        }
