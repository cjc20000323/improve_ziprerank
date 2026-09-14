#!/usr/bin/env python3
"""从已有相似度 JSON 和页面 parquet 补充导出四候选原图。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.similarity_image_export import (  # noqa: E402
    export_selected_similarity_candidate_images,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Restore the four candidate images for every sampled token-similarity "
            "example without rerunning model inference."
        )
    )
    parser.add_argument(
        "--examples_json",
        required=True,
        help=(
            "selected_token_similarity_examples_keepXX.json or "
            "selected_token_similarity_histograms_keepXX.json"
        ),
    )
    parser.add_argument(
        "--pages_parquet",
        required=True,
        help="Page parquet containing the image_binary column",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output root; defaults to the directory containing --examples_json",
    )
    return parser.parse_args(argv)


def _extract_example_groups(payload: Mapping[str, Any]) -> Dict[str, Sequence[Mapping[str, Any]]]:
    """同时兼容紧凑样例 JSON 和包含逐 bin 计数的直方图 JSON。"""
    if "correct_examples" in payload and "incorrect_examples" in payload:
        return {
            "correct": payload["correct_examples"],
            "incorrect": payload["incorrect_examples"],
        }

    groups = payload.get("groups")
    if isinstance(groups, Mapping) and "correct" in groups and "incorrect" in groups:
        return {
            "correct": groups["correct"],
            "incorrect": groups["incorrect"],
        }

    raise ValueError(
        "Input JSON must contain correct_examples/incorrect_examples or "
        "groups.correct/groups.incorrect"
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    examples_path = Path(args.examples_json).expanduser().resolve()
    output_path = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else examples_path.parent
    )

    payload = json.loads(examples_path.read_text(encoding="utf-8"))
    if "keep_ratio" not in payload:
        raise ValueError(f"Input JSON has no keep_ratio: {examples_path}")
    example_groups = _extract_example_groups(payload)

    # 这个入口只恢复已抽中样例的图片，不加载模型，也不重新运行推理。候选记录中的
    # global_idx 直接定位页面 parquet 的行；边界检查用于尽早发现 JSON 与 parquet
    # 来自不同数据版本的情况，避免悄悄导出错误页面。
    parquet_df = pd.read_parquet(args.pages_parquet, columns=["image_binary"])

    def load_image_binary(global_idx: int) -> bytes:
        row_index = int(global_idx)
        if not 0 <= row_index < len(parquet_df):
            raise IndexError(
                f"Candidate global_idx {row_index} is outside parquet row range "
                f"[0, {len(parquet_df)})"
            )
        return bytes(parquet_df.iloc[row_index]["image_binary"])

    paths = export_selected_similarity_candidate_images(
        example_groups=example_groups,
        output_dir=str(output_path),
        keep_ratio=float(payload["keep_ratio"]),
        image_binary_loader=load_image_binary,
    )
    print(json.dumps(paths, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
