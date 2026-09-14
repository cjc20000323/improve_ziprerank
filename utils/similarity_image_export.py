"""把 token 相似度诊断样例中的候选页恢复为可直接查看的图片。"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

from PIL import Image


ImageBinaryLoader = Callable[[int], bytes]


def _safe_path_component(value: Any, fallback: str) -> str:
    """把候选角色等元数据转换成适合 Windows 和 Linux 的短文件名片段。"""
    cleaned = re.sub(r"[^\w.-]+", "_", str(value), flags=re.UNICODE).strip("._")
    return (cleaned or fallback)[:80]


def _decode_source_image(image_binary: bytes) -> Tuple[Image.Image, Dict[str, Any]]:
    """解码 parquet 字节，并保留转换前的格式、色彩模式和原始像素尺寸。"""
    with Image.open(io.BytesIO(bytes(image_binary))) as opened:
        source_metadata = {
            "format": opened.format or "unknown",
            "mode": opened.mode,
            "width": int(opened.width),
            "height": int(opened.height),
        }
        # convert() 会立即生成一份独立图像，因此关闭 BytesIO 和 opened 后仍可安全保存。
        # 这里只统一色彩模式，不执行缩放或裁剪，输出像素宽高与 parquet 原图一致。
        restored_image = opened.convert("RGB")
    return restored_image, source_metadata


def _candidate_identity(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    """只复制图片清单需要的候选身份字段，避免把逐 token tensor 写入 JSON。"""
    integer_fields = (
        "candidate_pos",
        "final_rank",
        "global_idx",
        "local_page_id",
    )
    identity: Dict[str, Any] = {
        "role": str(candidate["role"]),
        "role_label": str(candidate.get("role_label", candidate["role"])),
        "candidate_letter": str(candidate["candidate_letter"]),
        "is_gt": bool(candidate["is_gt"]),
    }
    for field_name in integer_fields:
        identity[field_name] = int(candidate[field_name])
    return identity


def _example_identity(example: Mapping[str, Any]) -> Dict[str, Any]:
    """复制能够把候选图片关联回 query 的紧凑样例元数据。"""
    return {
        "qid": str(example["qid"]),
        "doc_name": str(example.get("doc_name", "")),
        "domain": str(example.get("domain", "")),
        "q_idx": int(example.get("q_idx", 0)),
        "query": str(example.get("query", "")),
        "ground_truth_page_ids": [
            int(page_id) for page_id in example.get("ground_truth_page_ids", [])
        ],
        "recall_at_1": float(example.get("recall_at_1", 0.0)),
        "recall1_correct": bool(example.get("recall1_correct", False)),
    }


def export_selected_similarity_candidate_images(
    *,
    example_groups: Mapping[str, Sequence[Mapping[str, Any]]],
    output_dir: str,
    keep_ratio: float,
    image_binary_loader: ImageBinaryLoader,
) -> Dict[str, str]:
    """保存正确/错误诊断样例的四张候选原图以及文件到候选身份的映射。"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    keep_percent = int(round(keep_ratio * 100))
    directory_names = {
        "correct": f"recall1_correct_examples_keep{keep_percent}",
        "incorrect": f"recall1_incorrect_examples_keep{keep_percent}",
    }

    # 图片目录与现有的 example_XX.png 相似度分布图放在同一组目录中。
    # example_XX_candidates 下的候选顺序也与分布图从左到右的四个子图一致，
    # 因此既可以通过序号直接对照，也可以借助 manifest 精确核对页号和最终排名。
    exported_groups: Dict[str, Any] = {}
    for group_name in ("correct", "incorrect"):
        examples = list(example_groups.get(group_name, []))
        group_dir = output_path / directory_names[group_name]
        group_dir.mkdir(parents=True, exist_ok=True)
        exported_examples = []

        for example_index, example in enumerate(examples, start=1):
            candidates = list(example.get("candidates", []))
            if len(candidates) != 4:
                raise ValueError(
                    f"Similarity example {example.get('qid', example_index)!r} in "
                    f"group {group_name!r} must contain exactly four candidates; "
                    f"got {len(candidates)}"
                )

            candidate_dir = group_dir / f"example_{example_index:02d}_candidates"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            exported_candidates = []
            for candidate_index, candidate in enumerate(candidates, start=1):
                identity = _candidate_identity(candidate)
                role_component = _safe_path_component(identity["role"], "candidate")
                gt_component = "gt" if identity["is_gt"] else "non_gt"
                filename = (
                    f"candidate_{candidate_index:02d}_{identity['candidate_letter']}_"
                    f"{role_component}_page_{identity['local_page_id']}_"
                    f"rank_{identity['final_rank']}_{gt_component}.png"
                )
                image_path = candidate_dir / filename

                # global_idx 是 parquet 的位置索引；只对 reservoir sampling 最终留下的
                # 少量样例执行读取和解码，不会在统计全测试集直方图时复制所有页面。
                image_binary = image_binary_loader(identity["global_idx"])
                restored_image, source_metadata = _decode_source_image(image_binary)
                restored_image.save(image_path, format="PNG")

                exported_candidates.append({
                    "candidate_index_in_similarity_figure": candidate_index,
                    **identity,
                    "image_file": image_path.relative_to(output_path).as_posix(),
                    "source_image": source_metadata,
                    "saved_image": {
                        "format": "PNG",
                        "mode": "RGB",
                        "width": restored_image.width,
                        "height": restored_image.height,
                        "resized": False,
                    },
                })

            exported_examples.append({
                "example_index": example_index,
                **_example_identity(example),
                "similarity_figure_file": (
                    group_dir / f"example_{example_index:02d}.png"
                ).relative_to(output_path).as_posix(),
                "candidate_image_directory": candidate_dir.relative_to(
                    output_path
                ).as_posix(),
                "candidates": exported_candidates,
            })

        exported_groups[group_name] = exported_examples

    manifest_path = output_path / f"selected_candidate_images_keep{keep_percent}.json"
    manifest = {
        "schema_version": 1,
        "keep_ratio": keep_ratio,
        "description": (
            "Source candidate pages for the sampled Recall@1-correct and "
            "Recall@1-incorrect token-similarity examples."
        ),
        "image_semantics": {
            "source": "image_binary at each candidate global_idx in the page parquet",
            "output_format": "PNG",
            "color_mode": "RGB",
            "resize_or_crop_applied": False,
            "candidate_order": (
                "candidate_index_in_similarity_figure matches the left-to-right "
                "candidate subplot order in example_XX.png"
            ),
            "path_base": "directory containing this manifest",
        },
        "groups": exported_groups,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"candidate_images_manifest": str(manifest_path)}
