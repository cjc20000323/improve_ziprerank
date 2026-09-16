import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from utils.similarity_analysis import TokenSimilarityAnalysisCollector
from utils.similarity_image_export import export_selected_similarity_candidate_images


class SimilarityImageExportTest(unittest.TestCase):
    @staticmethod
    def _example(group: str):
        candidates = []
        for candidate_index in range(4):
            candidates.append({
                "role": f"candidate_role_{candidate_index}",
                "role_label": f"Candidate role {candidate_index}",
                "candidate_pos": candidate_index,
                "candidate_letter": chr(ord("A") + candidate_index),
                "final_rank": candidate_index + 1,
                "global_idx": 10 + candidate_index,
                "local_page_id": 20 + candidate_index,
                "is_gt": candidate_index == (0 if group == "correct" else 3),
            })
        return {
            "qid": f"query_{group}",
            "doc_name": "doc",
            "domain": "News",
            "q_idx": 2,
            "query": "Which page answers the query?",
            "ground_truth_page_ids": [20],
            "recall_at_1": 1.0 if group == "correct" else 0.0,
            "recall1_correct": group == "correct",
            "candidates": candidates,
        }

    @staticmethod
    def _png_bytes(width: int, height: int) -> bytes:
        buffer = io.BytesIO()
        Image.new("RGB", (width, height), color=(20, 40, 60)).save(
            buffer, format="PNG"
        )
        return buffer.getvalue()

    def test_exports_four_native_size_images_and_manifest_per_example(self):
        image_bytes = {
            global_idx: self._png_bytes(global_idx, global_idx - 3)
            for global_idx in range(10, 14)
        }

        with tempfile.TemporaryDirectory() as temporary_dir:
            paths = export_selected_similarity_candidate_images(
                example_groups={
                    "correct": [self._example("correct")],
                    "incorrect": [self._example("incorrect")],
                },
                output_dir=temporary_dir,
                keep_ratio=0.5,
                image_binary_loader=image_bytes.__getitem__,
            )

            manifest_path = Path(paths["candidate_images_manifest"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["keep_ratio"], 0.5)
            self.assertFalse(
                manifest["image_semantics"]["resize_or_crop_applied"]
            )

            for group_name in ("correct", "incorrect"):
                example = manifest["groups"][group_name][0]
                self.assertEqual(len(example["candidates"]), 4)
                for candidate in example["candidates"]:
                    image_path = Path(temporary_dir) / candidate["image_file"]
                    self.assertTrue(image_path.is_file())
                    global_idx = candidate["global_idx"]
                    with Image.open(image_path) as saved_image:
                        self.assertEqual(saved_image.size, (global_idx, global_idx - 3))
                        self.assertEqual(saved_image.mode, "RGB")
                    self.assertFalse(candidate["saved_image"]["resized"])

    def test_collector_save_invokes_optional_image_export(self):
        collector = TokenSimilarityAnalysisCollector(num_examples=1, seed=7)

        def loader(_global_idx: int) -> bytes:
            return b"unused"

        exported_path = "candidate_manifest.json"

        # 本测试只检查 collector 与图片导出模块的接线，不需要真正加载绘图后端。
        with tempfile.TemporaryDirectory() as temporary_dir, \
                patch("utils.similarity_analysis.ensure_matplotlib_available"), \
                patch.object(collector, "_plot_overall"), \
                patch.object(collector, "_plot_top1_outcome"), \
                patch.object(collector, "_plot_examples", side_effect=[[], []]), \
                patch(
                    "utils.similarity_analysis.export_selected_similarity_candidate_images",
                    return_value={"candidate_images_manifest": exported_path},
                ) as image_export:
            paths = collector.save(
                temporary_dir,
                keep_ratio=0.5,
                image_binary_loader=loader,
            )

        self.assertEqual(paths["candidate_images_manifest"], exported_path)
        image_export.assert_called_once()
        self.assertIs(
            image_export.call_args.kwargs["image_binary_loader"],
            loader,
        )


if __name__ == "__main__":
    unittest.main()
