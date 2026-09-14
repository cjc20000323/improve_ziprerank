import io
import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from models.qwen3vl_with_qi_early_logit_lens import (
    select_spaced_late_layer_indices,
)
from scripts.evaluate_visual_logit_lens import (
    _consume_visual_logit_lens_stats,
)
from utils.visual_logit_lens_analysis import (
    LogitLensVocabularyProjector,
    VisualLogitLensCollector,
    draw_kept_patch_overlay,
    select_visual_logit_lens_query_subset,
    visual_token_patch_geometry,
)


class _FakeTokenizer:
    all_special_ids = [99]

    @staticmethod
    def convert_ids_to_tokens(token_id):
        return f"vocab_{token_id}"

    @staticmethod
    def decode(token_ids, **_kwargs):
        return f"decoded_{token_ids[0]}"


class _FakeProjector:
    """Return deterministic top-2 vocabulary results without a real LM head."""

    @staticmethod
    def project(hidden_states):
        num_tokens = hidden_states.shape[0]
        first_ids = 10 + torch.arange(num_tokens) % 2
        second_ids = torch.full((num_tokens,), 99)
        top_ids = torch.stack([first_ids, second_ids], dim=1)
        top_logits = torch.stack([
            torch.linspace(5.0, 4.0, num_tokens),
            torch.linspace(2.0, 1.0, num_tokens),
        ], dim=1)
        return top_ids, top_logits


def _image_bytes(width=100, height=50):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=(120, 120, 120)).save(
        buffer, format="PNG"
    )
    return buffer.getvalue()


class VisualLogitLensAnalysisTest(unittest.TestCase):
    @staticmethod
    def _eligible_result(index):
        start_idx = 1000 + index * 10
        return {
            "qid": f"doc_{index}",
            "doc_name": f"doc_{index}",
            "domain": "News",
            "q_idx": index,
            "query": f"query {index}",
            "page_id": [1],
            "start_idx": start_idx,
            "end_idx": start_idx + 9,
            "top_k_global_indices": [
                start_idx + 1,
                start_idx + 2,
                start_idx + 3,
                start_idx + 4,
            ],
        }

    @staticmethod
    def _candidate_stats():
        records = []
        kept_mask = torch.tensor(
            [True, False, True, False, True, False, True, False]
        )
        selected_indices = torch.tensor([0, 2, 4, 6])
        for candidate_pos in range(5):
            records.append({
                "candidate_pos": candidate_pos,
                "scores": torch.linspace(0.1, 0.8, 8) + candidate_pos * 0.01,
                "kept_mask": kept_mask,
                "selected_indices": selected_indices,
                "visual_sequence_positions": torch.tensor([20, 21, 22, 23]),
                "threshold": 0.4 + candidate_pos * 0.01,
                "num_original_tokens": 8,
                "num_kept_tokens": 4,
                "image_grid_thw": [1, 4, 8],
                "layer_hidden_states": {
                    6: torch.arange(12, dtype=torch.float32).reshape(4, 3),
                    7: torch.arange(12, dtype=torch.float32).reshape(4, 3) + 1,
                },
            })
        return records

    @staticmethod
    def _reranked_result():
        return {
            "qid": "doc_0",
            "doc_name": "doc",
            "domain": "News",
            "q_idx": 0,
            "query": "test query",
            # Global rows map to local pages [1, 2, 3, 4, 5]; page 3 is GT.
            "page_id": [3],
            "start_idx": 100,
            "end_idx": 110,
            "top_k_global_indices": [101, 102, 103, 104, 105],
            # Best wrong=pos1, best GT=pos2, then wrong pos4 and pos0.
            "ranked_indices": [1, 2, 4, 0, 3],
        }

    def test_sampling_stops_after_twenty_eligible_queries(self):
        first_stage = {
            f"key_{index}": self._eligible_result(index) for index in range(30)
        }

        selected_a, metadata_a = select_visual_logit_lens_query_subset(
            first_stage, num_queries=20, seed=7
        )
        selected_b, metadata_b = select_visual_logit_lens_query_subset(
            first_stage, num_queries=20, seed=7
        )

        self.assertEqual(list(selected_a), list(selected_b))
        self.assertEqual(len(selected_a), 20)
        self.assertEqual(metadata_a["queries_inspected_before_early_stop"], 20)
        self.assertEqual(metadata_a["queries_not_inspected"], 10)
        self.assertEqual(metadata_a, metadata_b)

    def test_spaced_layer_selection_concentrates_four_points_late(self):
        layer_indices = select_spaced_late_layer_indices(
            num_layers=36,
            num_selected_layers=4,
            start_ratio=0.6,
        )

        # For a 36-block model, the selection covers depths 61%-100%, includes
        # the final block, and deliberately differs from four consecutive tails.
        self.assertEqual(layer_indices, (21, 26, 30, 35))
        self.assertNotEqual(layer_indices, tuple(range(32, 36)))
        self.assertEqual(len(set(layer_indices)), 4)
        self.assertTrue(all(index >= 18 for index in layer_indices))

    def test_visual_token_geometry_maps_merged_grid_to_source_pixels(self):
        geometry = visual_token_patch_geometry(
            visual_token_index=5,
            image_grid_thw=[1, 4, 8],
            spatial_merge_size=2,
            image_size=(100, 50),
        )

        self.assertEqual(geometry["merged_grid_row"], 1)
        self.assertEqual(geometry["merged_grid_column"], 1)
        self.assertEqual(
            geometry["source_image_pixel_box_xyxy"], [25, 25, 50, 50]
        )

    def test_patch_overlay_preserves_source_resolution(self):
        source = Image.new("RGB", (100, 50), color=(120, 120, 120))
        annotated = draw_kept_patch_overlay(
            source,
            kept_visual_token_indices=[0, 5],
            image_grid_thw=[1, 4, 8],
            spatial_merge_size=2,
        )

        self.assertEqual(annotated.size, source.size)
        self.assertNotEqual(annotated.getpixel((10, 10)), source.getpixel((10, 10)))

    def test_projector_matches_direct_final_norm_and_lm_head(self):
        class FakeLanguageModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.LayerNorm(3)

        class FakeBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = FakeLanguageModel()

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = FakeBackbone()
                self.lm_head = torch.nn.Linear(3, 4, bias=False)

        model = FakeModel()
        with torch.no_grad():
            model.lm_head.weight.copy_(torch.tensor([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-1.0, -1.0, -1.0],
            ]))
        hidden_states = torch.tensor([
            [3.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 4.0],
        ])

        # A chunk size smaller than the token count exercises the bounded-memory
        # path.  Its result must still equal a direct final-norm/head projection.
        projector = LogitLensVocabularyProjector(model, top_k=2, chunk_size=2)
        actual_ids, actual_logits = projector.project(hidden_states)
        expected_logits = model.lm_head(model.model.language_model.norm(hidden_states))
        expected_logits, expected_ids = torch.topk(expected_logits, k=2, dim=-1)

        self.assertTrue(torch.equal(actual_ids, expected_ids))
        self.assertTrue(torch.allclose(actual_logits, expected_logits))

    def test_collector_selects_gt_and_three_best_wrong_candidates(self):
        image_binary = _image_bytes()
        collector = VisualLogitLensCollector(
            tokenizer=_FakeTokenizer(),
            projector=_FakeProjector(),
            image_binary_loader=lambda _global_idx: image_binary,
            spatial_merge_size=2,
        )
        collector.add_query(self._reranked_result(), self._candidate_stats())

        self.assertEqual(len(collector.query_records), 1)
        candidates = collector.query_records[0]["candidates"]
        self.assertEqual(
            [candidate["candidate_pos"] for candidate in candidates],
            [2, 1, 4, 0],
        )
        self.assertEqual(
            [candidate["role"] for candidate in candidates],
            [
                "highest_ranked_correct",
                "highest_ranked_incorrect_1",
                "highest_ranked_incorrect_2",
                "highest_ranked_incorrect_3",
            ],
        )
        self.assertEqual(candidates[0]["num_kept_visual_tokens"], 4)
        self.assertEqual(len(candidates[0]["logit_lens_layers"]), 2)
        first_prediction = candidates[0]["logit_lens_layers"][0][
            "token_predictions"
        ][0]
        self.assertEqual(first_prediction["original_visual_token_index"], 0)
        self.assertEqual(
            first_prediction["top_vocabulary"][0]["vocab_token"], "vocab_10"
        )

        # Saving exercises the complete artifact layout: four untouched source
        # images, four overlays and one JSON file with relative image paths.
        with tempfile.TemporaryDirectory() as temporary_dir:
            paths = collector.save(temporary_dir, run_metadata={"test": True})
            payload = json.loads(
                Path(paths["analysis_json"]).read_text(encoding="utf-8")
            )
            self.assertEqual(payload["schema_version"], 2)
            self.assertEqual(payload["num_queries_exported"], 1)
            compact_candidate = payload["queries"][0]["candidates"][0]
            self.assertNotIn("kept_visual_tokens", compact_candidate)
            self.assertNotIn("logit_lens_layers", compact_candidate)
            self.assertEqual(len(compact_candidate["visual_tokens"]), 8)

            # Every original token now has one compact row. Retained tokens carry
            # per-layer word lists, while pruned tokens explicitly carry no lens.
            retained_token = compact_candidate["visual_tokens"][0]
            pruned_token = compact_candidate["visual_tokens"][1]
            self.assertFalse(retained_token["pruned"])
            self.assertEqual(
                retained_token["logit_lens_words_by_layer"]["7"][0],
                "decoded_10",
            )
            self.assertTrue(pruned_token["pruned"])
            self.assertEqual(pruned_token["logit_lens_words_by_layer"], {})
            for candidate in payload["queries"][0]["candidates"]:
                for relative_path in candidate["image_files"].values():
                    self.assertTrue((Path(temporary_dir) / relative_path).is_file())

    def test_consumer_splits_concatenated_layer_states_by_image(self):
        class FakePruner:
            last_similarity_stats = [
                {
                    "scores": torch.tensor([0.8, 0.1, 0.7, 0.2]),
                    "kept_mask": torch.tensor([True, False, True, False]),
                    "threshold": torch.tensor(0.7),
                    "num_original_tokens": 4,
                    "num_kept_tokens": 2,
                },
                {
                    "scores": torch.tensor([0.6, 0.1, 0.5, 0.2]),
                    "kept_mask": torch.tensor([True, False, True, False]),
                    "threshold": torch.tensor(0.5),
                    "num_original_tokens": 4,
                    "num_kept_tokens": 2,
                },
            ]

        class FakeModel:
            def __init__(self):
                self.qi_early_pruner = FakePruner()
                self.capture = {
                    "capture_complete": True,
                    "image_grid_thw": torch.tensor([[1, 4, 4], [1, 4, 4]]),
                    "selected_indices_by_image": [
                        torch.tensor([0, 2]),
                        torch.tensor([0, 2]),
                    ],
                    "visual_sequence_positions_by_image": [
                        torch.tensor([10, 11]),
                        torch.tensor([20, 21]),
                    ],
                    "pruned_tokens_per_image": [2, 2],
                    "layer_hidden_states": {
                        7: torch.arange(12, dtype=torch.float32).reshape(4, 3)
                    },
                }

            def pop_last_logit_lens_capture(self):
                capture = self.capture
                self.capture = None
                return capture

        model = FakeModel()
        copied = _consume_visual_logit_lens_stats(model, [4, 9])

        self.assertEqual([item["candidate_pos"] for item in copied], [4, 9])
        self.assertEqual(copied[0]["layer_hidden_states"][7].shape, (2, 3))
        self.assertEqual(copied[1]["layer_hidden_states"][7][0, 0].item(), 6.0)
        self.assertIsNone(model.qi_early_pruner.last_similarity_stats)
        self.assertIsNone(model.capture)


if __name__ == "__main__":
    unittest.main()
