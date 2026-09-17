import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from utils.pruning_recall_analysis import (
    PRUNING_TO_KEEP_RATIO,
    RECALL_CUTOFFS,
    analyze_pruning_recalls,
    build_correct_query_curve,
    build_correct_query_higher_rate_breakdown,
    get_recall_at_k_outcome,
    get_top1_outcome,
    save_pruning_recall_analysis,
)
from scripts.plot_pruning_recall_top1 import (
    DEFAULT_BREAKDOWN_OUTPUT_FILENAMES,
    DEFAULT_OUTPUT_FILENAMES,
    load_analysis_json,
    save_all_recall_correct_query_plots,
    save_all_recall_higher_rate_breakdown_plots,
)


class PruningRecallAnalysisTest(unittest.TestCase):
    @staticmethod
    def _result(
            doc_name,
            q_idx,
            ground_truth_page_ids,
            top1_candidate_pos=0,
            candidate_global_indices=None,
            ranked_indices=None,
    ):
        if candidate_global_indices is None:
            # With start_idx=100 these candidates have local page IDs [1, ..., 5].
            candidate_global_indices = [101, 102, 103, 104, 105]
        if ranked_indices is None:
            remaining_positions = [
                position
                for position in range(len(candidate_global_indices))
                if position != top1_candidate_pos
            ]
            ranked_indices = [top1_candidate_pos, *remaining_positions]
        return {
            "qid": f"{doc_name}_{q_idx}",
            "doc_name": doc_name,
            "domain": "News",
            "q_idx": q_idx,
            "query": f"query {doc_name} {q_idx}",
            "page_id": ground_truth_page_ids,
            "start_idx": 100,
            "end_idx": 109,
            "top_k_global_indices": candidate_global_indices,
            "ranked_indices": ranked_indices,
        }

    def test_pruning_sweep_covers_zero_through_ninety_by_ten(self):
        self.assertEqual(list(PRUNING_TO_KEEP_RATIO), list(range(0, 100, 10)))
        self.assertEqual(PRUNING_TO_KEEP_RATIO[0], 1.0)
        self.assertEqual(PRUNING_TO_KEEP_RATIO[50], 0.5)
        self.assertEqual(PRUNING_TO_KEEP_RATIO[90], 0.1)
        self.assertEqual(RECALL_CUTOFFS, (1, 3, 5))

    def test_get_top1_outcome_converts_candidate_position_to_local_page(self):
        result = self._result(
            doc_name="doc",
            q_idx=0,
            ground_truth_page_ids=[2],
            top1_candidate_pos=1,
        )

        outcome = get_top1_outcome(result)

        self.assertTrue(outcome["correct"])
        self.assertEqual(outcome["top1_candidate_pos"], 1)
        self.assertEqual(outcome["top1_global_page_id"], 102)
        self.assertEqual(outcome["top1_local_page_id"], 2)
        self.assertEqual(outcome["recall_at_1"], 1.0)

    def test_multiple_ground_truth_pages_use_hit_boolean_and_fractional_recall(self):
        result = self._result(
            doc_name="doc",
            q_idx=0,
            ground_truth_page_ids=[1, 3],
            top1_candidate_pos=2,
        )

        outcome = get_top1_outcome(result)

        self.assertTrue(outcome["correct"])
        self.assertEqual(outcome["recall_at_1"], 0.5)

    def test_recall_at_k_uses_all_ranked_pages_and_official_fraction(self):
        result = self._result(
            doc_name="doc",
            q_idx=0,
            ground_truth_page_ids=[2, 4],
            ranked_indices=[0, 3, 1, 2, 4],
        )

        recall1 = get_recall_at_k_outcome(result, 1)
        recall3 = get_recall_at_k_outcome(result, 3)

        self.assertFalse(recall1["correct"])
        self.assertEqual(recall1["recall_at_k"], 0.0)
        self.assertTrue(recall3["correct"])
        self.assertEqual(recall3["recall_at_k"], 1.0)
        self.assertEqual(recall3["retrieved_local_page_ids"], [1, 4, 2])
        self.assertEqual(recall3["hit_ground_truth_page_ids"], [2, 4])

    def test_analysis_groups_correct_and_incorrect_queries_for_every_cutoff(self):
        # Under the default order, A hits at rank 1, B at rank 2, C at rank 5,
        # and D never hits. Prune90 reverses the first-stage candidate order.
        specifications = {
            "a": [1],
            "b": [2],
            "c": [5],
            "d": [9],
        }
        results_by_pruning = {
            pruning_percent: []
            for pruning_percent in PRUNING_TO_KEEP_RATIO
        }
        for q_idx, (doc_name, ground_truth) in enumerate(specifications.items()):
            for pruning_percent in PRUNING_TO_KEEP_RATIO:
                ranked_indices = (
                    [4, 3, 2, 1, 0]
                    if pruning_percent == 90
                    else [0, 1, 2, 3, 4]
                )
                results_by_pruning[pruning_percent].append(
                    self._result(
                        doc_name=doc_name,
                        q_idx=q_idx,
                        ground_truth_page_ids=ground_truth,
                        ranked_indices=ranked_indices,
                    )
                )

        analysis = analyze_pruning_recalls(results_by_pruning)

        self.assertEqual(analysis["num_aligned_samples"], 4)
        conditions = analysis["conditions"]
        self.assertEqual(len(conditions), 10)
        self.assertEqual(conditions["prune0"]["keep_ratio"], 1.0)

        prune0_metrics = conditions["prune0"]["metrics"]
        self.assertEqual(prune0_metrics["recall_at_1"]["correct_count"], 1)
        self.assertEqual(prune0_metrics["recall_at_3"]["correct_count"], 2)
        self.assertEqual(prune0_metrics["recall_at_5"]["correct_count"], 3)
        self.assertEqual(
            [item["doc_name"] for item in prune0_metrics["recall_at_1"]["correct_queries"]],
            ["a"],
        )
        self.assertEqual(
            [item["doc_name"] for item in prune0_metrics["recall_at_1"]["incorrect_queries"]],
            ["b", "c", "d"],
        )

        prune90_metrics = conditions["prune90"]["metrics"]
        self.assertEqual(prune90_metrics["recall_at_1"]["correct_count"], 1)
        self.assertEqual(prune90_metrics["recall_at_3"]["correct_count"], 1)
        self.assertEqual(prune90_metrics["recall_at_5"]["correct_count"], 3)
        self.assertNotIn("transitions", analysis)
        self.assertNotIn("pairwise_outcome_tables", analysis)

        recall1_points = build_correct_query_curve(analysis, 1)["points"]
        self.assertEqual(
            [point["correct_at_pruning_rate"] for point in recall1_points],
            [1] * 10,
        )
        self.assertEqual(
            [
                point["correct_at_current_or_higher_pruning_union"]
                for point in recall1_points
            ],
            [2] * 9 + [1],
        )
        recall3_points = build_correct_query_curve(analysis, 3)["points"]
        self.assertEqual(
            [point["correct_at_pruning_rate"] for point in recall3_points],
            [2] * 9 + [1],
        )
        self.assertEqual(
            [
                point["correct_at_current_or_higher_pruning_union"]
                for point in recall3_points
            ],
            [3] * 9 + [1],
        )
        recall5_points = build_correct_query_curve(analysis, 5)["points"]
        self.assertEqual(
            [point["correct_at_pruning_rate"] for point in recall5_points],
            [3] * 10,
        )
        self.assertEqual(
            [
                point["correct_at_current_or_higher_pruning_union"]
                for point in recall5_points
            ],
            [3] * 10,
        )

        recall1_breakdown = build_correct_query_higher_rate_breakdown(
            analysis, 1
        )["points"]
        self.assertEqual(
            [
                point["current_correct_higher_rates_all_wrong"]
                for point in recall1_breakdown
            ],
            [0] * 8 + [1, 1],
        )
        self.assertEqual(
            [
                point["current_correct_and_any_higher_rate_correct"]
                for point in recall1_breakdown
            ],
            [1] * 8 + [0, 0],
        )
        self.assertEqual(
            [point["current_correct_total"] for point in recall1_breakdown],
            [1] * 10,
        )

        recall3_breakdown = build_correct_query_higher_rate_breakdown(
            analysis, 3
        )["points"]
        self.assertEqual(
            [
                point["current_correct_higher_rates_all_wrong"]
                for point in recall3_breakdown
            ],
            [0] * 8 + [2, 1],
        )
        self.assertEqual(
            [
                point["current_correct_and_any_higher_rate_correct"]
                for point in recall3_breakdown
            ],
            [2] * 8 + [0, 0],
        )

        recall5_breakdown = build_correct_query_higher_rate_breakdown(
            analysis, 5
        )["points"]
        self.assertEqual(
            [
                point["current_correct_higher_rates_all_wrong"]
                for point in recall5_breakdown
            ],
            [0] * 9 + [3],
        )
        self.assertEqual(
            [
                point["current_correct_and_any_higher_rate_correct"]
                for point in recall5_breakdown
            ],
            [3] * 9 + [0],
        )

    def test_compare_rejects_different_query_sets(self):
        common = self._result("doc", 0, [1], 0)
        analysis_input = {
            pruning_percent: [common]
            for pruning_percent in PRUNING_TO_KEEP_RATIO
        }
        analysis_input[90] = []

        with self.assertRaisesRegex(ValueError, "Query set mismatch"):
            analyze_pruning_recalls(analysis_input)

    def test_compare_rejects_different_first_stage_candidates(self):
        reference = self._result("doc", 0, [1], 0)
        different_candidates = self._result(
            "doc",
            0,
            [1],
            0,
            candidate_global_indices=[101, 102, 104],
        )

        analysis_input = {
            pruning_percent: [reference]
            for pruning_percent in PRUNING_TO_KEEP_RATIO
        }
        analysis_input[10] = [different_candidates]
        with self.assertRaisesRegex(ValueError, "First-stage candidates differ"):
            analyze_pruning_recalls(analysis_input)

    def test_save_writes_one_json_with_all_query_groups(self):
        result = self._result("doc", 0, [1], 0)
        results_by_pruning = {
            pruning_percent: [result]
            for pruning_percent in PRUNING_TO_KEEP_RATIO
        }
        analysis = analyze_pruning_recalls(results_by_pruning)

        with tempfile.TemporaryDirectory() as temporary_dir:
            output_path = save_pruning_recall_analysis(
                analysis,
                Path(temporary_dir),
            )
            summary = json.loads(
                Path(output_path).read_text(encoding="utf-8")
            )
            loaded_analysis = load_analysis_json(Path(output_path))
            fake_figure = MagicMock()
            fake_axis = MagicMock()
            fake_matplotlib = MagicMock()
            fake_pyplot = MagicMock()
            fake_ticker = MagicMock()
            fake_pyplot.subplots.return_value = (fake_figure, fake_axis)
            fake_figure.savefig.side_effect = (
                lambda path, **_: Path(path).write_bytes(b"\x89PNG\r\n\x1a\n")
            )

            def fake_import_module(module_name):
                if module_name == "matplotlib":
                    return fake_matplotlib
                if module_name == "matplotlib.pyplot":
                    return fake_pyplot
                if module_name == "matplotlib.ticker":
                    return fake_ticker
                raise AssertionError(f"Unexpected module import: {module_name}")

            with patch(
                "scripts.plot_pruning_recall_top1.importlib.import_module",
                side_effect=fake_import_module,
            ):
                plot_paths = save_all_recall_correct_query_plots(
                    loaded_analysis,
                    Path(temporary_dir),
                )
                breakdown_plot_paths = (
                    save_all_recall_higher_rate_breakdown_plots(
                        loaded_analysis,
                        Path(temporary_dir),
                    )
                )
            plot_signatures = [
                Path(path).read_bytes()[:8]
                for path in [
                    *plot_paths.values(),
                    *breakdown_plot_paths.values(),
                ]
            ]
            output_files = sorted(
                path.name for path in Path(temporary_dir).iterdir()
            )

        self.assertEqual(
            set(output_files),
            {
                "pruning_recall_query_groups.json",
                DEFAULT_OUTPUT_FILENAMES[3],
                DEFAULT_OUTPUT_FILENAMES[5],
                DEFAULT_OUTPUT_FILENAMES[1],
                DEFAULT_BREAKDOWN_OUTPUT_FILENAMES[1],
                DEFAULT_BREAKDOWN_OUTPUT_FILENAMES[3],
                DEFAULT_BREAKDOWN_OUTPUT_FILENAMES[5],
            },
        )
        self.assertEqual(
            plot_signatures,
            [b"\x89PNG\r\n\x1a\n"] * 6,
        )
        self.assertEqual(fake_matplotlib.use.call_count, 6)
        self.assertEqual(fake_pyplot.subplots.call_count, 6)
        self.assertEqual(fake_axis.plot.call_count, 6)
        self.assertEqual(fake_axis.bar.call_count, 9)
        first_curve = fake_axis.plot.call_args_list[0].args
        second_curve = fake_axis.plot.call_args_list[1].args
        self.assertEqual(first_curve, (list(range(0, 100, 10)), [1] * 10))
        self.assertEqual(second_curve, (list(range(0, 100, 10)), [1] * 10))
        self.assertEqual(
            [
                fake_axis.plot.call_args_list[index].kwargs["label"]
                for index in (0, 2, 4)
            ],
            [
                "Recall@1 correct at exact pruning rate",
                "Recall@3 correct at exact pruning rate",
                "Recall@5 correct at exact pruning rate",
            ],
        )
        first_bar = fake_axis.bar.call_args_list[0].args
        second_bar = fake_axis.bar.call_args_list[1].args
        total_bar = fake_axis.bar.call_args_list[2].args
        self.assertEqual(first_bar[1], [0] * 9 + [1])
        self.assertEqual(second_bar[1], [1] * 9 + [0])
        self.assertEqual(total_bar[1], [1] * 10)
        self.assertEqual(summary["recall_cutoffs"], [1, 3, 5])
        self.assertIn("top1_correct_query_curve", summary)
        self.assertEqual(
            sorted(summary["correct_query_curves"]),
            ["recall_at_1", "recall_at_3", "recall_at_5"],
        )
        self.assertEqual(
            sorted(summary["correct_query_higher_rate_breakdowns"]),
            ["recall_at_1", "recall_at_3", "recall_at_5"],
        )
        self.assertEqual(len(summary["conditions"]), 10)
        self.assertEqual(
            summary["conditions"]["prune50"]["metrics"]["recall_at_5"]
            ["correct_queries"][0]["qid"],
            "doc_0",
        )
        self.assertNotIn("transitions", summary)


if __name__ == "__main__":
    unittest.main()
