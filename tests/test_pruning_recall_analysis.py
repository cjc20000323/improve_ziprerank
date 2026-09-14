import json
import tempfile
import unittest
from pathlib import Path

from utils.pruning_recall_analysis import (
    compare_pruning_recall1,
    get_top1_outcome,
    save_pruning_recall1_analysis,
)


class PruningRecallAnalysisTest(unittest.TestCase):
    @staticmethod
    def _result(
            doc_name,
            q_idx,
            ground_truth_page_ids,
            top1_candidate_pos,
            candidate_global_indices=None,
    ):
        if candidate_global_indices is None:
            # With start_idx=100 these candidates have local page IDs [1, 2, 3].
            candidate_global_indices = [101, 102, 103]
        remaining_positions = [
            position
            for position in range(len(candidate_global_indices))
            if position != top1_candidate_pos
        ]
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
            "ranked_indices": [top1_candidate_pos, *remaining_positions],
        }

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

    def test_compare_reports_both_requested_transition_groups(self):
        # Query A is recovered by lighter 10% pruning; query B is broken by the
        # aggressive 90% pruning. C and D provide unchanged correct/wrong cases.
        specifications = {
            "a": {"gt": [1], "prune10": 0, "prune50": 1, "prune90": 2},
            "b": {"gt": [2], "prune10": 1, "prune50": 1, "prune90": 0},
            "c": {"gt": [3], "prune10": 2, "prune50": 2, "prune90": 2},
            "d": {"gt": [1], "prune10": 1, "prune50": 1, "prune90": 1},
        }
        results_by_pruning = {10: [], 50: [], 90: []}
        for q_idx, (doc_name, spec) in enumerate(specifications.items()):
            for pruning_percent in (10, 50, 90):
                results_by_pruning[pruning_percent].append(
                    self._result(
                        doc_name=doc_name,
                        q_idx=q_idx,
                        ground_truth_page_ids=spec["gt"],
                        top1_candidate_pos=spec[f"prune{pruning_percent}"],
                    )
                )

        analysis = compare_pruning_recall1(results_by_pruning)

        self.assertEqual(analysis["num_aligned_samples"], 4)
        summaries = analysis["condition_summaries"]
        self.assertEqual(summaries["prune10"]["recall1_correct_count"], 3)
        self.assertEqual(summaries["prune50"]["recall1_correct_count"], 2)
        self.assertEqual(summaries["prune90"]["recall1_correct_count"], 1)

        transitions = analysis["transitions"]
        recovered = transitions["prune50_wrong_prune10_correct"]
        degraded = transitions["prune50_correct_prune90_wrong"]
        self.assertEqual(recovered["count"], 1)
        self.assertEqual(degraded["count"], 1)
        self.assertEqual(recovered["samples"][0]["doc_name"], "a")
        self.assertEqual(degraded["samples"][0]["doc_name"], "b")
        self.assertFalse(
            recovered["samples"][0]["outcomes"]["prune50"]["correct"]
        )
        self.assertTrue(
            recovered["samples"][0]["outcomes"]["prune10"]["correct"]
        )

        table = analysis["pairwise_outcome_tables"]["prune50_vs_prune10"]
        self.assertEqual(table["prune50_wrong_prune10_correct"], 1)
        self.assertEqual(table["prune50_correct_prune10_correct"], 2)

    def test_compare_rejects_different_query_sets(self):
        common = self._result("doc", 0, [1], 0)
        analysis_input = {
            10: [common],
            50: [common],
            90: [],
        }

        with self.assertRaisesRegex(ValueError, "Query set mismatch"):
            compare_pruning_recall1(analysis_input)

    def test_compare_rejects_different_first_stage_candidates(self):
        reference = self._result("doc", 0, [1], 0)
        different_candidates = self._result(
            "doc",
            0,
            [1],
            0,
            candidate_global_indices=[101, 102, 104],
        )

        with self.assertRaisesRegex(ValueError, "First-stage candidates differ"):
            compare_pruning_recall1({
                10: [different_candidates],
                50: [reference],
                90: [reference],
            })

    def test_save_writes_summary_and_both_transition_jsonl_files(self):
        result = self._result("doc", 0, [1], 0)
        analysis = compare_pruning_recall1({
            10: [result],
            50: [self._result("doc", 0, [1], 1)],
            90: [self._result("doc", 0, [1], 1)],
        })

        with tempfile.TemporaryDirectory() as temporary_dir:
            paths = save_pruning_recall1_analysis(
                analysis,
                Path(temporary_dir),
            )
            summary = json.loads(
                Path(paths["summary"]).read_text(encoding="utf-8")
            )
            recovered_lines = Path(
                paths["prune50_wrong_prune10_correct"]
            ).read_text(encoding="utf-8").splitlines()
            degraded_lines = Path(
                paths["prune50_correct_prune90_wrong"]
            ).read_text(encoding="utf-8").splitlines()

        self.assertEqual(
            summary["transitions"]["prune50_wrong_prune10_correct"]["count"],
            1,
        )
        self.assertEqual(len(recovered_lines), 1)
        self.assertEqual(degraded_lines, [])


if __name__ == "__main__":
    unittest.main()
