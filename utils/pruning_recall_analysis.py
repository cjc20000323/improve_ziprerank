"""Compare page-level Recall@1 outcomes across visual-token pruning rates.

The reranker stores rankings as positions in the original first-stage candidate
list, while MMDocIR ground-truth page IDs are local to a document.  This module
centralizes the index conversion and performs strict cross-run validation before
reporting samples whose Recall@1 outcome changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple, cast


# The experiment is expressed in pruning percentages, but the model receives the
# complementary visual-token keep ratio.  Keeping the mapping here prevents a
# 90% pruning experiment from accidentally being run as 90% retention.
PRUNING_TO_KEEP_RATIO: Dict[int, float] = {
    10: 0.9,
    50: 0.5,
    90: 0.1,
}
REQUIRED_PRUNING_PERCENTS = tuple(PRUNING_TO_KEEP_RATIO)

QueryKey = Tuple[str, int]


def make_query_key(result: Mapping[str, Any]) -> QueryKey:
    """Return the stable key used to align the same query across three runs."""
    try:
        return str(result["doc_name"]), int(result["q_idx"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Each reranking result must contain valid doc_name and q_idx fields"
        ) from exc


def _ground_truth_page_ids(result: Mapping[str, Any]) -> Tuple[int, ...]:
    """Normalize the document-local ground-truth page IDs."""
    raw_page_ids = result.get("page_id")
    if raw_page_ids is None:
        raise ValueError(f"Missing page_id for query {make_query_key(result)!r}")

    if isinstance(raw_page_ids, (list, tuple, set)):
        values = raw_page_ids
    else:
        values = [raw_page_ids]

    normalized_page_ids = set()
    for page_id in values:
        if page_id is None:
            raise ValueError(
                f"Invalid page_id values for query {make_query_key(result)!r}: "
                f"{raw_page_ids!r}"
            )
        try:
            normalized_page_ids.add(int(cast(int, page_id)))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid page_id values for query {make_query_key(result)!r}: "
                f"{raw_page_ids!r}"
            ) from exc
    page_ids = tuple(sorted(normalized_page_ids))

    if not page_ids:
        raise ValueError(f"Empty page_id for query {make_query_key(result)!r}")
    return page_ids


def get_top1_outcome(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve the final Top-1 candidate and determine whether it hits any GT.

    ``ranked_indices`` contains positions in ``top_k_global_indices``.  The
    selected global parquet index must therefore be looked up first and then
    shifted by ``start_idx`` before it can be compared with MMDocIR ``page_id``.
    """
    query_key = make_query_key(result)
    ranked_indices = result.get("ranked_indices")
    candidate_global_indices = result.get("top_k_global_indices")
    if not isinstance(ranked_indices, (list, tuple)) or not ranked_indices:
        raise ValueError(f"Empty or invalid ranked_indices for query {query_key!r}")
    if not isinstance(candidate_global_indices, (list, tuple)) or not candidate_global_indices:
        raise ValueError(
            f"Empty or invalid top_k_global_indices for query {query_key!r}"
        )

    try:
        top1_candidate_pos = int(ranked_indices[0])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid Top-1 candidate position for query {query_key!r}: "
            f"{ranked_indices[0]!r}"
        ) from exc
    if not 0 <= top1_candidate_pos < len(candidate_global_indices):
        raise ValueError(
            f"Top-1 candidate position {top1_candidate_pos} is outside "
            f"[0, {len(candidate_global_indices)}) for query {query_key!r}"
        )

    try:
        start_idx = int(result["start_idx"])
        end_idx = int(result["end_idx"])
        top1_global_page_id = int(candidate_global_indices[top1_candidate_pos])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid document or candidate indices for query {query_key!r}"
        ) from exc
    if end_idx < start_idx:
        raise ValueError(
            f"end_idx={end_idx} precedes start_idx={start_idx} for query {query_key!r}"
        )

    top1_local_page_id = top1_global_page_id - start_idx
    num_document_pages = end_idx - start_idx + 1
    if not 0 <= top1_local_page_id < num_document_pages:
        raise ValueError(
            f"Top-1 global page {top1_global_page_id} lies outside document range "
            f"[{start_idx}, {end_idx}] for query {query_key!r}"
        )

    ground_truth_page_ids = _ground_truth_page_ids(result)
    recall1_correct = top1_local_page_id in ground_truth_page_ids
    return {
        "correct": recall1_correct,
        # Official recall divides the single hit by the number of GT pages.  The
        # boolean above is the appropriate correct/incorrect grouping criterion.
        "recall_at_1": (
            1.0 / len(ground_truth_page_ids) if recall1_correct else 0.0
        ),
        "top1_candidate_pos": top1_candidate_pos,
        "top1_global_page_id": top1_global_page_id,
        "top1_local_page_id": top1_local_page_id,
    }


def _index_results(
        results: Sequence[Mapping[str, Any]],
        condition_name: str,
) -> Dict[QueryKey, Mapping[str, Any]]:
    """Index a result list and reject duplicate query records."""
    indexed: Dict[QueryKey, Mapping[str, Any]] = {}
    for result in results:
        key = make_query_key(result)
        if key in indexed:
            raise ValueError(f"Duplicate query {key!r} in {condition_name}")
        indexed[key] = result
    return indexed


def _validate_static_query_data(
        reference: Mapping[str, Any],
        candidate: Mapping[str, Any],
        condition_name: str,
) -> None:
    """Ensure that only the final ranking changes between pruning conditions."""
    key = make_query_key(reference)
    scalar_fields = ("query", "domain", "start_idx", "end_idx")
    for field in scalar_fields:
        if reference.get(field) != candidate.get(field):
            raise ValueError(
                f"Field {field!r} differs for query {key!r} in {condition_name}"
            )

    if _ground_truth_page_ids(reference) != _ground_truth_page_ids(candidate):
        raise ValueError(
            f"Ground-truth pages differ for query {key!r} in {condition_name}"
        )

    reference_candidates = [int(value) for value in reference["top_k_global_indices"]]
    candidate_candidates = [int(value) for value in candidate["top_k_global_indices"]]
    if reference_candidates != candidate_candidates:
        raise ValueError(
            f"First-stage candidates differ for query {key!r} in {condition_name}"
        )


def _condition_summary(outcomes: Mapping[QueryKey, Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize binary Top-1 hits and the official multi-GT Recall@1 value."""
    num_samples = len(outcomes)
    correct_count = sum(bool(outcome["correct"]) for outcome in outcomes.values())
    recall_sum = sum(float(outcome["recall_at_1"]) for outcome in outcomes.values())
    return {
        "num_samples": num_samples,
        "recall1_correct_count": correct_count,
        "recall1_incorrect_count": num_samples - correct_count,
        "recall1_hit_rate": correct_count / num_samples if num_samples else 0.0,
        "mean_official_recall_at_1": recall_sum / num_samples if num_samples else 0.0,
    }


def _pairwise_outcome_table(
        first_name: str,
        first: Mapping[QueryKey, Mapping[str, Any]],
        second_name: str,
        second: Mapping[QueryKey, Mapping[str, Any]],
) -> Dict[str, int]:
    """Build a complete 2x2 hit/miss table for auditing a requested transition."""
    counts: Dict[str, int] = {}
    for first_correct in (False, True):
        for second_correct in (False, True):
            first_state = "correct" if first_correct else "wrong"
            second_state = "correct" if second_correct else "wrong"
            name = f"{first_name}_{first_state}_{second_name}_{second_state}"
            counts[name] = sum(
                bool(first[key]["correct"]) is first_correct
                and bool(second[key]["correct"]) is second_correct
                for key in first
            )
    return counts


def _sample_record(
        reference: Mapping[str, Any],
        outcomes_by_pruning: Mapping[int, Mapping[QueryKey, Mapping[str, Any]]],
) -> Dict[str, Any]:
    """Create a compact, directly inspectable record for one changed query."""
    key = make_query_key(reference)
    return {
        "qid": reference.get("qid", f"{key[0]}_{key[1]}"),
        "doc_name": key[0],
        "q_idx": key[1],
        "domain": reference.get("domain"),
        "query": reference.get("query"),
        "ground_truth_page_ids": list(_ground_truth_page_ids(reference)),
        "outcomes": {
            f"prune{pruning_percent}": dict(outcomes_by_pruning[pruning_percent][key])
            for pruning_percent in REQUIRED_PRUNING_PERCENTS
        },
    }


def compare_pruning_recall1(
        results_by_pruning: Mapping[int, Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    """Compare the required 10%, 50%, and 90% visual-token pruning runs.

    The result contains both requested transition lists and complete pairwise
    contingency tables.  Query sets and immutable input data must match exactly;
    silently comparing only an intersection could otherwise produce misleading
    counts.
    """
    missing_conditions = set(REQUIRED_PRUNING_PERCENTS) - set(results_by_pruning)
    extra_conditions = set(results_by_pruning) - set(REQUIRED_PRUNING_PERCENTS)
    if missing_conditions or extra_conditions:
        raise ValueError(
            "Expected pruning conditions 10, 50, and 90; "
            f"missing={sorted(missing_conditions)}, extra={sorted(extra_conditions)}"
        )

    indexed = {
        pruning_percent: _index_results(
            results_by_pruning[pruning_percent],
            condition_name=f"prune{pruning_percent}",
        )
        for pruning_percent in REQUIRED_PRUNING_PERCENTS
    }
    reference = indexed[50]
    reference_keys = set(reference)
    for pruning_percent in (10, 90):
        condition_keys = set(indexed[pruning_percent])
        if condition_keys != reference_keys:
            raise ValueError(
                f"Query set mismatch between prune50 and prune{pruning_percent}: "
                f"missing_from_prune{pruning_percent}="
                f"{sorted(reference_keys - condition_keys)!r}, "
                f"extra_in_prune{pruning_percent}="
                f"{sorted(condition_keys - reference_keys)!r}"
            )

    # The insertion order of the 50% result file is retained in every exported
    # sample list, making the JSONL outputs deterministic and easy to cross-check.
    for key, reference_result in reference.items():
        for pruning_percent in (10, 90):
            _validate_static_query_data(
                reference_result,
                indexed[pruning_percent][key],
                condition_name=f"prune{pruning_percent}",
            )

    outcomes_by_pruning = {
        pruning_percent: {
            key: get_top1_outcome(result)
            for key, result in condition_results.items()
        }
        for pruning_percent, condition_results in indexed.items()
    }

    wrong50_correct10 = []
    correct50_wrong90 = []
    for key, reference_result in reference.items():
        outcome10 = outcomes_by_pruning[10][key]
        outcome50 = outcomes_by_pruning[50][key]
        outcome90 = outcomes_by_pruning[90][key]
        if not outcome50["correct"] and outcome10["correct"]:
            wrong50_correct10.append(
                _sample_record(reference_result, outcomes_by_pruning)
            )
        if outcome50["correct"] and not outcome90["correct"]:
            correct50_wrong90.append(
                _sample_record(reference_result, outcomes_by_pruning)
            )

    return {
        "schema_version": 1,
        "definitions": {
            "pruning_percent": "percentage of original visual tokens removed",
            "keep_ratio_by_pruning_percent": {
                str(pruning_percent): keep_ratio
                for pruning_percent, keep_ratio in PRUNING_TO_KEEP_RATIO.items()
            },
            "recall1_correct": "final Top-1 page hits any ground-truth page",
            "official_recall_at_1": (
                "1 / number_of_ground_truth_pages for a Top-1 hit, otherwise 0"
            ),
        },
        "num_aligned_samples": len(reference),
        "condition_summaries": {
            f"prune{pruning_percent}": {
                "pruning_percent": pruning_percent,
                "keep_ratio": PRUNING_TO_KEEP_RATIO[pruning_percent],
                **_condition_summary(outcomes_by_pruning[pruning_percent]),
            }
            for pruning_percent in REQUIRED_PRUNING_PERCENTS
        },
        "pairwise_outcome_tables": {
            "prune50_vs_prune10": _pairwise_outcome_table(
                "prune50",
                outcomes_by_pruning[50],
                "prune10",
                outcomes_by_pruning[10],
            ),
            "prune50_vs_prune90": _pairwise_outcome_table(
                "prune50",
                outcomes_by_pruning[50],
                "prune90",
                outcomes_by_pruning[90],
            ),
        },
        "transitions": {
            "prune50_wrong_prune10_correct": {
                "count": len(wrong50_correct10),
                "samples": wrong50_correct10,
            },
            "prune50_correct_prune90_wrong": {
                "count": len(correct50_wrong90),
                "samples": correct50_wrong90,
            },
        },
    }


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Write reranking or transition records without losing non-ASCII queries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_pruning_recall1_analysis(
        analysis: Mapping[str, Any],
        output_dir: Path,
) -> Dict[str, str]:
    """Save the complete summary plus one convenient JSONL per transition."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "pruning_recall1_transition_summary.json"
    wrong50_correct10_path = output_dir / "prune50_wrong_prune10_correct.jsonl"
    correct50_wrong90_path = output_dir / "prune50_correct_prune90_wrong.jsonl"

    summary_path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    transitions = analysis["transitions"]
    write_jsonl(
        wrong50_correct10_path,
        transitions["prune50_wrong_prune10_correct"]["samples"],
    )
    write_jsonl(
        correct50_wrong90_path,
        transitions["prune50_correct_prune90_wrong"]["samples"],
    )
    return {
        "summary": str(summary_path),
        "prune50_wrong_prune10_correct": str(wrong50_correct10_path),
        "prune50_correct_prune90_wrong": str(correct50_wrong90_path),
    }
