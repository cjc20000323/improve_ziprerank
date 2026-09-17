"""Group page-level Recall@1/3/5 outcomes by visual-token pruning rate.

The reranker stores rankings as positions in the original first-stage candidate
list, while MMDocIR ground-truth page IDs are local to a document.  This module
centralizes the index conversion and performs strict cross-run validation before
reporting the correct and incorrect queries for every pruning condition.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Set, Tuple, cast


# The experiment is expressed in pruning percentages, but the model receives the
# complementary visual-token keep ratio.  Record every 10-point condition from
# no pruning through 90% pruning.  Computing the complement from integer
# percentages prevents a 90% pruning experiment from being run as 90% retention.
PRUNING_TO_KEEP_RATIO: Dict[int, float] = {
    pruning_percent: (100 - pruning_percent) / 100
    for pruning_percent in range(0, 100, 10)
}
REQUIRED_PRUNING_PERCENTS = tuple(PRUNING_TO_KEEP_RATIO)
RECALL_CUTOFFS = (1, 3, 5)

QueryKey = Tuple[str, int]


def make_query_key(result: Mapping[str, Any]) -> QueryKey:
    """Return the stable key used to align the same query across pruning runs."""
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


def get_recall_at_k_outcome(
        result: Mapping[str, Any],
        k: int,
) -> Dict[str, Any]:
    """Resolve the final Top-K candidates and compute their page recall.

    ``ranked_indices`` contains positions in ``top_k_global_indices``.  The
    selected global parquet indices must therefore be looked up first and then
    shifted by ``start_idx`` before they can be compared with MMDocIR
    ``page_id``.  A query is classified as correct when Top-K contains at least
    one ground-truth page; ``recall_at_k`` retains the official fractional value.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

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
        start_idx = int(result["start_idx"])
        end_idx = int(result["end_idx"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid document indices for query {query_key!r}") from exc
    if end_idx < start_idx:
        raise ValueError(
            f"end_idx={end_idx} precedes start_idx={start_idx} for query {query_key!r}"
        )

    num_document_pages = end_idx - start_idx + 1
    candidate_positions = []
    global_page_ids = []
    local_page_ids = []
    for rank, raw_candidate_pos in enumerate(ranked_indices[:k], start=1):
        try:
            candidate_pos = int(raw_candidate_pos)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid rank-{rank} candidate position for query {query_key!r}: "
                f"{raw_candidate_pos!r}"
            ) from exc
        if not 0 <= candidate_pos < len(candidate_global_indices):
            raise ValueError(
                f"Rank-{rank} candidate position {candidate_pos} is outside "
                f"[0, {len(candidate_global_indices)}) for query {query_key!r}"
            )
        try:
            global_page_id = int(candidate_global_indices[candidate_pos])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid rank-{rank} global page for query {query_key!r}"
            ) from exc
        local_page_id = global_page_id - start_idx
        if not 0 <= local_page_id < num_document_pages:
            raise ValueError(
                f"Rank-{rank} global page {global_page_id} lies outside document "
                f"range [{start_idx}, {end_idx}] for query {query_key!r}"
            )
        candidate_positions.append(candidate_pos)
        global_page_ids.append(global_page_id)
        local_page_ids.append(local_page_id)

    ground_truth_page_ids = _ground_truth_page_ids(result)
    hit_ground_truth_page_ids = sorted(
        set(local_page_ids) & set(ground_truth_page_ids)
    )
    return {
        "correct": bool(hit_ground_truth_page_ids),
        "recall_at_k": (
            len(hit_ground_truth_page_ids) / len(ground_truth_page_ids)
        ),
        "requested_k": k,
        "evaluated_rank_count": len(candidate_positions),
        "retrieved_candidate_positions": candidate_positions,
        "retrieved_global_page_ids": global_page_ids,
        "retrieved_local_page_ids": local_page_ids,
        "hit_ground_truth_page_ids": hit_ground_truth_page_ids,
    }


def get_top1_outcome(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Backward-compatible Top-1 view of :func:`get_recall_at_k_outcome`."""
    outcome = get_recall_at_k_outcome(result, 1)
    return {
        "correct": outcome["correct"],
        "recall_at_1": outcome["recall_at_k"],
        "top1_candidate_pos": outcome["retrieved_candidate_positions"][0],
        "top1_global_page_id": outcome["retrieved_global_page_ids"][0],
        "top1_local_page_id": outcome["retrieved_local_page_ids"][0],
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


def _query_outcome_record(
        result: Mapping[str, Any],
        outcome: Mapping[str, Any],
        k: int,
) -> Dict[str, Any]:
    """Create a self-contained query record for one pruning rate and cutoff."""
    key = make_query_key(result)
    return {
        "qid": result.get("qid", f"{key[0]}_{key[1]}"),
        "doc_name": key[0],
        "q_idx": key[1],
        "domain": result.get("domain"),
        "query": result.get("query"),
        "ground_truth_page_ids": list(_ground_truth_page_ids(result)),
        "correct": bool(outcome["correct"]),
        f"recall_at_{k}": float(outcome["recall_at_k"]),
        "evaluated_rank_count": int(outcome["evaluated_rank_count"]),
        "retrieved_candidate_positions": list(
            outcome["retrieved_candidate_positions"]
        ),
        "retrieved_global_page_ids": list(outcome["retrieved_global_page_ids"]),
        "retrieved_local_page_ids": list(outcome["retrieved_local_page_ids"]),
        "hit_ground_truth_page_ids": list(outcome["hit_ground_truth_page_ids"]),
    }


def _group_queries_for_cutoff(
        condition_results: Mapping[QueryKey, Mapping[str, Any]],
        query_order: Sequence[QueryKey],
        k: int,
) -> Dict[str, Any]:
    """Group every query into correct/incorrect lists for one Recall@K cutoff."""
    correct_queries = []
    incorrect_queries = []
    recall_sum = 0.0
    for key in query_order:
        result = condition_results[key]
        outcome = get_recall_at_k_outcome(result, k)
        recall_sum += float(outcome["recall_at_k"])
        record = _query_outcome_record(result, outcome, k)
        target = correct_queries if outcome["correct"] else incorrect_queries
        target.append(record)

    num_queries = len(query_order)
    correct_count = len(correct_queries)
    return {
        "k": k,
        "num_queries": num_queries,
        "correct_count": correct_count,
        "incorrect_count": len(incorrect_queries),
        "hit_rate": correct_count / num_queries if num_queries else 0.0,
        "mean_official_recall_at_k": (
            recall_sum / num_queries if num_queries else 0.0
        ),
        "correct_queries": correct_queries,
        "incorrect_queries": incorrect_queries,
    }


def _correct_query_keys_by_pruning(
        analysis: Mapping[str, Any],
        k: int,
) -> Dict[int, Set[QueryKey]]:
    """Validate and index Recall@K-correct query identities by pruning rate."""
    if k not in RECALL_CUTOFFS:
        raise ValueError(
            f"Unsupported recall cutoff {k}; expected one of {RECALL_CUTOFFS}"
        )
    conditions = analysis.get("conditions")
    if not isinstance(conditions, Mapping):
        raise ValueError("Analysis does not contain a valid conditions mapping")

    metric_name = f"recall_at_{k}"
    correct_keys_by_pruning: Dict[int, Set[QueryKey]] = {}
    for pruning_percent in REQUIRED_PRUNING_PERCENTS:
        condition_name = f"prune{pruning_percent}"
        try:
            metric = conditions[condition_name]["metrics"][metric_name]
            correct_queries = metric["correct_queries"]
            declared_count = int(metric["correct_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Missing valid Recall@{k} data for {condition_name}"
            ) from exc
        if not isinstance(correct_queries, (list, tuple)):
            raise ValueError(
                f"correct_queries for {condition_name} must be a list"
            )

        correct_keys = {make_query_key(record) for record in correct_queries}
        if len(correct_keys) != len(correct_queries):
            raise ValueError(
                f"Duplicate Recall@{k}-correct queries found in {condition_name}"
            )
        if declared_count != len(correct_keys):
            raise ValueError(
                f"Recall@{k} correct_count mismatch in {condition_name}: "
                f"declared={declared_count}, records={len(correct_keys)}"
            )
        correct_keys_by_pruning[pruning_percent] = correct_keys
    return correct_keys_by_pruning


def build_correct_query_curve(
        analysis: Mapping[str, Any],
        k: int,
) -> Dict[str, Any]:
    """Build exact-rate and current-or-higher Recall@K query counts.

    The cumulative curve is a set union, so a query that is correct under
    several pruning rates contributes only once at each x-axis position.
    """
    correct_keys_by_pruning = _correct_query_keys_by_pruning(analysis, k)
    exact_counts = {
        pruning_percent: len(correct_keys)
        for pruning_percent, correct_keys in correct_keys_by_pruning.items()
    }

    cumulative_keys = set()
    union_counts = {}
    for pruning_percent in reversed(REQUIRED_PRUNING_PERCENTS):
        cumulative_keys.update(correct_keys_by_pruning[pruning_percent])
        union_counts[pruning_percent] = len(cumulative_keys)

    return {
        "recall_cutoff": k,
        "x_axis": "visual-token pruning rate (%)",
        "y_axis": f"number of Recall@{k}-correct queries",
        "current_or_higher_definition": (
            f"number of distinct queries that are Recall@{k}-correct at the current "
            "pruning rate or at any higher configured pruning rate"
        ),
        "points": [
            {
                "pruning_percent": pruning_percent,
                "correct_at_pruning_rate": exact_counts[pruning_percent],
                "correct_at_current_or_higher_pruning_union": (
                    union_counts[pruning_percent]
                ),
            }
            for pruning_percent in REQUIRED_PRUNING_PERCENTS
        ],
    }


def build_correct_query_higher_rate_breakdown(
        analysis: Mapping[str, Any],
        k: int,
) -> Dict[str, Any]:
    """Partition current Recall@K hits by whether any higher rate also hits.

    For each pruning rate, the first category contains queries that are correct
    now but wrong at every strictly higher configured pruning rate.  The second
    contains queries that are correct now and correct at one or more strictly
    higher rates.  These disjoint categories sum to the current correct count.
    """
    correct_keys_by_pruning = _correct_query_keys_by_pruning(analysis, k)
    higher_rate_correct_keys: Set[QueryKey] = set()
    counts_by_pruning = {}
    for pruning_percent in reversed(REQUIRED_PRUNING_PERCENTS):
        current_correct_keys = correct_keys_by_pruning[pruning_percent]
        higher_wrong_keys = current_correct_keys - higher_rate_correct_keys
        higher_correct_keys = current_correct_keys & higher_rate_correct_keys
        counts_by_pruning[pruning_percent] = {
            "current_correct_higher_rates_all_wrong": len(higher_wrong_keys),
            "current_correct_and_any_higher_rate_correct": len(
                higher_correct_keys
            ),
            "current_correct_total": len(current_correct_keys),
        }
        higher_rate_correct_keys.update(current_correct_keys)

    return {
        "recall_cutoff": k,
        "x_axis": "visual-token pruning rate (%)",
        "y_axis": f"number of Recall@{k}-correct queries",
        "higher_rate_definition": (
            "all configured pruning rates strictly greater than the current rate"
        ),
        "points": [
            {
                "pruning_percent": pruning_percent,
                **counts_by_pruning[pruning_percent],
            }
            for pruning_percent in REQUIRED_PRUNING_PERCENTS
        ],
    }


def build_top1_correct_query_curve(
        analysis: Mapping[str, Any],
) -> Dict[str, Any]:
    """Backward-compatible wrapper for the Recall@1 curve."""
    return build_correct_query_curve(analysis, 1)


def analyze_pruning_recalls(
        results_by_pruning: Mapping[int, Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    """Group Recall@1/3/5 outcomes for every configured pruning condition.

    Query sets and immutable input data must match exactly; silently analyzing
    only an intersection could otherwise produce misleading counts.
    """
    missing_conditions = set(REQUIRED_PRUNING_PERCENTS) - set(results_by_pruning)
    extra_conditions = set(results_by_pruning) - set(REQUIRED_PRUNING_PERCENTS)
    if missing_conditions or extra_conditions:
        expected_conditions = ", ".join(
            str(value) for value in REQUIRED_PRUNING_PERCENTS
        )
        raise ValueError(
            f"Expected pruning conditions {expected_conditions}; "
            f"missing={sorted(missing_conditions)}, extra={sorted(extra_conditions)}"
        )

    indexed = {
        pruning_percent: _index_results(
            results_by_pruning[pruning_percent],
            condition_name=f"prune{pruning_percent}",
        )
        for pruning_percent in REQUIRED_PRUNING_PERCENTS
    }
    reference_percent = REQUIRED_PRUNING_PERCENTS[0]
    reference = indexed[reference_percent]
    query_order = tuple(reference)
    reference_keys = set(reference)
    for pruning_percent in REQUIRED_PRUNING_PERCENTS:
        if pruning_percent == reference_percent:
            continue
        condition_keys = set(indexed[pruning_percent])
        if condition_keys != reference_keys:
            raise ValueError(
                f"Query set mismatch between prune{reference_percent} and "
                f"prune{pruning_percent}: "
                f"missing_from_prune{pruning_percent}="
                f"{sorted(reference_keys - condition_keys)!r}, "
                f"extra_in_prune{pruning_percent}="
                f"{sorted(condition_keys - reference_keys)!r}"
            )

    # The insertion order of the 0% result file is retained in every exported
    # query list, making the JSON deterministic and easy to cross-check.
    for key, reference_result in reference.items():
        for pruning_percent in REQUIRED_PRUNING_PERCENTS:
            if pruning_percent == reference_percent:
                continue
            _validate_static_query_data(
                reference_result,
                indexed[pruning_percent][key],
                condition_name=f"prune{pruning_percent}",
            )

    analysis = {
        "schema_version": 2,
        "definitions": {
            "pruning_percent": "percentage of original visual tokens removed",
            "keep_ratio_by_pruning_percent": {
                str(pruning_percent): keep_ratio
                for pruning_percent, keep_ratio in PRUNING_TO_KEEP_RATIO.items()
            },
            "correct": "Top-K contains at least one ground-truth page",
            "official_recall_at_k": (
                "number of distinct ground-truth pages in Top-K divided by the "
                "total number of ground-truth pages"
            ),
        },
        "recall_cutoffs": list(RECALL_CUTOFFS),
        "num_aligned_samples": len(reference),
        "conditions": {
            f"prune{pruning_percent}": {
                "pruning_percent": pruning_percent,
                "keep_ratio": PRUNING_TO_KEEP_RATIO[pruning_percent],
                "metrics": {
                    f"recall_at_{k}": _group_queries_for_cutoff(
                        indexed[pruning_percent],
                        query_order,
                        k,
                    )
                    for k in RECALL_CUTOFFS
                },
            }
            for pruning_percent in REQUIRED_PRUNING_PERCENTS
        },
    }
    analysis["correct_query_curves"] = {
        f"recall_at_{k}": build_correct_query_curve(analysis, k)
        for k in RECALL_CUTOFFS
    }
    analysis["correct_query_higher_rate_breakdowns"] = {
        f"recall_at_{k}": build_correct_query_higher_rate_breakdown(analysis, k)
        for k in RECALL_CUTOFFS
    }
    analysis["top1_correct_query_curve"] = analysis["correct_query_curves"][
        "recall_at_1"
    ]
    return analysis


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Write reranking records without losing non-ASCII queries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_pruning_recall_analysis(
        analysis: Mapping[str, Any],
        output_dir: Path,
) -> str:
    """Save all per-condition Recall@1/3/5 query groups to one JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "pruning_recall_query_groups.json"

    summary_path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(summary_path)
