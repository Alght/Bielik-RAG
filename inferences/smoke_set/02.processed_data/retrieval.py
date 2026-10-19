"""Validate retrieval traces and compare top-k results across runs."""

import csv
import hashlib
import itertools
import json
import re
from collections import Counter

from parameters import CLAIM_COMPARISONS, ROOT, SET_SIZE


SOURCE = ROOT / "02.inference_output"
KEY_FILE = ROOT / "01.key" / f"key-{SET_SIZE}.csv"
OUTPUT = ROOT / "12.retrieval_validation"
RUN_NAME = re.compile(
    rf"^bielik11-{SET_SIZE}-cpu-(norag|rag-(\d+))-prompt-(\d+)_retrievals$"
)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source, delimiter=";"))


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target, fieldnames=columns, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def load_trace(path):
    records = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path.name}:{line_number}: {error}") from error

    manifests = [record for record in records if record.get("record_type") == "run"]
    retrievals = [record for record in records if record.get("record_type") == "retrieval"]
    if len(manifests) != 1:
        raise ValueError(f"Expected one run manifest in {path.name}; found {len(manifests)}")
    match = RUN_NAME.fullmatch(path.stem)
    if not match:
        raise ValueError(f"Unrecognized retrieval trace name: {path.name}")
    scenario = "no-RAG" if match[1] == "norag" else f"RAG k={match[2]}"
    return {
        "name": path.name,
        "prompt_id": match[3],
        "scenario": scenario,
        "manifest": manifests[0],
        "records": retrievals,
    }


def manifest_chunk_fingerprint(manifest):
    serialized = json.dumps(manifest.get("chunk_sha256", []), separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def analyze_run(run, key_ids):
    manifest = run["manifest"]
    retrieval_records = run["records"]
    record_by_id = {}
    duplicate_ids = Counter()
    query_rows = []
    chunk_rows = []
    distances_for_run = []
    hash_errors = rank_errors = distance_order_errors = index_errors = 0
    request_count_errors = rag_flag_errors = duplicate_chunks = 0
    over_request_errors = 0

    for record in retrieval_records:
        input_id = str(record.get("input_id", ""))
        if input_id in record_by_id:
            duplicate_ids[input_id] += 1
        record_by_id[input_id] = record

        chunks = record.get("retrieved_chunks") or []
        requested_k = int(record.get("top_k_requested") or 0)
        is_rag = bool(record.get("use_rag"))
        requested_limit = requested_k if is_rag else 0
        retrieval_performed = bool(record.get("retrieval_performed"))
        rag_flag_errors += is_rag != retrieval_performed
        request_count_errors += len(chunks) != requested_limit
        returned_more_than_requested = len(chunks) > requested_limit
        over_request_errors += returned_more_than_requested
        chunk_hashes = []
        distances = []
        ranks_valid = True
        hashes_valid = True
        indices_valid = True

        for expected_rank, chunk in enumerate(chunks, start=1):
            rank_valid = chunk.get("rank") == expected_rank
            ranks_valid &= rank_valid
            rank_errors += not rank_valid
            text = chunk.get("chunk_text") or ""
            calculated_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            hash_valid = calculated_hash == chunk.get("chunk_sha256")
            hashes_valid &= hash_valid
            hash_errors += not hash_valid
            chunk_hashes.append(chunk.get("chunk_sha256"))

            chunk_index = chunk.get("chunk_index")
            index_valid = isinstance(chunk_index, int) and 0 <= chunk_index < manifest.get("chunk_count", 0)
            indices_valid &= index_valid
            index_errors += not index_valid
            distance = float(chunk.get("distance"))
            distances.append(distance)
            distances_for_run.append(distance)
            chunk_rows.append(
                {
                    "trace_file": run["name"],
                    "prompt_id": run["prompt_id"],
                    "configuration": run["scenario"],
                    "input_id": input_id,
                    "rank": expected_rank,
                    "chunk_index": chunk_index,
                    "squared_l2_distance": distance,
                    "chunk_sha256": chunk.get("chunk_sha256"),
                    "chunk_hash_matches_text": hash_valid,
                    "chunk_index_in_manifest_range": index_valid,
                }
            )

        distance_order_valid = all(left <= right for left, right in zip(distances, distances[1:]))
        distance_order_errors += not distance_order_valid
        has_duplicate_chunks = len(chunk_hashes) != len(set(chunk_hashes))
        duplicate_chunks += has_duplicate_chunks
        query_text = record.get("query_text") or ""
        query_rows.append(
            {
                "trace_file": run["name"],
                "prompt_id": run["prompt_id"],
                "configuration": run["scenario"],
                "input_id": input_id,
                "query_sha256": hashlib.sha256(query_text.encode("utf-8")).hexdigest(),
                "use_rag": is_rag,
                "retrieval_performed": retrieval_performed,
                "requested_k": requested_k,
                "returned_chunks": len(chunks),
                "returned_count_matches_request": len(chunks) == requested_limit,
                "returned_more_than_requested": returned_more_than_requested,
                "ranks_sequential": ranks_valid,
                "distances_non_decreasing": distance_order_valid,
                "distance_sequence_checked": len(distances) > 1,
                "chunk_hashes_match_text": hashes_valid,
                "chunk_indices_in_manifest_range": indices_valid,
                "duplicate_chunk_hashes": has_duplicate_chunks,
                "minimum_squared_l2": min(distances) if distances else "",
                "mean_squared_l2": sum(distances) / len(distances) if distances else "",
                "maximum_squared_l2": max(distances) if distances else "",
            }
        )

    trace_ids = set(record_by_id)
    output_path = SOURCE / run["name"].replace("_retrievals.jsonl", ".csv")
    output_ids = {row["ID"] for row in read_csv(output_path)} if output_path.exists() else set()
    unique_ids = trace_ids & key_ids
    source_documents = manifest.get("source_documents", [])
    summary = {
        "trace_file": run["name"],
        "prompt_id": run["prompt_id"],
        "configuration": run["scenario"],
        "use_rag": bool(retrieval_records and retrieval_records[0].get("use_rag")),
        "source_pdf_sha256": ",".join(document.get("pdf_sha256", "") for document in source_documents),
        "embedding_model": manifest.get("embedding_model") or "",
        "embedding_revision": manifest.get("embedding_revision") or "",
        "faiss_index_sha256": manifest.get("index_sha256") or "",
        "chunk_count": manifest.get("chunk_count", 0),
        "chunk_manifest_sha256": manifest_chunk_fingerprint(manifest),
        "retrieval_records": len(retrieval_records),
        "unique_trace_ids": len(trace_ids),
        "duplicate_trace_ids": sum(duplicate_ids.values()),
        "key_ids_missing_from_trace": len(key_ids - trace_ids),
        "trace_ids_not_in_key": len(trace_ids - key_ids),
        "output_csv_ids": len(output_ids),
        "output_ids_missing_from_trace": len(output_ids - trace_ids),
        "trace_ids_missing_from_output": len(trace_ids - output_ids),
        "requested_k_values": ",".join(map(str, sorted({int(r.get("top_k_requested") or 0) for r in retrieval_records}))),
        "returned_chunk_count_errors": request_count_errors,
        "queries_returning_more_than_requested": over_request_errors,
        "rag_flag_errors": rag_flag_errors,
        "chunk_hashes_checked": len(chunk_rows),
        "chunk_hashes_matching_text": len(chunk_rows) - hash_errors,
        "chunk_hash_errors": hash_errors,
        "rank_entries_checked": len(chunk_rows),
        "rank_errors": rank_errors,
        "distance_sequences_checked": sum(row["distance_sequence_checked"] for row in query_rows),
        "distance_order_errors": distance_order_errors,
        "chunk_index_range_errors": index_errors,
        "queries_with_duplicate_chunks": duplicate_chunks,
        "minimum_squared_l2": min(distances_for_run) if distances_for_run else "",
        "mean_squared_l2": sum(distances_for_run) / len(distances_for_run) if distances_for_run else "",
        "maximum_squared_l2": max(distances_for_run) if distances_for_run else "",
    }
    run["record_by_id"] = record_by_id
    run["summary"] = summary
    return summary, query_rows, chunk_rows


def build_integrity_summary(runs, query_rows, chunk_rows):
    rag_runs = [run for run in runs if run["summary"]["use_rag"]]
    source_pdf_hashes = {
        document.get("pdf_sha256", "")
        for run in rag_runs
        for document in run["manifest"].get("source_documents", [])
    }
    chunk_manifest_hashes = {
        manifest_chunk_fingerprint(run["manifest"]) for run in rag_runs
    }
    index_hashes = {
        run["manifest"].get("index_sha256", "")
        for run in rag_runs
        if run["manifest"].get("index_sha256")
    }
    rag_queries = [row for row in query_rows if row["use_rag"]]
    checked_distance_sequences = [
        row for row in query_rows if row["distance_sequence_checked"]
    ]
    hash_errors = sum(not row["chunk_hash_matches_text"] for row in chunk_rows)
    rank_errors = sum(run["summary"]["rank_errors"] for run in runs)
    distance_order_errors = sum(run["summary"]["distance_order_errors"] for run in runs)
    return {
        "trace_files": len(runs),
        "retrieval_query_records": len(query_rows),
        "rag_queries_checked": len(rag_queries),
        "chunk_hashes_checked": len(chunk_rows),
        "chunk_hashes_matching_text": len(chunk_rows) - hash_errors,
        "chunk_hash_errors": hash_errors,
        "rank_entries_checked": len(chunk_rows),
        "rank_errors": rank_errors,
        "distance_sequences_checked": len(checked_distance_sequences),
        "distance_order_errors": distance_order_errors,
        "rag_queries_returning_more_than_requested": sum(
            row["returned_more_than_requested"] for row in rag_queries
        ),
        "distinct_source_pdf_hashes": len(source_pdf_hashes),
        "distinct_chunk_manifest_hashes": len(chunk_manifest_hashes),
        "distinct_faiss_index_hashes": len(index_hashes),
    }


def compare_claim_pairs(runs):
    run_by_prompt_k = {}
    for run in runs:
        if not run["summary"]["use_rag"]:
            continue
        k_values = {int(record.get("top_k_requested") or 0) for record in run["records"]}
        if len(k_values) == 1:
            run_by_prompt_k[(run["prompt_id"], k_values.pop())] = run

    rows = []
    for comparison_id, left_prompt, left_k, right_prompt, right_k, prefix_k in CLAIM_COMPARISONS:
        left = run_by_prompt_k.get((left_prompt, left_k))
        right = run_by_prompt_k.get((right_prompt, right_k))
        row = {
            "comparison_id": comparison_id,
            "left_trace": left["name"] if left else "",
            "right_trace": right["name"] if right else "",
            "prefix_k_compared": prefix_k,
            "comparison_available": left is not None and right is not None,
            "shared_input_ids": "",
            "identical_query_texts": "",
            "complete_prefix_comparisons": "",
            "ordered_chunk_hash_matches": "",
            "ordered_chunk_hash_mismatches": "",
            "incomplete_prefix_comparisons": "",
            "hash_match_rate": "",
            "source_documents_match": "",
            "chunk_manifest_matches": "",
            "faiss_index_hashes_match": "",
        }
        if left is None or right is None:
            rows.append(row)
            continue

        left_records = left["record_by_id"]
        right_records = right["record_by_id"]
        shared_ids = sorted(set(left_records) & set(right_records))
        identical_queries = complete_prefixes = matches = mismatches = incomplete = 0
        for input_id in shared_ids:
            first = left_records[input_id]
            second = right_records[input_id]
            if first.get("query_text") != second.get("query_text"):
                continue
            identical_queries += 1
            first_chunks = first.get("retrieved_chunks", [])
            second_chunks = second.get("retrieved_chunks", [])
            if len(first_chunks) < prefix_k or len(second_chunks) < prefix_k:
                incomplete += 1
                continue
            complete_prefixes += 1
            first_hashes = [chunk.get("chunk_sha256") for chunk in first_chunks[:prefix_k]]
            second_hashes = [chunk.get("chunk_sha256") for chunk in second_chunks[:prefix_k]]
            if first_hashes == second_hashes:
                matches += 1
            else:
                mismatches += 1

        row.update(
            {
                "shared_input_ids": len(shared_ids),
                "identical_query_texts": identical_queries,
                "complete_prefix_comparisons": complete_prefixes,
                "ordered_chunk_hash_matches": matches,
                "ordered_chunk_hash_mismatches": mismatches,
                "incomplete_prefix_comparisons": incomplete,
                "hash_match_rate": matches / complete_prefixes if complete_prefixes else "",
                "source_documents_match": left["manifest"].get("source_documents")
                == right["manifest"].get("source_documents"),
                "chunk_manifest_matches": left["manifest"].get("chunk_sha256")
                == right["manifest"].get("chunk_sha256"),
                "faiss_index_hashes_match": left["manifest"].get("index_sha256")
                == right["manifest"].get("index_sha256"),
            }
        )
        rows.append(row)
    return rows


def compare_runs(runs):
    detail_rows = []
    summary_rows = []
    rag_runs = [run for run in runs if run["summary"]["use_rag"]]
    for left, right in itertools.combinations(rag_runs, 2):
        left_manifest = left["manifest"]
        right_manifest = right["manifest"]
        source_match = left_manifest.get("source_documents") == right_manifest.get("source_documents")
        chunks_match = left_manifest.get("chunk_sha256") == right_manifest.get("chunk_sha256")
        embedding_match = (
            left_manifest.get("embedding_model"), left_manifest.get("embedding_revision")
        ) == (
            right_manifest.get("embedding_model"), right_manifest.get("embedding_revision")
        )
        index_match = left_manifest.get("index_sha256") == right_manifest.get("index_sha256")
        common_ids = sorted(set(left["record_by_id"]) & set(right["record_by_id"]))
        prefix_k_values = set()
        query_matches = hash_matches = hash_mismatches = 0
        for input_id in common_ids:
            first = left["record_by_id"][input_id]
            second = right["record_by_id"][input_id]
            same_query = first.get("query_text") == second.get("query_text")
            prefix_k = min(int(first.get("top_k_requested") or 0), int(second.get("top_k_requested") or 0))
            prefix_k_values.add(prefix_k)
            first_hashes = [chunk.get("chunk_sha256") for chunk in first.get("retrieved_chunks", [])[:prefix_k]]
            second_hashes = [chunk.get("chunk_sha256") for chunk in second.get("retrieved_chunks", [])[:prefix_k]]
            hashes_match = same_query and first_hashes == second_hashes
            query_matches += same_query
            hash_matches += hashes_match
            hash_mismatches += same_query and not hashes_match
            detail_rows.append(
                {
                    "left_trace": left["name"],
                    "right_trace": right["name"],
                    "input_id": input_id,
                    "same_query_text": same_query,
                    "prefix_k_compared": prefix_k,
                    "ordered_chunk_hashes_match": hashes_match if same_query else "",
                    "source_manifest_matches": source_match,
                    "chunk_manifest_matches": chunks_match,
                    "embedding_settings_match": embedding_match,
                    "faiss_index_hash_matches": index_match,
                }
            )
        comparable = query_matches
        summary_rows.append(
            {
                "left_trace": left["name"],
                "right_trace": right["name"],
                "shared_input_ids": len(common_ids),
                "identical_query_texts": query_matches,
                "prefix_k_values": ",".join(map(str, sorted(prefix_k_values))),
                "hash_matches": hash_matches,
                "hash_mismatches": hash_mismatches,
                "hash_match_rate": hash_matches / comparable if comparable else "",
                "source_manifest_matches": source_match,
                "chunk_manifest_matches": chunks_match,
                "embedding_settings_match": embedding_match,
                "faiss_index_hash_matches": index_match,
            }
        )
    return summary_rows, detail_rows


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    key_ids = {row["ID"] for row in read_csv(KEY_FILE)}
    runs = [load_trace(path) for path in sorted(SOURCE.glob("*_retrievals.jsonl"))]
    run_summaries = []
    query_metrics = []
    chunk_distances = []
    for run in runs:
        summary, queries, chunks = analyze_run(run, key_ids)
        run_summaries.append(summary)
        query_metrics.extend(queries)
        chunk_distances.extend(chunks)

    comparison_summaries, comparison_details = compare_runs(runs)
    integrity_summary = build_integrity_summary(runs, query_metrics, chunk_distances)
    claim_comparisons = compare_claim_pairs(runs)
    write_csv(OUTPUT / "retrieval_summary.csv", list(run_summaries[0]), run_summaries)
    write_csv(OUTPUT / "retrieval_query_metrics.csv", list(query_metrics[0]), query_metrics)
    write_csv(OUTPUT / "retrieval_chunk_distances.csv", list(chunk_distances[0]), chunk_distances)
    write_csv(
        OUTPUT / "top_k_comparison_summary.csv",
        list(comparison_summaries[0]) if comparison_summaries else [],
        comparison_summaries,
    )
    write_csv(
        OUTPUT / "top_k_hash_comparisons.csv",
        list(comparison_details[0]) if comparison_details else [],
        comparison_details,
    )
    write_csv(
        OUTPUT / "retrieval_integrity_summary.csv",
        list(integrity_summary),
        [integrity_summary],
    )
    write_csv(
        OUTPUT / "retrieval_claim_comparisons.csv",
        list(claim_comparisons[0]) if claim_comparisons else [],
        claim_comparisons,
    )
    print(f"Processed {len(runs)} traces, {len(query_metrics)} query records, and {len(chunk_distances)} retrieved chunks.")
    print(f"Wrote retrieval validation CSVs to {OUTPUT}")


if __name__ == "__main__":
    main()