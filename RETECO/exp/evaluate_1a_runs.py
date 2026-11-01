#!/usr/bin/env python3
"""Validate, score, and diagnose RETECO Track 1a runs.

Only the directory supplied with --run-dir is read. A top-500 run can provide
Recall@100, @300, and @500; Recall@1000 requires a top-1000 run.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


DOMAINS = (
    "bitcoin", "cardano", "economics", "genealogy", "history", "hsm",
    "iota", "law", "monero", "politics", "quant", "travel", "workplace",
)
TARGET_DEPTHS = (100, 300, 500, 1000)
METRICS = ("nDCG@10", "TP@10", "TR@10", "TC@10", "NDCG|FC@10")
RUN_DIR_RE = re.compile(
    r"^runs_(?P<method>.+)_top(?P<depth>\d+)(?:_(?P<variant>[A-Za-z0-9][A-Za-z0-9_-]*))?$"
)


def command_output(*args: object) -> str:
    result = subprocess.run(
        [str(arg) for arg in args], capture_output=True, text=True
    )
    if result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}): "
            f"{' '.join(map(str, args))}\n{result.stdout}{result.stderr}"
        )
    return result.stdout


def read_qrels(path: Path) -> dict[str, set[str]]:
    """Read TREC qrels and keep all documents with relevance > 0."""
    qrels: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 4:
                raise ValueError(f"Invalid qrels line {path}:{line_no}: {line!r}")
            qid, docid, rel = fields[0], fields[2], fields[3]
            try:
                if float(rel) > 0:
                    qrels[qid].add(docid)
            except ValueError as exc:
                raise ValueError(f"Invalid relevance at {path}:{line_no}") from exc
    return dict(qrels)


def read_run(path: Path) -> dict[str, list[str]]:
    """Read a TREC run, preserving rank order and removing duplicate docids."""
    runs: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 3:
                raise ValueError(f"Invalid run line {path}:{line_no}: {line!r}")
            qid, docid = fields[0], fields[2]
            if docid not in seen[qid]:
                runs[qid].append(docid)
                seen[qid].add(docid)
    return dict(runs)


def count_examples(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        return sum(bool(line.strip()) for line in stream)


def recall_row(
    method: str,
    source_depth: int,
    target_depth: int,
    domain: str,
    split: str,
    qrels: dict[str, set[str]],
    run: dict[str, list[str]],
) -> dict[str, object]:
    query_recalls: list[float] = []
    relevant_found = 0
    relevant_total = 0
    hit_queries = 0

    for qid, positives in qrels.items():
        retrieved = set(run.get(qid, [])[:target_depth])
        hits = len(retrieved & positives)
        relevant_found += hits
        relevant_total += len(positives)
        query_recalls.append(hits / len(positives) if positives else 0.0)
        if hits > 0:
            hit_queries += 1

    num_topics = len(qrels)
    return {
        "method": method,
        "source_depth": source_depth,
        "depth": target_depth,
        "domain": domain,
        "split": split,
        "num_topics": num_topics,
        "relevant_found": relevant_found,
        "relevant_total": relevant_total,
        "recall_query_macro": (
            round(sum(query_recalls) / num_topics, 6) if num_topics else None
        ),
        "recall_micro": (
            round(relevant_found / relevant_total, 6) if relevant_total else None
        ),
        "query_hit_rate": (
            round(hit_queries / num_topics, 6) if num_topics else None
        ),
    }


def aggregate_recall(rows: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate domain rows with every query weighted equally."""
    if not rows:
        raise ValueError("Cannot aggregate empty recall rows")

    num_topics = sum(int(row["num_topics"]) for row in rows)
    relevant_found = sum(int(row["relevant_found"]) for row in rows)
    relevant_total = sum(int(row["relevant_total"]) for row in rows)
    query_recall_sum = sum(
        float(row["recall_query_macro"]) * int(row["num_topics"])
        for row in rows
        if row["recall_query_macro"] is not None
    )
    hit_sum = sum(
        float(row["query_hit_rate"]) * int(row["num_topics"])
        for row in rows
        if row["query_hit_rate"] is not None
    )
    first = rows[0]
    return {
        "method": first["method"],
        "source_depth": first["source_depth"],
        "depth": first["depth"],
        "domain": "ALL_DOMAINS",
        "split": first["split"],
        "num_topics": num_topics,
        "relevant_found": relevant_found,
        "relevant_total": relevant_total,
        "recall_query_macro": (
            round(query_recall_sum / num_topics, 6) if num_topics else None
        ),
        "recall_micro": (
            round(relevant_found / relevant_total, 6)
            if relevant_total else None
        ),
        "query_hit_rate": round(hit_sum / num_topics, 6) if num_topics else None,
    }


def parse_run_dir(run_dir: Path) -> tuple[str, int, Path]:
    """Validate one explicitly supplied run directory and infer its metadata."""
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    match = RUN_DIR_RE.fullmatch(run_dir.name)
    if not match:
        raise ValueError(
            f"Run directory must be named runs_<method>_top<depth>[_<variant>]: {run_dir}"
        )
    return match.group("method"), int(match.group("depth")), run_dir


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=repo.parent / "reteco_data",
        help="dataset root containing track1_tempo",
    )
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="one concrete directory, e.g. runs/runs_qwen3_embedding_8b_top500",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=repo / "diagnostics",
        help="single output directory for all reports",
    )
    args = parser.parse_args()

    data_root = args.data / "track1_tempo"
    if not data_root.is_dir():
        parser.error(f"Data directory does not exist: {data_root}")

    args.run_dir = args.run_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = [parse_run_dir(args.run_dir)]
    domains = [name for name in DOMAINS if (data_root / name).is_dir()]
    if not domains:
        parser.error(f"No Track 1 domains found under {data_root}")

    format_blocks: list[str] = []
    score_rows: list[dict[str, object]] = []
    run_cache: dict[
        tuple[str, int, str, str], tuple[Path, dict[str, list[str]]]
    ] = {}
    qrels_cache: dict[tuple[str, str], dict[str, set[str]]] = {}

    # Score every available source depth. The same run is also reused below
    # to calculate shallower Recall@K values.
    for method, source_depth, run_dir in run_dirs:
        for domain in domains:
            for split in ("train", "dev"):
                label = f"{method}/top{source_depth}/{domain}_{split}"
                run = run_dir / f"{method}_1a_{domain}_{split}.txt"
                domain_dir = data_root / domain
                qrels_path = domain_dir / f"qrels_{split}.txt"
                examples_path = domain_dir / f"examples_{split}.jsonl"
                steps_path = domain_dir / f"steps_{split}.jsonl"
                corpus_path = domain_dir / "documents.jsonl"

                if not run.is_file():
                    raise FileNotFoundError(f"Missing run: {run}")

                report = command_output(
                    sys.executable,
                    repo / "starter_kit" / "format_checker.py",
                    run,
                    "--qrels", qrels_path,
                    "--corpus", corpus_path,
                )
                format_blocks.append(f"===== {label} =====\n{report.strip()}\n")
                if "RESULT: VALID" not in report or "errors: 0" not in report:
                    raise RuntimeError(f"Format check failed for {label}:\n{report}")

                score = json.loads(command_output(
                    sys.executable,
                    repo / "starter_kit" / "scorer.py",
                    "--track", "1",
                    "--run", run,
                    "--qrels", qrels_path,
                    "--steps", steps_path,
                    "--json",
                ))
                expected = count_examples(examples_path)
                if score["num_topics"] != expected:
                    raise RuntimeError(
                        f"{label}: scored {score['num_topics']} of {expected} queries"
                    )

                score_rows.append({
                    "method": method,
                    "source_depth": source_depth,
                    "domain": domain,
                    "split": split,
                    "run": str(run),
                    "num_topics": score["num_topics"],
                    **{metric: score.get(metric) for metric in METRICS},
                    "num_full_coverage": score.get("num_full_coverage"),
                })
                score_rows[-1]["format_valid"] = True
                qrels_cache[(domain, split)] = read_qrels(qrels_path)
                run_cache[(method, source_depth, domain, split)] = (
                    run, read_run(run)
                )
                print(f"{label}: nDCG@10={score['nDCG@10']:.4f}", flush=True)

    recall_rows: list[dict[str, object]] = []
    methods = sorted({method for method, _, _ in run_dirs})
    for method in methods:
        available_depths = sorted(
            depth for current_method, depth, _ in run_dirs
            if current_method == method
        )
        for split in ("train", "dev"):
            for target_depth in TARGET_DEPTHS:
                usable = [depth for depth in available_depths if depth >= target_depth]
                if not usable:
                    continue
                source_depth = min(usable)
                domain_rows: list[dict[str, object]] = []
                for domain in domains:
                    key = (method, source_depth, domain, split)
                    if key not in run_cache:
                        continue
                    row = recall_row(
                        method, source_depth, target_depth, domain, split,
                        qrels_cache[(domain, split)], run_cache[key][1],
                    )
                    recall_rows.append(row)
                    domain_rows.append(row)
                if domain_rows:
                    recall_rows.append(aggregate_recall(domain_rows))

    (args.out_dir / "format_checks_1a.txt").write_text(
        "\n".join(format_blocks) + "\n", encoding="utf-8"
    )

    summary = {
        "task": "RETECO Track 1a",
        "run_dir": str(args.run_dir),
        "data_root": str(data_root.resolve()),
        "score_rows": score_rows,
        "recall_rows": recall_rows,
        "recall_definition": {
            "recall_query_macro": (
                "mean over queries of retrieved relevant documents divided "
                "by all qrels-positive documents"
            ),
            "recall_micro": (
                "all retrieved relevant documents divided by all qrels-positive "
                "documents"
            ),
            "query_hit_rate": (
                "fraction of queries with at least one retrieved qrels-positive "
                "document"
            ),
            "source_selection": (
                "for each target K, use the smallest available top-K directory "
                "that contains at least K ranks"
            ),
        },
    }
    (args.out_dir / "evaluation_1a.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    score_fields = (
        "method", "source_depth", "domain", "split", "run", "num_topics",
        "format_valid", *METRICS, "num_full_coverage",
    )
    with (args.out_dir / "evaluation_1a.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=score_fields)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field) for field in score_fields}
            for row in score_rows
        )

    recall_fields = (
        "method", "source_depth", "depth", "domain", "split", "num_topics",
        "relevant_found", "relevant_total", "recall_query_macro",
        "recall_micro", "query_hit_rate",
    )
    with (args.out_dir / "recall_1a.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=recall_fields)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field) for field in recall_fields}
            for row in recall_rows
        )

    # Wide table for quick inspection: one row per method/domain/split and
    # one column for each requested depth.  ALL_DOMAINS is the query-weighted
    # total for that method and split.
    wide: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in recall_rows:
        key = (str(row["method"]), str(row["domain"]), str(row["split"]))
        item = wide.setdefault(key, {
            "method": row["method"],
            "domain": row["domain"],
            "split": row["split"],
        })
        depth = int(row["depth"])
        item[f"source_top{depth}"] = row["source_depth"]
        item[f"recall_query_macro@{depth}"] = row["recall_query_macro"]
        item[f"recall_micro@{depth}"] = row["recall_micro"]
        item[f"query_hit_rate@{depth}"] = row["query_hit_rate"]
    wide_fields = ["method", "domain", "split"]
    for depth in TARGET_DEPTHS:
        wide_fields.extend([
            f"source_top{depth}",
            f"recall_query_macro@{depth}",
            f"recall_micro@{depth}",
            f"query_hit_rate@{depth}",
        ])
    with (args.out_dir / "recall_summary_1a.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=wide_fields)
        writer.writeheader()
        writer.writerows(
            {field: item.get(field) for field in wide_fields}
            for item in sorted(wide.values(), key=lambda item: (
                str(item["method"]), str(item["split"]), str(item["domain"])
            ))
        )

    print(f"Validated {len(score_rows)} runs")
    print(f"Recall rows: {len(recall_rows)}")
    print(f"Reports: {args.out_dir}")


if __name__ == "__main__":
    main()

