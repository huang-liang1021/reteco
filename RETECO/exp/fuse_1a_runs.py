#!/usr/bin/env python3
"""Fuse BM25 with ONE embedding or reranker TREC run (stdlib only).

Default: weighted reciprocal rank fusion (RRF). Optional minmax fusion
normalizes model scores per query/source. No qrels or model loading is used.
Input files: <prefix>_1a_<domain>_<train|dev>.txt. An explicit prefix may
be supplied; otherwise exactly one suffix-matching file must exist.
Output directory must be fresh and named runs_<method>_top<K>.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

DOMAINS = (
    "bitcoin", "cardano", "economics", "genealogy", "history", "hsm",
    "iota", "law", "monero", "politics", "quant", "travel", "workplace",
)


@dataclass(frozen=True)
class Hit:
    doc_id: str
    rank: int
    score: float


def finite_nonnegative(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def find_run(directory, prefix, domain, split):
    suffix = f"_1a_{domain}_{split}.txt"
    if prefix:
        matches = [directory / f"{prefix}{suffix}"]
        if not matches[0].is_file():
            raise FileNotFoundError(f"Missing run: {matches[0]}")
    else:
        matches = sorted(p for p in directory.iterdir() if p.is_file() and p.name.endswith(suffix))
        if not matches:
            raise FileNotFoundError(f"Missing run: no *{suffix} under {directory}")
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one *{suffix} under {directory}; found {len(matches)}. "
                "Use --bm25-prefix or --neural-prefix to disambiguate."
            )
    return matches[0]


def read_run(path):
    """Use explicit ranks, not file order; reject ambiguous or non-finite runs."""
    groups = defaultdict(list)
    seen_docs, seen_ranks = defaultdict(set), defaultdict(set)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            digest.update(raw)
            fields = raw.decode("utf-8-sig").split()
            if not fields:
                continue
            where = f"{path}:{line_number}"
            if len(fields) != 6:
                raise ValueError(f"{where}: expected six TREC columns")
            qid, q0, doc_id, rank_text, score_text, _tag = fields
            try:
                rank, score = int(rank_text), float(score_text)
            except ValueError as exc:
                raise ValueError(f"{where}: invalid rank or score") from exc
            if q0 != "Q0" or rank < 1 or not math.isfinite(score):
                raise ValueError(f"{where}: invalid Q0, rank, or non-finite score")
            if doc_id in seen_docs[qid] or rank in seen_ranks[qid]:
                raise ValueError(f"{where}: duplicate doc ID or rank for query {qid}")
            seen_docs[qid].add(doc_id)
            seen_ranks[qid].add(rank)
            groups[qid].append(Hit(doc_id, rank, score))
    if not groups:
        raise ValueError(f"Empty run: {path}")
    for qid, hits in groups.items():
        hits.sort(key=lambda h: h.rank)
        if [h.rank for h in hits] != list(range(1, len(hits) + 1)):
            raise ValueError(f"{path}: query {qid} ranks must be contiguous from 1")
        if any(b.score > a.score + 1e-8 for a, b in zip(hits, hits[1:])):
            raise ValueError(f"{path}: query {qid} scores increase with rank")
    return dict(groups), digest.hexdigest()


def has_negative_rank_scores(groups):
    return all(h.score == -h.rank for hits in groups.values() for h in hits)


def minmax_scores(hits):
    low, high = min(h.score for h in hits), max(h.score for h in hits)
    if high == low:
        # A constant source supplies equal evidence for its retrieved documents.
        return {h.doc_id: 1.0 for h in hits}
    # Rescale first to avoid overflow when finite extreme scores span both signs.
    scale = max(abs(low), abs(high))
    lo, hi = low / scale, high / scale
    return {h.doc_id: (h.score / scale - lo) / (hi - lo) for h in hits}


def fuse_query(bm25, neural, bm25_weight, neural_weight, fusion, rrf_k, top_k):
    """Missing source contribution is zero. Zero-weight sources add no docs."""
    b_rank = {h.doc_id: h.rank for h in bm25}
    n_rank = {h.doc_id: h.rank for h in neural}
    weights_sum = bm25_weight + neural_weight
    bw, nw = bm25_weight / weights_sum, neural_weight / weights_sum
    if fusion == "rrf":
        b_values = {h.doc_id: 1.0 / (rrf_k + h.rank) for h in bm25}
        n_values = {h.doc_id: 1.0 / (rrf_k + h.rank) for h in neural}
    else:
        b_values, n_values = minmax_scores(bm25), minmax_scores(neural)
    candidates = (set(b_values) if bw else set()) | (set(n_values) if nw else set())
    scores = {d: bw * b_values.get(d, 0.0) + nw * n_values.get(d, 0.0) for d in candidates}
    inf = float("inf")
    order = sorted(candidates, key=lambda d: (
        -scores[d], n_rank.get(d, inf) if nw else inf,
        b_rank.get(d, inf) if bw else inf, d,
    ))[:top_k]
    return [(doc_id, scores[doc_id]) for doc_id in order]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bm25-dir", type=Path, required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--reranker-dir", type=Path)
    source.add_argument("--embedding-dir", type=Path)
    p.add_argument("--bm25-prefix", help="optional filename prefix, e.g. bm25")
    p.add_argument("--neural-prefix", help="optional reranker/embedding filename prefix")
    p.add_argument("--bm25-weight", type=finite_nonnegative, default=0.3)
    p.add_argument("--neural-weight", type=finite_nonnegative, default=1.0,
                   help="reranker or embedding weight (default: 1)")
    p.add_argument("--fusion", choices=("rrf", "minmax"), default="rrf")
    p.add_argument("--rrf-k", type=finite_nonnegative, default=60.0,
                   help="RRF rank smoothing constant, unrelated to retrieval depth")
    p.add_argument("--bm25-depth", type=int, default=500,
                   help="maximum BM25 input rank to use (default: 500)")
    p.add_argument("--neural-depth", type=int, default=500)
    p.add_argument("--top-k", type=int, default=500, help="fused output depth")
    p.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    p.add_argument("--splits", nargs="+", choices=("train", "dev"), default=["train", "dev"])
    p.add_argument("--method", help="output filename prefix; inferred from --out-dir if omitted")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="fresh directory named runs_<method>_top<K>")
    p.add_argument("--dry-run", action="store_true",
                   help="validate inputs and compute statistics without writing outputs")
    args = p.parse_args()
    if min(args.bm25_depth, args.neural_depth, args.top_k) < 1:
        p.error("input depths and --top-k must be positive")
    total_weight = args.bm25_weight + args.neural_weight
    if not math.isfinite(total_weight) or total_weight <= 0:
        p.error("sum of weights must be finite and positive")
    args.neural_kind = "reranker" if args.reranker_dir else "embedding"
    args.neural_dir = (args.reranker_dir or args.embedding_dir).resolve()
    args.bm25_dir, args.out_dir = args.bm25_dir.resolve(), args.out_dir.resolve()
    for directory in (args.bm25_dir, args.neural_dir):
        if not directory.is_dir():
            p.error(f"Input directory missing: {directory}")
    match = re.fullmatch(r"runs_(.+)_top(\d+)", args.out_dir.name)
    if not match or int(match[2]) != args.top_k:
        p.error("--out-dir must be named runs_<method>_top<top-k>")
    args.method = args.method or match[1]
    if args.method != match[1] or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.method):
        p.error("--method must match output directory and use letters/digits/underscores/dots/hyphens")
    for prefix in (args.bm25_prefix, args.neural_prefix):
        if prefix is not None and (not prefix or any(c.isspace() or c in "/\\" for c in prefix)):
            p.error("input prefixes must be nonempty filename-safe strings")
    if len(set(args.domains)) != len(args.domains) or len(set(args.splits)) != len(args.splits):
        p.error("duplicate domains/splits are not allowed")
    if args.out_dir in (args.bm25_dir, args.neural_dir):
        p.error("output must differ from input directories")
    return args


def main():
    args = parse_args()
    start = time.monotonic()
    staging = args.out_dir.with_name(args.out_dir.name + ".partial")
    if not args.dry_run and (args.out_dir.exists() or staging.exists()):
        raise ValueError(f"Use a fresh output directory; {args.out_dir} or {staging} already exists")
    # Fail on missing/ambiguous files before writing anything. Only explicit dirs are searched.
    jobs = [(domain, split,
             find_run(args.bm25_dir, args.bm25_prefix, domain, split),
             find_run(args.neural_dir, args.neural_prefix, domain, split))
            for domain in args.domains for split in args.splits]
    report = {
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "weights_normalized": {
            "bm25": args.bm25_weight / (args.bm25_weight + args.neural_weight),
            "neural": args.neural_weight / (args.bm25_weight + args.neural_weight),
        },
        "missing_document_contribution": 0,
        "tie_break": "neural rank, BM25 rank (active sources only), document ID",
        "normalization": "per query, per source, within retained input depth",
        "constant_minmax_source": "all retrieved documents receive 1, absent documents 0",
        "labels_loaded": False,
        "metrics_computed": False,
        "files": [],
    }
    query_rows = []
    for domain, split, b_path, n_path in jobs:
        bm25, b_sha = read_run(b_path)
        neural, n_sha = read_run(n_path)
        if set(bm25) != set(neural):
            raise ValueError(
                f"{domain}/{split}: query sets differ. "
                f"BM25-only={sorted(set(bm25) - set(neural))[:10]}, "
                f"neural-only={sorted(set(neural) - set(bm25))[:10]}"
            )
        b_rank_only, n_rank_only = has_negative_rank_scores(bm25), has_negative_rank_scores(neural)
        if args.fusion == "minmax" and (
            (b_rank_only and args.bm25_weight > 0) or (n_rank_only and args.neural_weight > 0)
        ):
            raise ValueError(
                f"{domain}/{split}: scores are -rank, not model scores. Use --fusion rrf. "
                "The reranker .scores.jsonl must be converted separately for actual score fusion."
            )
        filename = f"{args.method}_1a_{domain}_{split}.txt"
        output_rows = []
        for qid in sorted(bm25):
            b_hits, n_hits = bm25[qid][:args.bm25_depth], neural[qid][:args.neural_depth]
            fused = fuse_query(b_hits, n_hits, args.bm25_weight, args.neural_weight,
                               args.fusion, args.rrf_k, args.top_k)
            if len(fused) < args.top_k:
                raise ValueError(
                    f"{domain}/{split}/{qid}: active candidate union has only {len(fused)} docs; "
                    f"cannot output top-{args.top_k}. Lower --top-k or provide deeper inputs."
                )
            b_ids, n_ids = {h.doc_id for h in b_hits}, {h.doc_id for h in n_hits}
            chosen = {d for d, _ in fused}
            query_rows.append({
                "domain": domain, "split": split, "query_id": qid,
                "bm25_input_count": len(b_hits), "neural_input_count": len(n_hits),
                "bm25_input_short": len(b_hits) < args.bm25_depth,
                "neural_input_short": len(n_hits) < args.neural_depth,
                "intersection_count": len(b_ids & n_ids), "union_count": len(b_ids | n_ids),
                "output_count": len(fused),
                "output_both": len(chosen & b_ids & n_ids),
                "output_bm25_only": len((chosen & b_ids) - n_ids),
                "output_neural_only": len((chosen & n_ids) - b_ids),
                "output_top10_bm25_only": sum(d in b_ids and d not in n_ids for d, _ in fused[:10]),
                "output_top10_neural_only": sum(d in n_ids and d not in b_ids for d, _ in fused[:10]),
            })
            for rank_no, (doc_id, score) in enumerate(fused, 1):
                if not math.isfinite(score):
                    raise FloatingPointError(f"Non-finite fused score: {domain}/{split}/{qid}")
                output_rows.append(f"{qid}\tQ0\t{doc_id}\t{rank_no}\t{score:.17g}\t{args.method}\n")
        if not args.dry_run:
            staging.mkdir(parents=True, exist_ok=True)
            (staging / filename).write_text("".join(output_rows), encoding="utf-8")
        report["files"].append({
            "domain": domain, "split": split, "bm25_file": str(b_path), "neural_file": str(n_path),
            "bm25_sha256": b_sha, "neural_sha256": n_sha,
            "bm25_scores_are_negative_rank": b_rank_only,
            "neural_scores_are_negative_rank": n_rank_only,
            "num_queries": len(bm25), "output_lines": len(output_rows),
            "output_file": str(args.out_dir / filename),
        })
        print(f"{domain}/{split}: {len(bm25)} queries, top-{args.top_k}, {args.fusion}", flush=True)
    report["elapsed_seconds"] = time.monotonic() - start
    report["num_queries"] = len(query_rows)
    report["queries_with_short_inputs"] = sum(r["bm25_input_short"] or r["neural_input_short"] for r in query_rows)
    if not args.dry_run:
        (staging / "fusion_config.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        with (staging / "fusion_query_stats.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(query_rows[0]))
            writer.writeheader()
            writer.writerows(query_rows)
        os.rename(staging, args.out_dir)
    print(f"{'DRY RUN PASSED' if args.dry_run else 'COMPLETED'}: {len(jobs)} files, "
          f"{len(query_rows)} queries, {report['elapsed_seconds']:.2f}s; "
          f"short inputs={report['queries_with_short_inputs']}", flush=True)
    if not args.dry_run:
        print(f"Runs and fusion reports: {args.out_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, FloatingPointError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
