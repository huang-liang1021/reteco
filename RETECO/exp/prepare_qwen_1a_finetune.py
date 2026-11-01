#!/usr/bin/env python3
"""Build RETECO Track 1a TRAIN query groups as one JSON array.

Uses only examples_train.jsonl, qrels_train.txt and *_train.txt runs.
All known positives are retained, including positives outside the source run.
After excluding positives and copies with the same normalized text, sample
2/4/2 hard candidates from ORIGINAL source ranks 1-50/51-200/201-500.
Also sample ``5 * number_of_positives`` ordinary candidates uniformly from
the corresponding domain corpus. Hard and ordinary candidates are kept in
separate JSON fields. Unjudged candidates are not verified negatives. No model
or GPU is required.

Requires Python 3.9+ and only the standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


DOMAINS = (
    "bitcoin", "cardano", "economics", "genealogy", "history", "hsm",
    "iota", "law", "monero", "politics", "quant", "travel", "workplace",
)
BUCKETS = ((1, 50), (51, 200), (201, 500))
BUCKET_NAMES = ("1-50", "51-200", "201-500")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl_rows(path):
    with path.open(encoding="utf-8-sig") as stream:
        for line_no, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_no}")
            yield line_no, row


def require_id(value, where):
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise ValueError(f"Expected a string/integer ID at {where}")
    value = str(value)
    if not value.strip():
        raise ValueError(f"Empty ID at {where}")
    return value


def load_queries(path):
    queries = []
    seen = set()
    for line_no, row in jsonl_rows(path):
        where = f"{path}:{line_no}"
        qid = require_id(row.get("id"), where)
        if qid in seen:
            raise ValueError(f"Duplicate query ID {qid!r} in {path}")
        seen.add(qid)
        query = row.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Missing/empty query at {where}")
        gold = row.get("gold_ids")
        if not isinstance(gold, list):
            raise ValueError(f"gold_ids must be an array at {where}")
        gold = list(dict.fromkeys(require_id(d, where) for d in gold))
        queries.append({"id": qid, "query": query, "gold_ids": gold})
    if not queries:
        raise ValueError(f"No training queries in {path}")
    return queries


def load_qrels(path, query_ids):
    labels = defaultdict(dict)
    with path.open(encoding="utf-8-sig") as stream:
        for line_no, raw in enumerate(stream, 1):
            fields = raw.split()
            if not fields:
                continue
            if len(fields) != 4:
                raise ValueError(f"Expected four qrels columns at {path}:{line_no}")
            qid, _, doc_id, rel_raw = fields
            if qid not in query_ids:
                raise ValueError(f"Unknown train query {qid!r} at {path}:{line_no}")
            try:
                rel = float(rel_raw)
            except ValueError as exc:
                raise ValueError(f"Invalid relevance at {path}:{line_no}") from exc
            if not math.isfinite(rel):
                raise ValueError(f"Non-finite relevance at {path}:{line_no}")
            if doc_id in labels[qid] and labels[qid][doc_id] != rel:
                raise ValueError(f"Conflicting qrels for {qid}/{doc_id} in {path}")
            labels[qid][doc_id] = rel
    return dict(labels)


def load_run(path, query_ids):
    """Use explicit ranks, not physical file order or scores as probabilities."""
    runs = defaultdict(list)
    ranks = defaultdict(set)
    with path.open(encoding="utf-8-sig") as stream:
        for line_no, raw in enumerate(stream, 1):
            fields = raw.split()
            if not fields:
                continue
            if len(fields) != 6:
                raise ValueError(f"Expected six run columns at {path}:{line_no}")
            qid, q0, doc_id, rank_raw, score_raw, _ = fields
            if qid not in query_ids:
                raise ValueError(f"Unknown train query {qid!r} at {path}:{line_no}")
            try:
                rank, score = int(rank_raw), float(score_raw)
            except ValueError as exc:
                raise ValueError(f"Invalid rank/score at {path}:{line_no}") from exc
            if q0 != "Q0" or rank < 1:
                raise ValueError(f"Invalid Q0/rank at {path}:{line_no}")
            if not math.isfinite(score):
                raise ValueError(
                    f"NaN/Inf score at {path}:{line_no}; regenerate the invalid run "
                    "before mining training data"
                )
            if rank in ranks[qid]:
                raise ValueError(f"Duplicate rank {rank} for {qid} in {path}")
            ranks[qid].add(rank)
            runs[qid].append({"doc_id": doc_id, "rank": rank, "score": score})
    for qid in sorted(query_ids):
        if qid not in runs:
            raise ValueError(f"Missing train query {qid!r} in {path}")
        rows = sorted(runs[qid], key=lambda r: r["rank"])
        missing = set(range(1, 501)) - ranks[qid]
        if missing:
            raise ValueError(
                f"{path}: query {qid} does not have complete ranks 1-500; "
                f"first missing ranks: {sorted(missing)[:8]}"
            )
        for previous, current in zip(rows, rows[1:]):
            if current["score"] > previous["score"] + 1e-7:
                raise ValueError(f"Score increases with explicit rank for {qid} in {path}")
        # A top-1000 file can be used, but only its original first 500 ranks
        # belong to this fixed Top-500 sampling experiment.
        runs[qid] = [r for r in rows if r["rank"] <= 500]
    return dict(runs)


def text_fingerprint(text):
    """Detect exact/whitespace/Unicode copies, not semantic near-duplicates."""
    normalized = " ".join(unicodedata.normalize("NFKC", text).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_documents(path, needed_ids):
    """Stream one corpus at a time; retain only this domain's needed texts."""
    documents, fingerprints = {}, {}
    seen = set()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for line_no, raw in enumerate(stream, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except (ValueError, UnicodeError) as exc:
                raise ValueError(f"Invalid corpus JSON at {path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a corpus object at {path}:{line_no}")
            doc_id = require_id(row.get("id"), f"{path}:{line_no}")
            if doc_id in seen:
                raise ValueError(f"Duplicate corpus document ID {doc_id!r} in {path}")
            seen.add(doc_id)
            if doc_id in needed_ids:
                content = row.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError(f"Missing/empty content for {doc_id!r} in {path}")
                documents[doc_id] = content
                fingerprints[doc_id] = text_fingerprint(content)
    missing = needed_ids - documents.keys()
    if missing:
        raise ValueError(
            f"{path}: {len(missing)} positive/run document IDs are missing; "
            f"examples: {sorted(missing)[:8]}. Check corpus/run versions; "
            "no positives or candidates are silently dropped."
        )
    return documents, fingerprints, digest.hexdigest(), len(seen)


def query_rng(seed, domain, qid, purpose="hard"):
    # Stable across --domains order and separate per-domain invocations.
    value = f"{seed}\0{domain}\0{qid}\0{purpose}".encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(value).digest(), "big"))


def bucket_index(rank):
    for index, (low, high) in enumerate(BUCKETS):
        if low <= rank <= high:
            return index
    raise ValueError(f"Rank outside sampling policy: {rank}")


def select_negatives(candidates, counts, rng):
    pools = [[] for _ in BUCKETS]
    for row in candidates:
        pools[bucket_index(row["rank"])].append(row)
    selected, initial_shortages = [], {}
    for index, requested in enumerate(counts):
        take = min(requested, len(pools[index]))
        selected.extend(rng.sample(pools[index], take))
        initial_shortages[BUCKET_NAMES[index]] = requested - take
    selected_ids = {r["doc_id"] for r in selected}
    remaining = sum(counts) - len(selected)
    # Prefer unused middle ranks when another bucket cannot meet its quota.
    for index in (1, 0, 2):
        pool = [r for r in pools[index] if r["doc_id"] not in selected_ids]
        extra = rng.sample(pool, min(remaining, len(pool)))
        selected.extend(extra)
        selected_ids.update(r["doc_id"] for r in extra)
        remaining -= len(extra)
    selected.sort(key=lambda r: r["rank"])
    return selected, {
        "available_by_bucket": dict(zip(BUCKET_NAMES, map(len, pools))),
        "initial_shortages_by_bucket": initial_shortages,
        "selected_by_bucket": dict(zip(BUCKET_NAMES, [
            sum(bucket_index(r["rank"]) == i for r in selected) for i in range(3)
        ])),
        "remaining_shortage": remaining,
    }


def index_corpus(path):
    """Return byte offsets for non-empty JSONL rows without retaining texts."""
    offsets = []
    seen_ids = set()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            offset = stream.tell()
            raw = stream.readline()
            if not raw:
                break
            digest.update(raw)
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except (ValueError, UnicodeError) as exc:
                raise ValueError(f"Invalid corpus JSON at {path}:{len(offsets) + 1}") from exc
            doc_id = require_id(row.get("id"), f"{path}:{len(offsets) + 1}")
            if doc_id in seen_ids:
                raise ValueError(f"Duplicate corpus document ID {doc_id!r} in {path}")
            seen_ids.add(doc_id)
            if not isinstance(row.get("content"), str) or not row["content"].strip():
                raise ValueError(f"Missing/empty content for {doc_id!r} in {path}")
            offsets.append(offset)
    return offsets, digest.hexdigest(), len(seen_ids)


def read_corpus_row_at(stream, offset, path):
    stream.seek(offset)
    raw = stream.readline()
    try:
        row = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Invalid corpus JSON at {path} offset {offset}") from exc
    doc_id = require_id(row.get("id"), f"{path} offset {offset}")
    content = row.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"Missing/empty content for {doc_id!r} in {path}")
    return doc_id, content


def sample_ordinary_documents(
    corpus_path, offsets, positive_ids, positive_fingerprints,
    hard_ids, hard_fingerprints, requested, seed, domain, qid,
):
    """Sample ordinary negatives uniformly from this domain's corpus.

    Byte offsets are indexed once per domain. Random access avoids scanning a
    large corpus once per query. Exact positive/hard IDs and text copies are
    excluded for the current query; ordinary samples may be reused by another
    query because each query has its own negative set.
    """
    rng = query_rng(seed, domain, qid, "ordinary")
    forbidden_ids = set(positive_ids) | set(hard_ids)
    forbidden_fps = set(positive_fingerprints) | set(hard_fingerprints)
    selected, selected_ids, selected_fps, tried_positions = [], set(), set(), set()
    if requested <= 0:
        return [], {
            "requested": 0,
            "selected": 0,
            "shortage": 0,
            "corpus_documents": len(offsets),
            "sampling_method": "uniform_random_domain_corpus",
        }
    if not offsets:
        return [], {
            "requested": requested,
            "selected": 0,
            "shortage": requested,
            "corpus_documents": 0,
            "sampling_method": "uniform_random_domain_corpus",
        }
    max_attempts = min(max(len(offsets) * 2, requested * 200 + 1000), 200000)
    with corpus_path.open("rb") as stream:
        while len(selected) < requested and len(tried_positions) < max_attempts:
            position = rng.randrange(len(offsets))
            if position in tried_positions:
                continue
            tried_positions.add(position)
            doc_id, content = read_corpus_row_at(stream, offsets[position], corpus_path)
            content_fp = text_fingerprint(content)
            if doc_id in forbidden_ids or doc_id in selected_ids:
                continue
            if content_fp in forbidden_fps or content_fp in selected_fps:
                continue
            selected.append({"doc_id": doc_id, "context": content, "content_fp": content_fp})
            selected_ids.add(doc_id)
            selected_fps.add(content_fp)
        if len(selected) < requested:
            # Deterministic fallback guarantees a shortage is reported only
            # when the corpus truly has too few eligible unique documents.
            start = rng.randrange(len(offsets)) if offsets else 0
            for step in range(len(offsets)):
                if len(selected) >= requested:
                    break
                position = (start + step) % len(offsets)
                if position in tried_positions:
                    continue
                tried_positions.add(position)
                doc_id, content = read_corpus_row_at(stream, offsets[position], corpus_path)
                content_fp = text_fingerprint(content)
                if doc_id in forbidden_ids or doc_id in selected_ids:
                    continue
                if content_fp in forbidden_fps or content_fp in selected_fps:
                    continue
                selected.append({"doc_id": doc_id, "context": content, "content_fp": content_fp})
                selected_ids.add(doc_id)
                selected_fps.add(content_fp)
    for row in selected:
        row.pop("content_fp", None)
    return selected, {
        "requested": requested,
        "selected": len(selected),
        "shortage": requested - len(selected),
        "corpus_documents": len(offsets),
        "sampling_method": "uniform_random_domain_corpus",
    }


def domain_paths(args, domain):
    folder = args.data / "track1_tempo" / domain
    return {
        "examples": folder / "examples_train.jsonl",
        "qrels": folder / "qrels_train.txt",
        "corpus": folder / "documents.jsonl",
        "run": args.run_dir / f"{args.run_prefix}_1a_{domain}_train.txt",
    }


def build_domain(args, domain, paths):
    queries = load_queries(paths["examples"])
    query_ids = {q["id"] for q in queries}
    labels = load_qrels(paths["qrels"], query_ids)
    runs = load_run(paths["run"], query_ids)
    positive_ids, differences = {}, []
    needed = set()
    for query in queries:
        qid = query["id"]
        gold = set(query["gold_ids"])
        judged_positive = {d for d, rel in labels.get(qid, {}).items() if rel > 0}
        conflicts = {d for d in gold if d in labels.get(qid, {}) and labels[qid][d] <= 0}
        if conflicts:
            raise ValueError(f"{domain}/{qid}: gold_ids conflict with qrels: {sorted(conflicts)}")
        # Exclude ALL known positive IDs, not just the positives in the run.
        positive_ids[qid] = sorted(gold | judged_positive)
        if not positive_ids[qid]:
            raise ValueError(f"{domain}/{qid}: no known positive documents")
        if gold != judged_positive:
            differences.append({"query_id": qid,
                                "only_in_examples": sorted(gold - judged_positive),
                                "only_in_qrels": sorted(judged_positive - gold)})
        needed.update(positive_ids[qid])
        needed.update(r["doc_id"] for r in runs[qid])
    documents, fingerprints, corpus_hash, corpus_count = load_documents(paths["corpus"], needed)

    records, query_audits = [], []
    prepared = []
    hard_selected_by_qid = {}
    hard_fingerprints_by_qid = {}
    for query in queries:
        qid = query["id"]
        positives = positive_ids[qid]
        positive_set = set(positives)
        positive_texts = {fingerprints[d] for d in positives}
        seen_ids, seen_texts = set(), set()
        excluded = Counter()
        candidates = []
        for row in runs[qid]:
            doc_id = row["doc_id"]
            if doc_id in positive_set:
                excluded["known_positive_id"] += 1
            elif fingerprints[doc_id] in positive_texts:
                excluded["positive_text_copy"] += 1
            elif doc_id in seen_ids:
                excluded["duplicate_candidate_id"] += 1
            elif fingerprints[doc_id] in seen_texts:
                excluded["duplicate_candidate_text"] += 1
            else:
                candidates.append(row)
                seen_ids.add(doc_id)
                seen_texts.add(fingerprints[doc_id])
        selected, sampling_audit = select_negatives(
            candidates, args.bucket_counts, query_rng(args.seed, domain, qid)
        )
        if not selected:
            raise ValueError(f"{domain}/{qid}: no usable negative candidates after exclusions")
        hard_selected_by_qid[qid] = [row["doc_id"] for row in selected]
        hard_fingerprints_by_qid[qid] = [fingerprints[row["doc_id"]] for row in selected]
        prepared.append({
            "domain": domain,
            "qid": qid,
            "positive_fingerprints": {fingerprints[d] for d in positives},
            "sampling_audit": sampling_audit,
            "num_negative_candidates": len(candidates),
            "excluded": excluded,
        })

    offsets, indexed_corpus_hash, indexed_corpus_count = index_corpus(paths["corpus"])
    if indexed_corpus_hash != corpus_hash or indexed_corpus_count != corpus_count:
        raise RuntimeError(f"Corpus changed while building {domain}; retry with a stable corpus")
    ordinary_by_qid = {}
    ordinary_audits = {}
    for query in queries:
        qid = query["id"]
        positives = positive_ids[qid]
        prepared_item = next(item for item in prepared if item["qid"] == qid)
        ordinary_negatives, ordinary_audit = sample_ordinary_documents(
            corpus_path=paths["corpus"],
            offsets=offsets,
            positive_ids=positives,
            positive_fingerprints=prepared_item["positive_fingerprints"],
            hard_ids=hard_selected_by_qid[qid],
            hard_fingerprints=hard_fingerprints_by_qid[qid],
            requested=len(positives) * args.ordinary_multiplier,
            seed=args.seed,
            domain=domain,
            qid=qid,
        )
        ordinary_by_qid[qid] = ordinary_negatives
        ordinary_audits[qid] = ordinary_audit
        run_by_doc = {item["doc_id"]: item for item in runs[qid]}
        hard_negatives = []
        for doc_id in hard_selected_by_qid[qid]:
            row = run_by_doc[doc_id]
            # rel==0 explicitly means judged nonrelevant; missing or negative
            # qrels values (e.g. the conventional -1) remain unjudged.
            status = "judged_nonrelevant" if labels.get(qid, {}).get(doc_id) == 0 else "unjudged"
            hard_negatives.append({
                "doc_id": doc_id,
                "context": documents[doc_id],
                "source_rank": row["rank"],
                "source_score": row["score"],
                "label_status": status,
            })
        ordinary_negatives = [
            {
                "doc_id": item["doc_id"],
                "context": item["context"],
                "label_status": "unjudged",
                "sampling_method": "uniform_random_domain_corpus",
            }
            for item in ordinary_by_qid[qid]
        ]
        records.append({
            "domain": domain,
            "query_id": qid,
            "query": query["query"],
            "negative_source": args.negative_source,
            "positives": [{"doc_id": d, "context": documents[d]} for d in positives],
            "hard_negatives": hard_negatives,
            "ordinary_negatives": ordinary_negatives,
        })
        query_audits.append({
            "query_id": qid, "num_positives": len(positives),
            "num_negative_candidates": prepared_item["num_negative_candidates"],
            "num_hard_negatives": len(hard_negatives),
            "num_ordinary_negatives": len(ordinary_negatives),
            "ordinary_requested": ordinary_audits[qid]["requested"],
            "ordinary_shortage": ordinary_audits[qid]["shortage"],
            "ordinary_sampling_method": ordinary_audits[qid]["sampling_method"],
            "excluded": dict(prepared_item["excluded"]),
            **prepared_item["sampling_audit"],
        })

    # Keep the output ordering independent of physical examples file order.
    records.sort(key=lambda row: row["query_id"])
    query_audits.sort(key=lambda row: row["query_id"])
    input_files = {}
    for key, path in paths.items():
        input_files[key] = {
            "path": str(path), "bytes": path.stat().st_size,
            "sha256": corpus_hash if key == "corpus" else sha256_file(path),
        }
    audit = {
        "domain": domain, "num_queries": len(records), "num_corpus_documents": corpus_count,
        "num_positives": sum(len(r["positives"]) for r in records),
        "num_hard_negatives": sum(len(r["hard_negatives"]) for r in records),
        "num_ordinary_negatives": sum(len(r["ordinary_negatives"]) for r in records),
        "queries_with_refill": sum(any(q["initial_shortages_by_bucket"].values()) for q in query_audits),
        "queries_with_remaining_shortage": sum(q["remaining_shortage"] > 0 for q in query_audits),
        "queries_with_ordinary_shortage": sum(q["ordinary_shortage"] > 0 for q in query_audits),
        "positive_label_differences": differences,
        "input_files": input_files, "queries": query_audits,
    }
    return records, audit


def stage_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=path.name + ".", suffix=".partial", delete=False,
        ) as stream:
            temp_name = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        return temp_name
    except BaseException:
        if temp_name is not None:
            temp_name.unlink(missing_ok=True)
        raise


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=repo.parent / "reteco_data")
    parser.add_argument("--run-dir", type=Path, required=True, help="explicit source run directory")
    parser.add_argument("--run-prefix", default="qwen3_embedding_8b", help="prefix before _1a_<domain>_train.txt")
    parser.add_argument("--negative-source", default="qwen3_embedding_8b_bf16_top500", help="source label stored in each query record")
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument("--bucket-counts", nargs=3, type=int, default=[2, 4, 2], metavar=("HIGH", "MIDDLE", "LOW"))
    parser.add_argument(
        "--ordinary-multiplier",
        type=int,
        default=5,
        help="number of ordinary corpus negatives per known positive (default: 5)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True, help="one JSON-array training file")
    parser.add_argument("--audit", type=Path, help="default: <out stem>.audit.json")
    parser.add_argument("--overwrite", action="store_true", help="replace existing output reports")
    args = parser.parse_args()
    if min(args.bucket_counts) < 0 or sum(args.bucket_counts) < 1:
        parser.error("bucket counts must be nonnegative, with a positive total")
    if args.ordinary_multiplier < 0:
        parser.error("--ordinary-multiplier must be nonnegative")
    if len(args.domains) != len(set(args.domains)):
        parser.error("--domains must not contain duplicates")
    if not args.negative_source.strip() or not args.run_prefix.strip():
        parser.error("source label and run prefix must be nonempty")
    if any(c in args.run_prefix for c in ("/", "\\")):
        parser.error("--run-prefix must be a filename prefix, not a path")
    args.data, args.run_dir = args.data.resolve(), args.run_dir.resolve()
    args.out = args.out.resolve()
    args.audit = (args.audit or args.out.with_suffix(".audit.json")).resolve()
    if args.out == args.audit:
        parser.error("training output and audit output must be different files")
    # Validate ALL requested domains first. Never silently build a partial
    # all-domain dataset because a run or a corpus file has not arrived yet.
    all_paths = {d: domain_paths(args, d) for d in args.domains}
    inputs = {path.resolve() for paths in all_paths.values() for path in paths.values()}
    if args.out in inputs or args.audit in inputs:
        parser.error("outputs must not overwrite input files")
    for output in (args.out, args.audit):
        if output.exists() and not args.overwrite:
            parser.error(f"Output exists: {output}; choose another name or use --overwrite")
    missing = sorted(str(p) for p in inputs if not p.is_file())
    if missing:
        parser.error("Missing required TRAIN files:\n" + "\n".join(missing))

    records, audits = [], []
    for domain in sorted(args.domains):
        print(f"[{domain}] validating train run and reading corpus ...", flush=True)
        domain_records, audit = build_domain(args, domain, all_paths[domain])
        records.extend(domain_records)
        audits.append(audit)
        print(
            f"[{domain}] queries={audit['num_queries']} positives={audit['num_positives']} "
            f"hard_negatives={audit['num_hard_negatives']} "
            f"ordinary_negatives={audit['num_ordinary_negatives']} "
            f"refill_queries={audit['queries_with_refill']} "
            f"shortage_queries={audit['queries_with_remaining_shortage']} "
            f"ordinary_shortage_queries={audit['queries_with_ordinary_shortage']}", flush=True,
        )
    statuses = Counter(n["label_status"] for r in records for n in r["hard_negatives"])
    ordinary_statuses = Counter(
        n["label_status"] for r in records for n in r["ordinary_negatives"]
    )
    report = {
        "schema": "reteco_track1a_query_group_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "train", "data_root": str(args.data), "source_run_dir": str(args.run_dir),
        "negative_source": args.negative_source, "run_prefix": args.run_prefix,
        "seed": args.seed,
        "sampling": {
            "original_rank_counts": dict(zip(BUCKET_NAMES, args.bucket_counts)),
            "ordinary_multiplier": args.ordinary_multiplier,
            "ordinary_sampling": "uniform_random_domain_corpus",
            "source_rank_limit": 500,
            "within_bucket": "uniform_without_replacement",
            "refill_order": ["51-200", "1-50", "201-500"],
            "shortage_policy": "keep_available_without_duplication_and_report",
            "rng": "sha256_of_seed_domain_query_id",
            "duplicate_text_policy": "NFKC_and_whitespace_normalized_SHA256",
        },
        "training_output": str(args.out),
        "num_domains": len(audits), "num_queries": len(records),
        "num_positive_documents": sum(len(r["positives"]) for r in records),
        "num_hard_negative_documents": sum(len(r["hard_negatives"]) for r in records),
        "num_ordinary_negative_documents": sum(len(r["ordinary_negatives"]) for r in records),
        "negative_label_status_counts": dict(statuses),
        "ordinary_negative_label_status_counts": dict(ordinary_statuses),
        "queries_with_remaining_shortage": sum(a["queries_with_remaining_shortage"] for a in audits),
        "warning": "Unjudged run candidates may be false negatives; rank is not a relevance judgment.",
        "domains": audits,
    }
    # No outputs are committed until every requested domain validates.
    staged = []
    try:
        data_temp = stage_json(args.out, records)
        staged.append(data_temp)
        report["training_output_sha256"] = sha256_file(data_temp)
        audit_temp = stage_json(args.audit, report)
        staged.append(audit_temp)
        os.replace(data_temp, args.out)
        os.replace(audit_temp, args.audit)
    finally:
        for path in staged:
            path.unlink(missing_ok=True)
    print(f"Wrote training JSON: {args.out}", flush=True)
    print(f"Wrote audit JSON:    {args.audit}", flush=True)
    print(f"Total: {len(audits)} domains, {len(records)} queries, labels={dict(statuses)}", flush=True)
    if report["queries_with_remaining_shortage"]:
        print("WARNING: some queries have fewer negatives than requested; inspect the audit.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
