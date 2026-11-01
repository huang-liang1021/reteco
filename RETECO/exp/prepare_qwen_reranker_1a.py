#!/usr/bin/env python3
"""Build RETECO Track 1a reranker groups from a Top-500 run.

All known positives are retained. For each query, candidates are taken from
the corresponding Top-500 run after removing positive IDs and duplicate text.
Hard negatives are sampled uniformly without replacement with a fixed seed.
The requested count is max(5 * positive_count, 20).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

DOMAINS = ("bitcoin", "cardano", "economics", "genealogy", "history", "hsm",
           "iota", "law", "monero", "politics", "quant", "travel", "workplace")


def fingerprint(text):
    return hashlib.sha256(" ".join(unicodedata.normalize("NFKC", text).split()).encode()).hexdigest()


def rng_for(seed, domain, qid):
    raw = f"{seed}\0{domain}\0{qid}\0reranker_hard_negatives".encode()
    return random.Random(int.from_bytes(hashlib.sha256(raw).digest(), "big"))


def rows(path):
    with path.open(encoding="utf-8-sig") as stream:
        for line_no, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            yield line_no, row


def require_id(value, where):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"Invalid ID at {where}")
    value = str(value)
    if not value.strip():
        raise ValueError(f"Empty ID at {where}")
    return value


def load_queries(path):
    output, seen = [], set()
    for line_no, row in rows(path):
        where = f"{path}:{line_no}"
        qid = require_id(row.get("id"), where)
        if qid in seen:
            raise ValueError(f"Duplicate query ID {qid} in {path}")
        query = row.get("query")
        gold = row.get("gold_ids")
        if not isinstance(query, str) or not query.strip() or not isinstance(gold, list):
            raise ValueError(f"Invalid query/gold_ids at {where}")
        output.append({"id": qid, "query": query,
                       "gold_ids": list(dict.fromkeys(require_id(x, where) for x in gold))})
        seen.add(qid)
    if not output:
        raise ValueError(f"No queries in {path}")
    return output


def load_qrels(path, query_ids):
    labels = defaultdict(dict)
    with path.open(encoding="utf-8-sig") as stream:
        for line_no, raw in enumerate(stream, 1):
            fields = raw.split()
            if not fields:
                continue
            if len(fields) != 4:
                raise ValueError(f"Expected four qrels fields at {path}:{line_no}")
            qid, _, did, rel_raw = fields
            if qid not in query_ids:
                raise ValueError(f"Unknown query {qid} at {path}:{line_no}")
            rel = float(rel_raw)
            if not math.isfinite(rel):
                raise ValueError(f"Non-finite relevance at {path}:{line_no}")
            if did in labels[qid] and labels[qid][did] != rel:
                raise ValueError(f"Conflicting qrels for {qid}/{did}")
            labels[qid][did] = rel
    return dict(labels)


def load_run(path, query_ids):
    runs, ranks, docs = defaultdict(list), defaultdict(set), defaultdict(set)
    with path.open(encoding="utf-8-sig") as stream:
        for line_no, raw in enumerate(stream, 1):
            fields = raw.split()
            if not fields:
                continue
            if len(fields) != 6:
                raise ValueError(f"Expected six run fields at {path}:{line_no}")
            qid, q0, did, rank_raw, score_raw, _ = fields
            if qid not in query_ids or q0 != "Q0":
                raise ValueError(f"Invalid query/Q0 at {path}:{line_no}")
            rank, score = int(rank_raw), float(score_raw)
            if rank < 1 or not math.isfinite(score) or rank in ranks[qid] or did in docs[qid]:
                raise ValueError(f"Invalid/duplicate rank or document at {path}:{line_no}")
            ranks[qid].add(rank)
            docs[qid].add(did)
            runs[qid].append({"doc_id": did, "rank": rank, "score": score})
    for qid in query_ids:
        ordered = sorted(runs.get(qid, []), key=lambda x: x["rank"])
        if [x["rank"] for x in ordered[:500]] != list(range(1, 501)):
            raise ValueError(f"{path}: {qid} does not contain complete ranks 1..500")
        runs[qid] = ordered[:500]
    return dict(runs)


def load_corpus(path, needed):
    docs, seen = {}, set()
    for line_no, row in rows(path):
        did = require_id(row.get("id"), f"{path}:{line_no}")
        if did in seen:
            raise ValueError(f"Duplicate corpus ID {did} in {path}")
        seen.add(did)
        if did in needed:
            text = row.get("content")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Empty content for {did} in {path}")
            docs[did] = text
    missing = needed - docs.keys()
    if missing:
        raise ValueError(f"Missing corpus documents: {sorted(missing)[:8]}")
    return docs


def build_domain(args, domain):
    folder = args.data / "track1_tempo" / domain
    queries = load_queries(folder / "examples_train.jsonl")
    query_ids = {x["id"] for x in queries}
    qrels = load_qrels(folder / "qrels_train.txt", query_ids)
    run_path = args.run_dir / f"{args.run_prefix}_1a_{domain}_train.txt"
    run = load_run(run_path, query_ids)
    positives, needed = {}, set()
    for query in queries:
        qid = query["id"]
        gold = set(query["gold_ids"])
        judged = {d for d, rel in qrels.get(qid, {}).items() if rel > 0}
        conflict = {d for d in gold if qrels.get(qid, {}).get(d, 1) <= 0}
        if conflict:
            raise ValueError(f"{domain}/{qid}: gold/qrels conflict {sorted(conflict)}")
        positives[qid] = sorted(gold | judged)
        if not positives[qid]:
            raise ValueError(f"{domain}/{qid}: no known positives")
        needed.update(positives[qid])
        needed.update(x["doc_id"] for x in run[qid])
    context = load_corpus(folder / "documents.jsonl", needed)
    records, audits = [], []
    for query in queries:
        qid = query["id"]
        pos = positives[qid]
        pos_set = set(pos)
        pos_fp = {fingerprint(context[d]) for d in pos}
        candidates, seen_fp = [], set()
        for item in run[qid]:
            did = item["doc_id"]
            fp = fingerprint(context[did])
            if did in pos_set or fp in pos_fp or fp in seen_fp:
                continue
            candidates.append(item)
            seen_fp.add(fp)
        requested = max(args.hard_multiplier * len(pos), args.min_hard_negatives)
        selected = rng_for(args.seed, domain, qid).sample(candidates, min(requested, len(candidates)))
        hard = [{
            "doc_id": x["doc_id"], "context": context[x["doc_id"]],
            "source_rank": x["rank"], "source_score": x["score"],
            "label_status": "judged_nonrelevant" if qrels.get(qid, {}).get(x["doc_id"]) == 0 else "unjudged",
        } for x in selected]
        records.append({"domain": domain, "query_id": qid, "query": query["query"],
                        "positives": [{"doc_id": d, "context": context[d]} for d in pos],
                        "hard_negatives": hard})
        audits.append({"query_id": qid, "num_positives": len(pos),
                       "requested_hard_negatives": requested,
                       "candidate_count_after_exclusion": len(candidates),
                       "selected_hard_negatives": len(selected),
                       "shortage": requested - len(selected),
                       "selected_source_ranks": [x["rank"] for x in selected]})
    return records, {"domain": domain, "num_queries": len(records),
                     "num_positives": sum(len(x["positives"]) for x in records),
                     "num_hard_negatives": sum(len(x["hard_negatives"]) for x in records),
                     "queries_with_shortage": sum(x["shortage"] > 0 for x in audits),
                     "queries": audits}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".partial", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=repo.parent / "reteco_data")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-prefix", default="qwen3_st_bf16_lora_v4")
    parser.add_argument("--negative-source", default="qwen3_st_bf16_lora_v4_top500")
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument("--hard-multiplier", type=int, default=5)
    parser.add_argument("--min-hard-negatives", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if min(args.hard_multiplier, args.min_hard_negatives) < 0:
        parser.error("negative counts must be nonnegative")
    args.data, args.run_dir, args.out = args.data.resolve(), args.run_dir.resolve(), args.out.resolve()
    args.audit = (args.audit or args.out.with_suffix(".audit.json")).resolve()
    for output in (args.out, args.audit):
        if output.exists() and not args.overwrite:
            parser.error(f"Output exists: {output}; use --overwrite")
    all_records, audits = [], []
    for domain in sorted(set(args.domains)):
        print(f"[{domain}] reading train data and Top-500 run ...", flush=True)
        records, audit = build_domain(args, domain)
        all_records.extend(records)
        audits.append(audit)
        print(f"[{domain}] queries={audit['num_queries']} positives={audit['num_positives']} "
              f"hard_negatives={audit['num_hard_negatives']} shortage_queries={audit['queries_with_shortage']}", flush=True)
    all_records.sort(key=lambda x: (x["domain"], x["query_id"]))
    report = {"schema": "reteco_track1a_reranker_groups_v1",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "split": "train", "data_root": str(args.data), "source_run_dir": str(args.run_dir),
              "run_prefix": args.run_prefix, "negative_source": args.negative_source, "seed": args.seed,
              "sampling": {"method": "uniform_without_replacement_after_positive_exclusion",
                            "requested_count": "max(5 * num_positives, 20)",
                            "hard_multiplier": args.hard_multiplier, "min_hard_negatives": args.min_hard_negatives,
                            "source_depth": 500, "warning": "unjudged candidates are not guaranteed true negatives"},
              "num_domains": len(audits), "num_queries": len(all_records),
              "num_positives": sum(len(x["positives"]) for x in all_records),
              "num_hard_negatives": sum(len(x["hard_negatives"]) for x in all_records), "domains": audits}
    write_json(args.out, all_records)
    write_json(args.audit, report)
    print(f"Wrote training JSON: {args.out}")
    print(f"Wrote audit JSON: {args.audit}")


if __name__ == "__main__":
    main()
