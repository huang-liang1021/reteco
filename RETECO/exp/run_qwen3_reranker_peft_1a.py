#!/usr/bin/env python3
"""Run a PEFT LoRA Qwen3 Reranker on existing RETECO Track 1a runs.

The script loads a local base Qwen3-Reranker causal LM, injects a LoRA
adapter with ``PeftModel.from_pretrained``, scores query/document pairs by
log P(yes | yes,no), and writes a reranked TREC run. Only the first
``--rerank-depth`` candidates are rescored; the remaining tail is preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

DOMAINS = (
    "bitcoin", "cardano", "economics", "genealogy", "history", "hsm",
    "iota", "law", "monero", "politics", "quant", "travel", "workplace",
)
DEFAULT_INSTRUCTION = (
    "Given a temporal information retrieval query, retrieve passages that "
    "provide evidence for the topic, entities, events, time periods, "
    "changes, causes, impacts, and requested evidence."
)
SYSTEM_PREFIX = (
    '<|im_start|>system\n'
    'Judge whether the Document meets the requirements based on the Query and '
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
    '<|im_end|>\n<|im_start|>user\n'
)
SYSTEM_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jsonl(path: Path):
    with path.open(encoding="utf-8-sig") as stream:
        for line_no, raw in enumerate(stream, 1):
            if raw.strip():
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_no}: expected a JSON object")
                yield row


def atomic_json(path: Path, obj: object) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def read_run(path: Path, query_ids: dict[str, str], depth: int) -> dict[str, list[str]]:
    runs = {qid: [] for qid in query_ids}
    seen_docs = {qid: set() for qid in query_ids}
    seen_ranks = {qid: set() for qid in query_ids}
    with path.open(encoding="utf-8-sig") as stream:
        for line_no, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            fields = raw.split()
            if len(fields) != 6:
                raise ValueError(f"{path}:{line_no}: expected six TREC columns")
            qid, q0, docid, rank, score, _tag = fields
            try:
                rank_no, score_value = int(rank), float(score)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: invalid rank or score") from exc
            if qid not in runs or q0 != "Q0" or rank_no < 1 or not math.isfinite(score_value):
                raise ValueError(f"{path}:{line_no}: invalid query, Q0, rank or score")
            if docid in seen_docs[qid] or rank_no in seen_ranks[qid]:
                raise ValueError(f"{path}:{line_no}: duplicate document or rank for {qid}")
            seen_docs[qid].add(docid)
            seen_ranks[qid].add(rank_no)
            runs[qid].append((rank_no, docid))
    output = {}
    for qid, rows in runs.items():
        rows.sort()
        if len(rows) < depth or [rank for rank, _ in rows[:depth]] != list(range(1, depth + 1)):
            raise ValueError(f"{path}: {qid} needs complete ranks 1..{depth}")
        output[qid] = [docid for _, docid in rows[:depth]]
    return output


def input_run(args, domain: str, split: str) -> Path:
    suffix = f"_1a_{domain}_{split}.txt"
    if args.run_prefix:
        matches = [args.run_dir / f"{args.run_prefix}{suffix}"]
    else:
        matches = sorted(args.run_dir.glob(f"*{suffix}"))
    if len(matches) != 1 or not matches[0].is_file():
        raise ValueError(
            f"Expected exactly one input for {domain}/{split}, found {matches}; "
            "use --run-prefix"
        )
    return matches[0]


def prepare(args, domain: str, split: str):
    folder = args.data / "track1_tempo" / domain
    queries = {}
    for row in jsonl(folder / f"examples_{split}.jsonl"):
        qid = str(row["id"])
        query = row.get("query")
        if qid in queries or not isinstance(query, str) or not query.strip():
            raise ValueError(f"{domain}/{split}: duplicate query ID or empty query")
        queries[qid] = query
    if not queries:
        raise ValueError(f"{domain}/{split}: empty query set")
    source = input_run(args, domain, split)
    runs = read_run(source, queries, args.output_depth)
    needed = {docid for docs in runs.values() for docid in docs[:args.rerank_depth]}
    contexts = {}
    corpus = folder / "documents.jsonl"
    all_ids = set()
    for row in jsonl(corpus):
        docid = str(row["id"])
        if docid in all_ids:
            raise ValueError(f"{corpus}: duplicate document ID {docid}")
        all_ids.add(docid)
        if docid in needed:
            text = row.get("content")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{corpus}: empty content for {docid}")
            contexts[docid] = text
    missing = needed - set(contexts)
    if missing:
        raise ValueError(f"{source}: missing corpus IDs: {sorted(missing)[:5]}")
    print(
        f"{domain}/{split}: {source.name}, {len(queries)} queries, "
        f"rerank {args.rerank_depth}, output {args.output_depth}",
        flush=True,
    )
    return source, queries, runs, contexts, corpus


def resolve_adapter(path: Path) -> Path:
    path = path.resolve()
    if (path / "adapter_config.json").is_file():
        return path
    if (path / "final" / "adapter_config.json").is_file():
        return path / "final"
    raise FileNotFoundError(
        f"Cannot find adapter_config.json under {path} or {path / 'final'}"
    )


def model_signature(root: Path):
    files = sorted(
        p for p in root.rglob("*")
        if p.is_file() and (p.suffix in (".json", ".safetensors", ".bin", ".model")
                            or p.name == "merges.txt")
    )
    if not files:
        raise ValueError(f"No model files under {root}")
    return [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": digest(path) if path.suffix == ".json" else None,
        }
        for path in files
    ]


def load_model(args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required")
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), local_files_only=True, padding_side="left"
    )
    tokenizer.truncation_side = "right"
    base = AutoModelForCausalLM.from_pretrained(
        str(args.model), local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=False)
    model = model.to("cuda:0").eval()
    model.config.use_cache = False
    for token in ("yes", "no"):
        if tokenizer.encode(token, add_special_tokens=False) != [tokenizer.convert_tokens_to_ids(token)]:
            raise ValueError(f"Expected {token!r} to be a single vocabulary token")
    return model, tokenizer


def score_pairs(model, tokenizer, query: str, documents: list[str], args):
    import torch

    prefix = tokenizer.encode(SYSTEM_PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(SYSTEM_SUFFIX, add_special_tokens=False)
    budget = args.max_length - len(prefix) - len(suffix)
    if budget < 32:
        raise ValueError("--max-length leaves fewer than 32 body tokens")
    no_id = tokenizer.convert_tokens_to_ids("no")
    yes_id = tokenizer.convert_tokens_to_ids("yes")
    scores, truncated = [], 0
    with torch.inference_mode():
        for start in range(0, len(documents), args.batch):
            pairs = [
                f"<Instruct>: {args.instruction}\n<Query>: {query}\n<Document>: {doc}"
                for doc in documents[start:start + args.batch]
            ]
            body = tokenizer(pairs, add_special_tokens=False, padding=False, truncation=False)
            body_ids = body["input_ids"]
            truncated += sum(len(ids) > budget for ids in body_ids)
            ids = [prefix + row[:budget] + suffix for row in body_ids]
            inputs = tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            try:
                logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :]
            except TypeError:
                logits = model(**inputs, use_cache=False).logits[:, -1, :]
            binary = logits.float()[:, [no_id, yes_id]]
            log_yes = torch.log_softmax(binary, dim=1)[:, 1]
            if not torch.isfinite(log_yes).all():
                raise FloatingPointError("Non-finite reranker score")
            scores.extend(log_yes.cpu().tolist())
    return scores, truncated


def process(args, domain, split, model, tokenizer, signatures):
    source, queries, runs, contexts, corpus = prepare(args, domain, split)
    stem = f"{args.method}_1a_{domain}_{split}"
    output = args.out_dir / f"{stem}.txt"
    progress = args.log_dir / f"{stem}.scores.jsonl"
    meta_path = args.log_dir / f"{stem}.meta.json"
    identity = {
        "source": str(source), "source_sha256": digest(source),
        "corpus_sha256": digest(corpus), "queries": queries,
        "model": str(args.model), "adapter": str(args.adapter),
        "model_signature": signatures, "instruction": args.instruction,
        "max_length": args.max_length, "batch": args.batch,
        "rerank_depth": args.rerank_depth, "output_depth": args.output_depth,
        "method": args.method,
        "script_sha256": digest(Path(__file__).resolve()),
        "score_definition": "log_P_yes_conditional_on_yes_no; output_score=-rank",
    }
    existing = any(path.exists() for path in (output, progress, meta_path))
    if existing:
        if not args.resume:
            raise ValueError(f"{stem}: existing output/progress; use --resume")
        if not meta_path.is_file() or json.loads(meta_path.read_text(encoding="utf-8")) != identity:
            raise ValueError(f"{stem}: resume metadata mismatch; use new output/log directories")
    else:
        atomic_json(meta_path, identity)

    completed = {}
    if progress.is_file():
        with progress.open("r+b") as stream:
            while True:
                start = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    stream.truncate(start)
                    break
                row = json.loads(line)
                qid = row["query_id"]
                if qid in completed or qid not in runs or row["doc_ids"] != runs[qid][:args.rerank_depth]:
                    raise ValueError(f"{progress}: invalid or duplicate query checkpoint")
                values = row["log_p_yes"]
                if len(values) != args.rerank_depth or not all(math.isfinite(v) for v in values):
                    raise ValueError(f"{progress}: invalid saved scores")
                completed[qid] = row

    started = time.monotonic()
    for qid, query in queries.items():
        if qid in completed:
            continue
        docs = runs[qid][:args.rerank_depth]
        values, truncated = score_pairs(model, tokenizer, query, [contexts[d] for d in docs], args)
        row = {"query_id": qid, "doc_ids": docs, "log_p_yes": values, "truncated_pairs": truncated}
        with progress.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
        completed[qid] = row
        print(
            f"{domain}/{split}: {len(completed)}/{len(queries)} queries, "
            f"elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )

    partial = output.with_suffix(".partial.txt")
    with partial.open("w", encoding="utf-8") as stream:
        for qid in queries:
            values = completed[qid]["log_p_yes"]
            head = runs[qid][:args.rerank_depth]
            order = sorted(range(len(head)), key=lambda index: (-values[index], index))
            documents = [head[index] for index in order] + runs[qid][args.rerank_depth:]
            if len(documents) != len(set(documents)) or len(documents) != args.output_depth:
                raise RuntimeError(f"{stem}: invalid output document count for {qid}")
            for rank_no, docid in enumerate(documents, 1):
                stream.write(f"{qid}\tQ0\t{docid}\t{rank_no}\t{-rank_no:.1f}\t{args.method}\n")
    os.replace(partial, output)
    print(f"Wrote {output}; truncated pairs={sum(row['truncated_pairs'] for row in completed.values())}", flush=True)


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True,
                        help="local base Qwen3-Reranker model directory")
    parser.add_argument("--adapter", type=Path, required=True,
                        help="LoRA adapter directory or training output directory")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-prefix", help="input prefix before _1a_<domain>_<split>.txt")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--method", default="qwen3_reranker_lora")
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument("--splits", nargs="+", choices=("train", "dev"), default=["train", "dev"])
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--rerank-depth", type=int, default=100)
    parser.add_argument("--output-depth", type=int, default=500)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--worker-domain", choices=DOMAINS, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not 1 <= args.rerank_depth <= args.output_depth or args.batch < 1 or args.max_length < 1:
        parser.error("require 1 <= rerank-depth <= output-depth and positive batch/max-length")
    if not args.method or any(ch.isspace() or ch in "/\\" for ch in args.method):
        parser.error("--method must be a nonempty filename-safe TREC tag")
    if not args.instruction.strip() or any(g < 0 for g in args.gpus):
        parser.error("instruction must be nonempty; GPU IDs must be nonnegative")
    for values in (args.gpus, args.domains, args.splits):
        if len(values) != len(set(values)):
            parser.error("GPU, domain and split arguments must not contain duplicates")

    for name in ("data", "model", "run_dir", "out_dir", "log_dir"):
        setattr(args, name, getattr(args, name).resolve())
    args.adapter = resolve_adapter(args.adapter)
    if args.out_dir.name != f"runs_{args.method}_top{args.output_depth}":
        parser.error("--out-dir must be named runs_<method>_top<output-depth>")
    if args.data_only:
        for domain in args.domains:
            for split in args.splits:
                prepare(args, domain, split)
        print("DATA CHECK PASSED; no model loaded or output written")
        return
    if not args.model.is_dir():
        parser.error(f"local base model not found: {args.model}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    if args.worker_domain:
        import torch
        model, tokenizer = load_model(args)
        signatures = {"model": model_signature(args.model), "adapter": model_signature(args.adapter)}
        start = time.monotonic()
        for split in args.splits:
            process(args, args.worker_domain, split, model, tokenizer, signatures)
        atomic_json(args.log_dir / f"{args.worker_domain}.runtime.json", {
            "seconds": time.monotonic() - start,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        })
        return

    for domain in args.domains:
        for split in args.splits:
            print(f"Input: {input_run(args, domain, split)}", flush=True)
    jobs = queue.Queue()
    for domain in args.domains:
        jobs.put(domain)
    failures, stop = [], threading.Event()

    def worker(gpu):
        while not stop.is_set():
            try:
                domain = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                log_path = args.log_dir / f"{domain}_gpu{gpu}.log"
                command = [sys.executable, "-u", str(Path(__file__).resolve()),
                           *sys.argv[1:], "--worker-domain", domain]
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                print(f"GPU {gpu}: {domain} -> {log_path}", flush=True)
                with log_path.open("w", encoding="utf-8") as stream:
                    result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode:
                    failures.append((domain, gpu, result.returncode, str(log_path)))
                    stop.set()
            finally:
                jobs.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in args.gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise SystemExit(f"FAILED: {failures}")
    print(f"Completed {len(args.domains)} domains, splits={args.splits}, top-{args.output_depth}", flush=True)


if __name__ == "__main__":
    main()
