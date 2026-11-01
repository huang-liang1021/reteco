#!/usr/bin/env python3
"""Qwen3-Embedding-8B retrieval for RETECO Track 1a.

The script encodes documents once per domain, caches normalized float16
embeddings with metadata, then retrieves train and dev queries against the
same cache. It uses the Qwen query instruction only for queries and leaves
documents unprompted, matching the model-card retrieval recipe.

Run from the RETECO repository root after copying this file to exp/:
  python -u exp/run_qwen3_embedding_1a.py --data ../reteco_data --gpus 0 1 \
      --top-k 500 --batch 1 --query-batch 1 --max-length 2048
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import torch


MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
DEFAULT_INSTRUCTION = (
    "Given a temporal information retrieval query, retrieve passages that "
    "provide evidence for the topic, entities, events, time periods, "
    "changes, causes, impacts, and requested evidence."
)
DOMAINS = (
    "bitcoin", "cardano", "economics", "genealogy", "history", "hsm",
    "iota", "law", "monero", "politics", "quant", "travel", "workplace",
)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def dtype_from_name(name: str) -> torch.dtype:
    values = {"float16": torch.float16, "fp16": torch.float16,
              "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}
    try:
        return values[name.lower()]
    except KeyError as exc:
        raise ValueError("--dtype must be float16 or bfloat16") from exc


def last_token_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pool the last non-padding token, independent of padding side."""
    reversed_mask = mask.flip(dims=(1,))
    distance = reversed_mask.long().argmax(dim=1)
    indices = mask.shape[1] - 1 - distance
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), indices]


def encode_texts(model, tokenizer, texts, device, batch_size, max_length, dtype):
    vectors = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            tokens = tokenizer(
                batch, padding=True, truncation=True, max_length=max_length,
                return_tensors="pt", add_special_tokens=True,
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            outputs = model(**tokens, return_dict=True)
            hidden = outputs.last_hidden_state
            pooled = last_token_pool(hidden, tokens["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            vectors.append(pooled.cpu().numpy().astype(np.float16))
    if not vectors:
        raise ValueError("Cannot encode an empty text list")
    return np.concatenate(vectors, axis=0)


def rank(doc_vectors, query_vectors, doc_ids, top_k, device, doc_chunk):
    """Exact cosine ranking with bounded GPU memory."""
    count = len(doc_vectors)
    k = min(top_k, count)
    for query_start in range(len(query_vectors)):
        query = torch.from_numpy(
            np.asarray(query_vectors[query_start], dtype=np.float32).copy()
        ).to(device).unsqueeze(0)
        best_scores = torch.full((k,), -float("inf"), device=device)
        best_indices = torch.full((k,), -1, dtype=torch.long, device=device)
        for offset in range(0, count, doc_chunk):
            block = torch.from_numpy(
                np.asarray(doc_vectors[offset:offset + doc_chunk], dtype=np.float32).copy()
            ).to(device)
            values, indices = (query @ block.T).squeeze(0).topk(
                min(k, len(block)), largest=True, sorted=True
            )
            candidates = torch.cat((best_scores, values))
            candidate_indices = torch.cat((best_indices, indices + offset))
            best_scores, selected = candidates.topk(k, largest=True, sorted=True)
            best_indices = candidate_indices[selected]
        yield best_scores.cpu().numpy(), best_indices.cpu().numpy()


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


def cache_paths(cache_dir: Path, domain: str, model_name: str, max_length: int, dtype_name: str):
    stem = f"{domain}__{model_slug(model_name)}__len{max_length}__{dtype_name}"
    return cache_dir / f"{stem}.npy", cache_dir / f"{stem}.json"


def load_model(args, device):
    from transformers import AutoModel, AutoTokenizer

    dtype = dtype_from_name(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir=args.model_cache, padding_side="left",
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        args.model, cache_dir=args.model_cache, dtype=dtype,
        low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(device)
    model.eval()
    return model, tokenizer, dtype


def load_or_build_cache(args, domain_dir, model, tokenizer, dtype, device):
    docs = list(read_jsonl(domain_dir / "documents.jsonl"))
    doc_ids = [str(row["id"]) for row in docs]
    if len(doc_ids) != len(set(doc_ids)):
        raise RuntimeError(f"Duplicate document IDs in {domain_dir / 'documents.jsonl'}")
    cache_path, meta_path = cache_paths(
        args.cache_dir, domain_dir.name, args.model, args.max_length, args.dtype
    )
    expected = {
        "model": args.model, "max_length": args.max_length,
        "num_documents": len(doc_ids), "pooling": "last_non_padding_token",
        "normalized": True, "dtype": args.dtype,
    }
    if cache_path.is_file() and meta_path.is_file():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"Cache metadata mismatch; remove or rename {cache_path}")
        vectors = np.load(cache_path, mmap_mode="r")
        if vectors.shape[0] != len(doc_ids) or vectors.ndim != 2:
            raise RuntimeError(f"Cache shape mismatch: {cache_path} {vectors.shape}")
        if metadata.get("dimension") != vectors.shape[1]:
            raise RuntimeError(f"Cache dimension metadata mismatch: {cache_path}")
        print(f"[{domain_dir.name}] Using verified cache {cache_path}", flush=True)
        return docs, doc_ids, vectors

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    partial = cache_path.with_suffix(".partial.npy")
    if partial.exists():
        partial.unlink()
    vectors_mem = None
    for start in range(0, len(docs), args.batch):
        encoded = encode_texts(
            model, tokenizer, [row["content"] for row in docs[start:start + args.batch]],
            device, args.batch, args.max_length, dtype,
        )
        if vectors_mem is None:
            vectors_mem = np.lib.format.open_memmap(
                partial, mode="w+", dtype=np.float16,
                shape=(len(docs), encoded.shape[1]),
            )
            expected["dimension"] = int(encoded.shape[1])
        vectors_mem[start:start + len(encoded)] = encoded
        if (start // args.batch) % 25 == 0:
            print(f"[{domain_dir.name}] encoded documents {min(start + len(encoded), len(docs))}/{len(docs)}", flush=True)
    if vectors_mem is None:
        raise RuntimeError(f"Empty corpus: {domain_dir}")
    vectors_mem.flush()
    del vectors_mem
    os.replace(partial, cache_path)
    meta_path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    return docs, doc_ids, np.load(cache_path, mmap_mode="r")


def run_domain(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        raise RuntimeError("Qwen3-Embedding-8B requires CUDA for this experiment")
    domain_dir = args.data / "track1_tempo" / args.domain
    model, tokenizer, dtype = load_model(args, device)
    _, doc_ids, doc_vectors = load_or_build_cache(
        args, domain_dir, model, tokenizer, dtype, device
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    instruction = args.query_instruction.strip()
    for split in args.splits:
        examples = list(read_jsonl(domain_dir / f"examples_{split}.jsonl"))
        queries = [f"Instruct: {instruction}\nQuery: {row['query']}" for row in examples]
        query_vectors = encode_texts(
            model, tokenizer, queries, device, args.query_batch, args.max_length, dtype
        )
        output = args.out_dir / f"qwen3_embedding_8b_1a_{args.domain}_{split}.txt"
        partial = output.with_suffix(".partial.txt")
        with partial.open("w", encoding="utf-8") as stream:
            for row, (scores, indices) in zip(examples, rank(
                doc_vectors, query_vectors, doc_ids, args.top_k, device, args.doc_chunk
            )):
                qid = str(row["id"])
                for rank_no, (score, index) in enumerate(zip(scores, indices), 1):
                    stream.write(
                        f"{qid}\tQ0\t{doc_ids[int(index)]}\t{rank_no}\t"
                        f"{float(score):.8f}\tqwen3_embedding_8b\n"
                    )
        os.replace(partial, output)
        print(f"[{args.domain}] wrote {output} ({len(examples)} queries, top-{args.top_k})", flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--model-cache", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("exp/cache_qwen3_embedding_8b"))
    parser.add_argument("--out-dir", type=Path, default=Path("exp/runs_qwen3_embedding_8b_top500"))
    parser.add_argument("--log-dir", type=Path, default=Path("exp/logs_qwen3_embedding_8b"))
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--splits", nargs="+", choices=("train", "dev"), default=["train", "dev"])
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--batch", type=int, default=1, help="document encoding batch per GPU")
    parser.add_argument("--query-batch", type=int, default=1)
    parser.add_argument("--doc-chunk", type=int, default=2048)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--dtype", default="float16", choices=("float16", "bfloat16"))
    parser.add_argument("--query-instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--domain", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.top_k < 1 or args.batch < 1 or args.query_batch < 1 or args.doc_chunk < 1:
        parser.error("top-k, batch, query-batch and doc-chunk must be positive")
    args.data = args.data.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    args.log_dir = args.log_dir.resolve()
    if args.worker:
        run_domain(args)
        return
    domains = args.domains or DOMAINS
    unknown = sorted(set(domains) - set(DOMAINS))
    if unknown:
        parser.error(f"Unknown domains: {unknown}")
    if len(args.gpus) < 1:
        parser.error("provide at least one GPU")
    jobs = queue.Queue()
    for domain in domains:
        folder = args.data / "track1_tempo" / domain
        for name in ("documents.jsonl", "examples_train.jsonl", "examples_dev.jsonl"):
            if not (folder / name).is_file():
                parser.error(f"Missing {folder / name}")
        jobs.put(domain)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    stop = threading.Event()

    def worker(gpu):
        while not stop.is_set():
            try:
                domain = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                log_path = args.log_dir / f"{domain}_gpu{gpu}.log"
                command = [sys.executable, "-u", str(Path(__file__).resolve()),
                           "--worker", "--domain", domain, "--data", str(args.data),
                           "--model", args.model, "--top-k", str(args.top_k),
                           "--batch", str(args.batch), "--query-batch", str(args.query_batch),
                           "--doc-chunk", str(args.doc_chunk), "--max-length", str(args.max_length),
                           "--dtype", args.dtype, "--cache-dir", str(args.cache_dir),
                           "--out-dir", str(args.out_dir), "--splits", *args.splits,
                           "--query-instruction", args.query_instruction,
                           "--model-cache", str(args.model_cache or "")]
                if args.model_cache is None:
                    command = [x for x in command if x != "--model-cache" and x != ""]
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                print(f"GPU {gpu}: {domain} -> {log_path}", flush=True)
                with log_path.open("w", encoding="utf-8") as stream:
                    result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode:
                    failures.append((domain, gpu, result.returncode, str(log_path)))
                    stop.set()
                else:
                    print(f"GPU {gpu}: {domain} complete", flush=True)
            finally:
                jobs.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in args.gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise SystemExit(f"Failed domains: {failures}")
    print(f"Completed {len(domains)} domains, splits={args.splits}, top-k={args.top_k}", flush=True)


if __name__ == "__main__":
    main()
