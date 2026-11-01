#!/usr/bin/env python3
"""Retrieve RETECO Track 1a with a Sentence Transformers Qwen3 LoRA adapter.

This is separate from run_qwen3_embedding_1a.py because that script loads an
AutoModel base checkpoint and cannot apply a Sentence Transformers/PEFT adapter.
The encoding path matches train_qwen3_embedding.py: raw documents, an instruction
plus ``Query:`` prefix for queries, left padding, last-token pooling and FP32
L2 normalization. Each worker handles domains assigned to one visible GPU.
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


DOMAINS = (
    "bitcoin", "cardano", "economics", "genealogy", "history", "hsm",
    "iota", "law", "monero", "politics", "quant", "travel", "workplace",
)
DEFAULT_INSTRUCTION = (
    "Given a temporal information retrieval query, retrieve passages that "
    "provide evidence for the topic, entities, events, time periods, "
    "changes, causes, impacts, and requested evidence."
)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8-sig") as stream:
        for line_no, raw in enumerate(stream, 1):
            if raw.strip():
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected JSON object at {path}:{line_no}")
                yield row


def resolve_model_path(path: Path) -> Path:
    """Accept either the training output directory or its final/ directory."""
    path = path.resolve()
    if (path / "modules.json").is_file():
        return path
    if (path / "final" / "modules.json").is_file():
        return path / "final"
    raise FileNotFoundError(
        f"Cannot find SentenceTransformer modules.json under {path} or {path / 'final'}"
    )


def load_encoder(adapter: Path, device: torch.device, max_length: int):
    from sentence_transformers import SentenceTransformer

    adapter = resolve_model_path(adapter)
    model = SentenceTransformer(
        str(adapter),
        device=str(device),
        local_files_only=True,
        model_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa"},
    )
    modules = list(model.children())
    names = [type(module).__name__ for module in modules]
    if names != ["Transformer", "Pooling", "Normalize"]:
        raise ValueError(f"Unexpected SentenceTransformer modules: {names}")
    if getattr(modules[1], "pooling_mode", None) != "lasttoken":
        raise ValueError("The adapter must use last-token pooling")
    model.max_seq_length = max_length
    model.default_prompt_name = None
    model.prompts = {"query": "", "document": ""}
    model[0].tokenizer.padding_side = "left"
    model[0].tokenizer.truncation_side = "right"
    model.eval()
    return model


def encode_texts(model, texts, device, batch_size, max_length):
    if not texts:
        raise ValueError("Cannot encode an empty text list")
    vectors = []
    tokenizer = model[0].tokenizer
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            tokens = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                add_special_tokens=True,
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            features = model[0](tokens)
            pooled = model[1](features)["sentence_embedding"].float()
            encoded = torch.nn.functional.normalize(pooled, p=2, dim=1)
            if not torch.isfinite(encoded).all():
                raise FloatingPointError("Non-finite embedding produced")
            vectors.append(encoded.cpu().numpy().astype(np.float16))
    return np.concatenate(vectors, axis=0)


def rank(doc_vectors, query_vectors, doc_ids, top_k, device, doc_chunk):
    count = len(doc_vectors)
    k = min(top_k, count)
    for query_index in range(len(query_vectors)):
        query = torch.from_numpy(
            np.asarray(query_vectors[query_index], dtype=np.float32).copy()
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


def cache_paths(cache_dir, domain, adapter_name, max_length):
    safe = adapter_name.replace("/", "_").replace("\\", "_")
    stem = f"{domain}__{safe}__len{max_length}__bf16"
    return cache_dir / f"{stem}.npy", cache_dir / f"{stem}.json"


def load_or_build_cache(args, domain_dir, model, device):
    docs = list(read_jsonl(domain_dir / "documents.jsonl"))
    doc_ids = [str(row["id"]) for row in docs]
    if len(doc_ids) != len(set(doc_ids)):
        raise ValueError(f"Duplicate document IDs in {domain_dir / 'documents.jsonl'}")
    cache_path, meta_path = cache_paths(
        args.cache_dir, domain_dir.name, args.method, args.max_length
    )
    expected = {
        "adapter": str(resolve_model_path(args.adapter)),
        "method": args.method,
        "max_length": args.max_length,
        "num_documents": len(doc_ids),
        "pooling": "last_non_padding_token",
        "normalized": True,
        "dtype": "bfloat16",
    }
    if cache_path.is_file() and meta_path.is_file():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"Cache metadata mismatch; remove or rename {cache_path}")
        vectors = np.load(cache_path, mmap_mode="r")
        if vectors.shape[0] != len(doc_ids) or vectors.ndim != 2:
            raise RuntimeError(f"Cache shape mismatch: {cache_path} {vectors.shape}")
        if metadata.get("dimension") != vectors.shape[1]:
            raise RuntimeError(f"Cache dimension mismatch: {cache_path}")
        print(f"[{domain_dir.name}] using verified cache {cache_path}", flush=True)
        return doc_ids, vectors

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    partial = cache_path.with_suffix(".partial.npy")
    partial.unlink(missing_ok=True)
    vectors_mem = None
    contents = [row.get("content") for row in docs]
    if any(not isinstance(text, str) or not text.strip() for text in contents):
        raise ValueError(f"Empty document content in {domain_dir / 'documents.jsonl'}")
    for start in range(0, len(contents), args.batch):
        encoded = encode_texts(model, contents[start:start + args.batch], device,
                               args.batch, args.max_length)
        if vectors_mem is None:
            vectors_mem = np.lib.format.open_memmap(
                partial, mode="w+", dtype=np.float16,
                shape=(len(contents), encoded.shape[1]),
            )
            expected["dimension"] = int(encoded.shape[1])
        vectors_mem[start:start + len(encoded)] = encoded
        done = min(start + len(encoded), len(contents))
        if start == 0 or done == len(contents) or (start // args.batch) % 25 == 0:
            print(f"[{domain_dir.name}] encoded documents {done}/{len(contents)}", flush=True)
    if vectors_mem is None:
        raise ValueError(f"Empty corpus: {domain_dir}")
    vectors_mem.flush()
    del vectors_mem
    os.replace(partial, cache_path)
    meta_path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    return doc_ids, np.load(cache_path, mmap_mode="r")


def run_domain(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This Qwen3 adapter retrieval requires CUDA")
    adapter = resolve_model_path(args.adapter)
    model = load_encoder(adapter, device, args.max_length)
    domain_dir = args.data / "track1_tempo" / args.domain
    doc_ids, doc_vectors = load_or_build_cache(args, domain_dir, model, device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    instruction = args.query_instruction.strip()
    for split in args.splits:
        examples = list(read_jsonl(domain_dir / f"examples_{split}.jsonl"))
        queries = [f"Instruct: {instruction}\nQuery: {row['query']}" for row in examples]
        query_vectors = encode_texts(model, queries, device, args.query_batch, args.max_length)
        output = args.out_dir / f"{args.method}_1a_{args.domain}_{split}.txt"
        partial = output.with_suffix(".partial.txt")
        with partial.open("w", encoding="utf-8") as stream:
            for row, (scores, indices) in zip(examples, rank(
                doc_vectors, query_vectors, doc_ids, args.top_k, device, args.doc_chunk
            )):
                qid = str(row["id"])
                for rank_no, (score, index) in enumerate(zip(scores, indices), 1):
                    stream.write(
                        f"{qid}\tQ0\t{doc_ids[int(index)]}\t{rank_no}\t"
                        f"{float(score):.8f}\t{args.method}\n"
                    )
        os.replace(partial, output)
        print(f"[{args.domain}] wrote {output} ({len(examples)} queries, top-{args.top_k})", flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True,
                        help="checkpoint directory or its final/ directory")
    parser.add_argument("--method", default="qwen3_st_bf16_lora_v2",
                        help="run prefix; must match runs_<method>_top<depth>")
    parser.add_argument("--cache-dir", type=Path, default=repo / "cache" / "cache_qwen3_st_bf16_lora_v2")
    parser.add_argument("--out-dir", type=Path, default=repo / "runs" / "runs_qwen3_st_bf16_lora_v2_top500")
    parser.add_argument("--log-dir", type=Path, default=repo / "logs" / "qwen3_st_bf16_lora_v2")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--splits", nargs="+", choices=("train", "dev"), default=["train", "dev"])
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--query-batch", type=int, default=1)
    parser.add_argument("--doc-chunk", type=int, default=2048)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--query-instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--domain", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.top_k, args.batch, args.query_batch, args.doc_chunk, args.max_length) < 1:
        parser.error("top-k, batch, query-batch, doc-chunk and max-length must be positive")
    if not args.method or any(c in args.method for c in "/\\"):
        parser.error("--method must be a nonempty filename-safe prefix")
    args.data = args.data.resolve()
    args.adapter = args.adapter.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    args.log_dir = args.log_dir.resolve()
    if args.worker:
        run_domain(args)
        return
    domains = args.domains or list(DOMAINS)
    unknown = sorted(set(domains) - set(DOMAINS))
    if unknown:
        parser.error(f"Unknown domains: {unknown}")
    resolve_model_path(args.adapter)
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
                           "--adapter", str(args.adapter), "--method", args.method,
                           "--top-k", str(args.top_k), "--batch", str(args.batch),
                           "--query-batch", str(args.query_batch), "--doc-chunk", str(args.doc_chunk),
                           "--max-length", str(args.max_length), "--cache-dir", str(args.cache_dir),
                           "--out-dir", str(args.out_dir), "--splits", *args.splits,
                           "--query-instruction", args.query_instruction]
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
