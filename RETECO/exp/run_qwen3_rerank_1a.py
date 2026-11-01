#!/usr/bin/env python3
"""Qwen3-Reranker-8B reranking of existing RETECO 1a TREC runs.

No qrels or steps are read. Rescore only the first --rerank-depth candidates,
then append the untouched tail. Output score=-rank is a monotone surrogate;
actual yes/no log-probabilities are saved in the resumable progress JSONL.
Requires the inference environment's torch, transformers and model files.
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

DOMAINS = ('bitcoin', 'cardano', 'economics', 'genealogy', 'history', 'hsm',
           'iota', 'law', 'monero', 'politics', 'quant', 'travel', 'workplace')
INSTRUCTION = 'Given a web search query, retrieve relevant passages that answer the query'
PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements '
          'based on the Query and the Instruct provided. Note that the answer '
          'can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def jsonl(path):
    with path.open(encoding='utf-8-sig') as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f'{path}:{n}: expected an object')
                yield row


def atomic_json(path, obj):
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def read_run(path, query_ids, depth):
    runs = {qid: [] for qid in query_ids}
    ids = {qid: set() for qid in query_ids}
    ranks = {qid: set() for qid in query_ids}
    with path.open(encoding='utf-8-sig') as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            cols = line.split()
            if len(cols) != 6:
                raise ValueError(f'{path}:{n}: expected 6 TREC columns')
            qid, q0, did, rank, score, _ = cols
            rank, score = int(rank), float(score)
            if qid not in runs or q0 != 'Q0' or rank < 1 or not math.isfinite(score):
                raise ValueError(f'{path}:{n}: unknown query or invalid rank/score/Q0')
            if did in ids[qid] or rank in ranks[qid]:
                raise ValueError(f'{path}:{n}: duplicate document or rank')
            ids[qid].add(did)
            ranks[qid].add(rank)
            runs[qid].append((rank, did))
    for qid, rows in runs.items():
        rows.sort()
        if len(rows) < depth or [r for r, _ in rows[:depth]] != list(range(1, depth + 1)):
            raise ValueError(f'{path}: {qid} needs complete ranks 1..{depth}')
        runs[qid] = [did for _, did in rows[:depth]]
    return runs


def input_run(args, domain, split):
    suffix = f'_1a_{domain}_{split}.txt'
    matches = ([args.run_dir / f'{args.run_prefix}{suffix}'] if args.run_prefix else
               sorted(args.run_dir.glob(f'*{suffix}')))
    if len(matches) != 1 or not matches[0].is_file():
        raise ValueError(f'Expected exactly one input for {domain}/{split}, found {matches}; use --run-prefix')
    return matches[0]


def prepare(args, domain, split):
    folder = args.data / 'track1_tempo' / domain
    queries = {}
    for row in jsonl(folder / f'examples_{split}.jsonl'):
        qid = str(row['id'])
        if qid in queries or not isinstance(row.get('query'), str) or not row['query'].strip():
            raise ValueError(f'{domain}/{split}: duplicate query ID or empty query')
        queries[qid] = row['query']
    if not queries:
        raise ValueError(f'{domain}/{split}: empty queries')
    source = input_run(args, domain, split)
    runs = read_run(source, queries, args.output_depth)
    needed = {d for docs in runs.values() for d in docs[:args.rerank_depth]}
    all_ids, contexts = set(), {}
    corpus = folder / 'documents.jsonl'
    for row in jsonl(corpus):
        did = str(row['id'])
        if did in all_ids:
            raise ValueError(f'{corpus}: duplicate ID {did}')
        all_ids.add(did)
        if did in needed:
            text = row.get('content')
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f'{corpus}: empty content for {did}')
            contexts[did] = text
    missing = {d for docs in runs.values() for d in docs} - all_ids
    if missing:
        raise ValueError(f'{source}: unknown corpus IDs: {sorted(missing)[:5]}')
    print(f'{domain}/{split}: {source.name}, {len(queries)} queries, '
          f'rerank {args.rerank_depth}, output {args.output_depth}', flush=True)
    return source, queries, runs, contexts, corpus


def model_signature(root):
    files = sorted(p for p in root.rglob('*') if p.is_file() and
                   (p.suffix in ('.json', '.safetensors', '.bin', '.model') or p.name == 'merges.txt'))
    if not files:
        raise ValueError(f'No model files under {root}')
    # Weight size/mtime avoids hashing 16 GB on each worker; model files must
    # remain immutable during reranking. Small config files are content hashed.
    return [{'path': str(p.relative_to(root)), 'bytes': p.stat().st_size,
             'mtime_ns': p.stat().st_mtime_ns,
             'sha256': digest(p) if p.suffix == '.json' else None} for p in files]


def load_model(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A BF16-capable CUDA GPU is required')
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, padding_side='left')
    tokenizer.truncation_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='sdpa',
    ).to('cuda:0').eval()
    model.config.use_cache = False
    for token in ('yes', 'no'):
        if tokenizer.encode(token, add_special_tokens=False) != [tokenizer.convert_tokens_to_ids(token)]:
            raise ValueError(f'Expected a single vocabulary token for {token!r}')
    return model, tokenizer


def score_pairs(model, tokenizer, query, documents, args):
    import torch
    prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    budget = args.max_length - len(prefix) - len(suffix)
    if budget < 32:
        raise ValueError('--max-length is too short for the reranker template')
    scores, truncated = [], 0
    with torch.inference_mode():
        for start in range(0, len(documents), args.batch):
            pairs = [f'<Instruct>: {args.instruction}\n<Query>: {query}\n<Document>: {doc}'
                     for doc in documents[start:start + args.batch]]
            # Match the model-card input template and truncation, reserving
            # the assistant suffix even for long documents.
            encoded = tokenizer(pairs, padding=False, truncation=False, return_attention_mask=False)
            body_ids = encoded['input_ids']
            truncated += sum(len(ids) > budget for ids in body_ids)
            inputs = tokenizer.pad(
                {'input_ids': [prefix + ids[:budget] + suffix for ids in body_ids]},
                padding=True, return_tensors='pt',
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            # Qwen3 supports logits_to_keep: avoid allocating full sequence
            # vocabulary logits, which can otherwise exhaust a 3090's VRAM.
            logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
            binary = logits[:, [tokenizer.convert_tokens_to_ids('no'),
                                tokenizer.convert_tokens_to_ids('yes')]]
            log_yes = torch.log_softmax(binary, dim=1)[:, 1]
            if not torch.isfinite(log_yes).all():
                raise FloatingPointError('Non-finite reranker score')
            scores.extend(log_yes.cpu().tolist())
    # Log P(yes | yes or no) has exactly the same mathematical ordering as
    # the official P(yes), with fewer ties caused by floating point saturation.
    return scores, truncated


def process(args, domain, split, model, tokenizer, signature):
    source, queries, runs, contexts, corpus = prepare(args, domain, split)
    stem = f'{args.method}_1a_{domain}_{split}'
    output = args.out_dir / f'{stem}.txt'
    progress = args.log_dir / f'{stem}.scores.jsonl'
    meta_path = args.log_dir / f'{stem}.meta.json'
    identity = dict(source=str(source), source_sha256=digest(source), corpus_sha256=digest(corpus),
                    queries=queries, model=str(args.model), model_signature=signature,
                    instruction=args.instruction, max_length=args.max_length, batch=args.batch,
                    rerank_depth=args.rerank_depth, output_depth=args.output_depth,
                    method=args.method, script_sha256=digest(Path(__file__)),
                    score_definition='log_P_yes_conditional_on_yes_no; output_score=-rank')
    existing = any(p.exists() for p in (output, progress, meta_path))
    if existing:
        if not args.resume:
            raise ValueError(f'{stem}: existing output/progress; use fresh directories or --resume')
        if not meta_path.is_file() or json.loads(meta_path.read_text(encoding='utf-8')) != identity:
            raise ValueError(f'{stem}: resume metadata mismatch; use new output/log directories')
    else:
        atomic_json(meta_path, identity)
    completed = {}
    if progress.is_file():
        # Drop an interrupted final JSONL fragment, retaining all committed queries.
        with progress.open('r+b') as f:
            while True:
                start = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.endswith(b'\n'):
                    f.truncate(start)
                    break
                row = json.loads(line)
                qid = row['query_id']
                if qid in completed or qid not in runs or row['doc_ids'] != runs[qid][:args.rerank_depth]:
                    raise ValueError(f'{progress}: invalid/duplicate query checkpoint')
                if len(row['log_p_yes']) != args.rerank_depth or not all(math.isfinite(v) for v in row['log_p_yes']):
                    raise ValueError(f'{progress}: invalid saved scores')
                completed[qid] = row
    started = time.monotonic()
    for qid, query in queries.items():
        if qid in completed:
            continue
        docs = runs[qid][:args.rerank_depth]
        values, truncated = score_pairs(model, tokenizer, query, [contexts[d] for d in docs], args)
        row = dict(query_id=qid, doc_ids=docs, log_p_yes=values, truncated_pairs=truncated)
        with progress.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
            f.flush()
        completed[qid] = row
        print(f'{domain}/{split}: {len(completed)}/{len(queries)} queries, '
              f'elapsed={time.monotonic()-started:.1f}s', flush=True)
    temp = output.with_suffix('.partial.txt')
    with temp.open('w', encoding='utf-8') as f:
        for qid in queries:
            values = completed[qid]['log_p_yes']
            head = runs[qid][:args.rerank_depth]
            order = sorted(range(len(head)), key=lambda i: (-values[i], i))
            documents = [head[i] for i in order] + runs[qid][args.rerank_depth:]
            assert len(documents) == len(set(documents)) == args.output_depth
            for rank, did in enumerate(documents, 1):
                f.write(f'{qid}\tQ0\t{did}\t{rank}\t{-rank:.1f}\t{args.method}\n')
    os.replace(temp, output)
    print(f'Wrote {output}; truncated pairs={sum(r["truncated_pairs"] for r in completed.values())}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('../reteco_data'))
    p.add_argument('--model', type=Path, default=Path('/home/huangliang/models/Qwen3-Reranker-8B'))
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--run-prefix', help='optional explicit input prefix before _1a_<domain>_<split>.txt')
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--log-dir', type=Path, required=True)
    p.add_argument('--method', default='qwen_reranker8b_h100')
    p.add_argument('--domains', nargs='+', choices=DOMAINS, default=list(DOMAINS))
    p.add_argument('--splits', nargs='+', choices=('train', 'dev'), default=['train', 'dev'])
    p.add_argument('--gpus', nargs='+', type=int, default=[0, 1])
    p.add_argument('--rerank-depth', type=int, default=100)
    p.add_argument('--output-depth', type=int, default=500)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--max-length', type=int, default=2048)
    p.add_argument('--instruction', default=INSTRUCTION)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--data-only', action='store_true', help='validate inputs without torch/model/GPU')
    p.add_argument('--worker-domain', choices=DOMAINS, help=argparse.SUPPRESS)
    args = p.parse_args()
    for name in ('data', 'model', 'run_dir', 'out_dir', 'log_dir'):
        setattr(args, name, getattr(args, name).resolve())
    if not 1 <= args.rerank_depth <= args.output_depth or min(args.batch, args.max_length) < 1:
        p.error('require 1 <= rerank-depth <= output-depth and positive batch/max-length')
    if any(ch.isspace() or ch in '/\\' for ch in args.method) or not args.method:
        p.error('--method must be a nonempty filename-safe TREC tag')
    if args.out_dir == args.run_dir:
        p.error('input and output directories must differ')
    if args.out_dir.name != f'runs_{args.method}_top{args.output_depth}':
        p.error('--out-dir must be named runs_<method>_top<output-depth> to match evaluate_1a_runs.py')
    if not args.instruction.strip() or any(g < 0 for g in args.gpus):
        p.error('instruction must be nonempty; GPU IDs must be nonnegative')
    for values in (args.gpus, args.domains, args.splits):
        if len(values) != len(set(values)):
            p.error('GPU, domain and split arguments must not contain duplicates')
    if args.data_only:
        for domain in args.domains:
            for split in args.splits:
                prepare(args, domain, split)
        print('DATA CHECK PASSED; no model loaded or output runs written')
        return
    if not args.model.is_dir():
        p.error(f'local reranker model not found: {args.model}')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    if args.worker_domain:
        import torch
        model, tokenizer = load_model(args)
        signature = model_signature(args.model)
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        for split in args.splits:
            process(args, args.worker_domain, split, model, tokenizer, signature)
        atomic_json(args.log_dir / f'{args.worker_domain}.runtime.json', {
            'seconds': time.monotonic() - start,
            'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
            'peak_reserved_gib': torch.cuda.max_memory_reserved() / 1024**3,
        })
        return
    # Resolve every input before launching expensive GPU workers.
    for domain in args.domains:
        for split in args.splits:
            print(f'Input: {input_run(args, domain, split)}', flush=True)
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
                logfile = args.log_dir / f'{domain}_gpu{gpu}.log'
                cmd = [sys.executable, '-u', str(Path(__file__).resolve()),
                       *sys.argv[1:], '--worker-domain', domain]
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = str(gpu)
                print(f'GPU {gpu}: {domain} -> {logfile}', flush=True)
                with logfile.open('w', encoding='utf-8') as f:
                    result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
                if result.returncode:
                    failures.append((domain, result.returncode, str(logfile)))
                    stop.set()
            except Exception as exc:
                failures.append((domain, str(exc)))
                stop.set()
            finally:
                jobs.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in args.gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failures:
        raise SystemExit(f'FAILED: {failures}')
    print('ALL REQUESTED DOMAINS COMPLETED', flush=True)


if __name__ == '__main__':
    main()
