#!/usr/bin/env python3
"""RETECO dense retrieval starter (BGE + GPU cosine) -- upgrade over BM25.

Mirrors bm25_baseline.py's CLI so runs are directly comparable, then score with
the official pure-python scorer:

    TRACK 1 (1a whole-query / 1b step-wise):
        python dense_retrieval.py --track 1 --domain iota --split dev
        python dense_retrieval.py --track 1 --domain iota --split dev --level step
    TRACK 2 (2a; --strategy chooses conversation context):
        python dense_retrieval.py --track 2 --domain drones --split dev --strategy current
        python dense_retrieval.py --track 2 --domain drones --split dev --strategy history

    # score the produced run:
        python ../starter_kit/scorer.py --track 1 --run runs/dense_1a_iota_dev.txt \
            --qrels <data>/track1_tempo/iota/qrels_dev.txt \
            --steps <data>/track1_tempo/iota/steps_dev.jsonl

Corpus embeddings are cached under --cache so re-runs / new strategies are fast.
Set HF_ENDPOINT=https://hf-mirror.com before first run if HF is unreachable.
"""
import argparse
import json
import os

import torch


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def load_corpus(path, key):
    docs = read_jsonl(path)
    return [d[key] for d in docs], [d["content"] for d in docs]


def track1_queries(path, level):
    items = read_jsonl(path)
    if level == "step":
        return [(st["step_id"],
                 f'{it["query"]}\n\nStep: {st.get("step_instruction", "")}')
                for it in items for st in it.get("steps", [])]
    return [(it["id"], it["query"]) for it in items]


def track2_queries(path, strategy):
    bench = json.load(open(path))
    conv = [bench] if isinstance(bench, dict) else bench
    out = []
    for c in conv:
        for t in c["turns"]:
            qid = f"{c['id']}_turn_{t['turn_id']}"
            if strategy == "history":
                hist = t.get("conversation_history", "")
                if hist.lower().startswith("no previous"):
                    hist = ""
                q = (hist + " " + t["query"]).strip()
            else:
                q = t["query"]
            out.append((qid, q))
    return out


def write_run(out, rows, tag="dense"):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        for qid, ranked in rows:
            for rank, (docid, score) in enumerate(ranked, 1):
                f.write(f"{qid}\tQ0\t{docid}\t{rank}\t{score:.6f}\t{tag}\n")


def encode(model, texts, batch):
    emb = model.encode(texts, batch_size=batch)
    return torch.nn.functional.normalize(
        torch.from_numpy(emb).float(), p=2, dim=1
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="root of reteco_data")
    ap.add_argument("--track", type=int, choices=[1, 2], required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--split", choices=["train", "dev"], default="dev")
    ap.add_argument("--level", choices=["query", "step"], default="query")
    ap.add_argument("--strategy", choices=["current", "history"], default="current")
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--query-instruction",
                    default="Represent this sentence for searching relevant passages: ")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"))
    ap.add_argument("--cache", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    sub = os.path.join(args.data, "track1_tempo" if args.track == 1 else "track2_recor", args.domain)
    corpus_path = os.path.join(sub, "documents.jsonl")
    corpus_key = "id" if args.track == 1 else "doc_id"
    doc_ids, doc_texts = load_corpus(corpus_path, corpus_key)

    if args.track == 1:
        queries = track1_queries(os.path.join(sub, f"examples_{args.split}.jsonl"), args.level)
        tag = f"dense_1{'a' if args.level == 'query' else 'b'}"
    else:
        queries = track2_queries(os.path.join(sub, f"benchmark_{args.split}.json"), args.strategy)
        tag = f"dense_2a_{args.strategy}"

    from FlagEmbedding import FlagModel
    model = FlagModel(args.model,
                      query_instruction_for_retrieval=args.query_instruction,
                      use_fp16=(args.device == "cuda"),
                      device=args.device)

    # corpus embeddings (cached)
    os.makedirs(args.cache, exist_ok=True)
    cache = os.path.join(args.cache, f"{args.domain}__{args.model.replace('/', '_')}.pt")
    if os.path.exists(cache):
        doc_emb = torch.load(cache, map_location=args.device)
    else:
        doc_emb = encode(model, doc_texts, args.batch).to(args.device)
        torch.save(doc_emb.cpu(), cache)

    q_emb = encode(model, [q for _, q in queries], args.batch).to(args.device)

    # cosine similarity (embeddings already normalized) -> top-k
    scores = doc_emb @ q_emb.T                      # [N, Q]
    k = min(args.top_k, len(doc_ids))
    topv, topi = torch.topk(scores, k, dim=0)       # [k, Q]

    rows = []
    for qi, (qid, _) in enumerate(queries):
        ranked = [(doc_ids[i], s) for s, i in zip(topv[:, qi].tolist(), topi[:, qi].tolist())]
        rows.append((qid, ranked))

    out = os.path.join(args.out_dir, f"{tag}_{args.domain}_{args.split}.txt")
    write_run(out, rows)
    print(f"wrote {out}: {len(rows)} topics over {len(doc_ids)} docs "
          f"(track {args.track}, {args.level}/{args.strategy}, {args.model})")


if __name__ == "__main__":
    main()
