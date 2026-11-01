#!/usr/bin/env python3
"""LoRA fine-tuning for Qwen3-Reranker-8B on RETECO Track 1a.

Input is one JSON array. Each record contains ``query``, ``positives`` and
``hard_negatives`` with document ``context`` fields. For every query and
epoch, one positive is selected and ``--hard-negatives`` hard negatives are
selected with an epoch-dependent deterministic RNG. The model scores each
query-document pair by log P(yes) versus log P(no), and trains with pairwise
logistic loss. Ordinary negatives are intentionally not read or used.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

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


def text_fingerprint(text: str) -> str:
    import unicodedata
    normalized = " ".join(unicodedata.normalize("NFKC", text).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def rng_for(seed: int, *parts: object) -> random.Random:
    raw = json.dumps([seed, *parts], ensure_ascii=False, separators=(",", ":")).encode()
    return random.Random(int.from_bytes(hashlib.sha256(raw).digest(), "big"))


def validate_doc(row_key: object, field: str, doc: object) -> dict:
    if not isinstance(doc, dict):
        raise ValueError(f"{row_key}: every {field} item must be an object")
    doc_id, context = doc.get("doc_id"), doc.get("context")
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError(f"{row_key}: invalid doc_id in {field}")
    if not isinstance(context, str) or not context.strip():
        raise ValueError(f"{row_key}: empty context in {field}/{doc_id}")
    status = doc.get("label_status", "unjudged")
    if status not in ("unjudged", "judged_nonrelevant"):
        raise ValueError(f"{row_key}: invalid label_status in {field}/{doc_id}")
    if field == "hard_negatives":
        try:
            score = float(doc.get("source_score", float("nan")))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{row_key}: invalid source_score in {field}/{doc_id}") from exc
        if not math.isfinite(score):
            raise ValueError(f"{row_key}: source_score must be finite in {field}/{doc_id}")
    return doc


def read_groups(path: Path, required_hard_negatives: int) -> list[dict]:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(rows, list) or not rows:
        raise ValueError("Training JSON must be a non-empty JSON array")
    seen, output = set(), []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Every training record must be an object")
        for field in ("domain", "query_id", "query"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Missing or empty {field}")
        key = (row["domain"], row["query_id"])
        if key in seen:
            raise ValueError(f"Duplicate query group: {key}")
        seen.add(key)
        positives, hard = row.get("positives"), row.get("hard_negatives")
        if not isinstance(positives, list) or not positives:
            raise ValueError(f"{key}: positives must be a non-empty array")
        if not isinstance(hard, list) or len(hard) < required_hard_negatives:
            raise ValueError(f"{key}: need at least {required_hard_negatives} hard_negatives")
        positives = [validate_doc(key, "positives", doc) for doc in positives]
        hard = [validate_doc(key, "hard_negatives", doc) for doc in hard]
        pos_ids, pos_texts = set(), set()
        for doc in positives:
            fp = text_fingerprint(doc["context"])
            if doc["doc_id"] in pos_ids or fp in pos_texts:
                raise ValueError(f"{key}: duplicate positive ID or text")
            pos_ids.add(doc["doc_id"]); pos_texts.add(fp)
        neg_ids, neg_texts = set(), set()
        for doc in hard:
            fp = text_fingerprint(doc["context"])
            if doc["doc_id"] in pos_ids or fp in pos_texts:
                raise ValueError(f"{key}: positive/hard-negative overlap")
            if doc["doc_id"] in neg_ids or fp in neg_texts:
                raise ValueError(f"{key}: duplicate hard-negative ID or text")
            neg_ids.add(doc["doc_id"]); neg_texts.add(fp)
        output.append({"domain": row["domain"], "query_id": row["query_id"],
                       "query": row["query"], "positives": positives,
                       "hard_negatives": hard})
    return sorted(output, key=lambda row: (row["domain"], row["query_id"]))


def select_epoch_group(row: dict, epoch: int, seed: int, hard_count: int) -> dict:
    key = (row["domain"], row["query_id"])
    positives = sorted(row["positives"], key=lambda doc: doc["doc_id"])
    order = list(range(len(positives)))
    rng_for(seed, *key, "positive", epoch).shuffle(order)
    negatives = sorted(row["hard_negatives"], key=lambda doc: doc["doc_id"])
    selected = rng_for(seed, *key, "hard-negative", epoch).sample(negatives, hard_count)
    return {"query": row["query"], "positive": positives[order[0]],
            "hard_negatives": selected, "domain": row["domain"], "query_id": row["query_id"]}


def format_pair(instruction: str, query: str, document: str) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"


def save_adapter_checkpoint(model, tokenizer, path, metadata):
    """Publish only a completely written adapter; not an optimizer resume state."""
    staging = path.with_name(path.name + '.partial')
    if path.exists() or staging.exists():
        raise ValueError(f'Checkpoint already exists: {path} or {staging}')
    staging.mkdir(parents=True)
    model.save_pretrained(str(staging))
    tokenizer.save_pretrained(str(staging))
    (staging / 'checkpoint_info.json').write_text(
        json.dumps(metadata, indent=2) + '\n', encoding='utf-8'
    )
    os.replace(staging, path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-json", type=Path, required=True)
    p.add_argument("--model", type=Path, default=Path("/home/huangliang/models/Qwen3-Reranker-8B"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--hard-negatives", type=int, default=2)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--pair-batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--limit-queries", type=int, default=0)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--data-only", action="store_true")
    p.add_argument("--save-every-epoch", type=int, default=1,
                   help="save adapter every N complete epochs; 0 disables epoch adapters")
    args = p.parse_args()
    for name in ("epochs", "hard_negatives", "max_length", "pair_batch", "grad_accum", "lora_r", "lora_alpha", "log_every"):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_steps < 0 or args.limit_queries < 0:
        p.error("--max-steps and --limit-queries must be nonnegative")
    if args.save_every_epoch < 0:
        p.error('--save-every-epoch must be nonnegative')
    if args.lr <= 0 or not 0 <= args.warmup_ratio <= 1 or not 0 <= args.lora_dropout < 1:
        p.error("invalid learning rate, warmup ratio or LoRA dropout")
    if not args.instruction.strip():
        p.error("--instruction must not be empty")
    args.train_json, args.model, args.output_dir = (x.resolve() for x in (args.train_json, args.model, args.output_dir))
    return args


def model_score(model, tokenizer, instruction, query, documents, args, device):
    import torch
    prefix = tokenizer.encode(SYSTEM_PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(SYSTEM_SUFFIX, add_special_tokens=False)
    budget = args.max_length - len(prefix) - len(suffix)
    if budget < 32:
        raise ValueError("--max-length leaves too few tokens after the reranker template")
    yes_id, no_id = tokenizer.convert_tokens_to_ids("yes"), tokenizer.convert_tokens_to_ids("no")
    # Keep score tensors on the current device. During training these tensors
    # retain the computation graph back to the LoRA parameters. Converting to
    # Python floats or calling detach()/cpu() here would make the loss unable
    # to update the adapter.
    score_chunks = []
    for start in range(0, len(documents), args.pair_batch):
        pairs = [format_pair(instruction, query, doc) for doc in documents[start:start + args.pair_batch]]
        body = tokenizer(pairs, add_special_tokens=False, padding=False, truncation=True,
                         max_length=budget, return_attention_mask=False)
        ids = [prefix + item + suffix for item in body["input_ids"]]
        inputs = tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        try:
            logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :]
        except TypeError:
            logits = model(**inputs, use_cache=False).logits[:, -1, :]
        binary = logits.float()[:, [no_id, yes_id]]
        score = torch.log_softmax(binary, dim=-1)[:, 1]
        if not torch.isfinite(score).all():
            raise FloatingPointError("Non-finite reranker score")
        score_chunks.append(score)
    return torch.cat(score_chunks, dim=0)


def run(args, rows):
    import torch
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from peft import LoraConfig, get_peft_model
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=args.grad_accum)
    if accelerator.device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Qwen3-Reranker training requires a BF16-capable CUDA GPU")
    set_seed(args.seed)
    if accelerator.is_main_process:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise ValueError(f"Output directory is not empty: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, padding_side="left")
    tokenizer.truncation_side = "right"
    model = AutoModelForCausalLM.from_pretrained(str(args.model), local_files_only=True,
        dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM"))
    trainable = [x for x in model.parameters() if x.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable LoRA parameters found")
    model.print_trainable_parameters()
    class Groups(Dataset):
        epoch = 0
        def __len__(self): return len(rows)
        def __getitem__(self, index): return select_epoch_group(rows[index], self.epoch, args.seed, args.hard_negatives)
    def collate(batch):
        if len(batch) != 1: raise ValueError("One query group per local microbatch is required")
        return batch[0]
    dataset = Groups()
    loader = DataLoader(dataset, batch_size=1, shuffle=True, generator=torch.Generator().manual_seed(args.seed), num_workers=0, collate_fn=collate)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps: total_steps = min(total_steps, args.max_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, min(total_steps - 1, math.ceil(total_steps * args.warmup_ratio)), total_steps)
    raw = accelerator.unwrap_model(model)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"world_size": accelerator.num_processes, "num_queries": len(rows), "steps_per_epoch": steps_per_epoch,
                   "planned_updates": total_steps, "loss": "-mean(logsigmoid(score_positive-score_hard_negative))",
                   "score": "log_P_yes_conditional_on_yes_no", "train_json_sha256": hashlib.sha256(args.train_json.read_bytes()).hexdigest(),
                   "checkpoint_contents": "LoRA adapter and tokenizer only; no optimizer/scheduler/RNG resume state",
                   "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "accelerate")}})
    if accelerator.is_main_process:
        (args.output_dir / "training_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    accelerator.print(json.dumps(config, ensure_ascii=False, indent=2))
    start_time, step = time.monotonic(), 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        dataset.epoch = epoch
        if hasattr(loader, "set_epoch"): loader.set_epoch(epoch)
        epoch_loss_sum = torch.zeros((), device=accelerator.device, dtype=torch.float32); epoch_count = 0
        step_loss_sum = torch.zeros((), device=accelerator.device, dtype=torch.float32); step_count = 0
        raw.train()
        for sample in loader:
            with accelerator.accumulate(model):
                query = sample["query"]
                documents = [sample["positive"]["context"]] + [d["context"] for d in sample["hard_negatives"]]
                scores = model_score(model, tokenizer, args.instruction, query, documents, args, accelerator.device)
                # scores already carry the autograd graph from the model
                # forward pass; do not reconstruct them with torch.tensor().
                if not scores.requires_grad:
                    raise RuntimeError(
                        "Reranker scores do not require gradients; the model "
                        "forward path has detached the LoRA computation graph"
                    )
                pos = scores[0]
                neg = scores[1:]
                loss = -F.logsigmoid(pos - neg).mean()
                if not torch.isfinite(loss): raise FloatingPointError("Non-finite pairwise loss")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    norm = accelerator.clip_grad_norm_(trainable, 1.0)
                    if not torch.isfinite(norm) or norm.item() == 0: raise FloatingPointError("Invalid LoRA gradient norm")
                    optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            epoch_loss_sum += loss.detach().float(); epoch_count += 1; step_loss_sum += loss.detach().float(); step_count += 1
            if accelerator.sync_gradients:
                step += 1
                total = accelerator.reduce(step_loss_sum, reduction="sum")
                count = accelerator.reduce(torch.tensor(float(step_count), device=accelerator.device), reduction="sum")
                event = {"type": "optimizer_step", "step": step, "epoch": epoch + 1,
                         "microbatches_global": int(count.item()), "optimizer_step_loss_mean": (total / count).item(),
                         "lr": scheduler.get_last_lr()[0], "grad_norm_local": float(norm),
                         "elapsed_seconds": time.monotonic() - start_time}
                step_loss_sum.zero_(); step_count = 0
                if step % args.log_every == 0 or step == 1:
                    accelerator.print(json.dumps(event))
                if accelerator.is_main_process:
                    with (args.output_dir / "training_log.jsonl").open("a", encoding="utf-8") as f: f.write(json.dumps(event) + "\n")
                if step >= total_steps: break
        epoch_total = accelerator.reduce(epoch_loss_sum, reduction="sum")
        epoch_num = accelerator.reduce(torch.tensor(float(epoch_count), device=accelerator.device), reduction="sum")
        epoch_event = {"type": "epoch_summary", "epoch": epoch + 1, "microbatches_global": int(epoch_num.item()),
                       "epoch_loss_mean": (epoch_total / epoch_num).item(),
                       "complete_epoch": epoch_count == len(loader),
                       "elapsed_seconds": time.monotonic() - start_time}
        accelerator.print(json.dumps(epoch_event))
        if accelerator.is_main_process:
            with (args.output_dir / "training_log.jsonl").open("a", encoding="utf-8") as f: f.write(json.dumps(epoch_event) + "\n")
        if (args.save_every_epoch and epoch_event['complete_epoch']
                and (epoch + 1) % args.save_every_epoch == 0):
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                checkpoint_path = args.output_dir / f'epoch_{epoch + 1:03d}'
                save_adapter_checkpoint(raw, tokenizer, checkpoint_path, {
                    'epoch': epoch + 1, 'optimizer_step': step,
                    'complete_epoch': True,
                    'epoch_loss_mean': epoch_event['epoch_loss_mean'],
                    'base_model': str(args.model),
                    'instruction': args.instruction,
                    'optimizer_resume_supported': False,
                })
                accelerator.print(f'SAVED ADAPTER: {checkpoint_path}')
            accelerator.wait_for_everyone()
        if step >= total_steps: break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        raw.save_pretrained(str(args.output_dir / "final")); tokenizer.save_pretrained(str(args.output_dir / "final"))
        result = {"updates": step, "elapsed_seconds": time.monotonic() - start_time,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                  "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                  "adapter": str(args.output_dir / "final")}
        (args.output_dir / "training_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    accelerator.wait_for_everyone(); accelerator.print("COMPLETED"); accelerator.end_training()


def main() -> None:
    args = parse_args()
    rows = read_groups(args.train_json, args.hard_negatives)
    if args.limit_queries: rows = rng_for(args.seed, "limit").sample(rows, min(args.limit_queries, len(rows)))
    if args.data_only:
        first, second = select_epoch_group(rows[0], 0, args.seed, args.hard_negatives), select_epoch_group(rows[0], 1, args.seed, args.hard_negatives)
        print(json.dumps({"num_queries": len(rows), "domains": dict(Counter(r["domain"] for r in rows)),
            "hard_negatives_per_query": args.hard_negatives,
            "epoch0_selection": {"positive_id": first["positive"]["doc_id"], "negative_ids": [d["doc_id"] for d in first["hard_negatives"]]},
            "epoch1_selection": {"positive_id": second["positive"]["doc_id"], "negative_ids": [d["doc_id"] for d in second["hard_negatives"]]}}, ensure_ascii=False, indent=2))
        return
    if not args.model.is_dir(): raise ValueError(f"Local reranker model directory does not exist: {args.model}")
    run(args, rows)


if __name__ == "__main__":
    try: main()
    except (OSError, ValueError, RuntimeError, FloatingPointError) as exc:
        print(f"ERROR: {exc}", flush=True); raise SystemExit(1)
