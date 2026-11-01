#!/usr/bin/env python3
"""Train Qwen3 Reranker LoRA with all positives and query-balanced pairwise loss.

Despite the run_ filename this is TRAINING, not run-file inference.
Every complete epoch visits each (query, positive ID) exactly once.
Objective: mean_query(mean_positive(mean_negative(softplus(s_neg-s_pos)))).
DDP padding is masked, not counted as training data. Only a train JSON is read.
Each epoch saves an inference adapter snapshot and an actual coverage report.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import random
import time
from collections import Counter
from contextlib import nullcontext
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
            if doc["doc_id"] in pos_ids:
                raise ValueError(f"{key}: duplicate positive ID")
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


def positive_units(rows):
    """A stable index for every distinct query/positive ID, with no text copies."""
    return [(qi, pi) for qi, row in enumerate(rows) for pi in range(len(row["positives"]))]


def epoch_windows(unit_count, epoch, seed, world_size, grad_accum):
    """Global batches; None slots equalize ranks without duplicating training."""
    indices = list(range(unit_count))
    rng_for(seed, "epoch_order", epoch).shuffle(indices)
    width = world_size * grad_accum
    for start in range(0, unit_count, width):
        real = indices[start:start + width]
        slots = real + [None] * ((-len(real)) % world_size)
        yield slots, len(real)


def select_unit(rows, units, unit_index, epoch, seed, hard_count):
    qi, pi = units[unit_index]
    row = rows[qi]
    candidates = sorted(row["hard_negatives"], key=lambda d: d["doc_id"])
    hard = rng_for(seed, row["domain"], row["query_id"],
                   row["positives"][pi]["doc_id"], "negative", epoch).sample(candidates, hard_count)
    return row, row["positives"][pi], hard


def unit_weight(num_units, num_queries, positive_count):
    # Uniform sampling of expanded units: E[w * loss] = query-macro objective.
    return num_units / (num_queries * positive_count)


def write_json(path, value):
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def save_snapshot(model, tokenizer, path, metadata):
    staging = path.with_name(path.name + ".partial")
    if path.exists() or staging.exists():
        raise ValueError(f"Checkpoint exists: {path} or {staging}")
    staging.mkdir(parents=True)
    model.save_pretrained(str(staging))
    tokenizer.save_pretrained(str(staging))
    if not (staging / "adapter_config.json").is_file():
        raise RuntimeError("Saved checkpoint is missing adapter_config.json")
    write_json(staging / "checkpoint_info.json", {
        **metadata, "optimizer_resume_supported": False,
        "checkpoint_contents": "PEFT LoRA adapter and tokenizer; original base required",
    })
    os.rename(staging, path)


def coverage_report(rows, units, counts, query_loss_sums):
    per_query, per_positive = [], []
    for qi, row in enumerate(rows):
        per_query.append({"domain": row["domain"], "query_id": row["query_id"],
                          "expected_positives": len(row["positives"]),
                          "unique_positives_used": 0, "total_uses": 0,
                          "unused_positive_doc_ids": []})
    for ui, (qi, pi) in enumerate(units):
        used = int(counts[ui])
        row, query_row = rows[qi], per_query[qi]
        query_row["unique_positives_used"] += int(used > 0)
        query_row["total_uses"] += used
        doc_id = row["positives"][pi]["doc_id"]
        if not used:
            query_row["unused_positive_doc_ids"].append(doc_id)
        per_positive.append({"domain": row["domain"], "query_id": row["query_id"],
                             "positive_doc_id": doc_id, "used_count": used})
    seen_means, complete_queries = [], 0
    for qi, item in enumerate(per_query):
        item["coverage_rate"] = item["unique_positives_used"] / item["expected_positives"]
        item["observed_positive_loss_mean"] = (
            float(query_loss_sums[qi]) / item["total_uses"] if item["total_uses"] else None
        )
        if item["observed_positive_loss_mean"] is not None:
            seen_means.append(item["observed_positive_loss_mean"])
        complete_queries += int(item["unique_positives_used"] == item["expected_positives"])
    covered = sum(c > 0 for c in counts)
    return {
        "num_queries": len(rows), "total_positive_units": len(units),
        "unique_positive_units_used": covered,
        "positive_coverage_rate": covered / len(units),
        "positive_uses": sum(counts), "repeated_positive_uses": sum(max(0, c - 1) for c in counts),
        "queries_with_full_positive_coverage": complete_queries,
        "observed_query_macro_loss": sum(seen_means) / len(seen_means) if seen_means else None,
        "loss_note": "Online losses at changing weights; partial epoch averages only observed positives/queries; not nDCG or held-out loss",
        "queries": per_query,
    }, per_positive


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-json", type=Path, required=True)
    p.add_argument("--model", type=Path, default=Path("/home/huangliang/models/Qwen3-Reranker-8B"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--hard-negatives", type=int, default=2, help="negative count PER positive unit")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--pair-batch", type=int, default=1, help="query/document inputs per model forward")
    p.add_argument("--grad-accum", type=int, default=8, help="positive units per rank per full optimizer update")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    p.add_argument("--score-mode", choices=("log_p_yes", "logit_diff"), default="log_p_yes",
                   help="default preserves previous training score; logit_diff is an explicit ablation")
    p.add_argument("--max-steps", type=int, default=0, help="0=all epochs; smoke tests may stop partway")
    p.add_argument("--limit-queries", type=int, default=0)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--data-only", action="store_true")
    a = p.parse_args()
    for key in ("epochs", "hard_negatives", "max_length", "pair_batch", "grad_accum", "lora_r", "lora_alpha", "log_every"):
        if getattr(a, key) < 1:
            p.error(f"--{key.replace('_', '-')} must be positive")
    if a.max_steps < 0 or a.limit_queries < 0:
        p.error("step/query limits must be nonnegative")
    if not math.isfinite(a.lr) or a.lr <= 0 or not 0 <= a.warmup_ratio <= 1 or not 0 <= a.lora_dropout < 1:
        p.error("invalid learning rate/warmup/dropout")
    if not a.instruction.strip():
        p.error("instruction cannot be empty")
    for key in ("train_json", "model", "output_dir"):
        setattr(a, key, getattr(a, key).resolve())
    return a


def run(args, rows, units):
    import torch
    import torch.nn.functional as F
    from torch import nn
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, set_seed
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    # Accumulation and last-window normalization are explicit below. Leave
    # Accelerate's accumulation at 1 to avoid dividing gradients twice.
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[
        DistributedDataParallelKwargs(find_unused_parameters=False, broadcast_buffers=False)
    ])
    if accelerator.device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required")
    world, rank = accelerator.num_processes, accelerator.process_index
    # Read-only validation on every rank prevents a rank-0-only path error hang.
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Use a fresh output directory: {args.output_dir}")
    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, padding_side="left")
    tokenizer.truncation_side = "right"
    for token in ("yes", "no"):
        if tokenizer.encode(token, add_special_tokens=False) != [tokenizer.convert_tokens_to_ids(token)]:
            raise ValueError(f"Expected {token!r} to be a single vocabulary token")
    prefix = tokenizer.encode(SYSTEM_PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(SYSTEM_SUFFIX, add_special_tokens=False)
    budget = args.max_length - len(prefix) - len(suffix)
    if budget < 32:
        raise ValueError("max-length leaves fewer than 32 body tokens")
    no_id, yes_id = tokenizer.convert_tokens_to_ids("no"), tokenizer.convert_tokens_to_ids("yes")
    base = AutoModelForCausalLM.from_pretrained(str(args.model), local_files_only=True,
        dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="sdpa")
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    peft_model = get_peft_model(base, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM"))
    trainable = [(n, p) for n, p in peft_model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in n for n, p in trainable):
        raise RuntimeError("Expected only trainable LoRA parameters")
    for _, parameter in trainable:
        parameter.data = parameter.data.float()

    class Objective(nn.Module):
        """One DDP forward per positive unit, including all internal score chunks."""
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, row, positive, negatives):
            documents = [positive["context"]] + [d["context"] for d in negatives]
            scores = []
            for start in range(0, len(documents), args.pair_batch):
                texts = [f"<Instruct>: {args.instruction}\n<Query>: {row['query']}\n<Document>: {d}"
                         for d in documents[start:start + args.pair_batch]]
                body = tokenizer(texts, add_special_tokens=False, padding=False,
                                 truncation=True, max_length=budget, return_attention_mask=False)
                ids = [prefix + ids + suffix for ids in body["input_ids"]]
                inputs = tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")
                inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
                # No broad TypeError fallback: old runtimes should fail clearly
                # instead of allocating full sequence-vocabulary logits.
                logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
                binary = logits[:, [no_id, yes_id]]
                score = (binary[:, 1] - binary[:, 0] if args.score_mode == "logit_diff"
                         else F.log_softmax(binary, dim=-1)[:, 1])
                scores.append(score)
            scores = torch.cat(scores)
            if not scores.requires_grad or not torch.isfinite(scores).all():
                raise FloatingPointError("Detached or non-finite model scores")
            return F.softplus(scores[1:] - scores[0]).mean()

    objective = Objective(peft_model)
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=args.lr, weight_decay=0.0)
    objective, optimizer = accelerator.prepare(objective, optimizer)
    raw = accelerator.unwrap_model(objective)
    m, q = len(units), len(rows)
    steps_per_epoch = math.ceil(m / (world * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = min(total_steps - 1, math.ceil(total_steps * args.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"world_size": world, "num_queries": q, "positive_units_per_epoch": m,
        "steps_per_epoch": steps_per_epoch, "planned_updates": total_steps,
        "loss": "mean_q(mean_p(mean_n(softplus(s_neg-s_pos))))",
        "unit_weight": "M/(Q*positive_count_of_query)",
        "gradient_normalization": "world_size/actual_global_units_in_window; no Accelerate loss division",
        "padding": "zero loss weight and excluded from coverage/statistics",
        "score": args.score_mode, "instruction": args.instruction,
        "precision": "BF16 base; FP32 LoRA parameters, loss and Adam states",
        "train_json_sha256": hashlib.sha256(args.train_json.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "optimizer_resume_supported": False,
        "versions": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "peft", "accelerate")}})
    if accelerator.is_main_process:
        write_json(args.output_dir / "training_config.json", config)
    accelerator.print(json.dumps(config, ensure_ascii=False, indent=2))

    def log_event(event):
        if accelerator.is_main_process:
            with (args.output_dir / "training_log.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")

    started, step, summaries = time.monotonic(), 0, []
    for epoch in range(args.epochs):
        raw.train()
        torch.cuda.reset_peak_memory_stats(accelerator.device)
        epoch_start = time.monotonic()
        counts = torch.zeros(m, device=accelerator.device, dtype=torch.int64)
        query_sums = torch.zeros(q, device=accelerator.device, dtype=torch.float64)
        processed_windows = 0
        for slots, valid_count in epoch_windows(m, epoch, args.seed, world, args.grad_accum):
            optimizer.zero_grad(set_to_none=True)
            local_slots = slots[rank::world]
            local_stats = torch.zeros(3, device=accelerator.device, dtype=torch.float64)
            for micro_index, unit_index in enumerate(local_slots):
                valid = unit_index is not None
                actual_index = unit_index if valid else next(i for i in slots if i is not None)
                row, positive, negatives = select_unit(rows, units, actual_index, epoch,
                                                      args.seed, args.hard_negatives)
                final_micro = micro_index == len(local_slots) - 1
                context = nullcontext() if final_micro else accelerator.no_sync(objective)
                with context:
                    with accelerator.autocast():
                        loss = objective(row, positive, negatives)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite pairwise loss")
                    weight = unit_weight(m, q, len(row["positives"])) if valid else 0.0
                    # DDP averages across ranks. This gives (1/C) sum_i w_i*l_i
                    # for the ACTUAL global window size C, including the tail.
                    accelerator.backward(loss * (weight * world / valid_count))
                if valid:
                    qi, _ = units[unit_index]
                    counts[unit_index] += 1
                    query_sums[qi] += loss.detach().double()
                    local_stats[0] += loss.detach().double()
                    local_stats[1] += loss.detach().double() * weight
                    local_stats[2] += 1
                del loss  # release this unit's graph before the next forward
            params = [p for _, p in trainable]
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
                raise FloatingPointError("Non-finite LoRA gradients")
            norm = accelerator.clip_grad_norm_(params, 1.0)
            if not torch.isfinite(norm) or norm.item() == 0:
                raise FloatingPointError("Zero or non-finite LoRA gradient norm")
            lr_used = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            if any(not torch.isfinite(p).all() for p in params):
                raise FloatingPointError("Non-finite adapter weights after update")
            step += 1
            processed_windows += 1
            stats = accelerator.reduce(local_stats, reduction="sum").cpu().tolist()
            event = {"type": "optimizer_step", "epoch": epoch + 1, "step": step,
                "actual_positive_units_global": int(stats[2]),
                "microbatches_per_rank": len(local_slots),
                "pairwise_loss_mean": stats[0] / valid_count,
                "query_balanced_loss_estimate": stats[1] / valid_count,
                "lr_used": lr_used, "lr_next": scheduler.get_last_lr()[0],
                "grad_norm": float(norm), "elapsed_seconds": time.monotonic() - started}
            log_event(event)
            if step == 1 or step % args.log_every == 0:
                accelerator.print(json.dumps(event))
            if step >= total_steps:
                break

        all_counts = accelerator.reduce(counts, reduction="sum").cpu().tolist()
        all_sums = accelerator.reduce(query_sums, reduction="sum").cpu().tolist()
        report, positive_rows = coverage_report(rows, units, all_counts, all_sums)
        complete = processed_windows == steps_per_epoch
        if complete and (report["positive_coverage_rate"] != 1 or report["repeated_positive_uses"]):
            raise RuntimeError("Complete epoch failed exact-once positive coverage check")
        memory = torch.tensor([[torch.cuda.max_memory_allocated(accelerator.device) / 1024**3,
                               torch.cuda.max_memory_reserved(accelerator.device) / 1024**3]],
                              device=accelerator.device)
        memory_all = accelerator.gather(memory).cpu().tolist()
        event = {"type": "epoch_summary", "epoch": epoch + 1, "step": step,
            "complete_epoch": complete,
            "epoch_loss_query_macro": report["observed_query_macro_loss"],
            "positive_coverage_rate": report["positive_coverage_rate"],
            "positive_units_used": report["unique_positive_units_used"],
            "positive_units_expected": m,
            "repeated_positive_uses": report["repeated_positive_uses"],
            "epoch_seconds": time.monotonic() - epoch_start,
            "elapsed_seconds": time.monotonic() - started,
            "peak_allocated_reserved_gib_by_rank": memory_all,
            "evaluation": "training statistics only; no retrieval metrics computed"}
        log_event(event)
        accelerator.print(json.dumps(event, ensure_ascii=False))
        checkpoint_name = f"epoch_{epoch + 1:03d}" + ("" if complete else "_incomplete")
        if accelerator.is_main_process:
            write_json(args.output_dir / f"{checkpoint_name}_coverage.json", {**event, **report})
            with (args.output_dir / f"{checkpoint_name}_positive_coverage.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(positive_rows[0]))
                writer.writeheader()
                writer.writerows(positive_rows)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_snapshot(raw.model, tokenizer, args.output_dir / checkpoint_name, {
                **event, "base_model": str(args.model), "instruction": args.instruction,
                "score_mode": args.score_mode, "max_length": args.max_length,
                "train_json_sha256": config["train_json_sha256"],
            })
            accelerator.print(f"SAVED MODEL AND COVERAGE: {args.output_dir / checkpoint_name}")
        accelerator.wait_for_everyone()
        summaries.append(event)
        if step >= total_steps:
            break
    if accelerator.is_main_process:
        save_snapshot(raw.model, tokenizer, args.output_dir / "final", {
            **summaries[-1], "base_model": str(args.model), "instruction": args.instruction,
            "score_mode": args.score_mode, "max_length": args.max_length,
            "train_json_sha256": config["train_json_sha256"],
        })
        write_json(args.output_dir / "training_result.json", {
            "updates": step, "elapsed_seconds": time.monotonic() - started,
            "epochs": summaries, "optimizer_resume_supported": False,
            "evaluation": "No nDCG/recall computed; evaluate adapters with the inference script",
        })
    accelerator.wait_for_everyone()
    accelerator.print("COMPLETED")
    accelerator.end_training()


def main():
    args = parse_args()
    rows = read_groups(args.train_json, args.hard_negatives)
    if args.limit_queries:
        rows = rng_for(args.seed, "subset").sample(rows, min(args.limit_queries, len(rows)))
        rows.sort(key=lambda r: (r["domain"], r["query_id"]))
    units = positive_units(rows)
    if args.data_only:
        print(json.dumps({"num_queries": len(rows), "positive_units_per_epoch": len(units),
            "positive_count_min": min(len(r["positives"]) for r in rows),
            "positive_count_max": max(len(r["positives"]) for r in rows),
            "domains": dict(Counter(r["domain"] for r in rows)),
            "hard_negatives_per_positive": args.hard_negatives,
            "input_document_encodings_per_epoch": len(units) * (1 + args.hard_negatives),
            "objective": "mean_q(mean_p(mean_n(softplus(s_neg-s_pos))))",
            "all_positive_weight_sum": sum(unit_weight(len(units), len(rows), len(rows[qi]["positives"])) for qi, pi in units),
            "padding_uses_zero_weight": True,
            "no_model_loaded_or_files_written": True}, ensure_ascii=False, indent=2))
        return
    if not args.model.is_dir():
        raise ValueError(f"Local model missing: {args.model}")
    run(args, rows, units)


if __name__ == "__main__":
    main()


