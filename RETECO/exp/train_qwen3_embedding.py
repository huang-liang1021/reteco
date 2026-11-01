#!/usr/bin/env python3
"""Sentence Transformers 6.1 BF16 LoRA training for RETECO 1a.

One local query per microbatch; explicit negatives only (no cross-rank gather).
All original positives remain in JSON; one positive rotates per query/epoch.
Each training sample draws configurable hard negatives. Ordinary corpus
negatives and negative_source metadata are optional (ordinary default: zero).
Save an independent Sentence Transformers LoRA checkpoint after every complete
epoch, as well as final/. These are model snapshots, not optimizer resume states.
This is an Accelerate loop using SentenceTransformer and its MNRL loss,
not SentenceTransformerTrainer. No dev labels are loaded.
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

INSTRUCTION = (
    "Given a temporal information retrieval query, retrieve passages that "
    "provide evidence for the topic, entities, events, time periods, "
    "changes, causes, impacts, and requested evidence."
)


def fingerprint(text):
    import unicodedata
    return hashlib.sha256(" ".join(unicodedata.normalize("NFKC", text).split()).encode()).hexdigest()


def rng_for(seed, *parts):
    raw = json.dumps([seed, *parts], ensure_ascii=False).encode()
    return random.Random(int.from_bytes(hashlib.sha256(raw).digest(), "big"))


def _validate_documents(row_key, field, docs, required_count, require_label_status=False):
    if docs is None:
        docs = []
    if not isinstance(docs, list):
        raise ValueError(f"{row_key}: {field} must be an array")
    if required_count and len(docs) < required_count:
        raise ValueError(
            f"{row_key}: {field} has {len(docs)} documents, "
            f"but {required_count} are required"
        )
    ids = set()
    for doc in docs:
        if not isinstance(doc, dict):
            raise ValueError(f"{row_key}: document in {field} must be an object")
        if not isinstance(doc.get("doc_id"), str) or not doc["doc_id"]:
            raise ValueError(f"{row_key}: invalid document ID in {field}")
        if doc["doc_id"] in ids:
            raise ValueError(f"{row_key}: duplicate document ID in {field}")
        ids.add(doc["doc_id"])
        if not isinstance(doc.get("context"), str) or not doc["context"].strip():
            raise ValueError(f"{row_key}: empty context in {field}")
        if require_label_status and doc.get("label_status") not in ("unjudged", "judged_nonrelevant"):
            raise ValueError(f"{row_key}: unknown label_status in {field}")
    return docs


def read_groups(path, hard_negative_count, ordinary_negative_count=0):
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Training JSON must be a nonempty array")
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each training record must be an object")
        for key in ("domain", "query_id", "query"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Missing/non-string {key}")
        if "negative_source" in row and (
            not isinstance(row["negative_source"], str) or not row["negative_source"].strip()
        ):
            raise ValueError("negative_source must be a nonempty string when provided")
        key = (row["domain"], row["query_id"])
        if key in seen:
            raise ValueError(f"Duplicate query group: {key}")
        seen.add(key)
        positives = _validate_documents(key, "positives", row.get("positives"), 1)
        hard = _validate_documents(
            key, "hard_negatives", row.get("hard_negatives"), hard_negative_count,
            require_label_status=True,
        )
        row["hard_negatives"] = hard
        ordinary = _validate_documents(
            key, "ordinary_negatives", row.get("ordinary_negatives", []), ordinary_negative_count,
            require_label_status=True,
        )
        row["ordinary_negatives"] = ordinary
        for field, docs in (("hard_negatives", hard), ("ordinary_negatives", ordinary)):
            for doc in docs:
                if doc.get("sampling_method") and not isinstance(doc["sampling_method"], str):
                    raise ValueError(f"{key}: invalid sampling_method in {field}")
        pos_ids = {d["doc_id"] for d in positives}
        pos_text = {fingerprint(d["context"]) for d in positives}
        neg_text = set()
        neg_ids = set()
        for field, docs in (("hard_negatives", hard), ("ordinary_negatives", ordinary)):
            for doc in docs:
                fp = fingerprint(doc["context"])
                if doc["doc_id"] in pos_ids or fp in pos_text:
                    raise ValueError(f"{key}: positive/{field} overlap")
                if doc["doc_id"] in neg_ids or fp in neg_text:
                    raise ValueError(f"{key}: duplicate negative ID or text across negative fields")
                neg_ids.add(doc["doc_id"])
                neg_text.add(fp)
                if field == "hard_negatives" and not math.isfinite(float(doc.get("source_score", float("nan")))):
                    raise ValueError(f"{key}: non-finite source_score")
        if row["query"].startswith("Instruct:"):
            raise ValueError(f"{key}: expected raw query, found existing instruction")
    return sorted(rows, key=lambda r: (r["domain"], r["query_id"]))


def save_encoder_checkpoint(encoder, path, base_model, metadata):
    """Save ST layout + LoRA, then publish the completed directory together.

    The frozen base is referenced by path, not copied. Optimizer, scheduler and
    RNG states are not saved, so this is not an exact training resume checkpoint.
    A failed save remains under .partial for inspection.
    """
    staging = path.with_name(path.name + ".partial")
    if path.exists() or staging.exists():
        raise ValueError(f"Checkpoint already exists: {path} or {staging}")
    staging.mkdir(parents=True)
    encoder.save_pretrained(str(staging), create_model_card=False)
    adapter_files = list(staging.rglob("adapter_config.json"))
    if len(adapter_files) != 1:
        raise RuntimeError("Expected one saved PEFT adapter configuration")
    adapter_config = json.loads(adapter_files[0].read_text(encoding="utf-8"))
    if Path(adapter_config["base_model_name_or_path"]).resolve() != base_model.resolve():
        raise RuntimeError("Saved adapter does not point at the original local base model")
    checkpoint_info = dict(metadata)
    checkpoint_info.update({
        "base_model": str(base_model),
        "checkpoint_contents": "Sentence Transformers configs/tokenizer and LoRA adapter; frozen base referenced separately",
        "optimizer_resume_supported": False,
    })
    (staging / "checkpoint_info.json").write_text(
        json.dumps(checkpoint_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(staging, path)


def select_group(row, epoch, seed, hard_negative_count, ordinary_negative_count, instruction):
    key = (row["domain"], row["query_id"])
    positives = sorted(row["positives"], key=lambda d: d["doc_id"])
    rng_for(seed, *key, "positive_order").shuffle(positives)
    pos = positives[epoch % len(positives)]
    hard_candidates = sorted(row["hard_negatives"], key=lambda d: d["doc_id"])
    ordinary_candidates = sorted(row["ordinary_negatives"], key=lambda d: d["doc_id"])
    hard = rng_for(seed, *key, "hard_negatives", epoch).sample(
        hard_candidates, hard_negative_count
    )
    ordinary = rng_for(seed, *key, "ordinary_negatives", epoch).sample(
        ordinary_candidates, ordinary_negative_count
    )
    negatives = hard + ordinary
    return {
        "texts": [f"Instruct: {instruction}\nQuery: {row['query']}",
                  pos["context"], *(d["context"] for d in negatives)],
        "domain": row["domain"], "query_id": row["query_id"],
        "positive_id": pos["doc_id"],
        "hard_negative_ids": [d["doc_id"] for d in hard],
        "ordinary_negative_ids": [d["doc_id"] for d in ordinary],
        "negative_ids": [d["doc_id"] for d in negatives],
    }


def first_item(batch):
    if len(batch) != 1:
        raise ValueError("This v1 implementation requires one query per local microbatch")
    return batch[0]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-json", type=Path, required=True)
    p.add_argument("--model", type=Path, default=Path("/home/huangliang/models/Qwen3-Embedding-8B"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument(
        "--hard-negatives", type=int, default=1,
        help="hard negatives sampled per query (default: 1)",
    )
    p.add_argument(
        "--ordinary-negatives", type=int, default=0,
        help="optional ordinary negatives sampled per query (default: 0; field may be absent)",
    )
    p.add_argument("--max-length", type=int, default=2048, help="same limit for queries and documents")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--query-instruction", default=INSTRUCTION)
    p.add_argument("--max-steps", type=int, default=0, help="optimizer update cap; 0 = all epochs")
    p.add_argument("--limit-queries", type=int, default=0, help="deterministic subset for smoke test only")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--data-only", action="store_true", help="validate/sample JSON without torch or GPU")
    p.add_argument("--reload-check", action="store_true", help="free trained model, reload saved adapter and compare vectors")
    args = p.parse_args()
    for name in ("epochs", "max_length", "grad_accum", "lora_r", "lora_alpha", "log_every"):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if args.hard_negatives < 0 or args.ordinary_negatives < 0:
        p.error("--hard-negatives and --ordinary-negatives must be nonnegative")
    if args.hard_negatives + args.ordinary_negatives < 1:
        p.error("at least one explicit negative is required")
    if not (0 < args.lr and 0 < args.temperature and 0 <= args.warmup_ratio <= 1 and 0 <= args.lora_dropout < 1):
        p.error("Invalid learning rate, temperature, warmup or dropout")
    if args.max_steps < 0 or args.limit_queries < 0:
        p.error("step/query limits must be nonnegative")
    args.train_json = args.train_json.resolve()
    args.model = args.model.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.query_instruction.strip():
        p.error("query instruction must not be empty")
    return args


def load_encoder(path, device, max_length):
    import torch
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(
        str(path), device=str(device), local_files_only=True,
        model_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa"},
    )
    modules = list(model.children())
    names = [type(m).__name__ for m in modules]
    if names != ["Transformer", "Pooling", "Normalize"]:
        raise ValueError(f"Unexpected ST modules: {names}")
    pooling = modules[1]
    if getattr(pooling, "pooling_mode", None) != "lasttoken":
        raise ValueError("Expected only last-token pooling")
    model.max_seq_length = max_length
    model.default_prompt_name = None
    # All prefixes are constructed exactly once by select_group().
    model.prompts = {"query": "", "document": ""}
    model.tokenizer.padding_side = "left"
    model.tokenizer.truncation_side = "right"
    model[0].model.config.use_cache = False
    return model


def run(args, rows):
    import gc
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from peft import LoraConfig
    from transformers import get_linear_schedule_with_warmup
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss

    class Groups(Dataset):
        epoch = 0

        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            return select_group(rows[index], self.epoch, args.seed, args.hard_negatives,
                                args.ordinary_negatives, args.query_instruction)

    class Objective(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.loss = MultipleNegativesRankingLoss(
                encoder, scale=1 / args.temperature, gather_across_devices=False,
                directions=("query_to_doc",), hardness_mode=None,
            )

        @property
        def encoder(self):
            return self.loss.model

        def embeddings(self, features):
            # Use ST's Transformer and last-token Pooling. Normalize explicitly
            # in FP32 before any BF16 normalization can lose precision.
            hidden = self.encoder[0](dict(features))
            pooled = self.encoder[1](hidden)["sentence_embedding"].float()
            if not torch.isfinite(pooled).all() or (pooled.norm(dim=-1) == 0).any():
                raise FloatingPointError("Non-finite/zero pooled embedding")
            return F.normalize(pooled, p=2, dim=-1)

        def forward(self, columns):
            embeddings = [self.embeddings(f) for f in columns]
            # Disable autocast: similarity and logsumexp must stay FP32.
            with torch.autocast("cuda", enabled=False):
                loss = self.loss.compute_loss_from_embeddings(embeddings, None)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite contrastive loss")
            return loss

    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=args.grad_accum)
    if accelerator.device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This BF16 training script requires a BF16-capable CUDA GPU")
    set_seed(args.seed)
    out = args.output_dir
    if accelerator.is_main_process:
        if out.exists() and any(out.iterdir()):
            raise ValueError(f"Output directory is not empty: {out}; use a fresh directory")
        out.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    encoder = load_encoder(args.model, accelerator.device, args.max_length)
    base = encoder[0].model
    targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    available = {name.split(".")[-1] for name, _ in base.named_modules()}
    if not set(targets) <= available:
        raise ValueError("Expected Qwen attention projection modules not found")
    encoder.add_adapter(LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, target_modules=targets, bias="none",
    ))
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.config.use_cache = False
    trainable = [(name, p) for name, p in encoder.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise ValueError("Expected only LoRA parameters to be trainable")
    # Keep small adapter weights/gradients/Adam states in FP32.
    for _, parameter in trainable:
        parameter.data = parameter.data.float()
    objective = Objective(encoder)
    dataset = Groups()
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, generator=generator,
                        num_workers=0, collate_fn=first_item)
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=args.lr, weight_decay=0.0)
    objective, optimizer, loader = accelerator.prepare(objective, optimizer, loader)
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=min(total_steps - 1, math.ceil(total_steps * args.warmup_ratio)),
        num_training_steps=total_steps,
    )
    raw = accelerator.unwrap_model(objective)

    def tokenize(text):
        # Direct tokenizer call matches the existing retrieval script and
        # bypasses all implicit ST query/document prompt routing.
        encoded = raw.encoder.tokenizer(
            [text], padding=True, truncation=True, max_length=args.max_length,
            return_tensors="pt", add_special_tokens=True,
        )
        return {k: v.to(accelerator.device) for k, v in encoded.items()}

    probe = select_group(
        rows[0], 0, args.seed, args.hard_negatives, args.ordinary_negatives,
        args.query_instruction,
    )
    def probe_vectors():
        raw.eval()
        with torch.inference_mode(), accelerator.autocast():
            return torch.cat([raw.embeddings(tokenize(t)) for t in probe["texts"]]).cpu()

    # Before training check ST Normalize output versus explicit FP32 path and
    # disabled initial LoRA versus initialized LoRA in eval mode.
    initialized = probe_vectors()
    raw.encoder.disable_adapters()
    base_vectors = probe_vectors()
    raw.encoder.enable_adapters()
    initial_cos = F.cosine_similarity(initialized, base_vectors).min().item()
    if initial_cos < 0.9999:
        raise RuntimeError(f"Fresh adapter changed eval embeddings: cosine={initial_cos}")
    with torch.inference_mode(), accelerator.autocast():
        st_vector = raw.encoder(tokenize(probe["texts"][0]))["sentence_embedding"].float()
    st_cos = F.cosine_similarity(st_vector.cpu(), initialized[:1]).item()
    if st_cos < 0.999:
        raise RuntimeError(f"ST pipeline versus FP32 normalization differs: cosine={st_cos}")
    versions = {p: importlib.metadata.version(p) for p in (
        "torch", "transformers", "sentence-transformers", "peft", "accelerate"
    )}
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"versions": versions, "world_size": accelerator.num_processes,
                   "num_queries": len(rows), "steps_per_epoch": steps_per_epoch,
                   "planned_updates": total_steps, "trainable_parameters": sum(p.numel() for _, p in trainable),
                   "train_json_sha256": hashlib.sha256(args.train_json.read_bytes()).hexdigest(),
                   "initial_adapter_cosine": initial_cos, "st_pipeline_cosine": st_cos,
                   "microbatch_queries_per_rank": 1,
                   "effective_query_groups_per_full_update": args.grad_accum * accelerator.num_processes,
                   "distributed_padding": "Accelerate may repeat final samples to equalize ranks; never expands by positive count",
                   "precision": "BF16 frozen base; FP32 LoRA, normalization, loss and Adam states",
                   "checkpoint_policy": "every complete epoch plus final; ST LoRA snapshots only",
                   "resume_supported": False})
    if accelerator.is_main_process:
        (out / "training_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    accelerator.print(json.dumps(config, ensure_ascii=False, indent=2))
    before = {name: p.detach().cpu().clone() for name, p in trainable}
    torch.cuda.reset_peak_memory_stats(accelerator.device)
    start, step = time.monotonic(), 0
    optimizer.zero_grad(set_to_none=True)
    raw.train()
    for epoch in range(args.epochs):
        dataset.epoch = epoch
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)
        epoch_loss_sum = torch.zeros((), device=accelerator.device, dtype=torch.float32)
        epoch_microbatches = 0
        step_loss_sum = torch.zeros((), device=accelerator.device, dtype=torch.float32)
        step_microbatches = 0
        for micro_index, sample in enumerate(loader):
            features = [tokenize(text) for text in sample["texts"]]
            with accelerator.accumulate(objective):
                loss = objective(features)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite loss before backward")
                # Keep the unscaled loss for reporting. Accelerate may scale
                # the loss internally for gradient accumulation, but the log
                # should describe the actual contrastive loss value.
                loss_value = loss.detach().float()
                step_loss_sum += loss_value
                step_microbatches += 1
                epoch_loss_sum += loss_value
                epoch_microbatches += 1
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    active = [p for _, p in trainable if p.grad is not None]
                    if not active or any(not torch.isfinite(p.grad).all() for p in active):
                        raise FloatingPointError("Missing/non-finite LoRA gradients")
                    norm = accelerator.clip_grad_norm_([p for _, p in trainable], 1.0)
                    if not torch.isfinite(norm) or norm.item() == 0:
                        raise FloatingPointError("Zero/non-finite LoRA gradient norm")
                if accelerator.sync_gradients:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                step += 1
                if any(not torch.isfinite(p).all() for _, p in trainable):
                    raise FloatingPointError("Non-finite adapter weights after update")
                # All ranks participate even when only rank 0 writes logs.
                # The final accumulation window can contain fewer than
                # grad_accum microbatches, so divide by the actual count.
                step_sum = accelerator.reduce(step_loss_sum, reduction="sum")
                step_count_local = torch.tensor(
                    float(step_microbatches), device=accelerator.device
                )
                step_count = accelerator.reduce(step_count_local, reduction="sum")
                optimizer_step_loss_mean = (step_sum / step_count).item()
                logged_microbatches_per_rank = step_microbatches
                logged_microbatches = int(step_count.item())
                step_loss_sum.zero_()
                step_microbatches = 0
                event = {"type": "optimizer_step", "step": step, "epoch": epoch + 1,
                         "microbatches_per_rank": logged_microbatches_per_rank,
                         "microbatches_global": logged_microbatches,
                         "optimizer_step_loss_mean": optimizer_step_loss_mean,
                         "lr": scheduler.get_last_lr()[0],
                         "grad_norm_local": float(norm),
                         "elapsed_seconds": time.monotonic() - start}
                if step % args.log_every == 0 or step == 1:
                    accelerator.print(json.dumps(event))
                if accelerator.is_main_process:
                    with (out / "training_log.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps(event) + "\n")
                if step >= total_steps:
                    break
        # This is the mean of all microbatch losses actually processed in the
        # epoch. If max_steps stops the epoch early, the event is explicitly a
        # partial epoch summary rather than being presented as a full epoch.
        epoch_sum = accelerator.reduce(epoch_loss_sum, reduction="sum")
        epoch_count_local = torch.tensor(
            float(epoch_microbatches), device=accelerator.device
        )
        epoch_count = accelerator.reduce(epoch_count_local, reduction="sum")
        epoch_event = {
            "type": "epoch_summary",
            "epoch": epoch + 1,
            "microbatches_per_rank": epoch_microbatches,
            "microbatches_global": int(epoch_count.item()),
            "epoch_loss_mean": (epoch_sum / epoch_count).item(),
            "complete_epoch": epoch_microbatches == len(loader),
            "elapsed_seconds": time.monotonic() - start,
        }
        accelerator.print(json.dumps(epoch_event))
        if accelerator.is_main_process:
            with (out / "training_log.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(epoch_event) + "\n")
        if epoch_event["complete_epoch"]:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                checkpoint_path = out / f"epoch_{epoch + 1:03d}"
                save_encoder_checkpoint(raw.encoder, checkpoint_path, args.model, {
                    **epoch_event,
                    "optimizer_step": step,
                    "query_instruction": args.query_instruction,
                    "max_length": args.max_length,
                    "train_json_sha256": config["train_json_sha256"],
                })
                accelerator.print(f"SAVED EPOCH CHECKPOINT: {checkpoint_path}")
            accelerator.wait_for_everyone()
        if step >= total_steps:
            break
    accelerator.wait_for_everyone()
    changed = any(not torch.equal(before[n], p.detach().cpu()) for n, p in trainable)
    if not changed:
        raise RuntimeError("Training finished without changing adapter weights")
    expected_vectors = probe_vectors()
    if accelerator.is_main_process:
        save_encoder_checkpoint(raw.encoder, out / "final", args.model, {
            "epoch": epoch + 1,
            "optimizer_step": step,
            "complete_epoch": epoch_event["complete_epoch"],
            "epoch_loss_mean": epoch_event["epoch_loss_mean"],
            "query_instruction": args.query_instruction,
            "max_length": args.max_length,
            "train_json_sha256": config["train_json_sha256"],
        })
    memory = torch.tensor([torch.cuda.max_memory_allocated(accelerator.device) / 1024**3,
                           torch.cuda.max_memory_reserved(accelerator.device) / 1024**3], device=accelerator.device)
    memory_all = accelerator.gather(memory.unsqueeze(0)).cpu().tolist()
    result = {"updates": step, "elapsed_seconds": time.monotonic() - start,
              "adapter_weights_changed": changed, "peak_allocated_reserved_gib_by_rank": memory_all,
              "reload_check": "not_requested", "evaluation": "No retrieval metrics computed"}
    accelerator.wait_for_everyone()
    if args.reload_check:
        # Free optimizer/model references before loading another 8B copy.
        accelerator.free_memory()
        del objective, optimizer, scheduler, loader, raw, encoder, base, trainable, active
        del features, loss, initialized, base_vectors, st_vector
        gc.collect()
        torch.cuda.empty_cache()
        reloaded = load_encoder(out / "final", accelerator.device, args.max_length)
        reloaded.eval()
        vectors = []
        with torch.inference_mode(), accelerator.autocast():
            for text in probe["texts"]:
                tokens = reloaded.tokenizer([text], padding=True, truncation=True,
                    max_length=args.max_length, return_tensors="pt", add_special_tokens=True)
                tokens = {k: v.to(accelerator.device) for k, v in tokens.items()}
                pooled = reloaded[1](reloaded[0](tokens))["sentence_embedding"].float()
                vectors.append(F.normalize(pooled, p=2, dim=-1).cpu())
        actual = torch.cat(vectors)
        if not torch.isfinite(actual).all():
            raise FloatingPointError("Non-finite reloaded embeddings")
        cosine = F.cosine_similarity(expected_vectors, actual).min().item()
        if cosine < 0.9999:
            raise RuntimeError(f"Saved/reloaded adapter mismatch: cosine={cosine}")
        result.update({"reload_check": "passed", "reload_min_cosine": cosine})
    if accelerator.is_main_process:
        (out / "training_result.json").write_text(json.dumps(result, indent=2) + "\n")
    accelerator.wait_for_everyone()
    accelerator.print("COMPLETED " + json.dumps(result))
    accelerator.end_training()


def main():
    args = parse_args()
    rows = read_groups(args.train_json, args.hard_negatives, args.ordinary_negatives)
    if args.limit_queries:
        rows = rng_for(args.seed, "subset").sample(rows, min(args.limit_queries, len(rows)))
    if args.data_only:
        print(json.dumps({"num_queries": len(rows),
            "domains": dict(Counter(r["domain"] for r in rows)),
            "negative_statuses": dict(Counter(
                d["label_status"]
                for r in rows
                for field in ("hard_negatives", "ordinary_negatives")
                for d in r[field]
            )),
            "first_selection": {k: v for k, v in select_group(
                rows[0], 0, args.seed, args.hard_negatives, args.ordinary_negatives,
                args.query_instruction).items() if k != "texts"}
        }, ensure_ascii=False, indent=2))
        return
    if not args.model.is_dir():
        raise ValueError(f"Local model directory missing: {args.model}")
    run(args, rows)


if __name__ == "__main__":
    main()
