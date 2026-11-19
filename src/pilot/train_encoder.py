"""Full XLM-R fine-tuning on the saved 4,000-row development pilot."""
import argparse
import math
import os
from pathlib import Path
import random
import time

from .common import ROOT, RunLock, RunStore, digest, file_hash, read_json, write_json
from .evaluate import finish
from .inputs import align_first_subwords, first_word_predictions, load_panel, merge_window_tags
from .models import cache_dir, load_tokenizer
from .scoring import bio_spans


def features(task, rows, tokenizer, label2id):
    out = []
    for row in rows:
        if task == "sa":
            encoded = dict(tokenizer(row["visible_text"], truncation=False))
            encoded["labels"] = int(row["label"])
            out.append(encoded)
        else:
            for wi, window in enumerate(row["windows"]):
                encoded = tokenizer(window["tokens"], is_split_into_words=True, truncation=False)
                aligned = align_first_subwords(encoded.word_ids(), row["labels"][window["start"]:window["end"]], label2id)
                # Supervise each original word exactly once despite overlapping context.
                for i, word in enumerate(encoded.word_ids()):
                    if word is not None and row["owners"][window["start"] + word] != wi:
                        aligned[i] = -100
                if any(x != -100 for x in aligned):
                    out.append({**dict(encoded), "labels": aligned})
    if any(len(x["input_ids"]) > 384 for x in out):
        raise ValueError("Shared input exceeded frozen encoder budget")
    return out


def collate(task, batch, tokenizer, device):
    import torch
    labels = [r["labels"] for r in batch]
    inputs = tokenizer.pad([{k: v for k, v in r.items() if k != "labels"} for r in batch],
                           padding=True, return_tensors="pt")
    if task == "ner":
        width = inputs["input_ids"].shape[1]
        labels = [x + [-100] * (width - len(x)) for x in labels]
    inputs["labels"] = torch.tensor(labels, dtype=torch.long)
    return {k: v.to(device) for k, v in inputs.items()}


def save_checkpoint(path, model, optimizer, scheduler, progress, run_hash):
    import numpy as np
    import torch
    state = {"run_hash": run_hash, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "progress": progress,
             "random": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
             "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def train(task, model, tokenizer, rows, label2id, config, store, device):
    import numpy as np
    import torch
    from transformers import get_linear_schedule_with_warmup
    dataset = features(task, rows, tokenizer, label2id)
    batches = math.ceil(len(dataset) / config["microbatch"])
    updates_per_epoch = math.ceil(batches / config["gradient_accumulation"])
    total_updates = updates_per_epoch * config["epochs"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_updates * config["warmup_ratio"]), total_updates)
    checkpoint = store.path / "checkpoint.pt"
    progress = {"epoch": 0, "next_batch": 0, "updates": 0, "active_training_seconds": 0., "loss_sum": 0., "loss_batches": 0,
                "peak_gpu_allocated_bytes": 0, "peak_gpu_reserved_bytes": 0}
    if checkpoint.exists():
        # Only a checkpoint created in this local run directory; never load arbitrary downloaded pickle.
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["run_hash"] != store.hash:
            raise ValueError("Checkpoint identity mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        progress = state["progress"]
        random.setstate(state["random"]); np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if state["cuda"] is not None:
            torch.cuda.set_rng_state_all(state["cuda"])
        del state
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    segment_start = time.perf_counter()
    for epoch in range(progress["epoch"], config["epochs"]):
        indices = list(range(len(dataset)))
        random.Random(config["seed"] + epoch).shuffle(indices)
        start_batch = progress["next_batch"] if epoch == progress["epoch"] else 0
        for bi in range(start_batch, batches):
            positions = indices[bi * config["microbatch"]:(bi + 1) * config["microbatch"]]
            batch = collate(task, [dataset[i] for i in positions], tokenizer, device)
            group_start = (bi // config["gradient_accumulation"]) * config["gradient_accumulation"]
            group_size = min(config["gradient_accumulation"], batches - group_start)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss; run stopped rather than silently continued")
            (loss / group_size).backward()
            progress["loss_sum"] += loss.detach().item()
            progress["loss_batches"] += 1
            if (bi + 1) % config["gradient_accumulation"] == 0 or bi + 1 == batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                progress.update(epoch=epoch, next_batch=bi + 1, updates=progress["updates"] + 1)
                should_save = progress["updates"] % config["checkpoint_every_updates"] == 0 or bi + 1 == batches
                if should_save:
                    if device == "cuda":
                        torch.cuda.synchronize()
                        progress["peak_gpu_allocated_bytes"] = max(progress["peak_gpu_allocated_bytes"], torch.cuda.max_memory_allocated())
                        progress["peak_gpu_reserved_bytes"] = max(progress["peak_gpu_reserved_bytes"], torch.cuda.max_memory_reserved())
                    progress["active_training_seconds"] += time.perf_counter() - segment_start
                    save_checkpoint(checkpoint, model, optimizer, scheduler, progress, store.hash)
                    segment_start = time.perf_counter()
                    print({"task": task, "epoch": epoch + 1, "updates": progress["updates"], "total_updates": total_updates,
                           "mean_loss": progress["loss_sum"] / progress["loss_batches"]}, flush=True)
        progress.update(epoch=epoch + 1, next_batch=0)
    final_dir = store.path / "model"
    model.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)
    write_json(store.path / "trained.json", {"run_hash": store.hash, "progress": progress,
               "training_features": len(dataset), "model_sha256": {p.name: file_hash(p) for p in final_dir.glob("*.safetensors")}})
    del optimizer, scheduler
    if device == "cuda":
        torch.cuda.empty_cache()


def predict_one(task, row, model, tokenizer, labels, device):
    import torch
    raw = []
    if task == "sa":
        batch = tokenizer(row["visible_text"], return_tensors="pt", truncation=False).to(device)
        with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(**batch).logits.float()[0].cpu().tolist()
        label = max(range(2), key=lambda i: logits[i])
        return {"label": label, "failed": False, "errors": []}, {"logits": logits}
    for window in row["windows"]:
        encoded = tokenizer(window["tokens"], is_split_into_words=True, return_tensors="pt", truncation=False)
        word_ids = encoded.word_ids()
        batch = encoded.to(device)
        with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            pred = model(**batch).logits.argmax(-1)[0].cpu().tolist()
        words = first_word_predictions(word_ids, pred, len(window["tokens"]))
        raw.append([labels[i] for i in words])
    tags = merge_window_tags(row["windows"], raw, len(row["tokens"]))
    spans, repairs = bio_spans(tags, repair=True)
    return {"spans": spans, "invalid_candidates": 0, "failed": False, "repairs": repairs}, {"word_tags": tags, "window_tags": raw}


def run(task, run_dir, config_path, cpu=False):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, set_seed
    config = read_json(config_path)
    if config["stage"] != "development_only" or config["precision"] != "bf16":
        raise ValueError("Only the documented BF16 development configuration is supported")
    device = "cpu" if cpu else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --cpu only for an explicit CPU run")
    if device == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unavailable; do not silently change the configuration")
    set_seed(config["seed"])
    torch.use_deterministic_algorithms(config["strict_determinism"], warn_only=False)
    torch.backends.cudnn.benchmark = False
    rows, dev = load_panel(task, "train"), load_panel(task, "valid")
    labels = read_json(ROOT / "data/derived/ner_labels.json")["bio"] if task == "ner" else ["negative", "positive"]
    label2id = {v: i for i, v in enumerate(labels)}
    spec = read_json(ROOT / "configs/models.lock.json")["encoder"]
    identity = {"task": task, "system": "xlmr", "config": config, "model": spec, "device": device,
                "shared_inputs_sha256": file_hash(ROOT / "data/pilot/manifest.json"), "train_sha256": digest(rows),
                "panel_sha256": digest(dev), "panel_ids": [r["id"] for r in dev], "labels": labels}
    store = RunStore(run_dir, identity)
    with RunLock(store.path):
        started = time.perf_counter()
        tokenizer = load_tokenizer("encoder")
        klass = AutoModelForTokenClassification if task == "ner" else AutoModelForSequenceClassification
        trained_path = store.path / "trained.json"
        if trained_path.exists():
            trained = read_json(trained_path)
            if trained["run_hash"] != store.hash:
                raise ValueError("Trained model identity mismatch")
            for name, sha in trained["model_sha256"].items():
                if file_hash(store.path / "model" / name) != sha:
                    raise ValueError("Trained weights checksum mismatch")
            model = klass.from_pretrained(store.path / "model", local_files_only=True)
        else:
            model = klass.from_pretrained(spec["model_id"], revision=spec["revision"], cache_dir=cache_dir(),
                    num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=label2id,
                    use_safetensors=True, local_files_only=True, attn_implementation="sdpa")
        model.to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        write_json(store.path / f"load_{time.time_ns()}.json", {"model_load_seconds": time.perf_counter() - started,
                     "device": device, "gpu": torch.cuda.get_device_name() if device == "cuda" else None})
        if not trained_path.exists():
            train(task, model, tokenizer, rows, label2id, config, store, device)
        model.eval()
        model.gradient_checkpointing_disable()
        predict_one(task, dev[0], model, tokenizer, labels, device)  # discarded warmup
        for index, row in enumerate(dev):
            if store.get(row["id"]) is not None:
                continue
            if device == "cuda":
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            parsed, raw = predict_one(task, row, model, tokenizer, labels, device)
            if device == "cuda":
                torch.cuda.synchronize()
            store.put({"id": row["id"], "input_sha256": digest(row), "parsed": parsed, "raw": raw,
                       "seconds": time.perf_counter() - start,
                       "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
                       "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else 0})
            if (index + 1) % 25 == 0:
                print({"task": task, "predictions": index + 1, "total": len(dev)}, flush=True)
        finish(store, task, dev)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["ner", "sa"], required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/encoder.json")
    p.add_argument("--cpu", action="store_true")
    a = p.parse_args()
    try:
        run(a.task, a.run, a.config, a.cpu)
    except Exception as error:
        write_json(a.run / f"failure_{time.time_ns()}.json", {"error_type": type(error).__name__, "error": str(error)})
        raise
