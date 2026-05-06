#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import sys
import time
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT_NODE_LABELS = [
    "State: Nominal",
    "O-CU-CP: Cell Procedure Management",
    "O-CU-CP: UE Procedure Management",
    "O-CU-UP: eGTPU",
    "Procedure: SCTP Association",
    "Protocol: GTP-U/F1-U",
    "Protocol: SCTP",
]


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def import_from_path(module_name: str, path: str | Path):
    p = Path(path).resolve()
    if str(p.parent) not in sys.path:
        sys.path.insert(0, str(p.parent))
    spec = importlib.util.spec_from_file_location(module_name, p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def safe_torch_load(path: str | Path, map_location: Any):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def discover_ckpts(snapshot_dir: str | Path) -> list[Path]:
    ckpts = sorted(Path(snapshot_dir).glob("*.pt"), key=lambda p: natural_key(p.name))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints found in {snapshot_dir}")
    return ckpts


def infer_step(path: Path, ordinal: int) -> int:
    m = re.search(r"step_(\d+)", path.stem)
    return int(m.group(1)) if m else ordinal


def read_vocab(path: str | Path) -> tuple[int, int]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    vocab = obj["vocab"]
    return vocab.index("[PAD]"), vocab.index("[EOS]")


def load_model(transformer_script: str | Path, ckpt_path: str | Path, device: torch.device):
    mod = import_from_path("rc_probe_transformer", transformer_script)
    ckpt = safe_torch_load(ckpt_path, device)
    cfg_raw = ckpt.get("cfg") or ckpt.get("config") or ckpt.get("model_cfg")
    if cfg_raw is None:
        raise KeyError(f"{ckpt_path} has no cfg/config/model_cfg")
    if is_dataclass(cfg_raw):
        cfg_raw = asdict(cfg_raw)
    elif not isinstance(cfg_raw, dict):
        cfg_raw = vars(cfg_raw)
    cfg_keys = {f.name for f in fields(mod.GPTConfig)}
    cfg = mod.GPTConfig(**{k: v for k, v in cfg_raw.items() if k in cfg_keys})
    model = mod.TinyGPT(cfg).to(device)
    state = ckpt.get("model") or ckpt.get("state_dict") or ckpt.get("model_state_dict")
    if state is None:
        raise KeyError(f"{ckpt_path} has no model state")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg, ckpt


class RowDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def collate_rows(batch: list[dict[str, Any]], *, ctx_len: int, pad_id: int, eos_id: int):
    lengths = [min(ctx_len, len(r["tokens"]), len(r["token_times"])) for r in batch]
    L = max(lengths)
    x = torch.full((len(batch), L), pad_id, dtype=torch.long)
    tt = torch.zeros((len(batch), L), dtype=torch.float32)
    eos_pos = torch.zeros(len(batch), dtype=torch.long)
    meta = []
    for i, (r, n) in enumerate(zip(batch, lengths)):
        toks = list(r["tokens"][:n])
        tms = [float(v) for v in r["token_times"][:n]]
        x[i, :n] = torch.tensor(toks, dtype=torch.long)
        tt[i, :n] = torch.tensor(tms, dtype=torch.float32)
        matches = [j for j, tok in enumerate(toks) if int(tok) == eos_id]
        eos_pos[i] = matches[-1] if matches else max(0, n - 1)
        meta.append((r["scenario_id"], r["test_id"]))
    return x, tt, eos_pos, meta


def make_feature_batches(
    rows: list[dict[str, Any]],
    *,
    ctx_len: int,
    pad_id: int,
    eos_id: int,
    batch_size: int,
    sort_by_length: bool,
    pin_memory: bool,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[str, str]]]]:
    indexed: list[tuple[int, int, dict[str, Any]]] = []
    for idx, r in enumerate(rows):
        indexed.append((idx, min(ctx_len, len(r["tokens"]), len(r["token_times"])), r))
    if sort_by_length:
        indexed.sort(key=lambda item: item[1], reverse=True)

    batches = []
    for start in range(0, len(indexed), batch_size):
        batch = [item[2] for item in indexed[start:start + batch_size]]
        x, tt, eos_pos, meta = collate_rows(batch, ctx_len=ctx_len, pad_id=pad_id, eos_id=eos_id)
        if pin_memory:
            x = x.pin_memory()
            tt = tt.pin_memory()
            eos_pos = eos_pos.pin_memory()
        batches.append((x, tt, eos_pos, meta))
    return batches


@torch.inference_mode()
def layer1_eos_representations(
    model,
    rows: list[dict[str, Any]],
    *,
    probe_layer: int,
    pad_id: int,
    eos_id: int,
    ctx_len: int,
    batch_size: int,
    device: torch.device,
    amp: bool,
) -> dict[tuple[str, str], torch.Tensor]:
    loader = DataLoader(
        RowDataset(rows),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_rows(b, ctx_len=ctx_len, pad_id=pad_id, eos_id=eos_id),
        num_workers=0,
    )
    out: dict[tuple[str, str], torch.Tensor] = {}
    for x, tt, eos_pos, meta in loader:
        x = x.to(device)
        tt = tt.to(device)
        eos_pos = eos_pos.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(amp and device.type == "cuda")):
            h = model.tok(x)
            if getattr(model, "pos", None) is not None:
                pos = torch.arange(x.shape[1], device=device)
                h = h + model.pos(pos)[None, :, :]
            h = model.drop(h)
            tt = model._prep_times(tt)
            if probe_layer < 0 or probe_layer > len(model.blocks):
                raise ValueError(f"probe_layer={probe_layer} is outside 0..{len(model.blocks)}")
            for li in range(probe_layer):
                h = model.blocks[li](h, tt)
            if probe_layer == len(model.blocks):
                h = model.norm(h)
        picked = h[torch.arange(h.shape[0], device=device), eos_pos].detach().float().cpu()
        for key, vec in zip(meta, picked):
            out[(str(key[0]), str(key[1]))] = vec
    return out


@torch.inference_mode()
def layer1_eos_representations_from_batches(
    model,
    batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[str, str]]]],
    *,
    probe_layer: int,
    device: torch.device,
    amp: bool,
) -> dict[tuple[str, str], torch.Tensor]:
    out: dict[tuple[str, str], torch.Tensor] = {}
    if probe_layer < 0 or probe_layer > len(model.blocks):
        raise ValueError(f"probe_layer={probe_layer} is outside 0..{len(model.blocks)}")

    for x, tt, eos_pos, meta in batches:
        x = x.to(device, non_blocking=True)
        tt = tt.to(device, non_blocking=True)
        eos_pos = eos_pos.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(amp and device.type == "cuda")):
            h = model.tok(x)
            if getattr(model, "pos", None) is not None:
                pos = torch.arange(x.shape[1], device=device)
                h = h + model.pos(pos)[None, :, :]
            h = model.drop(h)
            tt = model._prep_times(tt)
            if probe_layer == 1:
                h = model.blocks[0](h, tt)
            else:
                for li in range(probe_layer):
                    h = model.blocks[li](h, tt)
                if probe_layer == len(model.blocks):
                    h = model.norm(h)
        picked = h[torch.arange(h.shape[0], device=device), eos_pos].detach().float().cpu()
        for key, vec in zip(meta, picked):
            out[(str(key[0]), str(key[1]))] = vec
    return out


def scenario_matrix(
    reps: dict[tuple[str, str], torch.Tensor],
    scenario_ids: list[str],
    test_ids: list[str],
    labels_by_scenario: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for sid in scenario_ids:
        parts = []
        for tid in test_ids:
            key = (sid, tid)
            if key not in reps:
                raise KeyError(f"missing representation for scenario={sid} test={tid}")
            parts.append(reps[key])
        xs.append(torch.cat(parts, dim=0))
        ys.append(labels_by_scenario[sid])
    return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)


def single_test_matrix(
    reps: dict[tuple[str, str], torch.Tensor],
    scenario_ids: list[str],
    test_ids: list[str],
    labels_by_scenario: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for sid in scenario_ids:
        for tid in test_ids:
            key = (sid, tid)
            if key not in reps:
                raise KeyError(f"missing representation for scenario={sid} test={tid}")
            xs.append(reps[key])
            ys.append(labels_by_scenario[sid])
    return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)


def stratified_splits(y: torch.Tensor, *, seed: int, train_frac: float, val_frac: float) -> dict[str, list[int]]:
    rng = __import__("random").Random(seed)
    buckets: dict[int, list[int]] = {}
    for i, label in enumerate(y.tolist()):
        buckets.setdefault(int(label), []).append(i)
    train, val, test = [], [], []
    for label in sorted(buckets):
        idxs = list(buckets[label])
        rng.shuffle(idxs)
        n = len(idxs)
        if n == 1:
            train.extend(idxs)
            continue
        if n == 2:
            train.extend(idxs[:1])
            test.extend(idxs[1:])
            continue
        n_test = max(1, int(round((1.0 - train_frac - val_frac) * n)))
        n_val = max(1, int(round(val_frac * n))) if n >= 4 and val_frac > 0 else 0
        n_train = n - n_val - n_test
        while n_train < 1:
            if n_val > 0:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            n_train = n - n_val - n_test
        train.extend(idxs[:n_train])
        val.extend(idxs[n_train:n_train + n_val])
        test.extend(idxs[n_train + n_val:])
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def spectral_k(X: torch.Tensor, var_threshold: float) -> int:
    X = X.float()
    if X.shape[0] < 2:
        return 0
    X = X - X.mean(dim=0, keepdim=True)
    eig = torch.linalg.eigvalsh(X.T @ X).clamp_min(0).flip(0).cpu().numpy()
    total = float(eig.sum())
    if total <= 0 or not np.isfinite(total):
        return 0
    return int(np.searchsorted(np.cumsum(eig) / total, var_threshold) + 1)


def train_probe(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    n_classes: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    standardize: bool = True,
    train_indices: list[int] | None = None,
    val_indices: list[int] | None = None,
    test_indices: list[int] | None = None,
    batch_size: int = 1024,
    eval_every: int = 5,
    early_stop_patience: int = 50,
) -> dict[str, float]:
    torch.manual_seed(seed)
    if standardize:
        mu = X_train.mean(dim=0, keepdim=True)
        sd = X_train.std(dim=0, keepdim=True).clamp_min(1e-6)
        X_train = (X_train - mu) / sd
        X_val = (X_val - mu) / sd
    X_train = X_train.to(device)
    X_val = X_val.to(device)
    y_train = y_train.to(device)
    y_val = y_val.to(device)

    probe = nn.Linear(X_train.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    if train_indices is None:
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(probe(X_train), y_train)
            loss.backward()
            opt.step()
    else:
        X_all = X_train
        y_all = y_train
        train_idx = torch.tensor(train_indices, dtype=torch.long, device=device)
        val_idx = torch.tensor(val_indices or train_indices, dtype=torch.long, device=device)
        g = torch.Generator(device="cpu")
        best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
        best_val = float("inf")
        bad_epochs = 0
        for epoch in range(1, epochs + 1):
            g.manual_seed(seed + epoch)
            order = torch.tensor(train_indices, dtype=torch.long)[torch.randperm(len(train_indices), generator=g)]
            for start in range(0, len(order), batch_size):
                idx = order[start:start + batch_size].to(device)
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(probe(X_all[idx]), y_all[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
                opt.step()
            if epoch % eval_every == 0 or epoch == epochs:
                with torch.no_grad():
                    val_loss = F.cross_entropy(probe(X_all[val_idx]), y_all[val_idx]).item()
                if val_loss < best_val - 1e-7:
                    best_val = val_loss
                    best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
                    bad_epochs = 0
                else:
                    bad_epochs += eval_every
                if early_stop_patience > 0 and bad_epochs >= early_stop_patience:
                    break
        probe.load_state_dict(best_state)

    with torch.no_grad():
        train_logits = probe(X_train)
        val_logits = probe(X_val)
        eval_logits = val_logits
        eval_y = y_val
        if train_indices is not None and test_indices is not None:
            test_idx = torch.tensor(test_indices, dtype=torch.long, device=device)
            eval_logits = probe(X_train[test_idx])
            eval_y = y_train[test_idx]
        val_probs = val_logits.softmax(dim=-1)
        eval_probs = eval_logits.softmax(dim=-1)
        true_probs = eval_probs[torch.arange(eval_y.numel(), device=device), eval_y]
        pred = eval_probs.argmax(dim=-1)
        if train_indices is not None:
            tr_idx = torch.tensor(train_indices, dtype=torch.long, device=device)
            train_acc = (train_logits[tr_idx].argmax(dim=-1) == y_train[tr_idx]).float().mean().item()
        else:
            train_acc = (train_logits.argmax(dim=-1) == y_train).float().mean().item()
        val_acc = (pred == eval_y).float().mean().item()
        val_loss = F.cross_entropy(eval_logits, eval_y).item()
    return {
        "probe_train_top1_accuracy": train_acc,
        "probe_val_top1_accuracy": val_acc,
        "probe_val_true_class_probability": true_probs.mean().item(),
        "probe_val_ce_loss": val_loss,
    }


def write_csv(rows: list[dict[str, Any]], path: Path):
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    t0 = time.perf_counter()
    p = argparse.ArgumentParser(description="Train scenario-level RC probes from layer-1 EOS test representations.")
    p.add_argument("--data", default="rc_dataset_v1_tokenized.parquet")
    p.add_argument("--vocab", default="tokenizer_vocab_locked_v1.json")
    p.add_argument("--transformer-script", default="train_transformer_v0_seqtime_probe_ckpts.py")
    p.add_argument("--snapshot-dir", required=True)
    p.add_argument("--split-json", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--ctx-len", type=int, default=1024)
    p.add_argument("--probe-layer", type=int, default=1, help="0=embedding stream, 1=post block 1, ...")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--probe-epochs", type=int, default=600)
    p.add_argument("--probe-lr", type=float, default=1e-2)
    p.add_argument("--probe-weight-decay", type=float, default=1e-4)
    p.add_argument("--single-test", action="store_true", help="train Linear(d_model,n_faults) on one scenario-test row at a time")
    p.add_argument("--orchestrator-split", action="store_true", help="ignore split-json train/val split and make 80/10/10 stratified scenario splits")
    p.add_argument("--no-standardize", action="store_true", help="train probe on raw features")
    p.add_argument("--probe-batch-size", type=int, default=1024)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--early-stop-patience", type=int, default=50)
    p.add_argument("--var-threshold", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--tf32", action="store_true", help="allow TF32 matmul/cuDNN for faster no-AMP CUDA inference/probes")
    p.add_argument("--deterministic", action="store_true", help="request deterministic CUDA algorithms for same-machine reruns")
    p.add_argument("--deterministic-warn-only", action="store_true", help="warn instead of erroring on nondeterministic ops")
    p.add_argument("--empty-cache-each", action="store_true", help="empty CUDA cache after each checkpoint")
    p.add_argument("--no-cache-feature-batches", action="store_true", help="disable prebuilt featurization batches")
    p.add_argument("--no-sort-feature-batches", action="store_true", help="preserve parquet row order instead of batching by similar lengths")
    p.add_argument("--pin-feature-batches", action="store_true", help="pin cached CPU feature batches before CUDA transfer")
    args = p.parse_args()

    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=args.deterministic_warn_only)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)

    pad_id, eos_id = read_vocab(args.vocab)
    df = pd.read_parquet(args.data)
    test_ids = sorted(df["test_id"].unique())
    label_to_id = {label: i for i, label in enumerate(ROOT_NODE_LABELS)}
    labels_by_scenario = df.drop_duplicates("scenario_id").set_index("scenario_id")["scenario_label"].map(label_to_id).to_dict()
    if args.orchestrator_split:
        all_sids = sorted(labels_by_scenario.keys())
        y_all = torch.tensor([labels_by_scenario[sid] for sid in all_sids], dtype=torch.long)
        split_idx = stratified_splits(y_all, seed=args.seed, train_frac=0.80, val_frac=0.10)
        train_sids = [all_sids[i] for i in split_idx["train"]]
        val_sids = [all_sids[i] for i in split_idx["val"]]
        test_sids = [all_sids[i] for i in split_idx["test"]]
    else:
        split = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
        train_sids = [str(s) for s in split["train_scenario_ids"]]
        val_sids = [str(s) for s in split["val_scenario_ids"]]
        test_sids = []
        df = df[df["scenario_id"].isin(set(train_sids + val_sids))].copy()
        missing_labels = [sid for sid in train_sids + val_sids if sid not in labels_by_scenario]
        if missing_labels:
            missing = set(missing_labels)
            train_sids = [sid for sid in train_sids if sid not in missing]
            val_sids = [sid for sid in val_sids if sid not in missing]
            print(f"[!] skipped {len(missing)} split scenarios not present in {args.data}")

    rows = df[["scenario_id", "test_id", "tokens", "token_times"]].to_dict("records")
    feature_batches = None
    if not args.no_cache_feature_batches:
        batch_t0 = time.perf_counter()
        feature_batches = make_feature_batches(
            rows,
            ctx_len=args.ctx_len,
            pad_id=pad_id,
            eos_id=eos_id,
            batch_size=args.batch_size,
            sort_by_length=not args.no_sort_feature_batches,
            pin_memory=args.pin_feature_batches,
        )
        max_batch_len = max(int(x.shape[1]) for x, _tt, _eos, _meta in feature_batches)
        mean_batch_len = sum(int(x.shape[1]) for x, _tt, _eos, _meta in feature_batches) / max(1, len(feature_batches))
        print(
            f"feature_batches={len(feature_batches)} build_sec={time.perf_counter() - batch_t0:.3f} "
            f"max_batch_len={max_batch_len} mean_batch_len={mean_batch_len:.1f}"
        )
    ckpts = discover_ckpts(args.snapshot_dir)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device={device} checkpoints={len(ckpts)} train_scenarios={len(train_sids)} val_scenarios={len(val_sids)} test_scenarios={len(test_sids)}")
    print(f"tests={len(test_ids)} feature_dim={len(test_ids)}*128")
    print(
        f"fast_options=amp={args.amp} tf32={args.tf32} deterministic={args.deterministic} "
        f"empty_cache_each={args.empty_cache_each} cache_feature_batches={feature_batches is not None} "
        f"sort_feature_batches={not args.no_sort_feature_batches} pin_feature_batches={args.pin_feature_batches}"
    )

    records: list[dict[str, Any]] = []
    for i, ckpt in enumerate(ckpts, start=1):
        ckpt_t0 = time.perf_counter()
        print(f"[*] {i}/{len(ckpts)} {ckpt}", flush=True)
        model, cfg, raw = load_model(args.transformer_script, ckpt, device)
        feat_t0 = time.perf_counter()
        if feature_batches is None:
            reps = layer1_eos_representations(
                model,
                rows,
                probe_layer=args.probe_layer,
                pad_id=pad_id,
                eos_id=eos_id,
                ctx_len=args.ctx_len,
                batch_size=args.batch_size,
                device=device,
                amp=args.amp,
            )
        else:
            reps = layer1_eos_representations_from_batches(
                model,
                feature_batches,
                probe_layer=args.probe_layer,
                device=device,
                amp=args.amp,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        feat_sec = time.perf_counter() - feat_t0
        if args.single_test:
            X_train, y_train = single_test_matrix(reps, train_sids, test_ids, labels_by_scenario)
            X_val, y_val = single_test_matrix(reps, val_sids, test_ids, labels_by_scenario)
        else:
            if args.orchestrator_split:
                X_all, y_all = scenario_matrix(reps, train_sids + val_sids + test_sids, test_ids, labels_by_scenario)
                ntr, nv = len(train_sids), len(val_sids)
                train_idx = list(range(0, ntr))
                val_idx = list(range(ntr, ntr + nv))
                test_idx = list(range(ntr + nv, X_all.shape[0]))
                X_train, y_train = X_all, y_all
                X_val, y_val = X_all[test_idx], y_all[test_idx]
            else:
                train_idx = val_idx = test_idx = None
                X_train, y_train = scenario_matrix(reps, train_sids, test_ids, labels_by_scenario)
                X_val, y_val = scenario_matrix(reps, val_sids, test_ids, labels_by_scenario)
        k = spectral_k(X_train, args.var_threshold)
        metrics = train_probe(
            X_train,
            y_train,
            X_val,
            y_val,
            n_classes=len(ROOT_NODE_LABELS),
            epochs=args.probe_epochs,
            lr=args.probe_lr,
            weight_decay=args.probe_weight_decay,
            seed=args.seed + i,
            device=device,
            standardize=not args.no_standardize,
            train_indices=train_idx if (args.orchestrator_split and not args.single_test) else None,
            val_indices=val_idx if (args.orchestrator_split and not args.single_test) else None,
            test_indices=test_idx if (args.orchestrator_split and not args.single_test) else None,
            batch_size=args.probe_batch_size,
            eval_every=args.eval_every,
            early_stop_patience=args.early_stop_patience,
        )
        rec = {
            "checkpoint_number": i,
            "checkpoint_name": ckpt.name,
            "checkpoint_step": infer_step(ckpt, i),
            "spectral_k_after_layer1": k,
            "probe_val_true_class_probability": metrics["probe_val_true_class_probability"],
            "probe_val_top1_accuracy": metrics["probe_val_top1_accuracy"],
            "probe_train_top1_accuracy": metrics["probe_train_top1_accuracy"],
            "probe_val_ce_loss": metrics["probe_val_ce_loss"],
            "n_train_scenarios": len(train_sids),
            "n_val_scenarios": len(val_sids) if not args.orchestrator_split else len(test_sids),
            "n_tests": len(test_ids),
            "d_model": int(cfg.d_model),
            "n_faults": len(ROOT_NODE_LABELS),
        }
        records.append(rec)
        print(
            f"  k95={k} val_true_prob={rec['probe_val_true_class_probability']:.4f} "
            f"val_acc={rec['probe_val_top1_accuracy']:.4f} feature_sec={feat_sec:.3f} "
            f"checkpoint_sec={time.perf_counter() - ckpt_t0:.3f}",
            flush=True,
        )
        del model
        if device.type == "cuda" and args.empty_cache_each:
            torch.cuda.empty_cache()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(records, out_csv)
    print(f"[*] wrote {out_csv}")
    print(f"total_sec={time.perf_counter() - t0:.3f}")


if __name__ == "__main__":
    main()
