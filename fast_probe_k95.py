#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import dataclasses
import glob
import importlib.util
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ---------------- generic helpers ----------------

def import_from_path(module_name: str, path: str | Path):
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"missing script: {p}")
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


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def expand_ckpts(items: list[str]) -> list[Path]:
    out: list[Path] = []
    for item in items:
        matches = [Path(p) for p in glob.glob(str(item))]
        out.extend(matches if matches else [Path(item)])
    out = sorted(dict.fromkeys(out), key=lambda p: natural_key(str(p)))
    missing = [str(p) for p in out if not p.exists()]
    if missing:
        raise FileNotFoundError("missing checkpoint(s):\n" + "\n".join(missing))
    return out


def discover_ckpts(args) -> list[Path]:
    """Return checkpoints in natural timeline order.

    Preferred restart-friendly path:
      --checkpoint-run-name tinygpt_L4_D128_H4_seqtime_factored_probe
    resolves to:
      checkpoints/tinygpt_L4_D128_H4_seqtime_factored_probe/snapshots/*.pt

    You can still pass --snapshot-dir or explicit --transformer-ckpts globs.
    """
    if args.transformer_ckpts:
        return expand_ckpts(args.transformer_ckpts)

    if args.snapshot_dir:
        snap_dir = Path(args.snapshot_dir)
    elif args.checkpoint_run_name:
        snap_dir = Path(args.checkpoint_root) / args.checkpoint_run_name / "snapshots"
    else:
        raise ValueError("pass one of --checkpoint-run-name, --snapshot-dir, or --transformer-ckpts")

    if not snap_dir.exists():
        raise FileNotFoundError(f"snapshot dir does not exist: {snap_dir}")
    ckpts = expand_ckpts([str(snap_dir / "*.pt")])
    if not ckpts:
        raise FileNotFoundError(f"no .pt checkpoints found in {snap_dir}")
    return ckpts


def read_vocab(vocab_path: str | Path) -> tuple[int, dict[str, int]]:
    vocab_obj = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    vocab_list = vocab_obj["vocab"]
    stoi = {s: i for i, s in enumerate(vocab_list)}
    return stoi["[PAD]"], stoi


# ---------------- transformer feature wrapper, no GNN dependency ----------------

class CheckpointTransformerFeatures(nn.Module):
    """Load TinyGPT from train_transformer_v0_seqtime.py and expose hidden_states().

    This is intentionally the tiny subset of the old GNN FrozenTransformer wrapper
    needed for spectral probing.  It assumes the transformer script exposes
    GPTConfig and TinyGPT, and that checkpoints contain cfg + model, matching the
    train_transformer_v0_seqtime snapshots.
    """

    def __init__(self, script_path: str | Path, ckpt_path: str | Path, device: torch.device):
        super().__init__()
        mod = import_from_path("diagnostic_transformer_for_k95", script_path)
        if not hasattr(mod, "GPTConfig") or not hasattr(mod, "TinyGPT"):
            raise AttributeError(
                f"{script_path} must expose GPTConfig and TinyGPT; found "
                f"GPTConfig={hasattr(mod, 'GPTConfig')} TinyGPT={hasattr(mod, 'TinyGPT')}"
            )

        ckpt_path = Path(ckpt_path)
        ckpt = safe_torch_load(ckpt_path, device)
        if not isinstance(ckpt, dict):
            raise TypeError(f"checkpoint should be a dict, got {type(ckpt)} from {ckpt_path}")

        cfg_raw = ckpt.get("cfg") or ckpt.get("config") or ckpt.get("model_cfg")
        if cfg_raw is None:
            raise KeyError(f"checkpoint {ckpt_path} has no cfg/config/model_cfg key")
        if dataclasses.is_dataclass(cfg_raw):
            cfg_raw = dataclasses.asdict(cfg_raw)
        elif not isinstance(cfg_raw, dict):
            # Some scripts save argparse Namespace-ish objects.
            cfg_raw = vars(cfg_raw)

        cfg_keys = {f.name for f in dataclasses.fields(mod.GPTConfig)}
        cfg = mod.GPTConfig(**{k: v for k, v in cfg_raw.items() if k in cfg_keys})
        model = mod.TinyGPT(cfg).to(device)

        state = (
            ckpt.get("model")
            or ckpt.get("state_dict")
            or ckpt.get("model_state_dict")
            or ckpt.get("transformer_model")
        )
        if state is None:
            raise KeyError(f"checkpoint {ckpt_path} has no model/state_dict/model_state_dict/transformer_model key")
        model.load_state_dict(state, strict=True)
        model.eval()

        self.model = model
        self.cfg = cfg
        self.d_model = int(cfg.d_model)
        self.n_layers = int(cfg.n_layers)
        self.ckpt_epoch = ckpt.get("epoch")
        self.ckpt_global_step = ckpt.get("global_step")
        self.ckpt_path = str(ckpt_path)

    @torch.no_grad()
    def hidden_states(self, idx: torch.Tensor, token_times: torch.Tensor | None) -> list[torch.Tensor]:
        m = self.model
        B, T = idx.shape
        if T > m.cfg.ctx_len:
            raise ValueError(f"sequence length {T} exceeds transformer ctx_len {m.cfg.ctx_len}")

        x = m.tok(idx)
        if getattr(m, "pos", None) is not None:
            pos = torch.arange(T, device=idx.device)
            x = x + m.pos(pos)[None, :, :]
        x = m.drop(x)

        if hasattr(m, "_prep_times"):
            token_times = m._prep_times(token_times)

        hids = [x]
        for block in m.blocks:
            try:
                x = block(x, token_times)
            except TypeError:
                x = block(x)
            hids.append(x)

        # Match TinyGPT.forward final normalization for the last stream.
        hids[-1] = m.norm(hids[-1])
        return hids


# ---------------- data ----------------

class ProbeRows(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i: int):
        return self.rows[i]


def load_probe_rows(
    data_path: str | Path,
    *,
    n_sequences: int,
    seed: int,
    min_tokens: int,
    stratify: str | None,
) -> list[dict[str, Any]]:
    cols = ["sample_id", "scenario_id", "test_id", "tokens", "token_times", "fault_variant", "verdict"]
    tab = pq.read_table(data_path, columns=cols)
    py = {c: tab[c].to_pylist() for c in cols}
    candidates: list[dict[str, Any]] = []
    for i in range(tab.num_rows):
        toks = list(py["tokens"][i])
        tms = [float(x) for x in py["token_times"][i]]
        n = min(len(toks), len(tms))
        if n < min_tokens:
            continue
        candidates.append({
            "row_idx": i,
            "sample_id": str(py["sample_id"][i]),
            "scenario_id": str(py["scenario_id"][i]),
            "test_id": str(py["test_id"][i]),
            "fault_variant": str(py["fault_variant"][i] or ""),
            "verdict": str(py["verdict"][i] or ""),
            "tokens": toks[:n],
            "times": tms[:n],
        })
    if not candidates:
        raise RuntimeError("no candidate rows survived min_tokens filter")

    rng = random.Random(seed)
    if not stratify:
        rng.shuffle(candidates)
        return candidates[: min(n_sequences, len(candidates))]

    buckets: dict[str, list[dict[str, Any]]] = {}
    for r in candidates:
        buckets.setdefault(str(r.get(stratify, "")), []).append(r)
    for b in buckets.values():
        rng.shuffle(b)
    keys = sorted(buckets)
    rng.shuffle(keys)
    out: list[dict[str, Any]] = []
    while len(out) < n_sequences and any(buckets.values()):
        for k in keys:
            if buckets[k] and len(out) < n_sequences:
                out.append(buckets[k].pop())
    return out


def collate_probe(batch: list[dict[str, Any]], *, ctx_len: int, pad_id: int):
    L = min(ctx_len, max(len(r["tokens"]) for r in batch))
    B = len(batch)
    x = torch.full((B, L), pad_id, dtype=torch.long)
    tt = torch.zeros((B, L), dtype=torch.float32)
    mask = torch.zeros((B, L), dtype=torch.bool)
    meta = []
    for i, r in enumerate(batch):
        toks = r["tokens"][:L]
        tms = r["times"][:L]
        n = min(len(toks), len(tms))
        x[i, :n] = torch.tensor(toks[:n], dtype=torch.long)
        tt[i, :n] = torch.tensor(tms[:n], dtype=torch.float32)
        mask[i, :n] = True
        meta.append({k: r[k] for k in ["row_idx", "sample_id", "scenario_id", "test_id", "fault_variant", "verdict"]})
    return x, tt, mask, meta


# ---------------- spectral collection ----------------

def hids_to_layer_list(hids: Any, n_layers_hint: int | None) -> list[torch.Tensor]:
    if isinstance(hids, (list, tuple)):
        return list(hids)
    if not torch.is_tensor(hids):
        raise TypeError(f"hidden_states returned unsupported type: {type(hids)}")
    if hids.dim() == 3:
        return [hids]
    if hids.dim() != 4:
        raise ValueError(f"hidden_states tensor should be rank 3 or 4, got shape {tuple(hids.shape)}")

    if n_layers_hint is not None:
        if hids.shape[0] in {n_layers_hint, n_layers_hint + 1}:
            return [hids[i] for i in range(hids.shape[0])]
        if hids.shape[1] in {n_layers_hint, n_layers_hint + 1}:
            return [hids[:, i] for i in range(hids.shape[1])]
    return [hids[i] for i in range(hids.shape[0])]


def spectral_metrics(X: torch.Tensor, *, var_threshold: float) -> dict[str, float]:
    X = X.float()
    if X.numel() == 0 or X.size(0) < 2:
        return {"k_var": 0, "participation_ratio": 0.0, "stable_rank": 0.0, "top1_frac": 0.0, "var_total": 0.0}
    X = X - X.mean(dim=0, keepdim=True)
    C = X.T @ X
    eig = torch.linalg.eigvalsh(C).clamp_min(0.0).flip(0).cpu().numpy()
    total = float(eig.sum())
    if total <= 0.0 or not np.isfinite(total):
        return {"k_var": 0, "participation_ratio": 0.0, "stable_rank": 0.0, "top1_frac": 0.0, "var_total": total}
    cev = np.cumsum(eig) / total
    k_var = int(np.searchsorted(cev, var_threshold) + 1)
    pr = float(total * total / max(float((eig * eig).sum()), 1e-30))
    stable = float(total / max(float(eig[0]), 1e-30))
    top1 = float(eig[0] / total)
    return {"k_var": k_var, "participation_ratio": pr, "stable_rank": stable, "top1_frac": top1, "var_total": total}


def collect_layer_token_mats(
    feat,
    rows: list[dict[str, Any]],
    *,
    pad_id: int,
    ctx_len: int,
    batch_size: int,
    device: torch.device,
    amp: bool,
) -> tuple[list[torch.Tensor], int]:
    loader = DataLoader(
        ProbeRows(rows),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_probe(b, ctx_len=ctx_len, pad_id=pad_id),
        num_workers=0,
    )
    per_layer_chunks: list[list[torch.Tensor]] | None = None
    total_valid = 0
    with torch.no_grad():
        for x, tt, mask, _meta in loader:
            x = x.to(device, non_blocking=True)
            tt = tt.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            total_valid += int(mask.sum().item())
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(amp and device.type == "cuda")):
                hids = feat.hidden_states(x, tt)
            layers = hids_to_layer_list(hids, getattr(feat, "n_layers", None))
            if per_layer_chunks is None:
                per_layer_chunks = [[] for _ in layers]
            if len(layers) != len(per_layer_chunks):
                raise RuntimeError(f"number of layers changed across batches: {len(layers)} vs {len(per_layer_chunks)}")
            for li, H in enumerate(layers):
                if H.shape[:2] != mask.shape:
                    raise RuntimeError(f"layer {li} shape {tuple(H.shape)} incompatible with mask {tuple(mask.shape)}")
                per_layer_chunks[li].append(H[mask].detach().float().cpu())
    if per_layer_chunks is None:
        raise RuntimeError("probe loader produced no batches")
    return [torch.cat(chunks, dim=0) for chunks in per_layer_chunks], total_valid


def maybe_downsample(X: torch.Tensor, *, max_tokens: int, seed: int) -> torch.Tensor:
    if max_tokens <= 0 or X.shape[0] <= max_tokens:
        return X
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    idx = torch.randperm(X.shape[0], generator=g)[:max_tokens]
    idx, _ = torch.sort(idx)
    return X[idx]


def infer_step_from_ckpt(path: Path, ordinal: int) -> int:
    m = re.search(r"(?:step|iter|update)[_\-]?(\d+)", path.stem, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    nums = re.findall(r"\d+", path.stem)
    if nums:
        return int(nums[-1])
    return ordinal


def write_csv(rows: list[dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def plot_k95(rows: list[dict[str, Any]], out_path: Path):
    import matplotlib.pyplot as plt

    by_layer: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        by_layer.setdefault(int(r["layer"]), []).append(r)
    fig, ax = plt.subplots(figsize=(9, 5))
    for layer, rs in sorted(by_layer.items()):
        rs = sorted(rs, key=lambda r: (int(r["ckpt_order"]), int(r["step"])))
        xs = [int(r["ckpt_order"]) for r in rs]
        ys = [float(r["k_var"]) for r in rs]
        ax.plot(xs, ys, marker="o", linewidth=1.5, label=f"layer {layer}")
    ax.set_xlabel("checkpoint order")
    ax.set_ylabel("k for requested variance threshold")
    ax.set_title("Transformer token representation spectral dimension by layer")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ---------------- main ----------------

def main():
    t0 = time.perf_counter()
    p = argparse.ArgumentParser(description="Probe k95/effective spectral dimension of transformer hidden states by layer.")
    p.add_argument("--data", default="dataset_v1_tokenized.parquet")
    p.add_argument("--vocab", default="tokenizer_vocab_locked_v1.json")
    p.add_argument("--transformer-script", default="train_transformer_v0_seqtime_probe_ckpts.py")

    # New restart-friendly checkpoint discovery. No GNN wrapper required.
    p.add_argument("--checkpoint-root", default="checkpoints")
    p.add_argument("--checkpoint-run-name", default="", help="Run dir name under --checkpoint-root; uses <root>/<name>/snapshots/*.pt")
    p.add_argument("--snapshot-dir", default="", help="Explicit snapshots dir containing *.pt")
    p.add_argument("--transformer-ckpts", nargs="*", default=None, help="Optional explicit checkpoint paths/globs; overrides run-name/snapshot-dir")

    p.add_argument("--out-dir", default="eval/transformer_k95_probe")
    p.add_argument("--ctx-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-sequences", type=int, default=100)
    p.add_argument("--min-tokens", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=50000, help="Downsample valid tokens per layer after collection; <=0 means use all.")
    p.add_argument("--var-threshold", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--stratify", default="fault_variant", choices=["", "fault_variant", "test_id", "verdict"], help="Optional metadata column for round-robin row sampling.")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--tf32", action="store_true", help="allow TF32 matmul/cuDNN for faster no-AMP CUDA inference")
    p.add_argument("--deterministic", action="store_true", help="request deterministic CUDA algorithms for same-machine reruns")
    p.add_argument("--deterministic-warn-only", action="store_true", help="warn instead of erroring on nondeterministic ops")
    p.add_argument("--empty-cache-each", action="store_true", help="empty CUDA cache after each checkpoint")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=args.deterministic_warn_only)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pad_id, _stoi = read_vocab(args.vocab)
    ckpts = discover_ckpts(args)
    rows = load_probe_rows(
        args.data,
        n_sequences=args.n_sequences,
        seed=args.seed,
        min_tokens=args.min_tokens,
        stratify=(args.stratify or None),
    )
    probe_meta = [{k: r[k] for k in ["row_idx", "sample_id", "scenario_id", "test_id", "fault_variant", "verdict"]} for r in rows]
    (out_dir / "probe_rows.json").write_text(json.dumps(probe_meta, indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device={device} n_sequences={len(rows)} ckpts={len(ckpts)} pad_id={pad_id}")
    print(f"out_dir={out_dir}")
    print(f"first_ckpt={ckpts[0]}")
    print(f"last_ckpt={ckpts[-1]}")
    print(f"fast_options=amp={args.amp} tf32={args.tf32} deterministic={args.deterministic} empty_cache_each={args.empty_cache_each}")

    all_metrics: list[dict[str, Any]] = []
    for ci, ckpt in enumerate(ckpts):
        ckpt_t0 = time.perf_counter()
        print(f"[*] checkpoint {ci + 1}/{len(ckpts)}: {ckpt}", flush=True)
        feat = CheckpointTransformerFeatures(args.transformer_script, ckpt, device).to(device)
        feat.eval()
        layer_mats, total_valid = collect_layer_token_mats(
            feat,
            rows,
            pad_id=pad_id,
            ctx_len=args.ctx_len,
            batch_size=args.batch_size,
            device=device,
            amp=args.amp,
        )
        step = infer_step_from_ckpt(ckpt, ci)
        for li, X in enumerate(layer_mats):
            Xp = maybe_downsample(X, max_tokens=args.max_tokens, seed=args.seed)
            m = spectral_metrics(Xp, var_threshold=args.var_threshold)
            rec = {
                "ckpt_order": ci,
                "step": step,
                "ckpt": str(ckpt),
                "ckpt_name": ckpt.name,
                "layer": li,
                "n_tokens_all": int(X.shape[0]),
                "n_tokens_used": int(Xp.shape[0]),
                "d_model": int(Xp.shape[1]),
                "var_threshold": float(args.var_threshold),
                **m,
            }
            all_metrics.append(rec)
            print(
                f"  layer={li:02d} k{int(args.var_threshold * 100):02d}={rec['k_var']:3d} "
                f"pr={rec['participation_ratio']:.1f} stable={rec['stable_rank']:.1f} top1={rec['top1_frac']:.3f}",
                flush=True,
            )
        del feat
        if device.type == "cuda" and args.empty_cache_each:
            torch.cuda.empty_cache()
        print(f"  checkpoint_sec={time.perf_counter() - ckpt_t0:.3f}", flush=True)

    write_csv(all_metrics, out_dir / "k_spectral_metrics.csv")
    summary = {
        "args": vars(args),
        "checkpoints": [str(p) for p in ckpts],
        "n_probe_rows": len(rows),
        "probe_rows_path": str(out_dir / "probe_rows.json"),
        "metrics_path": str(out_dir / "k_spectral_metrics.csv"),
    }
    (out_dir / "k_spectral_metrics.meta.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not args.no_plot:
        plot_k95(all_metrics, out_dir / "k95_by_layer.png")
        print(f"[*] wrote {out_dir / 'k95_by_layer.png'}")
    print(f"[*] wrote {out_dir / 'k_spectral_metrics.csv'}")
    print(f"[*] wrote {out_dir / 'probe_rows.json'}")
    print(f"total_sec={time.perf_counter() - t0:.3f}")


if __name__ == "__main__":
    main()
