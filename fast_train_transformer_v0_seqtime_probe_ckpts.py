#!/usr/bin/env python3
from __future__ import annotations

import argparse, functools, json, math, os, random, time
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------- config ----------------

@dataclass
class GPTConfig:
    vocab_size: int
    n_layers: int = 4
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    ctx_len: int = 1024
    rope_layout: str = "learned_abs"   # none | learned_abs | seq_rope | time_rope | seq_time_rope
    rope_seq_dim: int = 0              # per-head dims; 0 means auto/fill from layout
    rope_time_dim: int = 0             # per-head dims; 0 means auto/fill from layout
    rope_base: float = 10000.0
    rope_seq_scale: float = 1.0        # multiply token index before seq RoPE
    rope_time_scale: float = 10.0      # multiply seconds before time RoPE; 10 => 100ms units
    time_origin: str = "first"         # first | zero
    dropout: float = 0.1
    norm_style: str = "layernorm"      # layernorm | rmsnorm
    loss_weighting: str = "uniform"    # v0: uniform only
    pad_id: int = 0


def resolve_rope_dims(cfg: GPTConfig) -> tuple[int, int]:
    if cfg.d_model % cfg.n_heads != 0:
        raise ValueError("canonical heads require d_model % n_heads == 0")
    head_dim = cfg.d_model // cfg.n_heads
    s, t = int(cfg.rope_seq_dim), int(cfg.rope_time_dim)

    if cfg.rope_layout in {"none", "learned_abs"}:
        return 0, 0
    if cfg.rope_layout == "seq_rope":
        s, t = s or head_dim, 0
    elif cfg.rope_layout == "time_rope":
        s, t = 0, t or head_dim
    elif cfg.rope_layout == "seq_time_rope":
        if s == 0 and t == 0:
            s = (head_dim // 2) & ~1       # even half-ish split
            t = (head_dim - s) & ~1
        elif s == 0:
            s = (head_dim - t) & ~1
        elif t == 0:
            t = (head_dim - s) & ~1
    else:
        raise ValueError("rope_layout must be none | learned_abs | seq_rope | time_rope | seq_time_rope")

    if s < 0 or t < 0 or s + t > head_dim:
        raise ValueError(f"bad RoPE allocation: seq={s}, time={t}, head_dim={head_dim}")
    if (s and s % 2) or (t and t % 2):
        raise ValueError(f"RoPE dims must be even: seq={s}, time={t}")
    return s, t


# ---------------- data ----------------

class TokenRows(Dataset):
    def __init__(self, rows: list[tuple[list[int], list[float]]]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def make_scenario_split(
    scenario_ids: list[str],
    *,
    val_frac: float,
    seed: int,
    strata_by_scenario: dict[str, str] | None = None,
):
    scenarios = sorted(set(scenario_ids))
    rng = random.Random(seed)
    if not strata_by_scenario:
        rng.shuffle(scenarios)
        n_val = max(1, int(len(scenarios) * val_frac))
        val_scenarios = set(scenarios[:n_val])
        return sorted(set(scenarios) - val_scenarios), sorted(val_scenarios)

    buckets: dict[str, list[str]] = {}
    for sid in scenarios:
        buckets.setdefault(str(strata_by_scenario[sid]), []).append(sid)

    val_scenarios: set[str] = set()
    for bucket in buckets.values():
        rng.shuffle(bucket)
        n_val = max(1, int(len(bucket) * val_frac)) if bucket else 0
        val_scenarios.update(bucket[:n_val])
    return sorted(set(scenarios) - val_scenarios), sorted(val_scenarios)


def load_scenario_split(path: str, val_frac: float, seed: int, split_stratify_col: str = ""):
    cols = ["scenario_id", "tokens", "token_times"]
    if split_stratify_col:
        cols.append(split_stratify_col)
    tab = pq.read_table(path, columns=cols)
    scenario_ids = tab["scenario_id"].to_pylist()
    tokens = tab["tokens"].to_pylist()
    times = tab["token_times"].to_pylist()

    strata_by_scenario = None
    if split_stratify_col:
        strata = tab[split_stratify_col].to_pylist()
        strata_by_scenario = {}
        for sid, stratum in zip(scenario_ids, strata):
            sid, stratum = str(sid), str(stratum)
            old = strata_by_scenario.setdefault(sid, stratum)
            if old != stratum:
                raise ValueError(f"{split_stratify_col} is not scenario-constant for {sid}: {old!r} vs {stratum!r}")

    train_scenarios, val_scenarios_list = make_scenario_split(
        [str(sid) for sid in scenario_ids],
        val_frac=val_frac,
        seed=seed,
        strata_by_scenario=strata_by_scenario,
    )
    val_scenarios = set(val_scenarios_list)

    train, val = [], []
    for sid, toks, tms in zip(scenario_ids, tokens, times):
        n = min(len(toks), len(tms))
        row = (list(toks[:n]), [float(x) for x in tms[:n]])
        (val if str(sid) in val_scenarios else train).append(row)
    split = {
        "seed": int(seed),
        "val_frac": float(val_frac),
        "stratify_col": str(split_stratify_col),
        "train_scenario_ids": train_scenarios,
        "val_scenario_ids": val_scenarios_list,
    }
    return train, val, split


def collate_causal(batch, *, ctx_len: int, pad_id: int):
    # Each row becomes x[t] -> y[t] = x[t+1].  token_times align with x.
    rows = [(tok[: ctx_len + 1], tms[: ctx_len + 1]) for tok, tms in batch]
    L = min(ctx_len, max(len(tok) - 1 for tok, _ in rows))
    B = len(rows)

    x = torch.full((B, L), pad_id, dtype=torch.long)
    y = torch.full((B, L), -100, dtype=torch.long)      # ignored by CE
    tt = torch.zeros((B, L), dtype=torch.float32)

    for i, (tok, tms) in enumerate(rows):
        if len(tok) < 2:
            continue
        inp, tgt = tok[:-1][:L], tok[1:][:L]
        t_inp = tms[:-1][:L]
        x[i, : len(inp)] = torch.tensor(inp, dtype=torch.long)
        y[i, : len(tgt)] = torch.tensor(tgt, dtype=torch.long)
        tt[i, : len(t_inp)] = torch.tensor(t_inp, dtype=torch.float32)

    return x, tt, y


# ---------------- model ----------------

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def make_norm(cfg: GPTConfig):
    if cfg.norm_style == "layernorm":
        return nn.LayerNorm(cfg.d_model)
    if cfg.norm_style == "rmsnorm":
        return RMSNorm(cfg.d_model)
    raise ValueError(f"unknown norm_style={cfg.norm_style!r}")


class SeqTimeRotary(nn.Module):
    """RoPE over per-head seq and/or continuous-time subspaces of q/k."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        head_dim = cfg.d_model // cfg.n_heads
        seq_dim, time_dim = resolve_rope_dims(cfg)
        if seq_dim + time_dim > head_dim:
            raise ValueError("RoPE allocation exceeds head_dim")
        self.seq_dim = seq_dim
        self.time_dim = time_dim
        self.seq_scale = float(cfg.rope_seq_scale)
        self.time_scale = float(cfg.rope_time_scale)

        if seq_dim:
            inv = 1.0 / (cfg.rope_base ** (torch.arange(0, seq_dim, 2).float() / seq_dim))
            self.register_buffer("inv_seq", inv, persistent=False)
        else:
            self.inv_seq = None
        if time_dim:
            inv = 1.0 / (cfg.rope_base ** (torch.arange(0, time_dim, 2).float() / time_dim))
            self.register_buffer("inv_time", inv, persistent=False)
        else:
            self.inv_time = None

    @staticmethod
    def _rot(x, cos, sin):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        y = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return y.flatten(-2)

    def _seq_angles(self, T: int, device, dtype):
        pos = torch.arange(T, device=device, dtype=torch.float32) * self.seq_scale
        freqs = torch.outer(pos, self.inv_seq.to(device=device))
        return freqs.cos().to(dtype)[None, None], freqs.sin().to(dtype)[None, None]

    def _time_angles(self, token_times: torch.Tensor, device, dtype):
        # token_times: [B,T], already origin-normalized by the model.
        pos = token_times.to(device=device, dtype=torch.float32) * self.time_scale
        freqs = pos[..., None] * self.inv_time.to(device=device)[None, None, :]
        return freqs.cos().to(dtype)[:, None], freqs.sin().to(dtype)[:, None]

    def forward(self, q, k, token_times: torch.Tensor | None):
        # q/k: [B,H,T,D]
        B, H, T, D = q.shape
        out_q, out_k = [], []
        off = 0

        if self.seq_dim:
            cos, sin = self._seq_angles(T, q.device, q.dtype)
            sl = slice(off, off + self.seq_dim)
            out_q.append(self._rot(q[..., sl], cos, sin))
            out_k.append(self._rot(k[..., sl], cos, sin))
            off += self.seq_dim

        if self.time_dim:
            if token_times is None:
                raise ValueError("time RoPE requires token_times")
            cos, sin = self._time_angles(token_times, q.device, q.dtype)
            sl = slice(off, off + self.time_dim)
            out_q.append(self._rot(q[..., sl], cos, sin))
            out_k.append(self._rot(k[..., sl], cos, sin))
            off += self.time_dim

        if off < D:
            out_q.append(q[..., off:])
            out_k.append(k[..., off:])
        return torch.cat(out_q, dim=-1), torch.cat(out_k, dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, "canonical heads require d_model % n_heads == 0"
        self.cfg = cfg
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.rope = SeqTimeRotary(cfg) if cfg.rope_layout in {"seq_rope", "time_rope", "seq_time_rope"} else None

    def forward(self, x, token_times=None):
        B, T, C = x.shape
        H, D = self.cfg.n_heads, self.head_dim

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, H, D).transpose(1, 2)  # [B,H,T,D]
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        if self.rope is not None:
            q, k = self.rope(q, k, token_times)

        p = self.cfg.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=p)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.out(y))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n1 = make_norm(cfg)
        self.attn = CausalSelfAttention(cfg)
        self.n2 = make_norm(cfg)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x, token_times=None):
        x = x + self.attn(self.n1(x), token_times)
        x = x + self.mlp(self.n2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        if cfg.loss_weighting != "uniform":
            raise ValueError("v0 supports only loss_weighting='uniform'")
        if cfg.time_origin not in {"first", "zero"}:
            raise ValueError("time_origin must be first | zero")
        cfg.rope_seq_dim, cfg.rope_time_dim = resolve_rope_dims(cfg)

        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.ctx_len, cfg.d_model) if cfg.rope_layout == "learned_abs" else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = make_norm(cfg)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _prep_times(self, token_times):
        if token_times is None:
            return None
        token_times = token_times.to(dtype=torch.float32)
        if self.cfg.time_origin == "first":
            token_times = token_times - token_times[:, :1]
        return token_times.clamp_min(0.0)

    def forward(self, idx, token_times=None, targets=None):
        B, T = idx.shape
        if T > self.cfg.ctx_len:
            raise ValueError(f"sequence length {T} exceeds ctx_len {self.cfg.ctx_len}")

        x = self.tok(idx)
        if self.pos is not None:
            pos = torch.arange(T, device=idx.device)
            x = x + self.pos(pos)[None, :, :]
        x = self.drop(x)

        token_times = self._prep_times(token_times)
        for block in self.blocks:
            x = block(x, token_times)

        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return logits, loss


# ---------------- train / checkpoint probe schedule ----------------

def eval_epoch(model, loader, device, amp: bool):
    model.train(False)
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for x, tt, y in loader:
            x = x.to(device, non_blocking=True)
            tt = tt.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            ntok = (y != -100).sum().item()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(amp and device.type == "cuda")):
                _, loss = model(x, tt, y)
            total_loss += loss.item() * ntok
            total_tokens += ntok
    return total_loss / max(1, total_tokens)


def epoch1_snapshot_steps(steps_per_epoch: int, n_snapshots: int) -> list[int]:
    """Return optimizer-step numbers inside epoch 1 at which to save snapshots.

    Includes the first optimizer step and the final optimizer step when possible.
    If there are fewer training batches than requested snapshots, this returns the
    maximum number of unique snapshot steps available.
    """
    if steps_per_epoch <= 0 or n_snapshots <= 0:
        return []
    if n_snapshots == 1:
        return [steps_per_epoch]
    raw = [1 + round(i * (steps_per_epoch - 1) / (n_snapshots - 1)) for i in range(n_snapshots)]
    return sorted({max(1, min(steps_per_epoch, int(s))) for s in raw})


def checkpoint_payload(
    *,
    model,
    opt,
    cfg: GPTConfig,
    args,
    epoch: int,
    global_step: int,
    epoch_step: int,
    steps_per_epoch: int,
    train_loss,
    val_loss,
    n_params: int,
    tag: str,
    snapshot_index,
    include_optimizer: bool,
):
    state_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    ckpt = {
        # Original trainer compatibility fields.
        "epoch": int(epoch),
        "cfg": asdict(cfg),
        "model": state_model.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "n_params": int(n_params),

        # Probe/checkpoint bookkeeping fields.
        "args": vars(args),
        "global_step": int(global_step),
        "epoch_step": int(epoch_step),
        "steps_per_epoch": int(steps_per_epoch),
        "tag": str(tag),
        "snapshot_index": snapshot_index,
        "checkpoint_kind": "transformer_representation_probe",
    }
    if include_optimizer:
        ckpt["optimizer"] = opt.state_dict()
    return ckpt


def save_checkpoint(
    path: Path,
    *,
    model,
    opt,
    cfg: GPTConfig,
    args,
    epoch: int,
    global_step: int,
    epoch_step: int,
    steps_per_epoch: int,
    train_loss,
    val_loss,
    n_params: int,
    tag: str,
    snapshot_index,
    include_optimizer: bool,
    manifest: list[dict],
):
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = checkpoint_payload(
        model=model,
        opt=opt,
        cfg=cfg,
        args=args,
        epoch=epoch,
        global_step=global_step,
        epoch_step=epoch_step,
        steps_per_epoch=steps_per_epoch,
        train_loss=train_loss,
        val_loss=val_loss,
        n_params=n_params,
        tag=tag,
        snapshot_index=snapshot_index,
        include_optimizer=include_optimizer,
    )
    torch.save(ckpt, path)
    rec = {
        "path": str(path),
        "name": path.name,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "epoch_step": int(epoch_step),
        "steps_per_epoch": int(steps_per_epoch),
        "train_loss": None if train_loss is None else float(train_loss),
        "val_loss": None if val_loss is None else float(val_loss),
        "tag": str(tag),
        "snapshot_index": snapshot_index,
        "include_optimizer": bool(include_optimizer),
    }
    manifest.append(rec)
    print(f"[*] saved {path}")
    return rec


def write_manifest(path: Path, *, args, cfg: GPTConfig, n_params: int, entries: list[dict], schedule: dict):
    payload = {
        "args": vars(args),
        "cfg": asdict(cfg),
        "n_params": int(n_params),
        "schedule": schedule,
        "checkpoints": entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def train_epoch_with_snapshots(
    *,
    model,
    loader,
    opt,
    device,
    amp: bool,
    epoch: int,
    global_step: int,
    snapshot_steps: list[int],
    snapshot_step_to_index: dict[int, int],
    skip_snapshot_steps: set[int],
    snapshot_tag: str,
    save_snapshot_fn,
):
    model.train(True)
    total_loss, total_tokens = 0.0, 0
    snapshot_steps_set = set(snapshot_steps)

    for epoch_step, (x, tt, y) in enumerate(loader, start=1):
        x, tt, y = x.to(device), tt.to(device), y.to(device)
        ntok = (y != -100).sum().item()

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(amp and device.type == "cuda")):
            _, loss = model(x, tt, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        global_step += 1

        total_loss += loss.item() * ntok
        total_tokens += ntok
        running_loss = total_loss / max(1, total_tokens)

        if epoch_step in snapshot_steps_set and epoch_step not in skip_snapshot_steps:
            save_snapshot_fn(
                epoch=epoch,
                global_step=global_step,
                epoch_step=epoch_step,
                train_loss=running_loss,
                val_loss=None,
                tag=snapshot_tag,
                snapshot_index=snapshot_step_to_index[epoch_step],
            )

    return total_loss / max(1, total_tokens), global_step


def main():
    t0 = time.perf_counter()
    p = argparse.ArgumentParser(description="Train TinyGPT and save representation-probe checkpoints.")
    p.add_argument("--data", default="dataset_v1_tokenized.parquet")
    p.add_argument("--vocab", default="tokenizer_vocab_locked_v1.json")
    p.add_argument("--out-dir", default="checkpoints/transformer_v0_probe_steps")

    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=0)
    p.add_argument("--ctx-len", type=int, default=1024)
    p.add_argument("--rope-layout", default="learned_abs", choices=["none", "learned_abs", "seq_rope", "time_rope", "seq_time_rope"])
    p.add_argument("--rope-seq-dim", type=int, default=0, help="per-head seq RoPE dims; 0=auto/fill")
    p.add_argument("--rope-time-dim", type=int, default=0, help="per-head time RoPE dims; 0=auto/fill")
    p.add_argument("--rope-base", type=float, default=10000.0)
    p.add_argument("--rope-seq-scale", type=float, default=1.0)
    p.add_argument("--rope-time-scale", type=float, default=10.0, help="multiply seconds before time RoPE")
    p.add_argument("--time-origin", default="first", choices=["first", "zero"])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--norm-style", default="layernorm", choices=["layernorm", "rmsnorm"])
    p.add_argument("--loss-weighting", default="uniform", choices=["uniform"])

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--epoch1-snapshots", type=int, default=20, help="number of evenly spaced optimizer-step snapshots during epoch 1")
    p.add_argument("--snapshots-per-epoch", type=int, default=0, help="if >0, save this many evenly spaced snapshots in every epoch")
    p.add_argument("--snapshot-dir-name", default="snapshots", help="subdirectory inside out-dir that receives scheduled probe snapshots")
    p.add_argument("--no-save-initial", action="store_true", help="do not save the random-init epoch000_step_000000 checkpoint")
    p.add_argument("--omit-optimizer", action="store_true", help="omit optimizer state from scheduled snapshots to save disk")

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--split-stratify-col", default="", help="scenario-constant parquet column used to stratify the train/val split")
    p.add_argument("--split-json", default="", help="optional path for the scenario train/val split JSON")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers; 0 is often fastest on Windows for in-memory rows")
    p.add_argument("--pin-memory", action="store_true", help="pin DataLoader batches before CUDA transfer")
    p.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch factor when workers > 0")
    p.add_argument("--persistent-workers", action="store_true", help="keep DataLoader workers alive between epochs")
    p.add_argument("--fused-adamw", action="store_true", help="use fused CUDA AdamW when available")
    p.add_argument("--compile", action="store_true", help="torch.compile the model; usually only worth it for longer runs")
    p.add_argument("--tf32", action="store_true", help="allow TF32 matmul/cuDNN for faster no-AMP CUDA training")
    p.add_argument("--deterministic", action="store_true", help="request deterministic CUDA algorithms for same-machine reruns")
    p.add_argument("--deterministic-warn-only", action="store_true", help="warn instead of erroring on nondeterministic ops")
    args = p.parse_args()

    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=args.deterministic_warn_only)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)

    vocab = json.loads(Path(args.vocab).read_text())
    pad_id = vocab["vocab"].index("[PAD]")

    cfg = GPTConfig(
        vocab_size=vocab["size"],
        n_layers=args.n_layers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff or 4 * args.d_model,
        ctx_len=args.ctx_len,
        rope_layout=args.rope_layout,
        rope_seq_dim=args.rope_seq_dim,
        rope_time_dim=args.rope_time_dim,
        rope_base=args.rope_base,
        rope_seq_scale=args.rope_seq_scale,
        rope_time_scale=args.rope_time_scale,
        time_origin=args.time_origin,
        dropout=args.dropout,
        norm_style=args.norm_style,
        loss_weighting=args.loss_weighting,
        pad_id=pad_id,
    )
    cfg.rope_seq_dim, cfg.rope_time_dim = resolve_rope_dims(cfg)

    train_rows, val_rows, split = load_scenario_split(args.data, args.val_frac, args.seed, args.split_stratify_col)
    collate = functools.partial(collate_causal, ctx_len=cfg.ctx_len, pad_id=cfg.pad_id)
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": bool(args.pin_memory),
        "persistent_workers": bool(args.persistent_workers and args.num_workers > 0),
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    train_gen = torch.Generator()
    train_gen.manual_seed(args.seed)
    train_loader = DataLoader(
        TokenRows(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_gen,
        collate_fn=collate,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        TokenRows(val_rows),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        **loader_kwargs,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = TinyGPT(cfg).to(device)
    if args.compile:
        model = torch.compile(model)
    adamw_kwargs = {}
    if args.fused_adamw and device.type == "cuda":
        adamw_kwargs["fused"] = True
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, **adamw_kwargs)

    n_params = sum(p.numel() for p in model.parameters())
    out_dir = Path(args.out_dir)
    snapshot_dir = out_dir / args.snapshot_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    steps_per_epoch = len(train_loader)
    dense_snapshots = args.snapshots_per_epoch > 0
    requested_snapshots = args.snapshots_per_epoch if dense_snapshots else args.epoch1_snapshots
    snap_steps = epoch1_snapshot_steps(steps_per_epoch, requested_snapshots)
    snap_step_to_index = {s: i for i, s in enumerate(snap_steps, start=1)}
    final_scheduled_snapshot = bool(snap_steps and snap_steps[-1] == steps_per_epoch)
    # Save final scheduled snapshots after validation. Validation does not alter weights,
    # and this gives epoch-end scheduled snapshots a real val_loss in the checkpoint/manifest.
    in_loop_skip = {steps_per_epoch} if final_scheduled_snapshot else set()

    schedule = {
        "epochs": int(args.epochs),
        "steps_per_epoch": int(steps_per_epoch),
        "snapshots_per_epoch": int(args.snapshots_per_epoch),
        "epoch1_requested_snapshots": int(args.epoch1_snapshots),
        "epoch1_actual_snapshot_steps": snap_steps,
        "epoch1_actual_snapshots": len(snap_steps),
        "final_scheduled_snapshot_saved_after_validation": final_scheduled_snapshot,
        "save_initial": not args.no_save_initial,
        "save_end_of_epochs": [] if dense_snapshots else list(range(2, args.epochs + 1)),
        "snapshot_dir": str(snapshot_dir),
    }

    manifest: list[dict] = []
    manifest_path = out_dir / "snapshot_manifest.json"
    split_path = Path(args.split_json) if args.split_json else out_dir / "scenario_split.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")

    print(f"device={device}")
    print(f"train_rows={len(train_rows)} val_rows={len(val_rows)}")
    print(f"cfg={asdict(cfg)}")
    print(f"params={n_params:,}")
    print(f"steps_per_epoch={steps_per_epoch}")
    print(f"epoch1_snapshot_steps={snap_steps}")
    print(f"snapshot_dir={snapshot_dir}")
    print(f"split_json={split_path}")
    print(
        "fast_options="
        f"amp={args.amp} tf32={args.tf32} deterministic={args.deterministic} "
        f"fused_adamw={args.fused_adamw} compile={args.compile} "
        f"num_workers={args.num_workers} pin_memory={args.pin_memory}"
    )

    include_optimizer = not args.omit_optimizer

    def save_scheduled_snapshot(*, epoch, global_step, epoch_step, train_loss, val_loss, tag, snapshot_index):
        if tag == "init":
            name = "epoch000_step_000000.pt"
        elif tag in {"epoch1_snapshot", "epoch1_snapshot_end", "epoch_snapshot", "epoch_snapshot_end"}:
            idx = int(snapshot_index)
            denom = max(1, len(snap_steps))
            name = f"epoch{epoch:03d}_snapshot_{idx:02d}_of_{denom:02d}_step_{global_step:06d}.pt"
        elif tag == "epoch_end":
            name = f"epoch{epoch:03d}_end_step_{global_step:06d}.pt"
        else:
            name = f"epoch{epoch:03d}_{tag}_step_{global_step:06d}.pt"

        rec = save_checkpoint(
            snapshot_dir / name,
            model=model,
            opt=opt,
            cfg=cfg,
            args=args,
            epoch=epoch,
            global_step=global_step,
            epoch_step=epoch_step,
            steps_per_epoch=steps_per_epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            n_params=n_params,
            tag=tag,
            snapshot_index=snapshot_index,
            include_optimizer=include_optimizer,
            manifest=manifest,
        )
        write_manifest(manifest_path, args=args, cfg=cfg, n_params=n_params, entries=manifest, schedule=schedule)
        return rec

    if not args.no_save_initial:
        save_scheduled_snapshot(
            epoch=0,
            global_step=0,
            epoch_step=0,
            train_loss=None,
            val_loss=None,
            tag="init",
            snapshot_index=0,
        )

    best_val = float("inf")
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.perf_counter()
        tr, global_step = train_epoch_with_snapshots(
            model=model,
            loader=train_loader,
            opt=opt,
            device=device,
            amp=args.amp,
            epoch=epoch,
            global_step=global_step,
            snapshot_steps=snap_steps,
            snapshot_step_to_index=snap_step_to_index,
            skip_snapshot_steps=in_loop_skip,
            snapshot_tag="epoch_snapshot" if dense_snapshots else "epoch1_snapshot",
            save_snapshot_fn=save_scheduled_snapshot,
        )
        train_elapsed = time.perf_counter() - epoch_t0
        eval_t0 = time.perf_counter()
        va = eval_epoch(model, val_loader, device, args.amp)
        eval_elapsed = time.perf_counter() - eval_t0
        ppl = math.exp(min(va, 20.0))
        print(
            f"epoch {epoch:03d} step={global_step:06d} train_loss={tr:.4f} "
            f"val_loss={va:.4f} val_ppl={ppl:.2f} train_sec={train_elapsed:.3f} eval_sec={eval_elapsed:.3f}"
        )

        # Complete the scheduled snapshots with the final within-epoch snapshot.
        if final_scheduled_snapshot and (dense_snapshots or epoch == 1):
            save_scheduled_snapshot(
                epoch=epoch,
                global_step=global_step,
                epoch_step=steps_per_epoch,
                train_loss=tr,
                val_loss=va,
                tag="epoch_snapshot_end" if dense_snapshots else "epoch1_snapshot_end",
                snapshot_index=snap_step_to_index[steps_per_epoch],
            )

        # End-of-further-epoch checkpoints: epoch 2 through args.epochs.
        if epoch >= 2 and not dense_snapshots:
            save_scheduled_snapshot(
                epoch=epoch,
                global_step=global_step,
                epoch_step=steps_per_epoch,
                train_loss=tr,
                val_loss=va,
                tag="epoch_end",
                snapshot_index=None,
            )

        # Keep a conventional best.pt in out_dir for downstream GNN training.
        if va < best_val:
            best_val = va
            best_ckpt = checkpoint_payload(
                model=model,
                opt=opt,
                cfg=cfg,
                args=args,
                epoch=epoch,
                global_step=global_step,
                epoch_step=steps_per_epoch,
                steps_per_epoch=steps_per_epoch,
                train_loss=tr,
                val_loss=va,
                n_params=n_params,
                tag="best",
                snapshot_index=None,
                include_optimizer=True,
            )
            torch.save(best_ckpt, out_dir / "best.pt")
            print(f"[*] saved {out_dir / 'best.pt'}")

        write_manifest(manifest_path, args=args, cfg=cfg, n_params=n_params, entries=manifest, schedule=schedule)

    print(f"[*] wrote {manifest_path}")
    print(f"[*] scheduled probe checkpoints are under {snapshot_dir}")
    print(f"[*] best checkpoint is {out_dir / 'best.pt'}")
    print(f"total_sec={time.perf_counter() - t0:.3f}")


if __name__ == "__main__":
    main()
