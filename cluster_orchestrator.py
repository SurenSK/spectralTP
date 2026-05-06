#!/usr/bin/env python3
from __future__ import annotations

"""
Single-node SLURM orchestrator for the ORAN spectral-compression transformer sweep.

Controller mode:
    uv run python cluster_orchestrator.py --root "$SCRATCH/spectralTP_runs" --gpus 0,1,2

LM-eval child mode, normally called by the controller:
    python cluster_orchestrator.py eval-lm --snapshot-dir ... --split-json ... --out-csv ...

The controller keeps one sequential run pipeline per GPU:
    train -> LM validation over every checkpoint -> token k95 -> scenario probe -> summary/log.
"""

import argparse
import csv
import dataclasses
import functools
import glob
import hashlib
import importlib.util
import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------- common utilities ----------------


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return "MISSING"
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_float(x: Any) -> float | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        v = float(s)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def fmt(x: Any, nd: int = 4) -> str:
    v = safe_float(x)
    if v is None:
        return "NA"
    return f"{v:.{nd}f}"


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    with p.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_csv_row(path: Path, row: dict[str, Any], lock: threading.Lock | None = None) -> None:
    def _write():
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists() and path.stat().st_size > 0
        keys = list(row.keys())
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            if not exists:
                w.writeheader()
            w.writerow(row)

    if lock is None:
        _write()
    else:
        with lock:
            _write()


def import_from_path(module_name: str, path: str | Path):
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"missing module path: {p}")
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
    import torch
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


# ---------------- architecture grid ----------------


@dataclass(frozen=True)
class Arch:
    arch_id: int
    name: str
    n_layers: int = 4
    d_model: int = 128
    n_heads: int = 4
    ctx_len: int = 1024
    rope_layout: str = "seq_time_rope"
    rope_seq_dim: int = 0
    rope_time_dim: int = 0
    norm_style: str = "layernorm"
    dropout: float = 0.1
    extra_train_args: tuple[str, ...] = field(default_factory=tuple)

    def train_args(self) -> list[str]:
        args = [
            "--n-layers", str(self.n_layers),
            "--d-model", str(self.d_model),
            "--n-heads", str(self.n_heads),
            "--ctx-len", str(self.ctx_len),
            "--rope-layout", self.rope_layout,
            "--rope-seq-dim", str(self.rope_seq_dim),
            "--rope-time-dim", str(self.rope_time_dim),
            "--norm-style", self.norm_style,
            "--dropout", str(self.dropout),
        ]
        args.extend(self.extra_train_args)
        return args


def default_arches() -> list[Arch]:
    # Ten-architecture grid from the paper plan.  Per-head dim is 32 for d=128,H=4,
    # so seq_time_rope auto allocation is 16/16.
    return [
        Arch(1, "none_layernorm", rope_layout="none"),
        Arch(2, "learned_abs_layernorm", rope_layout="learned_abs"),
        Arch(3, "seq_rope_layernorm", rope_layout="seq_rope"),
        Arch(4, "time_rope_layernorm", rope_layout="time_rope"),
        Arch(5, "seq_time_rope_layernorm_base", rope_layout="seq_time_rope"),
        Arch(6, "seq_time_rope_rmsnorm", rope_layout="seq_time_rope", norm_style="rmsnorm"),
        Arch(7, "seq_time_rope_dims_8_24", rope_layout="seq_time_rope", rope_seq_dim=8, rope_time_dim=24),
        Arch(8, "seq_time_rope_dims_24_8", rope_layout="seq_time_rope", rope_seq_dim=24, rope_time_dim=8),
        Arch(9, "seq_time_rope_3layers", rope_layout="seq_time_rope", n_layers=3),
        Arch(10, "seq_time_rope_5layers", rope_layout="seq_time_rope", n_layers=5),
    ]


@dataclass
class RunTask:
    arch: Arch
    seed: int
    seed_index: int
    epochs: int

    @property
    def run_id(self) -> str:
        return f"arch{self.arch.arch_id:02d}_{self.arch.name}_seed{self.seed}"


# ---------------- controller logging ----------------


class ControllerLog:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.text_path = self.root / "log.txt"
        self.events_path = self.root / "events.jsonl"
        self.summary_path = self.root / "summary.csv"
        self._lock = threading.Lock()

    def line(self, msg: str) -> None:
        s = f"[{now()}] {msg}"
        with self._lock:
            with self.text_path.open("a", encoding="utf-8") as f:
                f.write(s + "\n")
            print(s, flush=True)

    def event(self, **payload: Any) -> None:
        payload = {"ts": now(), **payload}
        with self._lock:
            with self.events_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, sort_keys=True) + "\n")

    def summary(self, row: dict[str, Any]) -> None:
        append_csv_row(self.summary_path, row, self._lock)


# ---------------- subprocess phases ----------------


def run_subprocess(
    *,
    cmd: list[str],
    env: dict[str, str],
    cwd: Path,
    phase_log: Path,
    global_log: ControllerLog,
    label: str,
    relay_patterns: tuple[str, ...] = (),
    dry_run: bool = False,
) -> int:
    phase_log.parent.mkdir(parents=True, exist_ok=True)
    global_log.line(f"START {label}")
    global_log.event(kind="phase_start", label=label, cmd=cmd, cwd=str(cwd))

    if dry_run:
        with phase_log.open("a", encoding="utf-8") as f:
            f.write("DRY RUN\n")
            f.write(" ".join(cmd) + "\n")
        global_log.line(f"DRYRUN {label} :: {' '.join(cmd)}")
        return 0

    t0 = time.perf_counter()
    with phase_log.open("a", encoding="utf-8") as f:
        f.write(f"[{now()}] CMD {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            f.write(line)
            f.flush()
            stripped = line.rstrip("\n")
            if relay_patterns and any(pat in stripped for pat in relay_patterns):
                global_log.line(f"{label} | {stripped}")
        ret = proc.wait()

    sec = time.perf_counter() - t0
    if ret == 0:
        global_log.line(f"DONE {label} sec={sec:.1f}")
        global_log.event(kind="phase_done", label=label, sec=sec, returncode=ret)
    else:
        global_log.line(f"ERROR {label} returncode={ret} sec={sec:.1f} log={phase_log}")
        global_log.event(kind="phase_error", label=label, sec=sec, returncode=ret, log=str(phase_log))
    return ret


# ---------------- LM validation evaluator child mode ----------------


def eval_lm_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Evaluate LM validation loss over every saved checkpoint.")
    p.add_argument("--data", required=True)
    p.add_argument("--vocab", required=True)  # kept for metadata; checkpoints contain pad_id/cfg
    p.add_argument("--transformer-script", required=True)
    p.add_argument("--snapshot-dir", required=True)
    p.add_argument("--split-json", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--ctx-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--empty-cache-each", action="store_true")
    args = p.parse_args(argv)

    import pyarrow.parquet as pq
    import torch
    from torch.utils.data import DataLoader

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)

    mod = import_from_path("cluster_eval_lm_transformer", args.transformer_script)
    split = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
    val_sids = set(str(s) for s in split.get("val_scenario_ids", []))
    if not val_sids:
        raise RuntimeError(f"no val_scenario_ids found in {args.split_json}")

    tab = pq.read_table(args.data, columns=["scenario_id", "tokens", "token_times"])
    scenario_ids = tab["scenario_id"].to_pylist()
    tokens = tab["tokens"].to_pylist()
    times = tab["token_times"].to_pylist()
    val_rows = []
    for sid, toks, tms in zip(scenario_ids, tokens, times):
        if str(sid) not in val_sids:
            continue
        n = min(len(toks), len(tms))
        if n >= 2:
            val_rows.append((list(toks[:n]), [float(x) for x in tms[:n]]))
    if not val_rows:
        raise RuntimeError("no validation rows after filtering by split")

    ckpts = sorted(Path(args.snapshot_dir).glob("*.pt"), key=lambda x: natural_key(x.name))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints in {args.snapshot_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device={device} val_rows={len(val_rows)} checkpoints={len(ckpts)} out={args.out_csv}", flush=True)

    # cfg is re-read per checkpoint because architecture is in the checkpoint payload.
    rows: list[dict[str, Any]] = []
    for i, ckpt_path in enumerate(ckpts, start=1):
        t0 = time.perf_counter()
        ckpt = safe_torch_load(ckpt_path, device)
        cfg_raw = ckpt.get("cfg") or ckpt.get("config") or ckpt.get("model_cfg")
        if cfg_raw is None:
            raise KeyError(f"{ckpt_path} has no cfg/config/model_cfg")
        if dataclasses.is_dataclass(cfg_raw):
            cfg_raw = dataclasses.asdict(cfg_raw)
        elif not isinstance(cfg_raw, dict):
            cfg_raw = vars(cfg_raw)
        cfg_raw = dict(cfg_raw)
        cfg_raw["ctx_len"] = min(int(cfg_raw.get("ctx_len", args.ctx_len)), int(args.ctx_len))
        cfg_keys = {f.name for f in dataclasses.fields(mod.GPTConfig)}
        cfg = mod.GPTConfig(**{k: v for k, v in cfg_raw.items() if k in cfg_keys})
        model = mod.TinyGPT(cfg).to(device)
        state = ckpt.get("model") or ckpt.get("state_dict") or ckpt.get("model_state_dict")
        if state is None:
            raise KeyError(f"{ckpt_path} has no model state")
        model.load_state_dict(state, strict=True)
        model.eval()

        collate = functools.partial(mod.collate_causal, ctx_len=cfg.ctx_len, pad_id=cfg.pad_id)
        loader = DataLoader(
            mod.TokenRows(val_rows),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        val_loss = float(mod.eval_epoch(model, loader, device, args.amp))
        val_ppl = math.exp(min(val_loss, 20.0))
        train_loss = safe_float(ckpt.get("train_loss"))
        train_ppl = math.exp(min(train_loss, 20.0)) if train_loss is not None else None
        global_step = int(ckpt.get("global_step") or 0)
        rec = {
            "checkpoint_number": i,
            "checkpoint_name": ckpt_path.name,
            "checkpoint_step": global_step,
            "epoch": ckpt.get("epoch"),
            "global_step": global_step,
            "lm_train_loss": train_loss,
            "lm_train_ppl": train_ppl,
            "lm_val_loss": val_loss,
            "lm_val_ppl": val_ppl,
            "eval_sec": time.perf_counter() - t0,
        }
        rows.append(rec)
        print(
            f"[*] {i}/{len(ckpts)} {ckpt_path.name} step={global_step} "
            f"val_loss={val_loss:.6f} val_ppl={val_ppl:.4f} sec={rec['eval_sec']:.2f}",
            flush=True,
        )
        del model, ckpt
        if device.type == "cuda" and args.empty_cache_each:
            torch.cuda.empty_cache()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[*] wrote {out_csv}", flush=True)
    return 0


# ---------------- summary extraction ----------------


def summarize_run(
    *,
    task: RunTask,
    run_dir: Path,
    alpha: float,
    k95_layer: int,
    lm_eval_enabled: bool,
) -> dict[str, Any]:
    k95_csv = run_dir / "eval" / "k95" / "k_spectral_metrics.csv"
    probe_csv = run_dir / "eval" / "probe_hard.csv"
    lm_csv = run_dir / "eval" / "lm_metrics.csv"

    krows_all = read_csv_rows(k95_csv)
    krows = []
    for r in krows_all:
        if int(float(r.get("layer", -1))) == int(k95_layer):
            order = int(float(r["ckpt_order"]))
            krows.append({"order": order, "ckpt_num": order + 1, "k": int(float(r["k_var"])), "step": int(float(r.get("step", order)))})
    if not krows:
        raise RuntimeError(f"no k95 rows for layer={k95_layer} in {k95_csv}")
    krows.sort(key=lambda r: r["order"])

    probe_rows = read_csv_rows(probe_csv)
    probe_by_num: dict[int, dict[str, str]] = {}
    for r in probe_rows:
        probe_by_num[int(float(r["checkpoint_number"]))] = r

    lm_by_num: dict[int, dict[str, str]] = {}
    if lm_eval_enabled and lm_csv.exists():
        for r in read_csv_rows(lm_csv):
            lm_by_num[int(float(r["checkpoint_number"]))] = r

    first = krows[0]
    final = krows[-1]
    min_k = min(r["k"] for r in krows)
    nadir = next(r for r in krows if r["k"] == min_k)
    target = int(round(alpha * min_k))
    post = [r for r in krows if r["order"] >= nadir["order"]]
    eligible = [r for r in post if r["k"] >= target]
    pnc = eligible[0] if eligible else final

    def acc_at(ckpt_num: int) -> float | None:
        row = probe_by_num.get(ckpt_num)
        if not row:
            return None
        return safe_float(row.get("probe_val_top1_accuracy"))

    p0 = acc_at(first["ckpt_num"])
    pN = acc_at(nadir["ckpt_num"])
    pNC = acc_at(pnc["ckpt_num"])
    pC = acc_at(final["ckpt_num"])

    final_lm = lm_by_num.get(final["ckpt_num"], {})
    final_train_ppl = safe_float(final_lm.get("lm_train_ppl"))
    final_val_ppl = safe_float(final_lm.get("lm_val_ppl"))

    denom = None if p0 is None or pC is None else (pC - p0)
    f_pnc = None
    if denom is not None and denom > 0 and pNC is not None and p0 is not None:
        f_pnc = (pNC - p0) / denom

    row = {
        "arch_id": task.arch.arch_id,
        "arch_name": task.arch.name,
        "seed": task.seed,
        "epochs": task.epochs,
        "run_id": task.run_id,
        "run_dir": str(run_dir),
        "min_k95": min_k,
        "nadir_checkpoint_number": nadir["ckpt_num"],
        "nadir_checkpoint_step": nadir["step"],
        "pNC_target_k95": target,
        "pNC_checkpoint_number": pnc["ckpt_num"],
        "pNC_checkpoint_step": pnc["step"],
        "pNC_k95": pnc["k"],
        "final_k95": final["k"],
        "p0_random_init_accuracy": p0,
        "pN_nadir_accuracy": pN,
        "pNC_accuracy": pNC,
        "pC_final_accuracy": pC,
        "pNC_minus_pC": None if pNC is None or pC is None else pNC - pC,
        "f_compression_pNC": f_pnc,
        "final_lm_train_ppl": final_train_ppl,
        "final_lm_val_ppl": final_val_ppl,
        "n_checkpoints": len(krows),
    }
    return row


# ---------------- pipeline worker ----------------


def build_train_cmd(args: argparse.Namespace, task: RunTask, run_dir: Path) -> list[str]:
    ckpt_dir = run_dir / "ckpts"
    cmd = [
        sys.executable, args.train_script,
        "--data", args.data,
        "--vocab", args.vocab,
        "--out-dir", str(ckpt_dir),
        "--epochs", str(task.epochs),
        "--snapshots-per-epoch", str(args.snapshots_per_epoch),
        "--batch-size", str(args.train_batch_size),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--val-frac", str(args.val_frac),
        "--seed", str(task.seed),
        "--split-stratify-col", args.split_stratify_col,
        "--num-workers", str(args.train_num_workers),
        "--prefetch-factor", str(args.prefetch_factor),
        "--omit-optimizer",
    ]
    if args.amp:
        cmd.append("--amp")
    if args.tf32:
        cmd.append("--tf32")
    if args.pin_memory:
        cmd.append("--pin-memory")
    if args.persistent_workers:
        cmd.append("--persistent-workers")
    if args.fused_adamw:
        cmd.append("--fused-adamw")
    if args.deterministic:
        cmd.extend(["--deterministic", "--deterministic-warn-only"])
    cmd.extend(task.arch.train_args())
    return cmd


def build_lm_eval_cmd(args: argparse.Namespace, run_dir: Path) -> list[str]:
    ckpt_dir = run_dir / "ckpts"
    return [
        sys.executable, str(Path(__file__).resolve()), "eval-lm",
        "--data", args.data,
        "--vocab", args.vocab,
        "--transformer-script", args.train_script,
        "--snapshot-dir", str(ckpt_dir / "snapshots"),
        "--split-json", str(ckpt_dir / "scenario_split.json"),
        "--out-csv", str(run_dir / "eval" / "lm_metrics.csv"),
        "--ctx-len", str(args.ctx_len),
        "--batch-size", str(args.lm_eval_batch_size),
        "--num-workers", str(args.lm_eval_num_workers),
        *( ["--amp"] if args.amp else [] ),
        *( ["--tf32"] if args.tf32 else [] ),
        "--empty-cache-each",
    ]


def build_k95_cmd(args: argparse.Namespace, run_dir: Path) -> list[str]:
    ckpt_dir = run_dir / "ckpts"
    return [
        sys.executable, args.k95_script,
        "--data", args.data,
        "--vocab", args.vocab,
        "--transformer-script", args.train_script,
        "--snapshot-dir", str(ckpt_dir / "snapshots"),
        "--out-dir", str(run_dir / "eval" / "k95"),
        "--ctx-len", str(args.ctx_len),
        "--batch-size", str(args.k95_batch_size),
        "--n-sequences", str(args.k95_n_sequences),
        "--max-tokens", str(args.k95_max_tokens),
        "--var-threshold", str(args.var_threshold),
        "--seed", str(args.k95_seed),
        "--stratify", args.k95_stratify,
        *( ["--amp"] if args.amp else [] ),
        *( ["--tf32"] if args.tf32 else [] ),
        *( ["--deterministic", "--deterministic-warn-only"] if args.deterministic else [] ),
        "--empty-cache-each",
        "--no-plot",
    ]


def build_probe_cmd(args: argparse.Namespace, task: RunTask, run_dir: Path) -> list[str]:
    ckpt_dir = run_dir / "ckpts"
    cmd = [
        sys.executable, args.probe_script,
        "--data", args.data,
        "--vocab", args.vocab,
        "--transformer-script", args.train_script,
        "--snapshot-dir", str(ckpt_dir / "snapshots"),
        "--split-json", str(ckpt_dir / "scenario_split.json"),
        "--out-csv", str(run_dir / "eval" / "probe_hard.csv"),
        "--ctx-len", str(args.ctx_len),
        "--probe-layer", str(args.probe_layer),
        "--batch-size", str(args.probe_feature_batch_size),
        "--probe-epochs", str(args.probe_epochs),
        "--probe-lr", str(args.probe_lr),
        "--probe-weight-decay", str(args.probe_weight_decay),
        "--probe-batch-size", str(args.probe_batch_size),
        "--eval-every", str(args.probe_eval_every),
        "--early-stop-patience", str(args.probe_early_stop_patience),
        "--var-threshold", str(args.var_threshold),
        "--seed", str(task.seed),
        *( ["--amp"] if args.amp else [] ),
        *( ["--tf32"] if args.tf32 else [] ),
        *( ["--deterministic", "--deterministic-warn-only"] if args.deterministic else [] ),
        "--empty-cache-each",
    ]
    if args.probe_orchestrator_split:
        cmd.append("--orchestrator-split")
    return cmd


def run_task_on_gpu(
    *,
    task: RunTask,
    gpu_id: str,
    args: argparse.Namespace,
    global_log: ControllerLog,
    repo_root: Path,
) -> None:
    run_dir = args.root / args.run_tag / f"arch_{task.arch.arch_id:02d}_{task.arch.name}" / f"seed_{task.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    phase_dir = run_dir / "phase_logs"

    label_base = f"gpu={gpu_id} arch={task.arch.arch_id:02d}:{task.arch.name} seed={task.seed} epochs={task.epochs}"
    if args.resume and status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") == "done":
                global_log.line(f"SKIP done {label_base} run_dir={run_dir}")
                return
        except Exception:
            pass

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("OMP_NUM_THREADS", str(args.omp_num_threads))
    if args.deterministic:
        env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    run_t0 = time.perf_counter()
    status_path.write_text(json.dumps({"status": "running", "gpu": gpu_id, "task": dataclasses.asdict(task.arch), "seed": task.seed, "started": now()}, indent=2))
    global_log.line(f"RUN_START {label_base} run_dir={run_dir}")
    global_log.event(kind="run_start", arch_id=task.arch.arch_id, arch_name=task.arch.name, seed=task.seed, epochs=task.epochs, gpu=gpu_id, run_dir=str(run_dir))

    try:
        train_done = (run_dir / "ckpts" / "snapshot_manifest.json").exists()
        if not (args.resume and train_done):
            ret = run_subprocess(
                cmd=build_train_cmd(args, task, run_dir), env=env, cwd=repo_root,
                phase_log=phase_dir / "train.log", global_log=global_log,
                label=f"TRAIN {label_base}",
                relay_patterns=("device=", "steps_per_epoch=", "epoch ", "[*] saved", "total_sec"),
                dry_run=args.dry_run,
            )
            if ret != 0:
                raise RuntimeError("train failed")
        else:
            global_log.line(f"SKIP train existing {label_base}")

        if not args.no_lm_eval:
            lm_done = (run_dir / "eval" / "lm_metrics.csv").exists()
            if not (args.resume and lm_done):
                ret = run_subprocess(
                    cmd=build_lm_eval_cmd(args, run_dir), env=env, cwd=repo_root,
                    phase_log=phase_dir / "lm_eval.log", global_log=global_log,
                    label=f"LM_EVAL {label_base}",
                    relay_patterns=("device=", "val_loss=", "[*] wrote"),
                    dry_run=args.dry_run,
                )
                if ret != 0:
                    raise RuntimeError("lm eval failed")
            else:
                global_log.line(f"SKIP lm_eval existing {label_base}")

        if not args.no_k95:
            k95_done = (run_dir / "eval" / "k95" / "k_spectral_metrics.csv").exists()
            if not (args.resume and k95_done):
                ret = run_subprocess(
                    cmd=build_k95_cmd(args, run_dir), env=env, cwd=repo_root,
                    phase_log=phase_dir / "k95.log", global_log=global_log,
                    label=f"K95 {label_base}",
                    relay_patterns=("device=", "checkpoint ", "layer=01", "k95=", "[*] wrote", "total_sec"),
                    dry_run=args.dry_run,
                )
                if ret != 0:
                    raise RuntimeError("k95 failed")
            else:
                global_log.line(f"SKIP k95 existing {label_base}")

        if not args.no_probe:
            probe_done = (run_dir / "eval" / "probe_hard.csv").exists()
            if not (args.resume and probe_done):
                ret = run_subprocess(
                    cmd=build_probe_cmd(args, task, run_dir), env=env, cwd=repo_root,
                    phase_log=phase_dir / "probe_hard.log", global_log=global_log,
                    label=f"PROBE_HARD {label_base}",
                    relay_patterns=("device=", "checkpoints=", "k95=", "val_acc=", "[*] wrote", "total_sec"),
                    dry_run=args.dry_run,
                )
                if ret != 0:
                    raise RuntimeError("hard probe failed")
            else:
                global_log.line(f"SKIP probe_hard existing {label_base}")

        summary = {}
        if not args.dry_run and not args.no_k95 and not args.no_probe:
            summary = summarize_run(
                task=task,
                run_dir=run_dir,
                alpha=args.pnc_alpha,
                k95_layer=args.k95_summary_layer,
                lm_eval_enabled=not args.no_lm_eval,
            )
            summary["gpu"] = gpu_id
            summary["total_run_sec"] = time.perf_counter() - run_t0
            (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            global_log.summary(summary)
            global_log.line(
                "SUMMARY "
                f"arch={summary['arch_id']:02d}:{summary['arch_name']} seed={summary['seed']} epochs={summary['epochs']} "
                f"min_k95={summary['min_k95']} nadir_ckpt={summary['nadir_checkpoint_number']} "
                f"p0={fmt(summary['p0_random_init_accuracy'])} pN={fmt(summary['pN_nadir_accuracy'])} "
                f"pNC={fmt(summary['pNC_accuracy'])} pC={fmt(summary['pC_final_accuracy'])} "
                f"pNC_minus_pC={fmt(summary['pNC_minus_pC'])} f_pNC={fmt(summary['f_compression_pNC'])} "
                f"final_val_ppl={fmt(summary['final_lm_val_ppl'])} run_min={summary['total_run_sec']/60:.1f}"
            )

        status_path.write_text(json.dumps({"status": "done", "finished": now(), "gpu": gpu_id, "summary": summary}, indent=2, sort_keys=True), encoding="utf-8")
        global_log.line(f"RUN_DONE {label_base} total_sec={time.perf_counter() - run_t0:.1f}")
        global_log.event(kind="run_done", arch_id=task.arch.arch_id, arch_name=task.arch.name, seed=task.seed, gpu=gpu_id, run_dir=str(run_dir), sec=time.perf_counter() - run_t0)
    except Exception as e:
        status_path.write_text(json.dumps({"status": "error", "finished": now(), "gpu": gpu_id, "error": repr(e)}, indent=2), encoding="utf-8")
        global_log.line(f"RUN_ERROR {label_base} error={e!r} run_dir={run_dir}")
        global_log.event(kind="run_error", arch_id=task.arch.arch_id, arch_name=task.arch.name, seed=task.seed, gpu=gpu_id, run_dir=str(run_dir), error=repr(e))
        raise


def worker_loop(gpu_id: str, q: queue.Queue[RunTask], args: argparse.Namespace, global_log: ControllerLog, repo_root: Path, errors: list[BaseException], error_lock: threading.Lock) -> None:
    while True:
        try:
            task = q.get_nowait()
        except queue.Empty:
            return
        try:
            run_task_on_gpu(task=task, gpu_id=gpu_id, args=args, global_log=global_log, repo_root=repo_root)
        except BaseException as e:
            with error_lock:
                errors.append(e)
            if not args.keep_going:
                # Empty the queue to encourage other workers to wind down after their current task.
                try:
                    while True:
                        q.get_nowait()
                except queue.Empty:
                    pass
                return
        finally:
            q.task_done()


# ---------------- controller main ----------------


def controller_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="ORAN spectral-compression grid orchestrator for one SLURM node.")
    p.add_argument("--root", type=Path, default=Path("runs_cluster"))
    p.add_argument("--run-tag", default="ntia_grid")
    p.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"), help="comma list of GPU ids visible on the node, e.g. 0,1,2")
    p.add_argument("--arch-ids", default="all", help="all or comma list, e.g. 1,3,5")
    p.add_argument("--seeds", default="1337,2337,3337,4337,5337")
    p.add_argument("--long-seed-index", type=int, default=0, help="which seed index per architecture gets --long-epochs")
    p.add_argument("--one-epoch", type=int, default=1)
    p.add_argument("--long-epochs", type=int, default=10)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--keep-going", action="store_true", help="continue remaining tasks after a run fails")

    # Scripts and data.
    p.add_argument("--data", default="rc_dataset_v1_tokenized.parquet")
    p.add_argument("--vocab", default="tokenizer_vocab_locked_v1.json")
    p.add_argument("--train-script", default="fast_train_transformer_v0_seqtime_probe_ckpts.py")
    p.add_argument("--k95-script", default="fast_probe_k95.py")
    p.add_argument("--probe-script", default="fast_probe_rc_scenario_ckpts.py")
    p.add_argument("--ctx-len", type=int, default=1024)
    p.add_argument("--split-stratify-col", default="scenario_label")

    # Training.
    p.add_argument("--snapshots-per-epoch", type=int, default=50)
    p.add_argument("--train-batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--train-num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fused-adamw", action=argparse.BooleanOptionalAction, default=True)

    # Eval/probing.
    p.add_argument("--no-lm-eval", action="store_true")
    p.add_argument("--lm-eval-batch-size", type=int, default=64)
    p.add_argument("--lm-eval-num-workers", type=int, default=0)
    p.add_argument("--no-k95", action="store_true")
    p.add_argument("--k95-batch-size", type=int, default=128)
    p.add_argument("--k95-n-sequences", type=int, default=256)
    p.add_argument("--k95-max-tokens", type=int, default=50000)
    p.add_argument("--k95-seed", type=int, default=1337)
    p.add_argument("--k95-stratify", default="fault_variant", choices=["", "fault_variant", "test_id", "verdict"])
    p.add_argument("--k95-summary-layer", type=int, default=1)
    p.add_argument("--no-probe", action="store_true")
    p.add_argument("--probe-layer", type=int, default=1)
    p.add_argument("--probe-feature-batch-size", type=int, default=512)
    p.add_argument("--probe-batch-size", type=int, default=1024)
    p.add_argument("--probe-epochs", type=int, default=600)
    p.add_argument("--probe-lr", type=float, default=1e-2)
    p.add_argument("--probe-weight-decay", type=float, default=1e-4)
    p.add_argument("--probe-eval-every", type=int, default=5)
    p.add_argument("--probe-early-stop-patience", type=int, default=50)
    p.add_argument("--probe-orchestrator-split", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--var-threshold", type=float, default=0.95)
    p.add_argument("--pnc-alpha", type=float, default=1.2)

    # CUDA/perf.
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--omp-num-threads", type=int, default=4)

    args = p.parse_args(argv)
    args.root = args.root.resolve()
    repo_root = Path.cwd().resolve()
    run_root = args.root / args.run_tag
    log = ControllerLog(run_root)

    # Basic metadata.
    scripts = [args.train_script, args.k95_script, args.probe_script, Path(__file__).resolve()]
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        git_commit = "UNKNOWN"
    metadata = {
        "started": now(),
        "repo_root": str(repo_root),
        "git_commit": git_commit,
        "args": vars(args),
        "script_sha256": {str(s): sha256_file(s) for s in scripts},
    }
    (run_root / "orchestrator_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str), encoding="utf-8")
    log.line(f"ORCH_START root={run_root} git={git_commit} gpus={args.gpus} dry_run={args.dry_run} resume={args.resume}")
    log.line(f"WATCH: tail -f {run_root / 'log.txt'}")
    log.line(f"SUMMARY_CSV: {run_root / 'summary.csv'}")

    # Build task queue.
    arches = default_arches()
    if args.arch_ids != "all":
        wanted = {int(x) for x in args.arch_ids.split(",") if x.strip()}
        arches = [a for a in arches if a.arch_id in wanted]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    tasks: list[RunTask] = []
    for arch in arches:
        for i, seed in enumerate(seeds):
            epochs = args.long_epochs if i == args.long_seed_index else args.one_epoch
            tasks.append(RunTask(arch=arch, seed=seed, seed_index=i, epochs=epochs))

    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        raise RuntimeError("no GPUs specified; pass --gpus 0,1,2")

    log.line(f"TASKS n={len(tasks)} arches={[a.arch_id for a in arches]} seeds={seeds} long_seed_index={args.long_seed_index} workers={gpus}")
    q: queue.Queue[RunTask] = queue.Queue()
    for t in tasks:
        q.put(t)

    errors: list[BaseException] = []
    error_lock = threading.Lock()
    threads = []
    for gpu_id in gpus:
        th = threading.Thread(target=worker_loop, args=(gpu_id, q, args, log, repo_root, errors, error_lock), daemon=False)
        th.start()
        threads.append(th)
    for th in threads:
        th.join()

    if errors:
        log.line(f"ORCH_DONE_WITH_ERRORS n_errors={len(errors)}")
        return 2
    log.line("ORCH_DONE all tasks completed")
    return 0


# ---------------- entry point ----------------


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "eval-lm":
        raise SystemExit(eval_lm_main(sys.argv[2:]))
    raise SystemExit(controller_main(sys.argv[1:]))
