#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


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
    return v if math.isfinite(v) else None


@dataclass(frozen=True)
class Arch:
    arch_id: int
    name: str
    description: str
    axis: str
    n_layers: int = 4
    rope_layout: str = "seq_time_rope"
    norm_style: str = "layernorm"
    rope_seq_dim: int = 0
    rope_time_dim: int = 0

    def train_args(self) -> list[str]:
        return [
            "--n-layers", str(self.n_layers),
            "--d-model", "128",
            "--n-heads", "4",
            "--rope-layout", self.rope_layout,
            "--rope-seq-dim", str(self.rope_seq_dim),
            "--rope-time-dim", str(self.rope_time_dim),
            "--norm-style", self.norm_style,
        ]


def paper_arches() -> list[Arch]:
    return [
        Arch(1, "no_position", "No positional encoding", "position-control baseline", rope_layout="none"),
        Arch(2, "learned_absolute", "Learned absolute position embeddings", "standard learned-position control", rope_layout="learned_abs"),
        Arch(3, "sequence_rope", "Sequence-position RoPE only", "sequence order without time", rope_layout="seq_rope"),
        Arch(4, "time_rope", "Continuous-time RoPE only", "timing without sequence RoPE", rope_layout="time_rope"),
        Arch(5, "seq_time_rope", "Factored sequence-time RoPE", "base architecture", rope_layout="seq_time_rope"),
        Arch(6, "seq_time_rope_rmsnorm", "RMSNorm instead of LayerNorm", "normalization axis", rope_layout="seq_time_rope", norm_style="rmsnorm"),
        Arch(7, "seq_time_rope_8_24", "8 sequence dims, 24 time dims per head", "time-heavy RoPE allocation", rope_layout="seq_time_rope", rope_seq_dim=8, rope_time_dim=24),
        Arch(8, "seq_time_rope_24_8", "24 sequence dims, 8 time dims per head", "sequence-heavy RoPE allocation", rope_layout="seq_time_rope", rope_seq_dim=24, rope_time_dim=8),
        Arch(9, "seq_time_rope_3_layers", "Reduced depth", "depth axis", n_layers=3, rope_layout="seq_time_rope"),
        Arch(10, "seq_time_rope_5_layers", "Increased depth", "depth axis", n_layers=5, rope_layout="seq_time_rope"),
    ]


def run_cmd(cmd: list[str], *, cwd: Path, env: dict[str, str], log_path: Path, label: str, dry_run: bool) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{now()}] {label}\n")
        f.write("CMD " + " ".join(cmd) + "\n")
    if dry_run:
        return 0.0
    t0 = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as f:
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
            stripped = line.rstrip()
            if any(tok in stripped for tok in ("device=", "epoch 00", "feature_batches=", "total_sec=", "checkpoint_sec=", "val_acc=")):
                print(f"{label} | {stripped}", flush=True)
        ret = proc.wait()
    sec = time.perf_counter() - t0
    if ret != 0:
        raise RuntimeError(f"{label} failed with exit code {ret}; see {log_path}")
    return sec


def build_train_cmd(args: argparse.Namespace, arch: Arch, seed: int, out_dir: Path) -> list[str]:
    cmd = [
        args.python, args.train_script,
        "--data", args.data,
        "--vocab", args.vocab,
        "--out-dir", str(out_dir / "ckpts"),
        "--epochs", str(args.epochs),
        "--epoch1-snapshots", str(args.epoch1_snapshots),
        "--batch-size", str(args.train_batch_size),
        "--ctx-len", str(args.ctx_len),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--split-stratify-col", "scenario_label",
        "--seed", str(seed),
        "--omit-optimizer",
        "--tf32",
        "--deterministic",
        "--fused-adamw",
        "--pin-memory",
    ]
    cmd.extend(arch.train_args())
    return cmd


def build_k95_cmd(args: argparse.Namespace, seed: int, out_dir: Path) -> list[str]:
    return [
        args.python, args.k95_script,
        "--data", args.data,
        "--vocab", args.vocab,
        "--transformer-script", args.train_script,
        "--snapshot-dir", str(out_dir / "ckpts" / "snapshots"),
        "--out-dir", str(out_dir / "eval" / "k95"),
        "--ctx-len", str(args.ctx_len),
        "--batch-size", str(args.k95_batch_size),
        "--n-sequences", str(args.k95_n_sequences),
        "--max-tokens", str(args.k95_max_tokens),
        "--var-threshold", str(args.var_threshold),
        "--seed", str(seed),
        "--stratify", "fault_variant",
        "--tf32",
        "--deterministic",
        "--no-plot",
    ]


def build_probe_cmd(args: argparse.Namespace, seed: int, out_dir: Path) -> list[str]:
    return [
        args.python, args.probe_script,
        "--orchestrator-split",
        "--no-standardize",
        "--data", args.data,
        "--vocab", args.vocab,
        "--transformer-script", args.train_script,
        "--snapshot-dir", str(out_dir / "ckpts" / "snapshots"),
        "--split-json", str(out_dir / "ckpts" / "scenario_split.json"),
        "--out-csv", str(out_dir / "eval" / "probe_hard.csv"),
        "--ctx-len", str(args.ctx_len),
        "--probe-layer", "1",
        "--batch-size", str(args.probe_feature_batch_size),
        "--probe-epochs", str(args.probe_epochs),
        "--probe-lr", str(args.probe_lr),
        "--probe-weight-decay", str(args.probe_weight_decay),
        "--probe-batch-size", str(args.probe_batch_size),
        "--eval-every", str(args.probe_eval_every),
        "--early-stop-patience", str(args.probe_early_stop_patience),
        "--var-threshold", str(args.var_threshold),
        "--seed", str(seed),
        "--tf32",
        "--deterministic",
    ]


def merge_run(args: argparse.Namespace, arch: Arch, seed: int, out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    probe_rows = read_csv(out_dir / "eval" / "probe_hard.csv")
    k_rows = [r for r in read_csv(out_dir / "eval" / "k95" / "k_spectral_metrics.csv") if int(float(r["layer"])) == 1]
    manifest = json.loads((out_dir / "ckpts" / "snapshot_manifest.json").read_text(encoding="utf-8"))
    loss_by_name = {c["name"]: c for c in manifest["checkpoints"]}
    if len(probe_rows) != len(k_rows):
        raise RuntimeError(f"row mismatch for arch={arch.arch_id} seed={seed}: probe={len(probe_rows)} k95={len(k_rows)}")

    details: list[dict[str, Any]] = []
    for pr, kr in zip(probe_rows, k_rows):
        ckpt_name = pr["checkpoint_name"]
        loss = loss_by_name.get(ckpt_name, {})
        train_loss = safe_float(loss.get("train_loss"))
        val_loss = safe_float(loss.get("val_loss"))
        details.append({
            "arch_id": arch.arch_id,
            "arch_name": arch.name,
            "arch_description": arch.description,
            "arch_axis": arch.axis,
            "seed": seed,
            "checkpoint_number": int(float(pr["checkpoint_number"])),
            "checkpoint_name": ckpt_name,
            "checkpoint_step": int(float(pr["checkpoint_step"])),
            "spectral_k_after_layer1": int(float(kr["k_var"])),
            "probe_val_true_class_probability": safe_float(pr["probe_val_true_class_probability"]),
            "probe_val_top1_accuracy": safe_float(pr["probe_val_top1_accuracy"]),
            "probe_train_top1_accuracy": safe_float(pr["probe_train_top1_accuracy"]),
            "probe_val_ce_loss": safe_float(pr["probe_val_ce_loss"]),
            "lm_train_loss": train_loss,
            "lm_train_ppl": math.exp(min(train_loss, 20.0)) if train_loss is not None else None,
            "lm_val_loss": val_loss,
            "lm_val_ppl": math.exp(min(val_loss, 20.0)) if val_loss is not None else None,
        })
    write_csv(out_dir / "merged_metrics.csv", details)

    p0 = details[0]["probe_val_top1_accuracy"]
    pC = details[-1]["probe_val_top1_accuracy"]
    min_k = min(int(r["spectral_k_after_layer1"]) for r in details)
    nadir = next(r for r in details if int(r["spectral_k_after_layer1"]) == min_k)
    target_k = int(round(args.pnc_alpha * min_k))
    post = [r for r in details if int(r["checkpoint_number"]) >= int(nadir["checkpoint_number"])]
    eligible = [r for r in post if int(r["spectral_k_after_layer1"]) >= target_k]
    pnc = min(eligible if eligible else post, key=lambda r: (abs(int(r["spectral_k_after_layer1"]) - target_k), int(r["checkpoint_step"])))
    den = pC - p0 if p0 is not None and pC is not None else None
    summary = {
        "arch_id": arch.arch_id,
        "arch_name": arch.name,
        "arch_description": arch.description,
        "arch_axis": arch.axis,
        "seed": seed,
        "run_dir": str(out_dir),
        "min_k95": min_k,
        "nadir_checkpoint_number": nadir["checkpoint_number"],
        "nadir_checkpoint_step": nadir["checkpoint_step"],
        "p0_random_init_accuracy": p0,
        "pN_nadir_accuracy": nadir["probe_val_top1_accuracy"],
        "pC_final_accuracy": pC,
        "f_compression": ((nadir["probe_val_top1_accuracy"] - p0) / den) if den and abs(den) > 1e-12 else None,
        "pNC_target_k95": target_k,
        "pNC_checkpoint_number": pnc["checkpoint_number"],
        "pNC_checkpoint_step": pnc["checkpoint_step"],
        "pNC_k95": pnc["spectral_k_after_layer1"],
        "pNC_accuracy": pnc["probe_val_top1_accuracy"],
        "f_compression_pNC": ((pnc["probe_val_top1_accuracy"] - p0) / den) if den and abs(den) > 1e-12 else None,
        "final_lm_train_ppl": details[-1]["lm_train_ppl"],
        "final_lm_val_ppl": details[-1]["lm_val_ppl"],
        "final_k95": details[-1]["spectral_k_after_layer1"],
        "best_probe_accuracy": max(float(r["probe_val_top1_accuracy"]) for r in details),
        "n_checkpoints": len(details),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return details, summary


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Simple local/cluster runner for the paper architecture grid.")
    p.add_argument("--root", type=Path, default=Path("runs_paper_grid"))
    p.add_argument("--run-tag", default=datetime.now().strftime("paper_grid_%Y%m%d_%H%M%S"))
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--data", default="int_dataset_v1_tokenized.parquet")
    p.add_argument("--vocab", default="tokenizer_vocab_locked_v1.json")
    p.add_argument("--train-script", default="fast_train_transformer_v0_seqtime_probe_ckpts.py")
    p.add_argument("--k95-script", default="fast_probe_k95.py")
    p.add_argument("--probe-script", default="fast_probe_rc_scenario_ckpts.py")
    p.add_argument("--arch-ids", default="all")
    p.add_argument("--seeds", default="1401,1402,1403,1404,1405,1406,1407,1408,1409,1410", help="one seed per selected architecture, or one seed reused if only one value is provided")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--epoch1-snapshots", type=int, default=30)
    p.add_argument("--ctx-len", type=int, default=1024)
    p.add_argument("--train-batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--k95-batch-size", type=int, default=128)
    p.add_argument("--k95-n-sequences", type=int, default=100)
    p.add_argument("--k95-max-tokens", type=int, default=50000)
    p.add_argument("--probe-feature-batch-size", type=int, default=512)
    p.add_argument("--probe-batch-size", type=int, default=1024)
    p.add_argument("--probe-epochs", type=int, default=200)
    p.add_argument("--probe-lr", type=float, default=1e-3)
    p.add_argument("--probe-weight-decay", type=float, default=1e-4)
    p.add_argument("--probe-eval-every", type=int, default=5)
    p.add_argument("--probe-early-stop-patience", type=int, default=50)
    p.add_argument("--var-threshold", type=float, default=0.95)
    p.add_argument("--pnc-alpha", type=float, default=1.2)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    repo = Path.cwd().resolve()
    run_root = (args.root / args.run_tag).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    arches = paper_arches()
    if args.arch_ids != "all":
        keep = {int(x) for x in args.arch_ids.split(",") if x.strip()}
        arches = [a for a in arches if a.arch_id in keep]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if len(seeds) == 1:
        seeds = seeds * len(arches)
    if len(seeds) != len(arches):
        raise ValueError("--seeds must provide either one seed or one seed per selected architecture")

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    metadata = {
        "started": now(),
        "repo": str(repo),
        "args": vars(args),
        "script_sha256": {
            args.train_script: sha256_file(repo / args.train_script),
            args.k95_script: sha256_file(repo / args.k95_script),
            args.probe_script: sha256_file(repo / args.probe_script),
            Path(__file__).name: sha256_file(Path(__file__)),
        },
        "architectures": [a.__dict__ for a in arches],
    }
    (run_root / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    all_details: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    total_t0 = time.perf_counter()
    for arch, seed in zip(arches, seeds):
        run_dir = run_root / f"arch_{arch.arch_id:02d}_{arch.name}" / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{now()}] RUN arch={arch.arch_id}:{arch.name} seed={seed} dir={run_dir}", flush=True)
        t0 = time.perf_counter()
        train_sec = run_cmd(build_train_cmd(args, arch, seed, run_dir), cwd=repo, env=env, log_path=run_dir / "logs" / "train.log", label=f"train arch{arch.arch_id}", dry_run=args.dry_run)
        k95_sec = run_cmd(build_k95_cmd(args, seed, run_dir), cwd=repo, env=env, log_path=run_dir / "logs" / "k95.log", label=f"k95 arch{arch.arch_id}", dry_run=args.dry_run)
        probe_sec = run_cmd(build_probe_cmd(args, seed, run_dir), cwd=repo, env=env, log_path=run_dir / "logs" / "probe.log", label=f"probe arch{arch.arch_id}", dry_run=args.dry_run)
        if args.dry_run:
            continue
        details, summary = merge_run(args, arch, seed, run_dir)
        summary["train_sec"] = train_sec
        summary["k95_sec"] = k95_sec
        summary["probe_sec"] = probe_sec
        summary["total_sec"] = time.perf_counter() - t0
        summaries.append(summary)
        all_details.extend(details)
        print(
            f"[{now()}] DONE arch={arch.arch_id}:{arch.name} seed={seed} "
            f"min_k={summary['min_k95']} p0={summary['p0_random_init_accuracy']:.4f} "
            f"pN={summary['pN_nadir_accuracy']:.4f} pC={summary['pC_final_accuracy']:.4f} "
            f"sec={summary['total_sec']:.1f}",
            flush=True,
        )

    if not args.dry_run:
        write_csv(run_root / "paper_grid_summary.csv", summaries)
        write_csv(run_root / "paper_grid_details.csv", all_details)
        collated = {
            "metadata": metadata,
            "summary_rows": summaries,
            "detail_rows": all_details,
            "total_sec": time.perf_counter() - total_t0,
        }
        (run_root / "paper_grid_collated.json").write_text(json.dumps(collated, indent=2, default=str), encoding="utf-8")
        print(f"[{now()}] WROTE {run_root / 'paper_grid_summary.csv'}", flush=True)
        print(f"[{now()}] WROTE {run_root / 'paper_grid_collated.json'}", flush=True)
        print(f"[{now()}] TOTAL_SEC {time.perf_counter() - total_t0:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
