#!/usr/bin/env python3
"""
Train one base LLM on one dataset under three regimes and compare validation perplexity (PPL).

Methods (aligned with run_stp.py naming intent):
  1. regular   — `stp.py --regular` (standard causal LM fine-tuning).
  2. stp       — `stp.py` RepresentationTrainer, default two-view cosine auxiliary (no --linear).
  3. dynamics  — `stp.py --dynamics_tube` (Lyapunov tube + optional local TS term).

PPL is exp(mean NLL) over all non-masked label positions (labels != -100), using the same
chat template / masking as each training path.

Examples
--------
  # Train all three (each run uses torchrun internally). Do NOT wrap this script in torchrun.
  python compare_three_method.py \\
      --model_name meta-llama/Llama-3.2-1B-Instruct \\
      --dataset_name synth --data_prefix datasets/ \\
      --num_epochs 1 --batch_size 2 --grad_accum 4 --nproc 2

  # Evaluate existing checkpoints only
  python compare_three_method.py --skip_train \\
      --model_name meta-llama/Llama-3.2-1B-Instruct \\
      --dataset_name synth --data_prefix datasets/ \\
      --ckpt_regular ./out-regular --ckpt_stp ./out-stp --ckpt_dynamics ./out-dynamics
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling



# Project root (directory containing this file)
ROOT = Path(__file__).resolve().parent


def _pick_free_port() -> int:
    """Bind to port 0 to get an ephemeral free port (avoids EADDRINUSE on default 29500)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _import_stp_dataset():
    """Lazy import so --help works without full training deps in some envs."""
    import stp  # noqa: WPS433 — intentional side-effect import of project module

    return stp.load_and_prepare_dataset


def build_eval_dataset(
    eval_jsonl: Path,
    tokenizer,
    model_name: str,
    max_length: int,
    method: str,
    predictors: int = 0,
):
    """
    method in {"regular", "stp", "dynamics"} — must match how the checkpoint was trained.
    """
    load_and_prepare_dataset = _import_stp_dataset()
    if method == "regular":
        return load_and_prepare_dataset(
            str(eval_jsonl),
            tokenizer,
            model_name,
            max_length=max_length,
            predictors=predictors,
            regular=True,
            linear=None,
        )
    linear = "dynamics" if method == "dynamics" else None
    return load_and_prepare_dataset(
        str(eval_jsonl),
        tokenizer,
        model_name,
        max_length=max_length,
        predictors=predictors,
        regular=False,
        linear=linear,
    )


@torch.no_grad()
def compute_perplexity(
    checkpoint_dir: Path,
    eval_jsonl: Path,
    base_model_name: str,
    method: str,
    max_length: int = 512,
    batch_size: int = 4,
    predictors: int = 0,
) -> float:
    """Mean NLL over label tokens (labels != -100); returns PPL = exp(nll)."""
    checkpoint_dir = Path(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_dir),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()

    ds = build_eval_dataset(
        eval_jsonl, tokenizer, base_model_name, max_length, method, predictors=predictors
    )
    cols = ["input_ids", "labels", "attention_mask"]
    ds = ds.remove_columns([c for c in ds.column_names if c not in cols])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=None)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0

    for batch in tqdm(loader, desc=f"PPL[{method}]"):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["labels"]
        out = model(**batch)
        # HF returns mean loss over non-ignored labels in the batch
        n_tokens = (labels != -100).sum().item()
        if n_tokens == 0:
            continue
        # Recover sum of NLL: loss is mean over valid positions in the batch
        total_nll += out.loss.item() * n_tokens
        total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")
    mean_nll = total_nll / total_tokens
    return math.exp(mean_nll)


def run_torchrun_stp(
    *,
    stp_py: Path,
    output_dir: Path,
    train_file: Path,
    model_name: str,
    num_epochs: int,
    learning_rate: float,
    finetune_seed: int,
    batch_size: int,
    grad_accum: int,
    max_length: int,
    nproc: int,
    method: str,
    last_token: int,
    lbd: float,
    predictors: int,
    tube_gamma: float = 0.95,
    tube_tau: float = 1e-3,
    lbd_ts: float = 0.0,
    tube_log_interval: int = 50,
):
    """Invoke `torchrun stp.py` for one training mode."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stp_args = [
        str(stp_py),
        "--train_file",
        str(train_file),
        "--output_dir",
        str(output_dir),
        "--num_epochs",
        str(num_epochs),
        "--finetune_seed",
        str(finetune_seed),
        "--model_name",
        model_name,
        "--learning_rate",
        str(learning_rate),
        "--batch_size",
        str(batch_size),
        "--grad_accum",
        str(grad_accum),
        "--max_length",
        str(max_length),
        "--last_token",
        str(last_token),
        "--lbd",
        str(lbd),
        "--predictors",
        str(predictors),
    ]

    if method == "regular":
        stp_args.append("--regular")
    elif method == "stp":
        pass  # RepresentationTrainer, default two-view cosine (no --linear)
    elif method == "dynamics":
        stp_args.extend(
            [
                "--dynamics_tube",
                "--tube_gamma",
                str(tube_gamma),
                "--tube_tau",
                str(tube_tau),
                "--lbd_ts",
                str(lbd_ts),
                "--tube_log_interval",
                str(tube_log_interval),
            ]
        )
    else:
        raise ValueError(method)

    env = os.environ.copy()
    # Fresh port each subprocess: a shell-exported MASTER_PORT would otherwise be reused
    # across all three torchrun calls and can cause EADDRINUSE.
    master_addr = env.get("MASTER_ADDR", "127.0.0.1")
    master_port = str(_pick_free_port())
    env["MASTER_ADDR"] = master_addr
    env["MASTER_PORT"] = master_port

    rdzv_flags = [
        f"--master-addr={master_addr}",
        f"--master-port={master_port}",
    ]
    if shutil.which("torchrun"):
        cmd = ["torchrun", f"--nproc_per_node={nproc}"] + rdzv_flags + stp_args
    else:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={nproc}",
        ] + rdzv_flags + stp_args

    print(
        f"\n>>> rendezvous {master_addr}:{master_port} (set MASTER_ADDR/MASTER_PORT to override)",
        flush=True,
    )
    print(">>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)


def parse_args():
    p = argparse.ArgumentParser(description="Compare regular vs STP vs dynamics training via PPL.")
    p.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--dataset_name", type=str, default="synth", help="Stem; uses {data_prefix}{name}_train.jsonl")
    p.add_argument("--data_prefix", type=str, default="datasets/", help="Prefix for train/test JSONL paths.")
    p.add_argument("--output_root", type=str, default="compare_three_runs", help="Directory for three checkpoints.")
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--finetune_seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--nproc", type=int, default=2, help="torchrun --nproc_per_node")
    p.add_argument("--last_token", type=int, default=-2)
    p.add_argument("--lbd", type=float, default=0.02, help="Aux loss weight for regular (unused) / stp.")
    p.add_argument(
        "--lbd_dynamics",
        type=float,
        default=0.01,
        help="Aux weight for dynamics tube only; ablation suggests sweeping ~0.01–0.05.",
    )
    p.add_argument("--tube_gamma", type=float, default=0.95, help="Dynamics: Lyapunov decay factor (passed to stp.py).")
    p.add_argument("--tube_tau", type=float, default=1e-3, help="Dynamics: tube slack tau (passed to stp.py).")
    p.add_argument(
        "--lbd_ts",
        type=float,
        default=0.0,
        help="Dynamics: temporal-straightening curvature weight; default 0 for pure Lyapunov-tube ablation.",
    )
    p.add_argument(
        "--tube_log_interval",
        type=int,
        default=50,
        help="Dynamics: print [tube_diag] every N steps (0 = off). Passed to stp.py.",
    )
    p.add_argument(
        "--only_dynamics",
        action="store_true",
        help="Train and compute PPL only for dynamics (skip regular and STP).",
    )
    p.add_argument("--predictors", type=int, default=0)
    p.add_argument("--eval_batch_size", type=int, default=4, help="Batch size for PPL only.")
    p.add_argument("--skip_train", action="store_true", help="Only compute PPL on existing dirs.")
    p.add_argument("--ckpt_regular", type=str, default=None)
    p.add_argument("--ckpt_stp", type=str, default=None)
    p.add_argument("--ckpt_dynamics", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    prefix = args.data_prefix
    train_file = ROOT / f"{prefix}{args.dataset_name}_train.jsonl".replace("//", "/")
    eval_file = ROOT / f"{prefix}{args.dataset_name}_test.jsonl".replace("//", "/")
    if not train_file.is_file():
        sys.exit(f"Missing train file: {train_file}")
    if not eval_file.is_file():
        sys.exit(f"Missing eval file: {eval_file}")

    out_root = ROOT / args.output_root
    out_root.mkdir(parents=True, exist_ok=True)
    dirs = {
        "regular": Path(args.ckpt_regular) if args.ckpt_regular else out_root / "regular",
        "stp": Path(args.ckpt_stp) if args.ckpt_stp else out_root / "stp",
        "dynamics": Path(args.ckpt_dynamics) if args.ckpt_dynamics else out_root / "dynamics",
    }

    stp_py = ROOT / "stp.py"
    if not stp_py.is_file():
        sys.exit(f"Missing {stp_py}")

    train_methods = ("dynamics",) if args.only_dynamics else ("dynamics", "stp", "regular")
    ppl_methods = ("dynamics",) if args.only_dynamics else ("regular", "stp", "dynamics")

    if not args.skip_train:
        for method in train_methods:
            lbd = args.lbd_dynamics if method == "dynamics" else args.lbd
            run_torchrun_stp(
                stp_py=stp_py,
                output_dir=dirs[method],
                train_file=train_file,
                model_name=args.model_name,
                num_epochs=args.num_epochs,
                learning_rate=args.learning_rate,
                finetune_seed=args.finetune_seed,
                batch_size=args.batch_size,
                grad_accum=args.grad_accum,
                max_length=args.max_length,
                nproc=args.nproc,
                method=method,
                last_token=args.last_token,
                lbd=lbd,
                predictors=args.predictors,
                tube_gamma=args.tube_gamma,
                tube_tau=args.tube_tau,
                lbd_ts=args.lbd_ts,
                tube_log_interval=args.tube_log_interval,
            )

    results = {}
    for method in ppl_methods:
        ckpt = dirs[method]
        if not ckpt.is_dir():
            print(f"Skip PPL: missing checkpoint dir {ckpt}")
            results[method] = None
            continue
        ppl = compute_perplexity(
            ckpt,
            eval_file,
            args.model_name,
            method,
            max_length=args.max_length,
            batch_size=args.eval_batch_size,
            predictors=args.predictors,
        )
        results[method] = ppl

    print("\n========== Perplexity comparison ==========")
    for m in ppl_methods:
        v = results.get(m)
        print(f"  {m:12s}  PPL = {v if v is not None else 'N/A'}")
    summary_path = out_root / "ppl_comparison.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "dataset": args.dataset_name,
                "eval_file": str(eval_file),
                "ppl": {k: (float(v) if v is not None and not math.isinf(v) else None) for k, v in results.items()},
            },
            f,
            indent=2,
        )
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
