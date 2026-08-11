"""Train a clean, reproducible cohort of AutocorrRoPE models from scratch.

This script is deliberately separate from the canonical notebook.  Its default
cohort contains the models used in the current cross-model analysis:

    attn_4L, attn_5L, attn_6L, attn_7L, mlp_4L

All models receive the same initialization seed, training batches, lag split,
optimizer, and number of optimizer updates.  This makes depth/MLP comparisons
controlled.  Each model is independently reset before training; no existing
checkpoint is loaded.

Default protocol
----------------
Architecture:       d_model=64, d_head=64, one head per layer
Data:               T=200, rho=0.9
Training lags:      1..50 except 15, 27, 38
Interpolation lags: 15, 27, 38
High-lag lags:      52, 55, 58
Loss:               per-sequence masked MSE after position lag + 30
Optimizer:          AdamW, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                    weight_decay=0.1, amsgrad=False, foreach=False,
                    fused=False
Scheduler:          none
Gradient clipping:  none
Steps:              300,000
Batch size:         64
Validation:         every 10,000 steps
Best checkpoint:    highest held-interior mean prediction correlation

The high-lag bucket is reported but is not used to select the best checkpoint.

Examples
--------
Print the full resolved protocol without training:

    python retrain_all_models.py --dry-run

Smoke-test both architecture families:

    python retrain_all_models.py --models attn_4L,mlp_4L --smoke-test

Run the complete cohort:

    python retrain_all_models.py --models all --run-name clean_cohort_seed_20260810

Run one model or override the common step budget:

    python retrain_all_models.py --models attn_7L --steps 400000

Outputs are written under:

    models/retrained_from_scratch/<run-name>/<model-name>/

Each model directory contains config.json, metrics.csv, model-only best/final
weights, and complete training-state checkpoints.  The complete checkpoints
include optimizer and RNG states so the exact run state is auditable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import shlex
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW

from model_class import AutocorrRoPE
from util import make_dataset_lagset


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "attn_4L": dict(d_model=64, d_head=64, n_layers=4, n_heads=1, use_mlp=False),
    "attn_5L": dict(d_model=64, d_head=64, n_layers=5, n_heads=1, use_mlp=False),
    "attn_6L": dict(d_model=64, d_head=64, n_layers=6, n_heads=1, use_mlp=False),
    "attn_7L": dict(d_model=64, d_head=64, n_layers=7, n_heads=1, use_mlp=False),
    "mlp_4L": dict(d_model=64, d_head=64, n_layers=4, n_heads=1, use_mlp=True),
}

DEFAULT_MODELS = tuple(MODEL_SPECS)


@dataclass(frozen=True)
class Protocol:
    sequence_length: int = 200
    rho: float = 0.9
    train_lag_min: int = 1
    train_lag_max: int = 50
    held_interior: tuple[int, ...] = (15, 27, 38)
    held_high: tuple[int, ...] = (52, 55, 58)
    burn_in: int = 30
    batch_size: int = 64
    steps: int = 300_000
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    adam_amsgrad: bool = False
    adam_maximize: bool = False
    adam_foreach: bool = False
    adam_capturable: bool = False
    adam_differentiable: bool = False
    adam_fused: bool = False
    eval_every: int = 10_000
    checkpoint_every: int = 50_000
    eval_sequences_per_bucket: int = 3_000
    eval_batch_size: int = 64
    seed: int = 20260810
    selection_metric: str = "held_interior_mean_correlation"
    scheduler: str = "none"
    gradient_clipping: str = "none"
    precision: str = "float32"

    @property
    def train_lags(self) -> tuple[int, ...]:
        held = set(self.held_interior)
        return tuple(
            lag
            for lag in range(self.train_lag_min, self.train_lag_max + 1)
            if lag not in held
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_int_tuple(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def parse_models(text: str) -> tuple[str, ...]:
    if text.strip().lower() == "all":
        return DEFAULT_MODELS
    names = tuple(part.strip() for part in text.split(",") if part.strip())
    unknown = set(names) - set(MODEL_SPECS)
    if unknown:
        choices = ", ".join(MODEL_SPECS)
        raise argparse.ArgumentTypeError(
            f"unknown model(s): {sorted(unknown)}; choices are {choices}, or all"
        )
    if not names:
        raise argparse.ArgumentTypeError("select at least one model")
    return names


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                args,
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return completed.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = run("git", "status", "--porcelain")
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "branch", "--show-current"),
        "dirty": None if status is None else bool(status),
        "status_porcelain": status,
    }


def environment_metadata(project_root: Path) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    return {
        "created_utc": utc_now(),
        "command": " ".join(shlex.quote(arg) for arg in sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_version": torch.version.cuda,
        "mps_available": torch.backends.mps.is_available(),
        "git": git_metadata(project_root),
        "source_sha256": {
            script_path.name: sha256_file(script_path),
            "model_class.py": sha256_file(project_root / "model_class.py"),
            "util.py": sha256_file(project_root / "util.py"),
        },
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def build_model(spec: dict[str, Any], device: torch.device) -> AutocorrRoPE:
    model = AutocorrRoPE(
        d_model=spec["d_model"],
        d_head=spec["d_head"],
        n_layers=spec["n_layers"],
        n_heads=spec["n_heads"],
        use_mlp=spec["use_mlp"],
        act_function="GELU",
    )
    return model.to(device=device, dtype=torch.float32)


def masked_mse_per_sequence(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lags: torch.Tensor,
    burn_in: int,
) -> torch.Tensor:
    positions = torch.arange(prediction.shape[1], device=prediction.device)
    valid = positions.unsqueeze(0) >= lags.unsqueeze(1) + burn_in
    squared_error = (prediction - target).square()
    valid_count = valid.sum(dim=1).clamp_min(1)
    return (squared_error * valid).sum(dim=1) / valid_count


def masked_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lags: torch.Tensor,
    burn_in: int,
) -> torch.Tensor:
    return masked_mse_per_sequence(prediction, target, lags, burn_in).mean()


def make_fixed_eval_sets(protocol: Protocol) -> dict[str, tuple[torch.Tensor, ...]]:
    buckets = {
        "train": (protocol.train_lags, protocol.seed + 101),
        "held_interior": (protocol.held_interior, protocol.seed + 102),
        "held_high": (protocol.held_high, protocol.seed + 103),
    }
    return {
        name: make_dataset_lagset(
            protocol.eval_sequences_per_bucket,
            protocol.sequence_length,
            protocol.rho,
            lags,
            seed=seed,
        )
        for name, (lags, seed) in buckets.items()
    }


@torch.inference_mode()
def predict_in_batches(
    model: AutocorrRoPE,
    inputs: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    predictions: list[torch.Tensor] = []
    for start in range(0, len(inputs), batch_size):
        batch = inputs[start : start + batch_size].to(device)
        output = model(batch)
        predictions.append(output[0].detach().cpu())
        del output, batch
    return torch.cat(predictions, dim=0)


def evaluate_bucket(
    model: AutocorrRoPE,
    dataset: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    protocol: Protocol,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    inputs, targets, lags = dataset
    predictions = predict_in_batches(
        model, inputs, device, protocol.eval_batch_size
    )
    per_sequence_mse = masked_mse_per_sequence(
        predictions, targets, lags, protocol.burn_in
    )

    rows: list[dict[str, Any]] = []
    correlations: list[float] = []
    for lag_tensor in torch.unique(lags, sorted=True):
        lag = int(lag_tensor.item())
        selected = lags == lag
        lag_predictions = predictions[selected]
        lag_inputs = inputs[selected]
        start = lag + protocol.burn_in
        predicted_values = lag_predictions[:, start:].reshape(-1).numpy()
        lagged_values = lag_inputs[
            :, start - lag + 1 : protocol.sequence_length - lag + 1
        ].reshape(-1).numpy()
        correlation = float(np.corrcoef(predicted_values, lagged_values)[0, 1])
        correlations.append(correlation)
        rows.append(
            {
                "lag": lag,
                "n_sequences": int(selected.sum().item()),
                "correlation": correlation,
                "masked_mse": float(per_sequence_mse[selected].mean().item()),
            }
        )

    aggregate = {
        "mean_correlation": float(np.nanmean(correlations)),
        "mean_masked_mse": float(per_sequence_mse.mean().item()),
    }
    return rows, aggregate


def evaluate_model(
    model: AutocorrRoPE,
    eval_sets: dict[str, tuple[torch.Tensor, ...]],
    protocol: Protocol,
    device: torch.device,
    step: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    was_training = model.training
    model.eval()
    all_rows: list[dict[str, Any]] = []
    aggregates: dict[str, dict[str, float]] = {}
    for bucket, dataset in eval_sets.items():
        lag_rows, aggregate = evaluate_bucket(model, dataset, protocol, device)
        aggregates[bucket] = aggregate
        for row in lag_rows:
            all_rows.append({"step": step, "bucket": bucket, **row})
        all_rows.append(
            {
                "step": step,
                "bucket": bucket,
                "lag": "mean",
                "n_sequences": len(dataset[0]),
                "correlation": aggregate["mean_correlation"],
                "masked_mse": aggregate["mean_masked_mse"],
            }
        )
    model.train(was_training)
    return all_rows, aggregates


def append_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "step",
                "bucket",
                "lag",
                "n_sequences",
                "correlation",
                "masked_mse",
            ),
        )
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def rng_state(train_rng: np.random.Generator) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python_random_state": random.getstate(),
        "numpy_global_state": np.random.get_state(),
        "numpy_training_generator_state": train_rng.bit_generator.state,
        "torch_cpu_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def training_checkpoint(
    model: AutocorrRoPE,
    optimizer: AdamW,
    train_rng: np.random.Generator,
    step: int,
    best_step: int,
    best_score: float,
    model_name: str,
    protocol: Protocol,
    model_spec: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "model_name": model_name,
        "model_spec": model_spec,
        "protocol": asdict(protocol),
        "step": step,
        "best_step": best_step,
        "best_score": best_score,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": rng_state(train_rng),
    }


def format_evaluation(aggregates: dict[str, dict[str, float]]) -> str:
    return "  ".join(
        f"{name}: corr={values['mean_correlation']:.4f}, "
        f"mse={values['mean_masked_mse']:.4f}"
        for name, values in aggregates.items()
    )


def train_one_model(
    model_name: str,
    model_spec: dict[str, Any],
    protocol: Protocol,
    device: torch.device,
    model_dir: Path,
    environment: dict[str, Any],
) -> dict[str, Any]:
    if model_dir.exists() and any(model_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty model directory: {model_dir}"
        )
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = model_dir / "checkpoints"
    checkpoints_dir.mkdir()

    # Reset every source of randomness before every model.  Consequently, all
    # architectures see the same training sequences in the same order, and
    # compatible leading parameters receive paired initializations.
    seed_everything(protocol.seed)
    train_rng = np.random.default_rng(protocol.seed)
    model = build_model(model_spec, device)
    optimizer = AdamW(
        model.parameters(),
        lr=protocol.learning_rate,
        betas=(protocol.adam_beta1, protocol.adam_beta2),
        eps=protocol.adam_eps,
        weight_decay=protocol.weight_decay,
        amsgrad=protocol.adam_amsgrad,
        maximize=protocol.adam_maximize,
        foreach=protocol.adam_foreach,
        capturable=protocol.adam_capturable,
        differentiable=protocol.adam_differentiable,
        fused=protocol.adam_fused,
    )
    eval_sets = make_fixed_eval_sets(protocol)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    config = {
        "model_name": model_name,
        "model_spec": model_spec,
        "parameter_count": parameter_count,
        "protocol": {**asdict(protocol), "train_lags": protocol.train_lags},
        "optimizer": {
            "class": "torch.optim.AdamW",
            "learning_rate": protocol.learning_rate,
            "betas": (protocol.adam_beta1, protocol.adam_beta2),
            "eps": protocol.adam_eps,
            "weight_decay": protocol.weight_decay,
            "amsgrad": protocol.adam_amsgrad,
            "maximize": protocol.adam_maximize,
            "foreach": protocol.adam_foreach,
            "capturable": protocol.adam_capturable,
            "differentiable": protocol.adam_differentiable,
            "fused": protocol.adam_fused,
            "resolved_torch_defaults": optimizer.defaults,
        },
        "device": str(device),
        "environment": environment,
        "status": "running",
        "started_utc": utc_now(),
    }
    config_path = model_dir / "config.json"
    atomic_write_json(config_path, config)

    metrics_path = model_dir / "metrics.csv"
    best_score = -math.inf
    best_step = -1
    started = time.monotonic()

    print(f"\n{'=' * 76}")
    print(
        f"{model_name}: {parameter_count:,} parameters; "
        f"device={device}; steps={protocol.steps:,}"
    )
    print(f"output: {model_dir}")

    initial_rows, initial_aggregates = evaluate_model(
        model, eval_sets, protocol, device, step=0
    )
    append_metrics(metrics_path, initial_rows)
    print(f"step 0  {format_evaluation(initial_aggregates)}")

    initial_score = initial_aggregates["held_interior"]["mean_correlation"]
    if np.isfinite(initial_score):
        best_score = initial_score
        best_step = 0
        atomic_torch_save(model.state_dict(), model_dir / "best_model.pt")
        atomic_torch_save(
            training_checkpoint(
                model,
                optimizer,
                train_rng,
                0,
                best_step,
                best_score,
                model_name,
                protocol,
                model_spec,
            ),
            model_dir / "best_training.pt",
        )

    latest_aggregates = initial_aggregates
    latest_evaluated_step = 0
    model.train()
    for step in range(1, protocol.steps + 1):
        inputs, targets, lags = make_dataset_lagset(
            protocol.batch_size,
            protocol.sequence_length,
            protocol.rho,
            protocol.train_lags,
            seed=train_rng,
        )
        inputs = inputs.to(device)
        targets = targets.to(device)
        lags = lags.to(device)

        output = model(inputs)
        loss = masked_loss(output[0], targets, lags, protocol.burn_in)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        del output, inputs, targets, lags

        should_evaluate = step % protocol.eval_every == 0 or step == protocol.steps
        if should_evaluate:
            rows, latest_aggregates = evaluate_model(
                model, eval_sets, protocol, device, step
            )
            latest_evaluated_step = step
            append_metrics(metrics_path, rows)
            elapsed_minutes = (time.monotonic() - started) / 60
            print(
                f"step {step:>7,d}/{protocol.steps:,}  "
                f"train_batch_loss={loss.item():.5f}  "
                f"elapsed={elapsed_minutes:.1f}m  "
                f"{format_evaluation(latest_aggregates)}"
            )

            score = latest_aggregates["held_interior"]["mean_correlation"]
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_step = step
                atomic_torch_save(model.state_dict(), model_dir / "best_model.pt")
                atomic_torch_save(
                    training_checkpoint(
                        model,
                        optimizer,
                        train_rng,
                        step,
                        best_step,
                        best_score,
                        model_name,
                        protocol,
                        model_spec,
                    ),
                    model_dir / "best_training.pt",
                )
                print(f"  new best held-interior correlation: {best_score:.5f}")

        if step % protocol.checkpoint_every == 0:
            atomic_torch_save(
                training_checkpoint(
                    model,
                    optimizer,
                    train_rng,
                    step,
                    best_step,
                    best_score,
                    model_name,
                    protocol,
                    model_spec,
                ),
                checkpoints_dir / f"step_{step:07d}_training.pt",
            )

    if latest_evaluated_step != protocol.steps:
        rows, latest_aggregates = evaluate_model(
            model, eval_sets, protocol, device, protocol.steps
        )
        append_metrics(metrics_path, rows)

    atomic_torch_save(model.state_dict(), model_dir / "final_model.pt")
    atomic_torch_save(
        training_checkpoint(
            model,
            optimizer,
            train_rng,
            protocol.steps,
            best_step,
            best_score,
            model_name,
            protocol,
            model_spec,
        ),
        model_dir / "final_training.pt",
    )

    elapsed_seconds = time.monotonic() - started
    summary = {
        "model_name": model_name,
        "parameter_count": parameter_count,
        "steps_completed": protocol.steps,
        "best_step": best_step,
        "best_held_interior_mean_correlation": best_score,
        "final_metrics": latest_aggregates,
        "elapsed_seconds": elapsed_seconds,
        "completed_utc": utc_now(),
    }
    config.update(
        {
            "status": "complete",
            "completed_utc": summary["completed_utc"],
            "elapsed_seconds": elapsed_seconds,
            "best_step": best_step,
            "best_score": best_score,
            "final_metrics": latest_aggregates,
        }
    )
    atomic_write_json(config_path, config)
    atomic_write_json(model_dir / "summary.json", summary)
    return summary


def write_cohort_results(path: Path, summaries: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "model_name",
                "parameter_count",
                "steps_completed",
                "best_step",
                "best_held_interior_mean_correlation",
                "final_train_correlation",
                "final_held_interior_correlation",
                "final_held_high_correlation",
                "final_train_mse",
                "final_held_interior_mse",
                "final_held_high_mse",
                "elapsed_seconds",
            ),
        )
        writer.writeheader()
        for summary in summaries:
            metrics = summary["final_metrics"]
            writer.writerow(
                {
                    "model_name": summary["model_name"],
                    "parameter_count": summary["parameter_count"],
                    "steps_completed": summary["steps_completed"],
                    "best_step": summary["best_step"],
                    "best_held_interior_mean_correlation": summary[
                        "best_held_interior_mean_correlation"
                    ],
                    "final_train_correlation": metrics["train"]["mean_correlation"],
                    "final_held_interior_correlation": metrics["held_interior"][
                        "mean_correlation"
                    ],
                    "final_held_high_correlation": metrics["held_high"][
                        "mean_correlation"
                    ],
                    "final_train_mse": metrics["train"]["mean_masked_mse"],
                    "final_held_interior_mse": metrics["held_interior"][
                        "mean_masked_mse"
                    ],
                    "final_held_high_mse": metrics["held_high"]["mean_masked_mse"],
                    "elapsed_seconds": summary["elapsed_seconds"],
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a controlled cohort of AutocorrRoPE models from scratch."
    )
    parser.add_argument(
        "--models",
        type=parse_models,
        default=DEFAULT_MODELS,
        help="comma-separated model names or 'all' (default: all)",
    )
    parser.add_argument("--steps", type=int, default=Protocol.steps)
    parser.add_argument("--batch-size", type=int, default=Protocol.batch_size)
    parser.add_argument("--sequence-length", type=int, default=Protocol.sequence_length)
    parser.add_argument("--rho", type=float, default=Protocol.rho)
    parser.add_argument("--burn-in", type=int, default=Protocol.burn_in)
    parser.add_argument("--learning-rate", type=float, default=Protocol.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=Protocol.weight_decay)
    parser.add_argument("--eval-every", type=int, default=Protocol.eval_every)
    parser.add_argument(
        "--checkpoint-every", type=int, default=Protocol.checkpoint_every
    )
    parser.add_argument(
        "--eval-sequences-per-bucket",
        type=int,
        default=Protocol.eval_sequences_per_bucket,
    )
    parser.add_argument(
        "--eval-batch-size", type=int, default=Protocol.eval_batch_size
    )
    parser.add_argument("--seed", type=int, default=Protocol.seed)
    parser.add_argument(
        "--held-interior", type=parse_int_tuple, default=Protocol.held_interior
    )
    parser.add_argument("--held-high", type=parse_int_tuple, default=Protocol.held_high)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("models/retrained_from_scratch"),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="default: UTC timestamp plus seed",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="run two tiny steps and evaluations to verify the pipeline",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved configuration without creating files or training",
    )
    parser.add_argument(
        "--list-models", action="store_true", help="print available model names and exit"
    )
    return parser


def protocol_from_args(args: argparse.Namespace) -> Protocol:
    values = dict(
        sequence_length=args.sequence_length,
        rho=args.rho,
        held_interior=args.held_interior,
        held_high=args.held_high,
        burn_in=args.burn_in,
        batch_size=args.batch_size,
        steps=args.steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        eval_sequences_per_bucket=args.eval_sequences_per_bucket,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
    )
    if args.smoke_test:
        values.update(
            steps=2,
            batch_size=2,
            eval_every=1,
            checkpoint_every=1,
            eval_sequences_per_bucket=4,
            eval_batch_size=2,
        )
    protocol = Protocol(**values)
    if protocol.steps <= 0 or protocol.batch_size <= 0:
        raise ValueError("steps and batch size must be positive")
    if protocol.eval_every <= 0 or protocol.checkpoint_every <= 0:
        raise ValueError("evaluation and checkpoint intervals must be positive")
    if set(protocol.train_lags) & set(protocol.held_interior):
        raise ValueError("held-interior lags leaked into training lags")
    maximum_eval_lag = max((*protocol.train_lags, *protocol.held_interior, *protocol.held_high))
    if maximum_eval_lag + protocol.burn_in >= protocol.sequence_length:
        raise ValueError(
            "sequence length must exceed maximum evaluation lag plus burn-in"
        )
    return protocol


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.list_models:
        for name, spec in MODEL_SPECS.items():
            print(f"{name}: {spec}")
        return

    protocol = protocol_from_args(args)
    models = args.models
    device = resolve_device(args.device)
    project_root = Path(__file__).resolve().parent
    environment = environment_metadata(project_root)
    resolved = {
        "models": {name: MODEL_SPECS[name] for name in models},
        "protocol": {**asdict(protocol), "train_lags": protocol.train_lags},
        "device": str(device),
        "paired_design": (
            "Every model is reset to the same seed and receives the same ordered "
            "stream of generated training batches."
        ),
        "existing_checkpoints_loaded": False,
        "environment": environment,
    }

    if args.dry_run:
        print(json.dumps(json_safe(resolved), indent=2, sort_keys=True))
        return

    run_name = args.run_name or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"_seed{protocol.seed}"
    )
    run_dir = args.output_root.resolve() / run_name
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir(parents=True)

    cohort_manifest = {
        **resolved,
        "run_name": run_name,
        "run_directory": str(run_dir),
        "status": "running",
        "started_utc": utc_now(),
        "completed_models": [],
    }
    manifest_path = run_dir / "cohort_manifest.json"
    atomic_write_json(manifest_path, cohort_manifest)

    summaries: list[dict[str, Any]] = []
    try:
        for model_name in models:
            summary = train_one_model(
                model_name,
                MODEL_SPECS[model_name],
                protocol,
                device,
                run_dir / model_name,
                environment,
            )
            summaries.append(summary)
            cohort_manifest["completed_models"].append(model_name)
            atomic_write_json(manifest_path, cohort_manifest)

        write_cohort_results(run_dir / "cohort_results.csv", summaries)
        cohort_manifest.update(
            status="complete",
            completed_utc=utc_now(),
            summaries=summaries,
        )
        atomic_write_json(manifest_path, cohort_manifest)
        print(f"\nCohort complete: {run_dir}")
    except BaseException as error:
        cohort_manifest.update(
            status="failed",
            failed_utc=utc_now(),
            error=repr(error),
            traceback=traceback.format_exc(),
        )
        atomic_write_json(manifest_path, cohort_manifest)
        raise


if __name__ == "__main__":
    main()
