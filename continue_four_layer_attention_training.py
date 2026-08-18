"""Continue training a copy of the canonical 4L attention-only checkpoint.

The input checkpoint is never modified.  Every invocation creates a new run
directory containing:

* ordinary model ``state_dict`` checkpoints, compatible with existing code;
* resumable training bundles with model, optimizer, step, config, and history;
* aggregate and per-lag evaluation CSV files;
* a JSON manifest describing the run and the selected best checkpoint.

The original 4L lag split is preserved:

* train: lags 1--50 except 15, 27, and 38;
* interpolation validation: 15, 27, and 38;
* high-lag test: 52, 55, and 58.

The best checkpoint is selected by interpolation-validation MSE.  High-lag
results are logged but are not used for checkpoint selection.

Examples:
    python continue_four_layer_attention_training.py
    python continue_four_layer_attention_training.py --steps 100000 --lr 3e-4
    python continue_four_layer_attention_training.py --device cpu --steps 10 \
        --eval-every 5 --save-every 5 --eval-sequences 48

Resume an interrupted run from one of its ``*_training.pt`` bundles:
    python continue_four_layer_attention_training.py \
        --resume models/continued_4L/<run>/step_050000_training.pt
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW

from model_class import AutocorrRoPE
from util import make_dataset_lagset


TRAIN_LAGS = tuple(lag for lag in range(1, 51) if lag not in (15, 27, 38))
INTERPOLATION_LAGS = (15, 27, 38)
HIGH_TEST_LAGS = (52, 55, 58)


@dataclass
class RunConfig:
    input_checkpoint: str
    output_directory: str
    device: str
    total_steps: int
    starting_step: int
    batch_size: int
    sequence_length: int
    rho: float
    learning_rate: float
    weight_decay: float
    burn_in_extra: int
    eval_every: int
    save_every: int
    eval_sequences_per_bucket: int
    eval_batch_size: int
    seed: int
    train_lags: list[int]
    interpolation_lags: list[int]
    high_test_lags: list[int]
    optimizer_note: str
    selection_metric: str


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_model(device: torch.device) -> AutocorrRoPE:
    return AutocorrRoPE(
        d_model=64,
        d_head=64,
        n_layers=4,
        n_heads=1,
        use_mlp=False,
    ).to(device)


def masked_training_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    lags: torch.Tensor,
    burn_in_extra: int,
) -> torch.Tensor:
    positions = torch.arange(
        predictions.shape[1],
        device=predictions.device,
    ).unsqueeze(0)
    valid = positions >= lags.unsqueeze(1) + burn_in_extra
    squared_error = (predictions - targets).square()
    per_sequence = (squared_error * valid).sum(dim=1) / valid.sum(
        dim=1
    ).clamp_min(1)
    return per_sequence.mean()


def safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first = first - first.mean()
    second = second - second.mean()
    denominator = math.sqrt(float(first @ first) * float(second @ second))
    if denominator <= 1e-30:
        return float("nan")
    return float((first @ second) / denominator)


@torch.inference_mode()
def evaluate_bucket(
    model: AutocorrRoPE,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    lags: torch.Tensor,
    burn_in_extra: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, float | int]]]:
    """Return aggregate and per-lag prediction metrics on fixed sequences."""
    was_training = model.training
    model.eval()
    prediction_chunks = []
    for start in range(0, len(inputs), batch_size):
        batch = inputs[start : start + batch_size].to(device)
        predictions, *_ = model(batch)
        prediction_chunks.append(predictions.cpu())
    predictions = torch.cat(prediction_chunks, dim=0)
    if was_training:
        model.train()

    inputs_cpu = inputs.cpu()
    targets_cpu = targets.cpu()
    lags_cpu = lags.cpu()
    sequence_length = inputs.shape[1]
    per_lag_rows: list[dict[str, float | int]] = []

    all_predictions = []
    all_targets = []
    all_retrieval_values = []

    for lag_value in sorted(int(value) for value in lags_cpu.unique().tolist()):
        selected = torch.where(lags_cpu == lag_value)[0]
        first_position = lag_value + burn_in_extra
        if first_position >= sequence_length:
            raise ValueError(
                f"lag {lag_value} plus burn-in leaves no evaluation positions"
            )

        pred = predictions.index_select(0, selected)[:, first_position:]
        target = targets_cpu.index_select(0, selected)[:, first_position:]
        # Prediction at input index t targets sequence value t+1.  Under lag L,
        # the corresponding known-lag source is input index t-L+1.
        source = inputs_cpu.index_select(0, selected)[
            :,
            first_position - lag_value + 1 : sequence_length - lag_value + 1,
        ]

        pred_flat = pred.reshape(-1).numpy()
        target_flat = target.reshape(-1).numpy()
        source_flat = source.reshape(-1).numpy()
        mse = float(np.mean(np.square(pred_flat - target_flat)))
        target_corr = safe_correlation(pred_flat, target_flat)
        retrieval_corr = safe_correlation(pred_flat, source_flat)
        per_lag_rows.append(
            {
                "lag": lag_value,
                "n_sequences": int(len(selected)),
                "n_scalar_observations": int(len(pred_flat)),
                "mse_to_next_value": mse,
                "correlation_with_next_value": target_corr,
                "correlation_with_correct_lagged_source": retrieval_corr,
            }
        )
        all_predictions.append(pred_flat)
        all_targets.append(target_flat)
        all_retrieval_values.append(source_flat)

    aggregate_predictions = np.concatenate(all_predictions)
    aggregate_targets = np.concatenate(all_targets)
    aggregate_sources = np.concatenate(all_retrieval_values)
    aggregate = {
        "mse_to_next_value": float(
            np.mean(np.square(aggregate_predictions - aggregate_targets))
        ),
        "correlation_with_next_value": safe_correlation(
            aggregate_predictions,
            aggregate_targets,
        ),
        "correlation_with_correct_lagged_source": safe_correlation(
            aggregate_predictions,
            aggregate_sources,
        ),
        "mean_per_lag_mse": float(
            np.mean([row["mse_to_next_value"] for row in per_lag_rows])
        ),
        "mean_per_lag_next_value_correlation": float(
            np.mean(
                [row["correlation_with_next_value"] for row in per_lag_rows]
            )
        ),
        "mean_per_lag_retrieval_correlation": float(
            np.mean(
                [
                    row["correlation_with_correct_lagged_source"]
                    for row in per_lag_rows
                ]
            )
        ),
    }
    return aggregate, per_lag_rows


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_model_state(model: AutocorrRoPE, path: Path) -> None:
    """Save the state_dict format expected by the project's analysis code."""
    torch.save(model.state_dict(), path)


def save_training_bundle(
    model: AutocorrRoPE,
    optimizer: AdamW,
    completed_step: int,
    config: RunConfig,
    aggregate_history: list[dict[str, Any]],
    per_lag_history: list[dict[str, Any]],
    training_seed_generator: np.random.Generator,
    path: Path,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "completed_step": completed_step,
            "config": asdict(config),
            "aggregate_history": aggregate_history,
            "per_lag_history": per_lag_history,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "training_seed_generator_state": training_seed_generator.bit_generator.state,
        },
        path,
    )


def write_manifest(
    path: Path,
    config: RunConfig,
    status: str,
    completed_step: int,
    best_step: int,
    best_interpolation_mse: float,
    best_model_path: Path | None,
) -> None:
    manifest = {
        "status": status,
        "completed_step": completed_step,
        "best_step": best_step,
        "best_interpolation_mean_per_lag_mse": best_interpolation_mse,
        "best_model_checkpoint": str(best_model_path) if best_model_path else None,
        "config": asdict(config),
    }
    with path.open("w") as handle:
        json.dump(manifest, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/attn_d64_4L_int_ext.pt"),
        help="input state_dict; this file is never modified",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="optional *_training.pt bundle from an earlier invocation",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("models/continued_4L"),
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="subdirectory name; default includes the current timestamp",
    )
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=200)
    parser.add_argument("--rho", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--burn-in-extra", type=int, default=30)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--save-every", type=int, default=10_000)
    parser.add_argument("--eval-sequences", type=int, default=3_000)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    args = parser.parse_args()

    if args.steps < 0:
        raise ValueError("steps must be nonnegative")
    if args.eval_every <= 0 or args.save_every <= 0:
        raise ValueError("eval-every and save-every must be positive")
    if args.eval_sequences <= 0:
        raise ValueError("eval-sequences must be positive")
    if args.resume is not None and args.run_name is None:
        raise ValueError(
            "when using --resume, provide --run-name for the existing run directory"
        )

    device = choose_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"continue_{timestamp}"
    run_directory = args.output_root / run_name
    run_directory.mkdir(parents=True, exist_ok=args.resume is not None)

    model = create_model(device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    starting_step = 0
    aggregate_history: list[dict[str, Any]] = []
    per_lag_history: list[dict[str, Any]] = []
    resume_bundle: dict[str, Any] | None = None

    if args.resume is None:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)
        input_checkpoint = args.checkpoint
        optimizer_note = (
            "Fresh AdamW optimizer: the original checkpoint contains model weights only."
        )
    else:
        resume_bundle = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(resume_bundle["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume_bundle["optimizer_state_dict"])
        starting_step = int(resume_bundle["completed_step"])
        aggregate_history = list(resume_bundle.get("aggregate_history", []))
        per_lag_history = list(resume_bundle.get("per_lag_history", []))
        input_checkpoint = args.resume
        optimizer_note = "Optimizer and model restored from a training bundle."

    config = RunConfig(
        input_checkpoint=str(input_checkpoint),
        output_directory=str(run_directory),
        device=str(device),
        total_steps=args.steps,
        starting_step=starting_step,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        rho=args.rho,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        burn_in_extra=args.burn_in_extra,
        eval_every=args.eval_every,
        save_every=args.save_every,
        eval_sequences_per_bucket=args.eval_sequences,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        train_lags=list(TRAIN_LAGS),
        interpolation_lags=list(INTERPOLATION_LAGS),
        high_test_lags=list(HIGH_TEST_LAGS),
        optimizer_note=optimizer_note,
        selection_metric="lowest interpolation mean-per-lag MSE",
    )

    eval_buckets = {}
    for bucket_index, (bucket_name, bucket_lags) in enumerate(
        (
            ("train", TRAIN_LAGS),
            ("interpolation", INTERPOLATION_LAGS),
            ("high_test", HIGH_TEST_LAGS),
        )
    ):
        eval_buckets[bucket_name] = make_dataset_lagset(
            args.eval_sequences,
            args.sequence_length,
            args.rho,
            bucket_lags,
            seed=args.seed + 100_000 + bucket_index,
        )

    training_seed_generator = np.random.default_rng(args.seed + 999_999)
    if resume_bundle is not None:
        if "torch_rng_state" in resume_bundle:
            torch.set_rng_state(resume_bundle["torch_rng_state"].cpu())
        if (
            torch.cuda.is_available()
            and resume_bundle.get("cuda_rng_state_all") is not None
        ):
            torch.cuda.set_rng_state_all(resume_bundle["cuda_rng_state_all"])
        if "numpy_rng_state" in resume_bundle:
            np.random.set_state(resume_bundle["numpy_rng_state"])
        if "python_rng_state" in resume_bundle:
            random.setstate(resume_bundle["python_rng_state"])
        if "training_seed_generator_state" in resume_bundle:
            training_seed_generator.bit_generator.state = resume_bundle[
                "training_seed_generator_state"
            ]

    prior_interpolation_rows = [
        row for row in aggregate_history if row.get("bucket") == "interpolation"
    ]
    if prior_interpolation_rows:
        best_row = min(
            prior_interpolation_rows,
            key=lambda row: float(row["mean_per_lag_mse"]),
        )
        best_interpolation_mse = float(best_row["mean_per_lag_mse"])
        best_step = int(best_row["step"])
        candidate_best_path = run_directory / "best_model.pt"
        best_model_path: Path | None = (
            candidate_best_path if candidate_best_path.exists() else None
        )
    else:
        best_interpolation_mse = float("inf")
        best_step = -1
        best_model_path = None
    manifest_path = run_directory / "run_manifest.json"

    def evaluate_and_record(step: int, latest_training_loss: float | None) -> None:
        nonlocal best_interpolation_mse, best_step, best_model_path
        print(f"\nEvaluation at continuation step {step:,}")
        current_results = {}
        for bucket_name, (inputs, targets, lags) in eval_buckets.items():
            aggregate, per_lag = evaluate_bucket(
                model,
                inputs,
                targets,
                lags,
                args.burn_in_extra,
                args.eval_batch_size,
                device,
            )
            current_results[bucket_name] = aggregate
            aggregate_history.append(
                {
                    "step": step,
                    "bucket": bucket_name,
                    "latest_training_loss": latest_training_loss,
                    **aggregate,
                }
            )
            for row in per_lag:
                per_lag_history.append(
                    {
                        "step": step,
                        "bucket": bucket_name,
                        **row,
                    }
                )
            print(
                f"  {bucket_name:13s} "
                f"MSE={aggregate['mean_per_lag_mse']:.4f}  "
                f"target-r={aggregate['mean_per_lag_next_value_correlation']:+.3f}  "
                f"retrieval-r={aggregate['mean_per_lag_retrieval_correlation']:+.3f}"
            )

        interpolation_mse = current_results["interpolation"]["mean_per_lag_mse"]
        if interpolation_mse < best_interpolation_mse:
            best_interpolation_mse = interpolation_mse
            best_step = step
            best_model_path = run_directory / "best_model.pt"
            save_model_state(model, best_model_path)
            print(
                f"  new best interpolation MSE={best_interpolation_mse:.4f}; "
                f"saved {best_model_path}"
            )

        save_csv(run_directory / "evaluation_aggregate.csv", aggregate_history)
        save_csv(run_directory / "evaluation_per_lag.csv", per_lag_history)
        write_manifest(
            manifest_path,
            config,
            "running",
            step,
            best_step,
            best_interpolation_mse,
            best_model_path,
        )

    print("Continuing a copy of the 4L attention-only model")
    print(f"  input:      {input_checkpoint}")
    print(f"  output:     {run_directory}")
    print(f"  device:     {device}")
    print(f"  start step: {starting_step:,}")
    print(f"  add steps:  {args.steps:,}")
    print(f"  optimizer:  AdamW(lr={args.lr:g}, weight_decay={args.weight_decay:g})")
    print("  selection:  lowest interpolation-validation MSE")
    print("  original input checkpoint will not be modified")

    latest_training_loss: float | None = None
    completed_step = starting_step
    try:
        already_evaluated_start = any(
            int(row.get("step", -1)) == starting_step
            and row.get("bucket") == "interpolation"
            for row in aggregate_history
        )
        if not already_evaluated_start:
            evaluate_and_record(starting_step, latest_training_loss)
        model.train()
        for continuation_index in range(1, args.steps + 1):
            completed_step = starting_step + continuation_index
            batch_seed = int(
                training_seed_generator.integers(0, np.iinfo(np.int32).max)
            )
            inputs, targets, lags = make_dataset_lagset(
                args.batch_size,
                args.sequence_length,
                args.rho,
                TRAIN_LAGS,
                seed=batch_seed,
            )
            inputs = inputs.to(device)
            targets = targets.to(device)
            lags = lags.to(device)

            predictions, *_ = model(inputs)
            loss = masked_training_loss(
                predictions,
                targets,
                lags,
                args.burn_in_extra,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            latest_training_loss = float(loss.item())

            if continuation_index % args.eval_every == 0:
                evaluate_and_record(completed_step, latest_training_loss)
                model.train()

            if continuation_index % args.save_every == 0:
                model_path = run_directory / f"step_{completed_step:06d}_model.pt"
                bundle_path = (
                    run_directory / f"step_{completed_step:06d}_training.pt"
                )
                save_model_state(model, model_path)
                save_training_bundle(
                    model,
                    optimizer,
                    completed_step,
                    config,
                    aggregate_history,
                    per_lag_history,
                    training_seed_generator,
                    bundle_path,
                )
                print(f"  saved step checkpoint: {model_path}")

        if args.steps > 0 and args.steps % args.eval_every != 0:
            evaluate_and_record(completed_step, latest_training_loss)

        final_model_path = run_directory / "final_model.pt"
        final_bundle_path = run_directory / "final_training.pt"
        save_model_state(model, final_model_path)
        save_training_bundle(
            model,
            optimizer,
            completed_step,
            config,
            aggregate_history,
            per_lag_history,
            training_seed_generator,
            final_bundle_path,
        )
        write_manifest(
            manifest_path,
            config,
            "complete",
            completed_step,
            best_step,
            best_interpolation_mse,
            best_model_path,
        )
        print("\nTraining complete")
        print(f"  final model: {final_model_path}")
        print(f"  best model:  {best_model_path} (step {best_step:,})")
        print(f"  manifest:    {manifest_path}")

    except KeyboardInterrupt:
        interrupted_model_path = run_directory / "interrupted_model.pt"
        interrupted_bundle_path = run_directory / "interrupted_training.pt"
        save_model_state(model, interrupted_model_path)
        save_training_bundle(
            model,
            optimizer,
            completed_step,
            config,
            aggregate_history,
            per_lag_history,
            training_seed_generator,
            interrupted_bundle_path,
        )
        write_manifest(
            manifest_path,
            config,
            "interrupted",
            completed_step,
            best_step,
            best_interpolation_mse,
            best_model_path,
        )
        print("\nInterrupted safely")
        print(f"  resumable bundle: {interrupted_bundle_path}")
        raise


if __name__ == "__main__":
    main()
