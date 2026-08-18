"""Program the final attention stripe of the 4-layer attention-only model.

For every true data lag L and programmed final-attention lag M, this script
replaces the final head's learned attention matrix with a one-hot stripe at
query-minus-key offset M - 1.  It then measures next-value prediction MSE.

The diagonal L == M tests whether the same downstream value/output circuit can
be reused at many lags.  A flat diagonal means that performance does not decay
as the programmed lag increases.  Off-diagonal cells are specificity controls.

Usage:
    python four_layer_programmed_lag_mse.py --quick --no-show
    python four_layer_programmed_lag_mse.py --no-show
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from model_class import AutocorrRoPE
from util import make_dataset_lagset


def parse_int_tuple(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def load_model(checkpoint: Path, device: torch.device) -> AutocorrRoPE:
    model = AutocorrRoPE(
        d_model=64,
        d_head=64,
        n_layers=4,
        n_heads=1,
        use_mlp=False,
    ).to(device=device, dtype=torch.float64)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def stripe_attention(
    programmed_lag: int,
    sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """A causal, row-stochastic stripe at offset programmed_lag - 1."""
    offset = programmed_lag - 1
    query = torch.arange(sequence_length, device=device)
    key = (query - offset).clamp_min(0)
    attention = torch.zeros(
        1,
        sequence_length,
        sequence_length,
        device=device,
        dtype=dtype,
    )
    attention[0, query, key] = 1.0
    return attention


@torch.inference_mode()
def residual_entering_final_layer(
    model: AutocorrRoPE,
    inputs: torch.Tensor,
) -> torch.Tensor:
    """Run Layers 1--3 normally and return the residual entering Layer 4."""
    batch_size, sequence_length = inputs.shape
    positions = torch.arange(sequence_length, device=inputs.device).unsqueeze(0)
    positions = positions.expand(batch_size, sequence_length)
    index = torch.arange(sequence_length, device=inputs.device)
    causal_mask = index.unsqueeze(0) > index.unsqueeze(1)

    residual = model.W_r(inputs.unsqueeze(-1))
    for layer_index in range(model.n_layers - 1):
        attention, write = model._head(
            residual,
            model.layers[layer_index][0],
            positions,
            causal_mask,
        )
        del attention
        residual = residual + write
    return residual


@torch.inference_mode()
def prediction_with_programmed_final_attention(
    model: AutocorrRoPE,
    residual: torch.Tensor,
    attention: torch.Tensor,
) -> torch.Tensor:
    """Apply Layer 4's OV circuit using a supplied attention matrix."""
    _, _, value_matrix, output_matrix = model.layers[-1][0]
    final_write = (attention @ (residual @ value_matrix)) @ output_matrix
    return model.W_U(residual + final_write).squeeze(-1)


def sequence_mse(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    true_lag: int,
    programmed_lag: int,
    burn_in: int,
) -> torch.Tensor:
    """One MSE per sequence after both lags are fully available."""
    start = max(true_lag, programmed_lag) + burn_in
    if start >= prediction.shape[1]:
        raise ValueError(
            f"evaluation start {start} is beyond sequence length "
            f"{prediction.shape[1]}"
        )
    return (prediction[:, start:] - targets[:, start:]).square().mean(dim=1)


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(
    lags: tuple[int, ...],
    mse_grid: np.ndarray,
    diagonal_rows: list[dict],
    output_path: Path,
) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.3), constrained_layout=True)

    image = axes[0].imshow(mse_grid, cmap="magma", aspect="auto", origin="lower")
    axes[0].set_xticks(range(len(lags)), lags, rotation=60)
    axes[0].set_yticks(range(len(lags)), lags)
    axes[0].set_xlabel("programmed final-attention lag")
    axes[0].set_ylabel("true data lag")
    axes[0].set_title("Prediction MSE: full programming grid")
    figure.colorbar(image, ax=axes[0], label="mean next-value MSE")

    lag_values = np.asarray([row["lag"] for row in diagonal_rows])
    programmed = np.asarray([row["programmed_mse_mean"] for row in diagonal_rows])
    clean = np.asarray([row["clean_mse_mean"] for row in diagonal_rows])
    low = np.asarray([row["programmed_mse_p05"] for row in diagonal_rows])
    high = np.asarray([row["programmed_mse_p95"] for row in diagonal_rows])
    axes[1].plot(lag_values, programmed, marker="o", label="programmed stripe")
    axes[1].fill_between(lag_values, low, high, alpha=0.2, label="5th–95th percentile")
    axes[1].plot(lag_values, clean, marker="o", label="clean model")
    axes[1].axvline(50, color="gray", linestyle=":", label="training maximum")
    axes[1].axhline(1 - 0.9**2, color="black", linestyle="--", label="known-lag floor")
    axes[1].set_xlabel("matched true and programmed lag")
    axes[1].set_ylabel("mean next-value MSE")
    axes[1].set_title("Diagonal: does matched performance decay?")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    figure.suptitle("4-layer attention-only model: programmed final attention")
    figure.savefig(output_path, dpi=180)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/attn_d64_4L_int_ext.pt"),
    )
    parser.add_argument(
        "--lags",
        type=parse_int_tuple,
        default=(7, 10, 13, 19, 29, 40, 50, 60, 70, 80, 90, 100, 110, 120),
    )
    parser.add_argument("--n-sequences", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=240)
    parser.add_argument("--rho", type=float, default=0.9)
    parser.add_argument("--burn-in", type=int, default=30)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/four_layer_programmed_lag_mse"),
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if max(args.lags) + args.burn_in >= args.sequence_length:
        raise ValueError("sequence length is too short for the largest lag and burn-in")
    if args.quick:
        args.n_sequences = min(args.n_sequences, 12)

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint, device)

    programmed_attentions = {
        lag: stripe_attention(
            lag,
            args.sequence_length,
            device,
            torch.float64,
        )
        for lag in args.lags
    }

    grid_rows: list[dict] = []
    diagonal_rows: list[dict] = []
    mse_grid = np.empty((len(args.lags), len(args.lags)))

    for true_index, true_lag in enumerate(args.lags):
        print(f"true lag {true_lag}")
        inputs, targets, sampled_lags = make_dataset_lagset(
            args.n_sequences,
            args.sequence_length,
            args.rho,
            [true_lag],
            seed=700_000 + true_lag,
        )
        if not torch.all(sampled_lags == true_lag):
            raise RuntimeError("dataset returned an unexpected lag")
        inputs = inputs.to(device=device, dtype=torch.float64)
        targets = targets.to(device=device, dtype=torch.float64)

        with torch.inference_mode():
            clean_prediction, _, _, _ = model(inputs)
            prefinal_residual = residual_entering_final_layer(model, inputs)

        for programmed_index, programmed_lag in enumerate(args.lags):
            prediction = prediction_with_programmed_final_attention(
                model,
                prefinal_residual,
                programmed_attentions[programmed_lag],
            )
            per_sequence = sequence_mse(
                prediction,
                targets,
                true_lag,
                programmed_lag,
                args.burn_in,
            )
            mean_mse = float(per_sequence.mean().item())
            mse_grid[true_index, programmed_index] = mean_mse
            grid_rows.append(
                {
                    "true_lag": true_lag,
                    "programmed_lag": programmed_lag,
                    "mean_mse": mean_mse,
                    "median_mse": float(per_sequence.median().item()),
                    "p05_mse": float(torch.quantile(per_sequence, 0.05).item()),
                    "p95_mse": float(torch.quantile(per_sequence, 0.95).item()),
                    "n_sequences": args.n_sequences,
                }
            )

            if programmed_lag == true_lag:
                clean_per_sequence = sequence_mse(
                    clean_prediction,
                    targets,
                    true_lag,
                    true_lag,
                    args.burn_in,
                )
                diagonal_rows.append(
                    {
                        "lag": true_lag,
                        "programmed_mse_mean": mean_mse,
                        "programmed_mse_median": float(per_sequence.median().item()),
                        "programmed_mse_p05": float(torch.quantile(per_sequence, 0.05).item()),
                        "programmed_mse_p95": float(torch.quantile(per_sequence, 0.95).item()),
                        "clean_mse_mean": float(clean_per_sequence.mean().item()),
                        "clean_mse_median": float(clean_per_sequence.median().item()),
                    }
                )

    save_csv(args.output_dir / "programmed_lag_grid.csv", grid_rows)
    save_csv(args.output_dir / "matched_lag_diagonal.csv", diagonal_rows)
    np.save(args.output_dir / "mse_grid.npy", mse_grid)

    lag_values = np.asarray([row["lag"] for row in diagonal_rows], dtype=float)
    programmed_values = np.asarray(
        [row["programmed_mse_mean"] for row in diagonal_rows], dtype=float
    )
    slope, intercept = np.polyfit(lag_values, programmed_values, 1)
    off_diagonal = mse_grid[~np.eye(len(args.lags), dtype=bool)]
    summary = {
        "checkpoint": str(args.checkpoint),
        "lags": list(args.lags),
        "n_sequences_per_true_lag": args.n_sequences,
        "sequence_length": args.sequence_length,
        "burn_in": args.burn_in,
        "known_lag_floor": 1 - args.rho**2,
        "diagonal_mean_mse": float(np.mean(programmed_values)),
        "diagonal_min_mse": float(np.min(programmed_values)),
        "diagonal_max_mse": float(np.max(programmed_values)),
        "diagonal_mse_slope_per_lag": float(slope),
        "diagonal_mse_intercept": float(intercept),
        "off_diagonal_mean_mse": float(np.mean(off_diagonal)),
        "off_diagonal_median_mse": float(np.median(off_diagonal)),
    }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    figure = make_plot(
        args.lags,
        mse_grid,
        diagonal_rows,
        args.output_dir / "programmed_lag_mse.png",
    )
    if args.no_show:
        plt.close(figure)
    else:
        plt.show()

    print("\nMatched true/programmed lag:")
    for row in diagonal_rows:
        print(
            f"  lag {row['lag']:3d}: programmed MSE="
            f"{row['programmed_mse_mean']:.4f}, clean MSE="
            f"{row['clean_mse_mean']:.4f}"
        )
    print(
        f"\nDiagonal mean={summary['diagonal_mean_mse']:.4f}, "
        f"range=[{summary['diagonal_min_mse']:.4f}, "
        f"{summary['diagonal_max_mse']:.4f}], "
        f"slope/lag={summary['diagonal_mse_slope_per_lag']:+.6f}"
    )
    print(
        f"Off-diagonal mean={summary['off_diagonal_mean_mse']:.4f}; "
        f"known-lag floor={summary['known_lag_floor']:.4f}"
    )
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
