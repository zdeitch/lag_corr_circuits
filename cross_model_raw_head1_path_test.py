"""Test the raw-query / Head-1-only-key path pairing across model depths.

The four targeted path pairings are present in every attention-only model,
regardless of depth:

    raw -> raw
    raw -> Head 1 write
    Head 1 write -> raw
    Head 1 write -> Head 1 write

For every sequence, each pair receives a final-head pre-softmax contribution
curve over query-minus-key offsets.  We measure how often that curve peaks at
the prediction-correct offset ``lag - 1`` and how often it peaks at ``lag``.

The experiment also verifies that Head 1 itself has a fixed offset-1 attention
stripe and reports whether the complete final attention profile peaks at the
correct offset.

Usage:
    python cross_model_raw_head1_path_test.py --quick --no-show
    python cross_model_raw_head1_path_test.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from model_class import AutocorrRoPE
from util import make_dataset_lagset


MODEL_SPECS = {
    "4L": (4, Path("models/attn_d64_4L_int_ext.pt")),
    "5L": (5, Path("models/sweep_d64_5L_attn_lag1_50_best.pt")),
    "6L": (6, Path("models/sweep_d64_6L_attn_lag1_50_extended.pt")),
    "7L": (7, Path("models/sweep_d64_7L_attn_lag1_50_extended.pt")),
}

PAIR_NAMES = (
    "raw→raw",
    "raw→Head1",
    "Head1→raw",
    "Head1→Head1",
)


def parse_int_tuple(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def load_attention_only_model(
    n_layers: int,
    checkpoint: Path,
    device: torch.device,
) -> AutocorrRoPE:
    model = AutocorrRoPE(
        d_model=64,
        d_head=64,
        n_layers=n_layers,
        n_heads=1,
        use_mlp=False,
    ).to(device=device, dtype=torch.float64)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def attention_profiles(
    attention: torch.Tensor,
    query_positions: torch.Tensor,
    maximum_offset: int,
) -> torch.Tensor:
    offsets = torch.arange(maximum_offset + 1, device=attention.device)
    values = attention[
        :,
        query_positions[:, None],
        query_positions[:, None] - offsets[None, :],
    ]
    return values.mean(dim=1)


def peak_rates(
    curves: torch.Tensor,
    correct_offset: int,
) -> tuple[float, float, float]:
    """Correct, correct+1, and correct-vs-best-other score margin."""
    # curves: sequence x pair x candidate-offset, with offset axis starting at 1.
    peaks = curves.argmax(dim=-1) + 1
    correct_rate = (peaks == correct_offset).double().mean(dim=0)
    plus_one_rate = (peaks == correct_offset + 1).double().mean(dim=0)

    correct_index = correct_offset - 1
    correct_values = curves[..., correct_index]
    wrong = curves.clone()
    wrong[..., correct_index] = -torch.inf
    margins = correct_values - wrong.max(dim=-1).values
    return correct_rate, plus_one_rate, margins.mean(dim=0)


@torch.inference_mode()
def run_model_lag(
    model: AutocorrRoPE,
    lag: int,
    n_sequences: int,
    sequence_length: int,
    rho: float,
    query_start: int,
    query_stride: int,
    maximum_offset: int,
    device: torch.device,
) -> dict[str, np.ndarray | float | int]:
    inputs, _, sampled_lags = make_dataset_lagset(
        n_sequences,
        sequence_length,
        rho,
        [lag],
        seed=400_000 + 1_000 * model.n_layers + lag,
    )
    if not torch.all(sampled_lags == lag):
        raise RuntimeError("dataset returned an unexpected lag")
    inputs = inputs.to(device=device, dtype=torch.float64)

    _, attentions, post_attention, post_mlp = model(inputs)
    if any(not torch.equal(a, b) for a, b in zip(post_attention, post_mlp)):
        raise RuntimeError("checkpoint is not behaving as attention-only")

    query_positions = torch.arange(
        query_start,
        sequence_length,
        query_stride,
        device=device,
    )
    correct_offset = lag - 1

    head1_profiles = attention_profiles(
        attentions[0], query_positions, maximum_offset
    )
    head1_peaks = 1 + head1_profiles[:, 1:].argmax(dim=1)
    head1_offset1_rate = float((head1_peaks == 1).double().mean().item())

    final_profiles = attention_profiles(
        attentions[-1], query_positions, maximum_offset
    )
    final_peaks = 1 + final_profiles[:, 1:].argmax(dim=1)
    final_correct_rate = float(
        (final_peaks == correct_offset).double().mean().item()
    )

    raw = model.W_r(inputs.unsqueeze(-1))
    _, _, value1, output1 = model.layers[0][0]
    head1_only = (attentions[0] @ (raw @ value1)) @ output1

    final_query_matrix, final_key_matrix, _, _ = model.layers[-1][0]
    all_positions = torch.arange(sequence_length, device=device).unsqueeze(0)
    raw_query = model.apply_rope(raw @ final_query_matrix, all_positions)
    raw_key = model.apply_rope(raw @ final_key_matrix, all_positions)
    head1_query = model.apply_rope(
        head1_only @ final_query_matrix, all_positions
    )
    head1_key = model.apply_rope(head1_only @ final_key_matrix, all_positions)

    query_parts = (raw_query, raw_query, head1_query, head1_query)
    key_parts = (raw_key, head1_key, raw_key, head1_key)
    curves = torch.empty(
        n_sequences,
        len(PAIR_NAMES),
        maximum_offset,
        dtype=torch.float64,
        device=device,
    )

    for offset in range(1, maximum_offset + 1):
        for pair_index, (query_part, key_part) in enumerate(
            zip(query_parts, key_parts)
        ):
            contributions = (
                query_part[:, query_positions, :]
                * key_part[:, query_positions - offset, :]
            ).sum(dim=-1) / math.sqrt(model.d_head)
            curves[:, pair_index, offset - 1] = contributions.mean(dim=1)

    correct_rate, plus_one_rate, correct_margin = peak_rates(
        curves, correct_offset
    )
    return {
        "lag": lag,
        "head1_offset1_rate": head1_offset1_rate,
        "final_correct_rate": final_correct_rate,
        "pair_correct_rate": correct_rate.cpu().numpy(),
        "pair_plus_one_rate": plus_one_rate.cpu().numpy(),
        "pair_correct_margin": correct_margin.cpu().numpy(),
    }


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(
    aggregate_rows: list[dict],
    model_rows: list[dict],
    output_path: Path,
) -> plt.Figure:
    model_names = [name for name in MODEL_SPECS if any(row["model"] == name for row in aggregate_rows)]
    correct = np.asarray([
        [
            next(
                row["correct_peak_rate"]
                for row in aggregate_rows
                if row["model"] == model and row["pair"] == pair
            )
            for pair in PAIR_NAMES
        ]
        for model in model_names
    ])
    plus_one = np.asarray([
        [
            next(
                row["correct_plus_one_peak_rate"]
                for row in aggregate_rows
                if row["model"] == model and row["pair"] == pair
            )
            for pair in PAIR_NAMES
        ]
        for model in model_names
    ])
    head1_fixed = np.asarray([
        next(row["mean_head1_offset1_rate"] for row in model_rows if row["model"] == model)
        for model in model_names
    ])
    final_correct = np.asarray([
        next(row["mean_final_correct_rate"] for row in model_rows if row["model"] == model)
        for model in model_names
    ])

    figure, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    for axis, matrix, title in (
        (axes[0], correct, "Pair peak at correct offset (lag − 1)"),
        (axes[1], plus_one, "Pair peak at correct + 1 (lag)"),
    ):
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                color = "black" if matrix[row_index, column_index] > 0.58 else "white"
                axis.text(
                    column_index,
                    row_index,
                    f"{matrix[row_index, column_index]:.0%}",
                    ha="center",
                    va="center",
                    color=color,
                )
        axis.set_xticks(range(len(PAIR_NAMES)), PAIR_NAMES, rotation=30, ha="right")
        axis.set_yticks(range(len(model_names)), model_names)
        axis.set_title(title)
        axis.set_xlabel("query path → key path")

    positions = np.arange(len(model_names))
    width = 0.36
    axes[2].bar(
        positions - width / 2,
        head1_fixed,
        width,
        label="Head 1 peak at offset 1",
    )
    axes[2].bar(
        positions + width / 2,
        final_correct,
        width,
        label="final attention peak correct",
    )
    axes[2].set_xticks(positions, model_names)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("exact peak rate")
    axes[2].set_title("Behavior checks")
    axes[2].legend(fontsize=8)
    axes[2].grid(axis="y", alpha=0.3)

    figure.colorbar(image, ax=axes[:2], label="exact peak rate", shrink=0.85)
    figure.suptitle("Does the raw-query / Head-1-key mechanism recur across depths?")
    figure.savefig(output_path, dpi=180)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default="4L,5L,6L,7L",
        help="comma-separated subset of 4L,5L,6L,7L",
    )
    parser.add_argument(
        "--lags",
        type=parse_int_tuple,
        default=(25, 30, 35, 40, 45, 50),
    )
    parser.add_argument("--n-sequences", type=int, default=96)
    parser.add_argument("--sequence-length", type=int, default=200)
    parser.add_argument("--rho", type=float, default=0.9)
    parser.add_argument("--query-start", type=int, default=120)
    parser.add_argument("--query-stride", type=int, default=4)
    parser.add_argument("--maximum-offset", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/cross_model_raw_head1_path_test"),
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    selected_models = tuple(part.strip() for part in args.models.split(",") if part.strip())
    unknown = set(selected_models) - set(MODEL_SPECS)
    if unknown:
        raise ValueError(f"unknown model names: {sorted(unknown)}")
    if args.query_start < args.maximum_offset:
        raise ValueError("query_start must be at least maximum_offset")
    if args.quick:
        args.n_sequences = min(args.n_sequences, 16)

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_lag_rows: list[dict] = []
    aggregate_rows: list[dict] = []
    model_rows: list[dict] = []

    for model_name in selected_models:
        n_layers, checkpoint = MODEL_SPECS[model_name]
        print(f"\n{model_name}: loading {checkpoint}")
        model = load_attention_only_model(n_layers, checkpoint, device)
        model_results = []
        for lag in args.lags:
            print(f"  lag {lag}")
            result = run_model_lag(
                model=model,
                lag=lag,
                n_sequences=args.n_sequences,
                sequence_length=args.sequence_length,
                rho=args.rho,
                query_start=args.query_start,
                query_stride=args.query_stride,
                maximum_offset=args.maximum_offset,
                device=device,
            )
            model_results.append(result)
            for pair_index, pair_name in enumerate(PAIR_NAMES):
                per_lag_rows.append(
                    {
                        "model": model_name,
                        "lag": lag,
                        "pair": pair_name,
                        "correct_peak_rate": float(result["pair_correct_rate"][pair_index]),
                        "correct_plus_one_peak_rate": float(result["pair_plus_one_rate"][pair_index]),
                        "mean_correct_vs_best_other_margin": float(result["pair_correct_margin"][pair_index]),
                        "head1_offset1_rate": result["head1_offset1_rate"],
                        "final_correct_rate": result["final_correct_rate"],
                    }
                )

        for pair_index, pair_name in enumerate(PAIR_NAMES):
            aggregate_rows.append(
                {
                    "model": model_name,
                    "pair": pair_name,
                    "correct_peak_rate": float(np.mean([
                        result["pair_correct_rate"][pair_index]
                        for result in model_results
                    ])),
                    "correct_plus_one_peak_rate": float(np.mean([
                        result["pair_plus_one_rate"][pair_index]
                        for result in model_results
                    ])),
                    "mean_correct_vs_best_other_margin": float(np.mean([
                        result["pair_correct_margin"][pair_index]
                        for result in model_results
                    ])),
                }
            )
        model_rows.append(
            {
                "model": model_name,
                "mean_head1_offset1_rate": float(np.mean([
                    result["head1_offset1_rate"] for result in model_results
                ])),
                "mean_final_correct_rate": float(np.mean([
                    result["final_correct_rate"] for result in model_results
                ])),
            }
        )

    save_csv(args.output_dir / "per_lag_results.csv", per_lag_rows)
    save_csv(args.output_dir / "aggregate_pair_results.csv", aggregate_rows)
    save_csv(args.output_dir / "model_behavior_checks.csv", model_rows)
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(
            {
                "lags": list(args.lags),
                "n_sequences_per_lag": args.n_sequences,
                "models": list(selected_models),
                "pair_results": aggregate_rows,
                "behavior_checks": model_rows,
            },
            handle,
            indent=2,
        )

    figure = plot_results(
        aggregate_rows,
        model_rows,
        args.output_dir / "cross_model_raw_head1_path_test.png",
    )
    if args.no_show:
        plt.close(figure)
    else:
        plt.show()

    print("\nAggregate exact peak rates across lags:")
    for model_name in selected_models:
        checks = next(row for row in model_rows if row["model"] == model_name)
        raw_head1 = next(
            row for row in aggregate_rows
            if row["model"] == model_name and row["pair"] == "raw→Head1"
        )
        print(
            f"  {model_name}: Head1@D1={checks['mean_head1_offset1_rate']:.1%}, "
            f"final-correct={checks['mean_final_correct_rate']:.1%}, "
            f"raw→Head1 correct={raw_head1['correct_peak_rate']:.1%}, "
            f"raw→Head1 correct+1={raw_head1['correct_plus_one_peak_rate']:.1%}, "
            f"mean correct margin={raw_head1['mean_correct_vs_best_other_margin']:+.3f}"
        )
    print(f"\nSaved results to {args.output_dir}")


if __name__ == "__main__":
    main()
