"""Measure fixed-offset earlier heads across attention-only model depths.

For each model, lag, and layer, average attention over late query positions for
each sequence and record the offset with the most attention mass.  Earlier
layers are summarized by how often this peak remains at one absolute offset
across sequences with different lags.  The final layer is summarized by how
often its peak tracks the required offset ``lag - 1``.

Usage:
    python cross_model_fixed_offset_heads.py --quick --no-show
    python cross_model_fixed_offset_heads.py --no-show
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from cross_model_raw_head1_path_test import (
    MODEL_SPECS,
    attention_profiles,
    load_attention_only_model,
    parse_int_tuple,
)
from util import make_dataset_lagset


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def analyze_model(
    model,
    model_name: str,
    lags: tuple[int, ...],
    n_sequences: int,
    sequence_length: int,
    rho: float,
    query_start: int,
    query_stride: int,
    maximum_offset: int,
    device: torch.device,
) -> tuple[list[dict], list[dict]]:
    query_positions = torch.arange(
        query_start,
        sequence_length,
        query_stride,
        device=device,
    )
    per_lag_rows: list[dict] = []
    layer_peaks: dict[int, list[np.ndarray]] = {
        layer: [] for layer in range(model.n_layers)
    }
    layer_profile_masses: dict[int, list[np.ndarray]] = {
        layer: [] for layer in range(model.n_layers)
    }
    final_correct: list[np.ndarray] = []

    for lag in lags:
        print(f"  lag {lag}")
        inputs, _, sampled_lags = make_dataset_lagset(
            n_sequences,
            sequence_length,
            rho,
            [lag],
            seed=930_000 + 1_000 * model.n_layers + lag,
        )
        if not torch.all(sampled_lags == lag):
            raise RuntimeError("dataset returned an unexpected lag")
        inputs = inputs.to(device=device, dtype=torch.float64)
        _, attentions, post_attention, post_mlp = model(inputs)
        if any(not torch.equal(a, b) for a, b in zip(post_attention, post_mlp)):
            raise RuntimeError("checkpoint is not behaving as attention-only")
        if len(attentions) != model.n_layers:
            raise RuntimeError("this analysis expects one head per layer")

        for layer_index, attention in enumerate(attentions):
            profiles = attention_profiles(
                attention,
                query_positions,
                maximum_offset,
            )[:, 1:]
            peaks = 1 + profiles.argmax(dim=1)
            peaks_numpy = peaks.cpu().numpy()
            profiles_numpy = profiles.cpu().numpy()
            layer_peaks[layer_index].append(peaks_numpy)
            layer_profile_masses[layer_index].append(profiles_numpy)

            counts = Counter(peaks_numpy.tolist())
            modal_offset, modal_count = counts.most_common(1)[0]
            mean_profile = profiles_numpy.mean(axis=0)
            mean_profile_peak = int(mean_profile.argmax()) + 1
            correct_rate = float(np.mean(peaks_numpy == lag - 1))
            per_lag_rows.append(
                {
                    "model": model_name,
                    "lag": lag,
                    "layer": layer_index + 1,
                    "is_final_layer": layer_index == model.n_layers - 1,
                    "modal_sequence_peak_offset": modal_offset,
                    "modal_sequence_peak_rate": modal_count / n_sequences,
                    "mean_profile_peak_offset": mean_profile_peak,
                    "mean_mass_at_mean_profile_peak": float(
                        mean_profile[mean_profile_peak - 1]
                    ),
                    "correct_lag_minus_one_peak_rate": correct_rate,
                }
            )
            if layer_index == model.n_layers - 1:
                final_correct.append(peaks_numpy == lag - 1)

    aggregate_rows: list[dict] = []
    for layer_index in range(model.n_layers):
        pooled_peaks = np.concatenate(layer_peaks[layer_index])
        counts = Counter(pooled_peaks.tolist())
        global_mode, global_count = counts.most_common(1)[0]
        lag_modes = []
        for peaks in layer_peaks[layer_index]:
            lag_modes.append(Counter(peaks.tolist()).most_common(1)[0][0])
        pooled_profiles = np.concatenate(layer_profile_masses[layer_index], axis=0)
        is_final = layer_index == model.n_layers - 1
        aggregate_rows.append(
            {
                "model": model_name,
                "layer": layer_index + 1,
                "is_final_layer": is_final,
                "global_modal_offset": global_mode,
                "fixed_absolute_peak_rate": global_count / len(pooled_peaks),
                "fraction_lags_same_modal_offset": float(
                    np.mean(np.asarray(lag_modes) == global_mode)
                ),
                "mean_attention_mass_at_global_mode": float(
                    pooled_profiles[:, global_mode - 1].mean()
                ),
                "correct_lag_minus_one_peak_rate": (
                    float(np.concatenate(final_correct).mean()) if is_final else ""
                ),
            }
        )
    return per_lag_rows, aggregate_rows


def plot_results(
    selected_models: tuple[str, ...],
    lags: tuple[int, ...],
    per_lag_rows: list[dict],
    aggregate_rows: list[dict],
    output_path: Path,
) -> plt.Figure:
    figure, axes = plt.subplots(
        len(selected_models),
        2,
        figsize=(14, 3.6 * len(selected_models)),
        constrained_layout=True,
        squeeze=False,
    )
    for model_row, model_name in enumerate(selected_models):
        model_aggregate = [
            row for row in aggregate_rows if row["model"] == model_name
        ]
        n_layers = len(model_aggregate)
        offset_matrix = np.empty((n_layers, len(lags)))
        for layer_index in range(n_layers):
            for lag_index, lag in enumerate(lags):
                row = next(
                    item
                    for item in per_lag_rows
                    if item["model"] == model_name
                    and item["layer"] == layer_index + 1
                    and item["lag"] == lag
                )
                offset_matrix[layer_index, lag_index] = row[
                    "modal_sequence_peak_offset"
                ]

        image = axes[model_row, 0].imshow(
            offset_matrix,
            cmap="viridis",
            vmin=1,
            vmax=max(lags),
            aspect="auto",
        )
        for layer_index in range(n_layers):
            for lag_index in range(len(lags)):
                axes[model_row, 0].text(
                    lag_index,
                    layer_index,
                    f"{int(offset_matrix[layer_index, lag_index])}",
                    ha="center",
                    va="center",
                    color=(
                        "black"
                        if offset_matrix[layer_index, lag_index] > max(lags) * 0.55
                        else "white"
                    ),
                )
        axes[model_row, 0].set_xticks(range(len(lags)), lags)
        axes[model_row, 0].set_yticks(
            range(n_layers),
            [f"L{layer}" for layer in range(1, n_layers + 1)],
        )
        axes[model_row, 0].set_xlabel("data lag")
        axes[model_row, 0].set_ylabel("layer")
        axes[model_row, 0].set_title(
            f"{model_name}: modal attention offset per lag"
        )

        positions = np.arange(n_layers)
        fixed_rates = np.asarray(
            [row["fixed_absolute_peak_rate"] for row in model_aggregate]
        )
        masses = np.asarray(
            [row["mean_attention_mass_at_global_mode"] for row in model_aggregate]
        )
        width = 0.38
        axes[model_row, 1].bar(
            positions - width / 2,
            fixed_rates,
            width,
            label="fixed absolute peak rate",
        )
        axes[model_row, 1].bar(
            positions + width / 2,
            masses,
            width,
            label="mass at modal offset",
        )
        axes[model_row, 1].set_xticks(
            positions,
            [f"L{layer}" for layer in range(1, n_layers + 1)],
        )
        axes[model_row, 1].set_ylim(0, 1.05)
        axes[model_row, 1].set_ylabel("fraction / attention mass")
        axes[model_row, 1].set_title(f"{model_name}: fixedness and concentration")
        axes[model_row, 1].grid(axis="y", alpha=0.3)
        axes[model_row, 1].legend(fontsize=8)

    figure.colorbar(
        image,
        ax=axes[:, 0],
        label="modal query−key offset",
        shrink=0.8,
    )
    figure.suptitle(
        "Earlier attention-only layers use fixed offsets; final layers may track lag"
    )
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
        default=Path("/tmp/cross_model_fixed_offset_heads"),
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    selected_models = tuple(
        part.strip() for part in args.models.split(",") if part.strip()
    )
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

    for model_name in selected_models:
        n_layers, checkpoint = MODEL_SPECS[model_name]
        print(f"\n{model_name}: {checkpoint}")
        model = load_attention_only_model(n_layers, checkpoint, device)
        current_per_lag, current_aggregate = analyze_model(
            model=model,
            model_name=model_name,
            lags=args.lags,
            n_sequences=args.n_sequences,
            sequence_length=args.sequence_length,
            rho=args.rho,
            query_start=args.query_start,
            query_stride=args.query_stride,
            maximum_offset=args.maximum_offset,
            device=device,
        )
        per_lag_rows.extend(current_per_lag)
        aggregate_rows.extend(current_aggregate)

    save_csv(args.output_dir / "per_lag_layer_offsets.csv", per_lag_rows)
    save_csv(args.output_dir / "aggregate_layer_fixedness.csv", aggregate_rows)
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(
            {
                "models": list(selected_models),
                "lags": list(args.lags),
                "n_sequences_per_lag": args.n_sequences,
                "aggregate": aggregate_rows,
            },
            handle,
            indent=2,
        )

    figure = plot_results(
        selected_models,
        args.lags,
        per_lag_rows,
        aggregate_rows,
        args.output_dir / "cross_model_fixed_offsets.png",
    )
    if args.no_show:
        plt.close(figure)
    else:
        plt.show()

    print("\nAggregate layer behavior:")
    for model_name in selected_models:
        print(f"\n{model_name}")
        for row in aggregate_rows:
            if row["model"] != model_name:
                continue
            if row["is_final_layer"]:
                print(
                    f"  L{row['layer']} final: correct tracking="
                    f"{float(row['correct_lag_minus_one_peak_rate']):.1%}"
                )
            else:
                print(
                    f"  L{row['layer']}: mode D={row['global_modal_offset']}, "
                    f"fixed={row['fixed_absolute_peak_rate']:.1%}, "
                    f"lags-agree={row['fraction_lags_same_modal_offset']:.1%}, "
                    f"mass={row['mean_attention_mass_at_global_mode']:.3f}"
                )
    print(f"\nSaved results to {args.output_dir}")


if __name__ == "__main__":
    main()
