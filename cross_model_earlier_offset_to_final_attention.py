"""Map earlier-layer attention shifts into final-layer attention changes.

For every attention-only model and every layer before the final layer, shift
that layer's clean attention matrix by -1, 0, or +1 key position.  Recompute
all downstream layers normally and measure both the final attention peak and
the complete final attention matrix.

The shift-zero condition patches the original clean matrix back into the model
and serves as an exact implementation control.

Usage:
    python cross_model_earlier_offset_to_final_attention.py --quick --no-show
    python cross_model_earlier_offset_to_final_attention.py --no-show
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
from four_layer_earlier_attention_shift_patch import (
    forward_with_attention_patch,
    shift_attention_key_axis,
)
from util import make_dataset_lagset


SHIFT_CATEGORIES = ("≤−3", "−2", "−1", "0", "+1", "+2", "≥+3")


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def categorize_peak_shifts(shifts: np.ndarray) -> np.ndarray:
    counts = np.asarray(
        [
            np.sum(shifts <= -3),
            np.sum(shifts == -2),
            np.sum(shifts == -1),
            np.sum(shifts == 0),
            np.sum(shifts == 1),
            np.sum(shifts == 2),
            np.sum(shifts >= 3),
        ],
        dtype=float,
    )
    return counts / len(shifts)


@torch.inference_mode()
def run_model(
    model,
    model_name: str,
    lags: tuple[int, ...],
    shifts: tuple[int, ...],
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
    observations: list[dict] = []
    per_lag_rows: list[dict] = []

    for lag in lags:
        print(f"  lag {lag}")
        inputs, _, sampled_lags = make_dataset_lagset(
            n_sequences,
            sequence_length,
            rho,
            [lag],
            seed=1_030_000 + 1_000 * model.n_layers + lag,
        )
        if not torch.all(sampled_lags == lag):
            raise RuntimeError("dataset returned an unexpected lag")
        inputs = inputs.to(device=device, dtype=torch.float64)
        _, clean_attentions, _, _ = model(inputs)
        clean_final = clean_attentions[-1]
        clean_final_profiles = attention_profiles(
            clean_final,
            query_positions,
            maximum_offset,
        )[:, 1:]
        clean_final_peaks = 1 + clean_final_profiles.argmax(dim=1)

        for patch_layer in range(model.n_layers - 1):
            clean_layer_profiles = attention_profiles(
                clean_attentions[patch_layer],
                query_positions,
                maximum_offset,
            )[:, 1:]
            clean_layer_peaks = 1 + clean_layer_profiles.argmax(dim=1)
            clean_layer_mode = Counter(
                clean_layer_peaks.cpu().tolist()
            ).most_common(1)[0][0]

            for offset_shift in shifts:
                patched_attention = shift_attention_key_axis(
                    clean_attentions[patch_layer],
                    offset_shift,
                )
                _, patched_attentions, _, _ = forward_with_attention_patch(
                    model,
                    inputs,
                    patch_layer,
                    patched_attention,
                )
                patched_final = patched_attentions[-1]
                patched_final_profiles = attention_profiles(
                    patched_final,
                    query_positions,
                    maximum_offset,
                )[:, 1:]
                patched_final_peaks = 1 + patched_final_profiles.argmax(dim=1)
                final_peak_shifts = (
                    patched_final_peaks - clean_final_peaks
                ).cpu().numpy()

                rowwise_total_variation = 0.5 * (
                    patched_final[:, query_positions, :]
                    - clean_final[:, query_positions, :]
                ).abs().sum(dim=-1)
                sequence_total_variation = rowwise_total_variation.mean(dim=1)
                correct_rate = (
                    patched_final_peaks == lag - 1
                ).double().mean()
                clean_peak_mass = patched_final_profiles.gather(
                    1,
                    (clean_final_peaks - 1).unsqueeze(1),
                ).squeeze(1)
                original_clean_peak_mass = clean_final_profiles.gather(
                    1,
                    (clean_final_peaks - 1).unsqueeze(1),
                ).squeeze(1)
                mass_change = clean_peak_mass - original_clean_peak_mass

                observations.append(
                    {
                        "model": model_name,
                        "lag": lag,
                        "patch_layer": patch_layer + 1,
                        "clean_layer_modal_offset": clean_layer_mode,
                        "offset_shift": offset_shift,
                        "final_peak_shifts": final_peak_shifts,
                        "total_variation": sequence_total_variation.cpu().numpy(),
                        "correct_flags": (
                            patched_final_peaks == lag - 1
                        ).cpu().numpy(),
                        "clean_peak_mass_change": mass_change.cpu().numpy(),
                    }
                )

                patched_mode = Counter(
                    patched_final_peaks.cpu().tolist()
                ).most_common(1)[0][0]
                clean_mode = Counter(
                    clean_final_peaks.cpu().tolist()
                ).most_common(1)[0][0]
                distribution = categorize_peak_shifts(final_peak_shifts)
                per_lag_rows.append(
                    {
                        "model": model_name,
                        "lag": lag,
                        "patch_layer": patch_layer + 1,
                        "clean_layer_modal_offset": clean_layer_mode,
                        "offset_shift": offset_shift,
                        "clean_final_modal_offset": clean_mode,
                        "patched_final_modal_offset": patched_mode,
                        "modal_final_peak_shift": patched_mode - clean_mode,
                        "exact_clean_peak_retention_rate": float(
                            np.mean(final_peak_shifts == 0)
                        ),
                        "final_correct_lag_minus_one_rate": float(correct_rate.item()),
                        "mean_final_matrix_total_variation": float(
                            sequence_total_variation.mean().item()
                        ),
                        "mean_clean_peak_mass_change": float(mass_change.mean().item()),
                        **{
                            f"peak_shift_{category}_rate": float(rate)
                            for category, rate in zip(SHIFT_CATEGORIES, distribution)
                        },
                    }
                )

    aggregate_rows: list[dict] = []
    for patch_layer in range(1, model.n_layers):
        for offset_shift in shifts:
            selected = [
                row
                for row in observations
                if row["patch_layer"] == patch_layer
                and row["offset_shift"] == offset_shift
            ]
            peak_shifts = np.concatenate(
                [row["final_peak_shifts"] for row in selected]
            )
            total_variation = np.concatenate(
                [row["total_variation"] for row in selected]
            )
            correct_flags = np.concatenate(
                [row["correct_flags"] for row in selected]
            )
            mass_change = np.concatenate(
                [row["clean_peak_mass_change"] for row in selected]
            )
            distribution = categorize_peak_shifts(peak_shifts)
            mode_shift, mode_count = Counter(peak_shifts.tolist()).most_common(1)[0]
            clean_modes = [row["clean_layer_modal_offset"] for row in selected]
            natural_mode = Counter(clean_modes).most_common(1)[0][0]
            aggregate_rows.append(
                {
                    "model": model_name,
                    "patch_layer": patch_layer,
                    "clean_layer_modal_offset": natural_mode,
                    "offset_shift": offset_shift,
                    "nominal_patched_modal_offset": natural_mode + offset_shift,
                    "modal_final_peak_shift": int(mode_shift),
                    "modal_final_peak_shift_rate": mode_count / len(peak_shifts),
                    "exact_clean_peak_retention_rate": float(
                        np.mean(peak_shifts == 0)
                    ),
                    "final_correct_lag_minus_one_rate": float(correct_flags.mean()),
                    "mean_final_matrix_total_variation": float(total_variation.mean()),
                    "mean_clean_peak_mass_change": float(mass_change.mean()),
                    **{
                        f"peak_shift_{category}_rate": float(rate)
                        for category, rate in zip(SHIFT_CATEGORIES, distribution)
                    },
                }
            )
    return per_lag_rows, aggregate_rows


def metric_matrix(
    rows: list[dict],
    n_earlier_layers: int,
    shifts: tuple[int, ...],
    metric: str,
) -> np.ndarray:
    return np.asarray(
        [
            [
                next(
                    row[metric]
                    for row in rows
                    if row["patch_layer"] == layer
                    and row["offset_shift"] == shift
                )
                for shift in shifts
            ]
            for layer in range(1, n_earlier_layers + 1)
        ]
    )


def plot_model_summary(
    model_name: str,
    n_layers: int,
    shifts: tuple[int, ...],
    rows: list[dict],
    output_path: Path,
) -> plt.Figure:
    metrics = (
        (
            "exact_clean_peak_retention_rate",
            "Final peak remains at its clean offset",
            "viridis",
            0,
            1,
            "percent",
        ),
        (
            "final_correct_lag_minus_one_rate",
            "Final peak is at lag−1",
            "viridis",
            0,
            1,
            "percent",
        ),
        (
            "mean_final_matrix_total_variation",
            "Change in full final attention matrix",
            "magma",
            0,
            None,
            "decimal",
        ),
        (
            "modal_final_peak_shift",
            "Modal final peak shift",
            "coolwarm",
            None,
            None,
            "signed",
        ),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for axis, (metric, title, cmap, fixed_min, fixed_max, formatting) in zip(
        axes.flat, metrics
    ):
        matrix = metric_matrix(rows, n_layers - 1, shifts, metric)
        if cmap == "coolwarm":
            limit = max(float(np.max(np.abs(matrix))), 1.0)
            vmin, vmax = -limit, limit
        else:
            vmin, vmax = fixed_min, fixed_max
        image = axis.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                if formatting == "percent":
                    label = f"{value:.1%}"
                elif formatting == "signed":
                    label = f"{value:+.0f}"
                else:
                    label = f"{value:.3f}"
                axis.text(column_index, row_index, label, ha="center", va="center")
        axis.set_xticks(range(len(shifts)), [f"{shift:+d}" for shift in shifts])
        axis.set_yticks(
            range(n_layers - 1),
            [f"Layer {layer}" for layer in range(1, n_layers)],
        )
        axis.set_xlabel("shift applied to earlier attention offset")
        axis.set_ylabel("patched layer")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=0.8)
    figure.suptitle(
        f"{model_name}: how earlier attention addresses affect the final layer"
    )
    figure.savefig(output_path, dpi=180)
    return figure


def plot_peak_shift_distribution(
    model_name: str,
    n_layers: int,
    shifts: tuple[int, ...],
    rows: list[dict],
    output_path: Path,
) -> plt.Figure:
    conditions = [
        (layer, shift)
        for layer in range(1, n_layers)
        for shift in shifts
    ]
    matrix = np.asarray(
        [
            [
                next(
                    row[f"peak_shift_{category}_rate"]
                    for row in rows
                    if row["patch_layer"] == layer
                    and row["offset_shift"] == shift
                )
                for category in SHIFT_CATEGORIES
            ]
            for layer, shift in conditions
        ]
    )
    figure, axis = plt.subplots(
        figsize=(10, max(5, 0.45 * len(conditions))),
        constrained_layout=True,
    )
    image = axis.imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            if matrix[row_index, column_index] >= 0.005:
                axis.text(
                    column_index,
                    row_index,
                    f"{matrix[row_index, column_index]:.0%}",
                    ha="center",
                    va="center",
                    color="white" if matrix[row_index, column_index] < 0.55 else "black",
                )
    axis.set_xticks(range(len(SHIFT_CATEGORIES)), SHIFT_CATEGORIES)
    axis.set_yticks(
        range(len(conditions)),
        [f"Layer {layer}, patch {shift:+d}" for layer, shift in conditions],
    )
    axis.set_xlabel("patched final peak − clean final peak")
    axis.set_ylabel("earlier-layer intervention")
    axis.set_title(f"{model_name}: complete distribution of final attention peak shifts")
    figure.colorbar(image, ax=axis, label="fraction of sequences")
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
        "--lags", type=parse_int_tuple, default=(25, 30, 35, 40, 45, 50)
    )
    parser.add_argument(
        "--shifts", type=parse_int_tuple, default=(-1, 0, 1)
    )
    parser.add_argument("--n-sequences", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=200)
    parser.add_argument("--rho", type=float, default=0.9)
    parser.add_argument("--query-start", type=int, default=120)
    parser.add_argument("--query-stride", type=int, default=4)
    parser.add_argument("--maximum-offset", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/cross_model_earlier_offset_to_final_attention"),
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
        args.n_sequences = min(args.n_sequences, 8)

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_per_lag_rows: list[dict] = []
    all_aggregate_rows: list[dict] = []
    figures: list[plt.Figure] = []

    for model_name in selected_models:
        n_layers, checkpoint = MODEL_SPECS[model_name]
        print(f"\n{model_name}: {checkpoint}")
        model = load_attention_only_model(n_layers, checkpoint, device)
        per_lag_rows, aggregate_rows = run_model(
            model=model,
            model_name=model_name,
            lags=args.lags,
            shifts=args.shifts,
            n_sequences=args.n_sequences,
            sequence_length=args.sequence_length,
            rho=args.rho,
            query_start=args.query_start,
            query_stride=args.query_stride,
            maximum_offset=args.maximum_offset,
            device=device,
        )
        all_per_lag_rows.extend(per_lag_rows)
        all_aggregate_rows.extend(aggregate_rows)
        figures.append(
            plot_model_summary(
                model_name,
                n_layers,
                args.shifts,
                aggregate_rows,
                args.output_dir / f"{model_name}_final_attention_summary.png",
            )
        )
        figures.append(
            plot_peak_shift_distribution(
                model_name,
                n_layers,
                args.shifts,
                aggregate_rows,
                args.output_dir / f"{model_name}_final_peak_shift_distribution.png",
            )
        )

    save_csv(args.output_dir / "per_lag_results.csv", all_per_lag_rows)
    save_csv(args.output_dir / "aggregate_results.csv", all_aggregate_rows)
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(
            {
                "models": list(selected_models),
                "lags": list(args.lags),
                "n_sequences_per_lag": args.n_sequences,
                "offset_shifts": list(args.shifts),
                "peak_shift_definition": "patched final peak minus clean final peak",
                "matrix_change_metric": (
                    "mean rowwise total-variation distance between patched and "
                    "clean final attention matrices over analyzed query positions"
                ),
                "aggregate": all_aggregate_rows,
            },
            handle,
            indent=2,
        )

    if args.no_show:
        for figure in figures:
            plt.close(figure)
    else:
        plt.show()

    print("\nLargest downstream changes:")
    changed = [row for row in all_aggregate_rows if row["offset_shift"] != 0]
    for row in sorted(
        changed,
        key=lambda item: item["mean_final_matrix_total_variation"],
        reverse=True,
    )[:20]:
        print(
            f"  {row['model']} L{row['patch_layer']} "
            f"{row['clean_layer_modal_offset']}→"
            f"{row['nominal_patched_modal_offset']}: "
            f"retain={row['exact_clean_peak_retention_rate']:.1%}, "
            f"mode-shift={row['modal_final_peak_shift']:+d}, "
            f"TV={row['mean_final_matrix_total_variation']:.3f}, "
            f"final-correct={row['final_correct_lag_minus_one_rate']:.1%}"
        )
    print(f"\nSaved results to {args.output_dir}")


if __name__ == "__main__":
    main()
