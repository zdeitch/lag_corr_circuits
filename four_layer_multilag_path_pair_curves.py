"""Path-pair contribution curves across lags for the 4L attention-only model.

For each lag and sequence, this script:

1. Runs a clean forward pass and freezes the attention patterns in Layers 1--3.
2. Expands the residual entering Layer 4 into its eight computational paths.
3. Decomposes the final head's raw pre-softmax score at every candidate offset
   into 8 x 8 query-path/key-path contributions.

For each path pair, it reports only a small set of measurements:

* modal offset: the most common per-sequence peak location;
* peak contribution: the mean contribution at that modal offset;
* fixed-peak rate: fraction of sequences whose peak equals the modal offset;
* 5th--95th percentile band across individual sequence-level curves.

The full curves are saved in an NPZ file.  The analysis is exact conditional on
the clean attention patterns: summing all 64 path-pair curves reconstructs the
real final-head raw score curve.

Usage:
    python four_layer_multilag_path_pair_curves.py --quick
    python four_layer_multilag_path_pair_curves.py
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

from four_layer_attention_path_analysis import (
    expand_prefinal_paths,
    load_model,
    parse_int_tuple,
)
from util import make_dataset_lagset


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def modal_offset_and_rate(peaks: np.ndarray, maximum_offset: int) -> tuple[int, float]:
    counts = np.bincount(peaks.astype(np.int64), minlength=maximum_offset + 1)
    mode = int(counts.argmax())
    return mode, float(counts[mode] / len(peaks))


@torch.inference_mode()
def curves_for_lag(
    model,
    lag: int,
    n_sequences: int,
    sequence_length: int,
    rho: float,
    query_start: int,
    query_stride: int,
    maximum_offset: int,
    device: torch.device,
) -> tuple[np.ndarray, list[str], float, float]:
    """Return sequence x query-path x key-path x offset contribution curves."""
    if query_start < maximum_offset:
        raise ValueError("query_start must be at least maximum_offset")

    inputs, _, sampled_lags = make_dataset_lagset(
        n_sequences,
        sequence_length,
        rho,
        [lag],
        seed=300_000 + lag,
    )
    if not torch.all(sampled_lags == lag):
        raise RuntimeError("dataset returned an unexpected lag")
    inputs = inputs.to(device=device, dtype=torch.float64)

    _, attentions, post_attention, post_mlp = model(inputs)
    if any(not torch.equal(a, b) for a, b in zip(post_attention, post_mlp)):
        raise RuntimeError("checkpoint is not behaving as attention-only")

    embedding = model.W_r(inputs.unsqueeze(-1))
    path_dict = expand_prefinal_paths(
        embedding,
        attentions,
        model,
        post_attention[: model.n_layers - 1],
    )
    labels = list(path_dict)
    path_components = torch.stack(list(path_dict.values()))
    final_input = post_attention[-2]
    residual_error = (
        (path_components.sum(dim=0) - final_input).norm()
        / final_input.norm().clamp_min(1e-30)
    )

    query_matrix, key_matrix, _, _ = model.layers[-1][0]
    all_positions = torch.arange(sequence_length, device=device).view(1, 1, -1)
    path_queries = model.apply_rope(path_components @ query_matrix, all_positions)
    path_keys = model.apply_rope(path_components @ key_matrix, all_positions)
    full_queries = model.apply_rope(
        final_input @ query_matrix, all_positions.squeeze(0)
    )
    full_keys = model.apply_rope(
        final_input @ key_matrix, all_positions.squeeze(0)
    )

    query_positions = torch.arange(
        query_start,
        sequence_length,
        query_stride,
        device=device,
    )
    offsets = torch.arange(1, maximum_offset + 1, device=device)
    sequence_curves = torch.empty(
        n_sequences,
        len(labels),
        len(labels),
        len(offsets),
        dtype=torch.float64,
        device=device,
    )
    maximum_score_error = 0.0

    path_query_at_positions = path_queries[:, :, query_positions, :]
    full_query_at_positions = full_queries[:, query_positions, :]

    for offset_index, offset_tensor in enumerate(offsets):
        offset = int(offset_tensor.item())
        path_key_at_offset = path_keys[:, :, query_positions - offset, :]
        pair_scores = torch.einsum(
            "pbnh,qbnh->bnpq",
            path_query_at_positions,
            path_key_at_offset,
        ) / math.sqrt(model.d_head)
        sequence_curves[..., offset_index] = pair_scores.mean(dim=1)

        direct_scores = (
            full_query_at_positions
            * full_keys[:, query_positions - offset, :]
        ).sum(dim=-1) / math.sqrt(model.d_head)
        score_error = (pair_scores.sum(dim=(-1, -2)) - direct_scores).abs().max()
        maximum_score_error = max(maximum_score_error, float(score_error.item()))

    return (
        sequence_curves.cpu().numpy(),
        labels,
        float(residual_error.item()),
        maximum_score_error,
    )


def summarize_lag(
    lag: int,
    curves: np.ndarray,
    labels: list[str],
    offsets: np.ndarray,
) -> tuple[list[dict], dict[str, np.ndarray], np.ndarray]:
    n_sequences = curves.shape[0]
    mean = curves.mean(axis=0)
    low, high = np.percentile(curves, [5, 95], axis=0)
    sequence_peaks = offsets[curves.argmax(axis=-1)]

    rows: list[dict] = []
    for query_index, query_label in enumerate(labels):
        for key_index, key_label in enumerate(labels):
            peaks = sequence_peaks[:, query_index, key_index]
            modal_offset, fixed_peak_rate = modal_offset_and_rate(
                peaks, int(offsets[-1])
            )
            modal_index = int(np.where(offsets == modal_offset)[0][0])
            rows.append(
                {
                    "lag": lag,
                    "correct_offset": lag - 1,
                    "query_path": query_label,
                    "key_path": key_label,
                    "modal_offset": modal_offset,
                    "fixed_peak_rate": fixed_peak_rate,
                    "peak_contribution": float(
                        mean[query_index, key_index, modal_index]
                    ),
                    "peak_sequence_p05": float(
                        low[query_index, key_index, modal_index]
                    ),
                    "peak_sequence_p95": float(
                        high[query_index, key_index, modal_index]
                    ),
                }
            )

    arrays = {
        "mean": mean,
        "sequence_p05": low,
        "sequence_p95": high,
        "sequence_peaks": sequence_peaks,
    }
    return rows, arrays, sequence_peaks


def summarize_pooled_fixedness(
    curves_by_lag: dict[int, np.ndarray],
    peaks_by_lag: dict[int, np.ndarray],
    labels: list[str],
    offsets: np.ndarray,
) -> list[dict]:
    pooled_curves = np.concatenate(list(curves_by_lag.values()), axis=0)
    pooled_peaks = np.concatenate(list(peaks_by_lag.values()), axis=0)
    mean = pooled_curves.mean(axis=0)
    low, high = np.percentile(pooled_curves, [5, 95], axis=0)

    rows: list[dict] = []
    for query_index, query_label in enumerate(labels):
        for key_index, key_label in enumerate(labels):
            peaks = pooled_peaks[:, query_index, key_index]
            modal_offset, fixed_peak_rate = modal_offset_and_rate(
                peaks, int(offsets[-1])
            )
            modal_index = int(np.where(offsets == modal_offset)[0][0])
            peak = float(mean[query_index, key_index, modal_index])
            lag_modes = []
            for lag_peaks in peaks_by_lag.values():
                lag_mode, _ = modal_offset_and_rate(
                    lag_peaks[:, query_index, key_index], int(offsets[-1])
                )
                lag_modes.append(lag_mode)
            rows.append(
                {
                    "query_path": query_label,
                    "key_path": key_label,
                    "modal_offset_across_all_lags": modal_offset,
                    "fixed_peak_rate_across_all_lags": fixed_peak_rate,
                    "fraction_lags_with_same_modal_offset": float(
                        np.mean(np.asarray(lag_modes) == modal_offset)
                    ),
                    "peak_contribution_across_all_lags": peak,
                    "peak_sequence_p05": float(
                        low[query_index, key_index, modal_index]
                    ),
                    "peak_sequence_p95": float(
                        high[query_index, key_index, modal_index]
                    ),
                }
            )
    return rows


def plot_pair_pages_for_lag(
    lag: int,
    arrays: dict[str, np.ndarray],
    rows: list[dict],
    labels: list[str],
    offsets: np.ndarray,
    output_dir: Path,
) -> list[plt.Figure]:
    mean = arrays["mean"]
    low = arrays["sequence_p05"]
    high = arrays["sequence_p95"]
    lookup = {(row["query_path"], row["key_path"]): row for row in rows}
    figures: list[plt.Figure] = []

    for query_index, query_label in enumerate(labels):
        figure, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True)
        axes = axes.ravel()
        for key_index, key_label in enumerate(labels):
            axis = axes[key_index]
            axis.plot(
                offsets,
                mean[query_index, key_index],
                linewidth=1.35,
                label="mean contribution",
            )
            axis.fill_between(
                offsets,
                low[query_index, key_index],
                high[query_index, key_index],
                alpha=0.2,
                label="5th–95th percentile",
            )
            row = lookup[(query_label, key_label)]
            axis.axvline(
                row["modal_offset"],
                color="black",
                linestyle=":",
                linewidth=1.0,
                label="modal peak offset",
            )
            axis.axvline(
                lag - 1,
                color="red",
                linestyle="--",
                linewidth=0.9,
                label="correct offset",
            )
            axis.axhline(0, color="gray", linewidth=0.5)
            axis.set_title(
                f"Key path {key_label}\n"
                f"modal offset={row['modal_offset']}  |  "
                f"fixed-peak rate={row['fixed_peak_rate']:.0%}",
                fontsize=10,
            )
            axis.grid(alpha=0.18)
            if key_index >= 4:
                axis.set_xlabel("query − key offset")
            if key_index in (0, 4):
                axis.set_ylabel("raw-score contribution")
        axes[0].legend(fontsize=8, loc="best")
        figure.suptitle(
            f"Lag {lag}: query path {query_label} paired with every key path\n"
            "Each curve is a final-head pre-softmax score contribution",
            y=1.01,
        )
        figure.tight_layout()
        figure.savefig(
            output_dir
            / f"path_pair_curves_lag_{lag}_query_{query_label}.png",
            dpi=180,
        )
        figures.append(figure)
    return figures


def plot_pooled_fixedness(
    rows: list[dict],
    labels: list[str],
    output_path: Path,
) -> plt.Figure:
    mode = np.empty((len(labels), len(labels)))
    rate = np.empty_like(mode)
    lookup = {(row["query_path"], row["key_path"]): row for row in rows}
    for query_index, query_label in enumerate(labels):
        for key_index, key_label in enumerate(labels):
            row = lookup[(query_label, key_label)]
            mode[query_index, key_index] = row["modal_offset_across_all_lags"]
            rate[query_index, key_index] = row["fixed_peak_rate_across_all_lags"]

    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(rate, vmin=0, vmax=1, cmap="viridis")
    for query_index in range(len(labels)):
        for key_index in range(len(labels)):
            color = "white" if rate[query_index, key_index] < 0.55 else "black"
            axis.text(
                key_index,
                query_index,
                f"D={int(mode[query_index, key_index])}\n{rate[query_index, key_index]:.0%}",
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )
    axis.set_xticks(range(len(labels)), labels)
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("key path")
    axis.set_ylabel("query path")
    axis.set_title("Path-pair peak fixedness pooled across lags")
    figure.colorbar(image, ax=axis, label="fixed-peak rate")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    return figure


def plot_correct_offset_alignment(
    lags: tuple[int, ...],
    arrays_by_lag: dict[int, dict[str, np.ndarray]],
    labels: list[str],
    offsets: np.ndarray,
    output_path: Path,
) -> plt.Figure:
    """Compare path-pair matrices at each lag's own correct offset."""
    matrices = []
    for lag in lags:
        correct_index = int(np.where(offsets == lag - 1)[0][0])
        matrices.append(arrays_by_lag[lag]["mean"][..., correct_index])

    limit = max(float(np.abs(matrix).max()) for matrix in matrices)
    figure, axes = plt.subplots(
        1,
        len(lags),
        figsize=(6.0 * len(lags), 5.5),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for axis, lag, matrix in zip(axes[0], lags, matrices):
        image = axis.imshow(
            matrix,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        for query_index in range(len(labels)):
            for key_index in range(len(labels)):
                value = matrix[query_index, key_index]
                text_color = "white" if abs(value) > 0.55 * limit else "black"
                axis.text(
                    key_index,
                    query_index,
                    f"{value:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=text_color,
                )
        axis.set_xticks(range(len(labels)), labels)
        axis.set_yticks(range(len(labels)), labels)
        axis.set_xlabel("key path")
        axis.set_ylabel("query path")
        axis.set_title(
            f"Lag {lag}: correct offset D={lag - 1}\n"
            f"64 terms sum to {matrix.sum():+.2f}"
        )

    if image is not None:
        figure.colorbar(
            image,
            ax=axes[0],
            label="mean raw-score contribution",
            shrink=0.84,
        )
    subtitle = ""
    if len(matrices) == 2:
        correlation = float(
            np.corrcoef(matrices[0].ravel(), matrices[1].ravel())[0, 1]
        )
        subtitle = f"; cellwise Pearson correlation = {correlation:.3f}"
    figure.suptitle(
        "Path-pair contributions at each lag's correct RoPE offset" + subtitle
    )
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
        default=(25, 30, 35, 40, 45, 50),
    )
    parser.add_argument("--n-sequences", type=int, default=96)
    parser.add_argument("--sequence-length", type=int, default=200)
    parser.add_argument("--rho", type=float, default=0.9)
    parser.add_argument("--query-start", type=int, default=120)
    parser.add_argument("--query-stride", type=int, default=4)
    parser.add_argument("--maximum-offset", type=int, default=100)
    parser.add_argument("--plot-lag", type=int, default=40)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/four_layer_multilag_path_pair_curves"),
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="save figures without opening/displaying them",
    )
    args = parser.parse_args()

    if args.quick:
        args.n_sequences = min(args.n_sequences, 16)
    if args.plot_lag not in args.lags:
        raise ValueError("plot_lag must be included in lags")

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint, device)
    offsets = np.arange(1, args.maximum_offset + 1)

    curves_by_lag: dict[int, np.ndarray] = {}
    arrays_by_lag: dict[int, dict[str, np.ndarray]] = {}
    peaks_by_lag: dict[int, np.ndarray] = {}
    per_lag_rows: list[dict] = []
    labels: list[str] | None = None
    reconstruction: dict[int, dict[str, float]] = {}

    for lag in args.lags:
        print(f"lag {lag}: computing path-pair curves")
        curves, current_labels, residual_error, score_error = curves_for_lag(
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
        if labels is None:
            labels = current_labels
        elif labels != current_labels:
            raise RuntimeError("path labels changed between lags")

        rows, arrays, peaks = summarize_lag(lag, curves, labels, offsets)
        curves_by_lag[lag] = curves
        arrays_by_lag[lag] = arrays
        peaks_by_lag[lag] = peaks
        per_lag_rows.extend(rows)
        reconstruction[lag] = {
            "residual_relative_error": residual_error,
            "maximum_raw_score_error": score_error,
        }

    assert labels is not None
    pooled_rows = summarize_pooled_fixedness(
        curves_by_lag,
        peaks_by_lag,
        labels,
        offsets,
    )
    save_csv(args.output_dir / "per_lag_path_pair_summary.csv", per_lag_rows)
    save_csv(args.output_dir / "pooled_path_pair_fixedness.csv", pooled_rows)

    npz_values: dict[str, np.ndarray] = {"offsets": offsets}
    for lag, arrays in arrays_by_lag.items():
        for name in ("mean", "sequence_p05", "sequence_p95"):
            npz_values[f"lag_{lag}_{name}"] = arrays[name]
    np.savez_compressed(args.output_dir / "path_pair_curves.npz", **npz_values)

    selected_rows = [row for row in per_lag_rows if row["lag"] == args.plot_lag]
    curve_figures = plot_pair_pages_for_lag(
        args.plot_lag,
        arrays_by_lag[args.plot_lag],
        selected_rows,
        labels,
        offsets,
        args.output_dir,
    )
    fixedness_figure = plot_pooled_fixedness(
        pooled_rows,
        labels,
        args.output_dir / "pooled_path_pair_fixedness.png",
    )
    alignment_figure = plot_correct_offset_alignment(
        args.lags,
        arrays_by_lag,
        labels,
        offsets,
        args.output_dir / "correct_offset_path_pair_alignment.png",
    )

    if args.no_show:
        for curve_figure in curve_figures:
            plt.close(curve_figure)
        plt.close(fixedness_figure)
        plt.close(alignment_figure)
    else:
        plt.show()

    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "lags": list(args.lags),
                "n_sequences_per_lag": args.n_sequences,
                "offset_range": [1, args.maximum_offset],
                "path_labels": labels,
                "path_bit_meaning": "0=identity, 1=frozen attention/OV write in Layers 1--3",
                "observation_band": "5th--95th percentile across sequence-level curves",
                "peak_definition": "largest signed raw-score contribution",
                "reconstruction": reconstruction,
            },
            handle,
            indent=2,
        )

    print("\nMost fixed path-pair peaks across all lags:")
    strongest_fixed = sorted(
        pooled_rows,
        key=lambda row: row["fixed_peak_rate_across_all_lags"],
        reverse=True,
    )[:12]
    for row in strongest_fixed:
        print(
            f"  q={row['query_path']} k={row['key_path']}  "
            f"mode D={row['modal_offset_across_all_lags']:>3}  "
            f"fixed={row['fixed_peak_rate_across_all_lags']:.1%}  "
            f"peak={row['peak_contribution_across_all_lags']:+.4f}  "
            f"lags-agree={row['fraction_lags_with_same_modal_offset']:.1%}"
        )
    print(f"\nSaved results to {args.output_dir}")


if __name__ == "__main__":
    main()
