"""Visualize which path pairs build the final score at every offset.

This script reads the mean path-pair contribution curves produced by
``four_layer_multilag_path_pair_curves.py``.  It assigns each path an
approximate transported delay from the fixed attention offsets in Layers 1--3
and organizes the 64 pair contributions by key-delay minus query-delay.

For a candidate query-minus-key offset D, a pair with relative delay delta
compares input information separated by approximately D + delta.  On data with
lag L, the simple delay-basis prediction is therefore that this pair responds
near D = L - delta.  The plots test that prediction against the actual final
QK/RoPE score contributions.

Usage:
    python four_layer_offset_path_pair_atlas.py --lag 40 --no-show
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_PATH_DELAYS = {
    "000": 0,
    "001": 23,
    "010": 21,
    "011": 44,
    "100": 1,
    "101": 24,
    "110": 22,
    "111": 45,
}


def parse_int_tuple(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_curves(
    input_dir: Path,
    lag: int,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    with (input_dir / "summary.json").open() as handle:
        summary = json.load(handle)
    if lag not in summary["lags"]:
        raise ValueError(
            f"lag {lag} is not in the saved lags {summary['lags']}"
        )
    archive = np.load(input_dir / "path_pair_curves.npz")
    offsets = archive["offsets"]
    curves = archive[f"lag_{lag}_mean"]
    labels = summary["path_labels"]
    if curves.shape != (len(labels), len(labels), len(offsets)):
        raise RuntimeError("unexpected path-pair curve shape")
    return offsets, curves, labels, summary


def pair_metadata(labels: list[str], lag: int) -> list[dict]:
    rows = []
    for query_index, query_label in enumerate(labels):
        for key_index, key_label in enumerate(labels):
            query_delay = DEFAULT_PATH_DELAYS[query_label]
            key_delay = DEFAULT_PATH_DELAYS[key_label]
            relative_delay = key_delay - query_delay
            rows.append(
                {
                    "query_index": query_index,
                    "key_index": key_index,
                    "query_path": query_label,
                    "key_path": key_label,
                    "query_delay": query_delay,
                    "key_delay": key_delay,
                    "relative_delay": relative_delay,
                    "delay_basis_predicted_offset": lag - relative_delay,
                }
            )
    return rows


def plot_pair_atlas(
    offsets: np.ndarray,
    curves: np.ndarray,
    metadata: list[dict],
    lag: int,
    output_path: Path,
) -> plt.Figure:
    ordered = sorted(
        metadata,
        key=lambda row: (
            row["delay_basis_predicted_offset"],
            row["query_path"],
            row["key_path"],
        ),
    )
    raw = np.stack(
        [curves[row["query_index"], row["key_index"]] for row in ordered]
    )
    centered = raw - raw.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    normalized = centered / np.maximum(scale, 1e-12)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(18, 16),
        constrained_layout=True,
        sharey=True,
    )
    raw_limit = float(np.quantile(np.abs(raw), 0.99))
    images = (
        axes[0].imshow(
            raw,
            cmap="RdBu_r",
            vmin=-raw_limit,
            vmax=raw_limit,
            aspect="auto",
            extent=(offsets[0] - 0.5, offsets[-1] + 0.5, len(ordered) - 0.5, -0.5),
        ),
        axes[1].imshow(
            normalized,
            cmap="RdBu_r",
            vmin=-3,
            vmax=3,
            aspect="auto",
            extent=(offsets[0] - 0.5, offsets[-1] + 0.5, len(ordered) - 0.5, -0.5),
        ),
    )
    row_labels = [
        (
            f"q{row['query_path']}×k{row['key_path']}  "
            f"({row['query_delay']}→{row['key_delay']}, "
            f"Δ={row['relative_delay']:+d})"
        )
        for row in ordered
    ]
    for axis, title in zip(
        axes,
        (
            "Mean signed raw-score contribution",
            "Each pair centered and scaled across offsets",
        ),
    ):
        axis.axvline(lag - 1, color="lime", linestyle="--", linewidth=1.5)
        axis.axvline(lag, color="black", linestyle=":", linewidth=1.5)
        for row_index, row in enumerate(ordered):
            predicted = row["delay_basis_predicted_offset"]
            if offsets[0] <= predicted <= offsets[-1]:
                axis.scatter(
                    predicted,
                    row_index,
                    marker="|",
                    color="yellow",
                    s=32,
                    linewidths=1,
                )
        axis.set_xlabel("candidate query−key offset")
        axis.set_title(title)
    axes[0].set_yticks(range(len(ordered)), row_labels, fontsize=6)
    figure.colorbar(images[0], ax=axes[0], label="raw contribution", shrink=0.65)
    figure.colorbar(images[1], ax=axes[1], label="within-pair standardized contribution", shrink=0.65)
    figure.suptitle(
        f"Lag {lag}: offset-resolved atlas of all 64 path-pair contributions\n"
        "green = required lag−1; black = raw lag; yellow ticks = delay-basis prediction"
    )
    figure.savefig(output_path, dpi=180)
    return figure


def plot_selected_offset_matrices(
    offsets: np.ndarray,
    curves: np.ndarray,
    labels: list[str],
    lag: int,
    selected_offsets: tuple[int, ...],
    output_path: Path,
) -> plt.Figure:
    valid_offsets = tuple(
        offset for offset in selected_offsets if offset in set(offsets.tolist())
    )
    matrices = [curves[..., int(np.where(offsets == offset)[0][0])] for offset in valid_offsets]
    difference = (
        curves[..., int(np.where(offsets == lag - 1)[0][0])]
        - curves[..., int(np.where(offsets == lag)[0][0])]
    )
    plot_matrices = matrices + [difference]
    titles = [f"candidate offset {offset}" for offset in valid_offsets]
    titles.append(f"offset {lag - 1} minus offset {lag}")

    columns = 3
    rows = int(np.ceil(len(plot_matrices) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(13.5, 4.2 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, matrix, title in zip(axes.flat, plot_matrices, titles):
        limit = float(np.max(np.abs(matrix)))
        image = axis.imshow(
            matrix,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            aspect="auto",
        )
        axis.set_xticks(range(len(labels)), labels)
        axis.set_yticks(range(len(labels)), labels)
        axis.set_xlabel("key path")
        axis.set_ylabel("query path")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=0.75)
    for axis in axes.flat[len(plot_matrices):]:
        axis.axis("off")
    figure.suptitle(f"Lag {lag}: path-pair score matrix at selected offsets")
    figure.savefig(output_path, dpi=180)
    return figure


def plot_relative_delay_groups(
    offsets: np.ndarray,
    curves: np.ndarray,
    metadata: list[dict],
    lag: int,
    output_path: Path,
) -> tuple[plt.Figure, dict[int, np.ndarray]]:
    relative_delays = sorted({row["relative_delay"] for row in metadata})
    grouped: dict[int, np.ndarray] = {}
    for relative_delay in relative_delays:
        members = [
            curves[row["query_index"], row["key_index"]]
            for row in metadata
            if row["relative_delay"] == relative_delay
        ]
        grouped[relative_delay] = np.stack(members).sum(axis=0)
    matrix = np.stack([grouped[relative_delay] for relative_delay in relative_delays])
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    limit = float(np.quantile(np.abs(centered), 0.99))

    figure, axes = plt.subplots(2, 1, figsize=(15, 10), constrained_layout=True)
    image = axes[0].imshow(
        centered,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
        extent=(offsets[0] - 0.5, offsets[-1] + 0.5, len(relative_delays) - 0.5, -0.5),
    )
    axes[0].set_yticks(range(len(relative_delays)), relative_delays)
    axes[0].set_ylabel("key delay − query delay")
    axes[0].set_xlabel("candidate query−key offset")
    axes[0].axvline(lag - 1, color="lime", linestyle="--")
    axes[0].axvline(lag, color="black", linestyle=":")
    for row_index, relative_delay in enumerate(relative_delays):
        predicted = lag - relative_delay
        if offsets[0] <= predicted <= offsets[-1]:
            axes[0].scatter(predicted, row_index, marker="|", color="yellow", s=45)
    axes[0].set_title("Contribution grouped by relative transported delay")
    figure.colorbar(image, ax=axes[0], label="centered grouped contribution")

    total_score = curves.sum(axis=(0, 1))
    axes[1].plot(offsets, total_score, color="black", linewidth=2, label="all 64 pairs")
    for relative_delay in relative_delays:
        if abs(lag - relative_delay - (lag - 1)) <= 3:
            axes[1].plot(
                offsets,
                grouped[relative_delay],
                label=f"relative delay {relative_delay:+d}",
                alpha=0.8,
            )
    axes[1].axvline(lag - 1, color="lime", linestyle="--", label="required lag−1")
    axes[1].axvline(lag, color="gray", linestyle=":", label="raw lag")
    axes[1].set_xlabel("candidate query−key offset")
    axes[1].set_ylabel("summed raw-score contribution")
    axes[1].set_title("Total score and groups relevant near the correct offset")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8, ncol=3)
    figure.suptitle(f"Lag {lag}: does relative path delay organize the score?")
    figure.savefig(output_path, dpi=180)
    return figure, grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/tmp/four_layer_multilag_path_pair_curves"),
    )
    parser.add_argument("--lag", type=int, default=40)
    parser.add_argument(
        "--selected-offsets",
        type=parse_int_tuple,
        default=(31, 35, 37, 39, 40, 41, 43, 47),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/four_layer_offset_path_pair_atlas"),
    )
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    offsets, curves, labels, source_summary = load_curves(
        args.input_dir,
        args.lag,
    )
    metadata = pair_metadata(labels, args.lag)
    atlas_figure = plot_pair_atlas(
        offsets,
        curves,
        metadata,
        args.lag,
        args.output_dir / f"lag_{args.lag}_path_pair_atlas.png",
    )
    matrix_figure = plot_selected_offset_matrices(
        offsets,
        curves,
        labels,
        args.lag,
        args.selected_offsets,
        args.output_dir / f"lag_{args.lag}_selected_offset_matrices.png",
    )
    grouped_figure, grouped = plot_relative_delay_groups(
        offsets,
        curves,
        metadata,
        args.lag,
        args.output_dir / f"lag_{args.lag}_relative_delay_groups.png",
    )

    ranking_rows = []
    for offset in args.selected_offsets:
        if offset not in set(offsets.tolist()):
            continue
        offset_index = int(np.where(offsets == offset)[0][0])
        for row in metadata:
            contribution = float(
                curves[row["query_index"], row["key_index"], offset_index]
            )
            ranking_rows.append(
                {
                    "lag": args.lag,
                    "candidate_offset": offset,
                    "query_path": row["query_path"],
                    "key_path": row["key_path"],
                    "query_delay": row["query_delay"],
                    "key_delay": row["key_delay"],
                    "relative_delay": row["relative_delay"],
                    "effective_input_separation": offset + row["relative_delay"],
                    "matches_true_lag_by_delay_model": (
                        offset + row["relative_delay"] == args.lag
                    ),
                    "mean_raw_score_contribution": contribution,
                    "absolute_contribution": abs(contribution),
                }
            )
    save_csv(args.output_dir / "selected_offset_pair_contributions.csv", ranking_rows)
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(
            {
                "lag": args.lag,
                "source_n_sequences": source_summary["n_sequences_per_lag"],
                "path_delays": DEFAULT_PATH_DELAYS,
                "delay_model": (
                    "effective input separation = candidate offset + "
                    "key delay - query delay"
                ),
                "selected_offsets": list(args.selected_offsets),
                "relative_delay_groups": sorted(grouped),
                "total_score_peak_offset": int(
                    offsets[curves.sum(axis=(0, 1)).argmax()]
                ),
            },
            handle,
            indent=2,
        )

    if args.no_show:
        plt.close(atlas_figure)
        plt.close(matrix_figure)
        plt.close(grouped_figure)
    else:
        plt.show()

    print(f"lag {args.lag}: total score peaks at offset "
          f"{offsets[curves.sum(axis=(0, 1)).argmax()]}")
    for offset in (args.lag - 1, args.lag):
        rows = [row for row in ranking_rows if row["candidate_offset"] == offset]
        print(f"\nOffset {offset}: largest absolute contributions")
        for row in sorted(rows, key=lambda item: item["absolute_contribution"], reverse=True)[:10]:
            print(
                f"  q{row['query_path']}×k{row['key_path']} "
                f"({row['query_delay']}→{row['key_delay']}, "
                f"effective separation={row['effective_input_separation']}): "
                f"{row['mean_raw_score_contribution']:+.3f}"
            )
    print(f"\nSaved results to {args.output_dir}")


if __name__ == "__main__":
    main()
