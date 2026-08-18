"""Distribute the 4L locator score across its 64 frozen-attention path pairs.

Layers 1--3 give eight residual paths on each side of the Layer-4 QK score.
The bilinear score therefore decomposes exactly into 8 x 8 = 64 query-path /
key-path contributions.  This script measures their distributions for the
clean model and after programming Layer 1 to selected absolute offsets.

The primary histogram uses one value per path pair: that pair's signed raw
pre-softmax score contribution, averaged over query positions and sequences.

Usage:
    python four_layer_programmed_path_pair_distribution.py --quick --no-show
    python four_layer_programmed_path_pair_distribution.py --no-show
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from four_layer_attention_path_analysis import expand_prefinal_paths, load_model
from four_layer_earlier_attention_shift_patch import forward_with_attention_patch
from full_earlier_offset_programming import fixed_offset_attention, parse_offset_spec
from util import make_dataset_lagset


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def condition_label(programmed_offset: int | None) -> str:
    if programmed_offset is None:
        return "clean"
    return f"program_L1_D{programmed_offset}"


def final_attention_profiles(
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


@torch.inference_mode()
def analyze_condition(
    model,
    inputs: torch.Tensor,
    programmed_offset: int | None,
    lag: int,
    query_positions: torch.Tensor,
    maximum_offset: int,
) -> dict:
    batch_size, sequence_length = inputs.shape
    if programmed_offset is None:
        _, attentions, post_attention, post_mlp = model(inputs)
        if any(not torch.equal(a, b) for a, b in zip(post_attention, post_mlp)):
            raise RuntimeError("checkpoint is not behaving as attention-only")
    else:
        offsets = torch.full(
            (batch_size,),
            programmed_offset,
            dtype=torch.long,
            device=inputs.device,
        )
        stripe = fixed_offset_attention(
            batch_size,
            sequence_length,
            offsets,
            inputs.dtype,
            inputs.device,
        )
        _, attentions, post_attention, _ = forward_with_attention_patch(
            model,
            inputs,
            patch_layer=0,
            patched_attention=stripe,
        )

    embedding = model.W_r(inputs.unsqueeze(-1))
    path_dict = expand_prefinal_paths(
        embedding,
        attentions,
        model,
        post_attention[:3],
    )
    labels = list(path_dict)
    paths = torch.stack(list(path_dict.values()))
    locator_input = post_attention[2]
    residual_error = (
        (paths.sum(dim=0) - locator_input).norm()
        / locator_input.norm().clamp_min(1e-30)
    )

    query_matrix, key_matrix, _, _ = model.layers[3][0]
    positions = torch.arange(sequence_length, device=inputs.device).view(1, 1, -1)
    path_queries = model.apply_rope(paths @ query_matrix, positions)
    path_keys = model.apply_rope(paths @ key_matrix, positions)
    full_queries = model.apply_rope(
        locator_input @ query_matrix, positions.squeeze(0)
    )
    full_keys = model.apply_rope(
        locator_input @ key_matrix, positions.squeeze(0)
    )

    profiles = final_attention_profiles(
        attentions[3], query_positions, maximum_offset
    )
    candidate_profiles = profiles[:, 1:]
    sequence_peaks = candidate_profiles.argmax(dim=1) + 1
    modal_peak = Counter(sequence_peaks.cpu().tolist()).most_common(1)[0][0]

    sequence_matrices: dict[int, torch.Tensor] = {}
    mean_matrices: dict[int, torch.Tensor] = {}
    maximum_score_error = 0.0
    full_query_at_positions = full_queries[:, query_positions, :]
    path_query_at_positions = path_queries[:, :, query_positions, :]

    for offset in range(1, maximum_offset + 1):
        path_key_at_offset = path_keys[:, :, query_positions - offset, :]
        # batch x query-position x query-path x key-path
        pair_scores = torch.einsum(
            "pbnd,qbnd->bnpq",
            path_query_at_positions,
            path_key_at_offset,
        ) / math.sqrt(model.d_head)
        per_sequence = pair_scores.mean(dim=1)
        sequence_matrices[offset] = per_sequence
        mean_matrices[offset] = per_sequence.mean(dim=0)

        direct_scores = (
            full_query_at_positions
            * full_keys[:, query_positions - offset, :]
        ).sum(dim=-1) / math.sqrt(model.d_head)
        score_error = (
            pair_scores.sum(dim=(-1, -2)) - direct_scores
        ).abs().max()
        maximum_score_error = max(maximum_score_error, float(score_error.item()))

    correct_offset = lag - 1
    lag_offset = lag
    return {
        "condition": condition_label(programmed_offset),
        "programmed_offset": programmed_offset,
        "labels": labels,
        "modal_peak": int(modal_peak),
        "sequence_peaks": sequence_peaks.cpu().numpy(),
        "correct_rate": float(
            (sequence_peaks == correct_offset).double().mean().item()
        ),
        "mean_matrices": {
            offset: matrix.cpu().numpy() for offset, matrix in mean_matrices.items()
        },
        "sequence_matrices": {
            offset: matrix.cpu().numpy()
            for offset, matrix in sequence_matrices.items()
            if offset in {correct_offset, lag_offset, modal_peak}
        },
        "residual_reconstruction_error": float(residual_error.item()),
        "maximum_score_reconstruction_error": maximum_score_error,
    }


def shared_histogram_bins(arrays: list[np.ndarray], n_bins: int) -> np.ndarray:
    values = np.concatenate([array.reshape(-1) for array in arrays])
    low, high = float(values.min()), float(values.max())
    if math.isclose(low, high):
        padding = max(abs(low) * 0.05, 1e-6)
        low, high = low - padding, high + padding
    return np.linspace(low, high, n_bins + 1)


def annotate_histogram(axis: plt.Axes, values: np.ndarray) -> None:
    positive = int(np.sum(values > 0))
    negative = int(np.sum(values < 0))
    axis.text(
        0.98,
        0.95,
        f"sum={values.sum():+.2f}\npositive={positive}, negative={negative}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )
    axis.axvline(0, color="black", linewidth=1)
    axis.grid(axis="y", alpha=0.25)


def plot_clean_histograms(
    result: dict,
    lag: int,
    n_bins: int,
    output_path: Path,
) -> plt.Figure:
    correct = result["mean_matrices"][lag - 1]
    correct_minus_lag = correct - result["mean_matrices"][lag]
    arrays = [correct, correct_minus_lag]
    titles = [
        f"Raw score at correct offset {lag - 1}",
        f"Contribution to score({lag - 1}) − score({lag})",
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, values, title in zip(axes, arrays, titles):
        bins = shared_histogram_bins([values], n_bins)
        axis.hist(values.reshape(-1), bins=bins, color="#4C78A8", edgecolor="white")
        annotate_histogram(axis, values)
        axis.set_title(title)
        axis.set_xlabel("mean signed path-pair contribution to raw QK score")
        axis.set_ylabel("number of the 64 path pairs")
    figure.suptitle(
        "4L clean model, lag 40: distribution of the 8×8 path-pair contributions"
    )
    figure.savefig(output_path, dpi=180)
    return figure


def plot_programmed_histograms(
    results: list[dict],
    n_bins: int,
    output_path: Path,
) -> plt.Figure:
    arrays = [result["mean_matrices"][result["modal_peak"]] for result in results]
    bins = shared_histogram_bins(arrays, n_bins)
    n_columns = 3
    n_rows = math.ceil(len(results) / n_columns)
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(5.2 * n_columns, 4.0 * n_rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    flat_axes = np.atleast_1d(axes).reshape(-1)
    for axis, result, values in zip(flat_axes, results, arrays):
        axis.hist(values.reshape(-1), bins=bins, color="#4C78A8", edgecolor="white")
        annotate_histogram(axis, values)
        title = "clean" if result["programmed_offset"] is None else (
            f"program Layer 1 at D={result['programmed_offset']}"
        )
        axis.set_title(
            f"{title}\nLayer-4 modal peak D={result['modal_peak']}; "
            f"correct rate={result['correct_rate']:.0%}"
        )
        axis.set_xlabel("mean contribution at that modal offset")
        axis.set_ylabel("number of path pairs")
    for axis in flat_axes[len(results) :]:
        axis.remove()
    figure.suptitle(
        "How the 64 path-pair score contributions redistribute when Layer 1 is programmed"
    )
    figure.savefig(output_path, dpi=180)
    return figure


def plot_programmed_heatmaps(
    results: list[dict],
    output_path: Path,
) -> plt.Figure:
    arrays = [result["mean_matrices"][result["modal_peak"]] for result in results]
    limit = max(float(np.max(np.abs(array))) for array in arrays)
    n_columns = 3
    n_rows = math.ceil(len(results) / n_columns)
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(5.0 * n_columns, 4.3 * n_rows),
        constrained_layout=True,
    )
    flat_axes = np.atleast_1d(axes).reshape(-1)
    image = None
    for axis, result, matrix in zip(flat_axes, results, arrays):
        image = axis.imshow(
            matrix,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            aspect="equal",
        )
        labels = result["labels"]
        axis.set_xticks(range(8), labels)
        axis.set_yticks(range(8), labels)
        axis.set_xlabel("key path")
        axis.set_ylabel("query path")
        title = "clean" if result["programmed_offset"] is None else (
            f"program L1 D={result['programmed_offset']}"
        )
        axis.set_title(f"{title}: contributions at final D={result['modal_peak']}")
    for axis in flat_axes[len(results) :]:
        axis.remove()
    if image is not None:
        figure.colorbar(
            image,
            ax=flat_axes[: len(results)],
            label="mean signed raw-score contribution",
            shrink=0.85,
        )
    figure.suptitle("Identity of the 64 path pairs behind each histogram")
    figure.savefig(output_path, dpi=180)
    return figure


def plot_programmed_margin_heatmaps(
    results: list[dict],
    lag: int,
    output_path: Path,
) -> plt.Figure:
    arrays = [
        result["mean_matrices"][lag - 1] - result["mean_matrices"][lag]
        for result in results
    ]
    limit = max(float(np.max(np.abs(array))) for array in arrays)
    n_columns = 3
    n_rows = math.ceil(len(results) / n_columns)
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(5.0 * n_columns, 4.3 * n_rows),
        constrained_layout=True,
    )
    flat_axes = np.atleast_1d(axes).reshape(-1)
    image = None
    for axis, result, matrix in zip(flat_axes, results, arrays):
        image = axis.imshow(
            matrix,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            aspect="equal",
        )
        labels = result["labels"]
        axis.set_xticks(range(8), labels)
        axis.set_yticks(range(8), labels)
        axis.set_xlabel("key path")
        axis.set_ylabel("query path")
        title = "clean" if result["programmed_offset"] is None else (
            f"program L1 D={result['programmed_offset']}"
        )
        axis.set_title(
            f"{title}: total margin={matrix.sum():+.2f}"
        )
    for axis in flat_axes[len(results) :]:
        axis.remove()
    if image is not None:
        figure.colorbar(
            image,
            ax=flat_axes[: len(results)],
            label=f"contribution to score({lag - 1}) − score({lag})",
            shrink=0.85,
        )
    figure.suptitle(
        "Which path pairs make Layer 4 prefer the prediction-correct offset 39 over raw lag 40"
    )
    figure.savefig(output_path, dpi=180)
    return figure


def result_rows(results: list[dict], lag: int) -> tuple[list[dict], list[dict]]:
    pair_rows: list[dict] = []
    curve_rows: list[dict] = []
    for result in results:
        labels = result["labels"]
        correct = result["mean_matrices"][lag - 1]
        lag_matrix = result["mean_matrices"][lag]
        modal = result["mean_matrices"][result["modal_peak"]]
        correct_sequence = result["sequence_matrices"][lag - 1]
        modal_sequence = result["sequence_matrices"][result["modal_peak"]]
        for query_index, query_path in enumerate(labels):
            for key_index, key_path in enumerate(labels):
                pair_rows.append(
                    {
                        "condition": result["condition"],
                        "programmed_layer1_offset": result["programmed_offset"],
                        "final_modal_peak": result["modal_peak"],
                        "final_correct_rate": result["correct_rate"],
                        "query_path": query_path,
                        "key_path": key_path,
                        "mean_contribution_at_correct_D39": float(
                            correct[query_index, key_index]
                        ),
                        "sd_sequence_contribution_at_correct_D39": float(
                            correct_sequence[:, query_index, key_index].std(ddof=0)
                        ),
                        "mean_contribution_at_raw_lag_D40": float(
                            lag_matrix[query_index, key_index]
                        ),
                        "correct_D39_minus_D40_contribution": float(
                            correct[query_index, key_index]
                            - lag_matrix[query_index, key_index]
                        ),
                        "mean_contribution_at_condition_modal_peak": float(
                            modal[query_index, key_index]
                        ),
                        "sd_sequence_contribution_at_condition_modal_peak": float(
                            modal_sequence[:, query_index, key_index].std(ddof=0)
                        ),
                    }
                )
                for offset, matrix in result["mean_matrices"].items():
                    curve_rows.append(
                        {
                            "condition": result["condition"],
                            "programmed_layer1_offset": result[
                                "programmed_offset"
                            ],
                            "offset": offset,
                            "query_path": query_path,
                            "key_path": key_path,
                            "mean_raw_score_contribution": float(
                                matrix[query_index, key_index]
                            ),
                        }
                    )
    return pair_rows, curve_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/attn_d64_4L_int_ext.pt"),
    )
    parser.add_argument("--lag", type=int, default=40)
    parser.add_argument(
        "--programmed-offsets",
        type=parse_offset_spec,
        default=(1, 10, 21, 41),
    )
    parser.add_argument("--n-sequences", type=int, default=96)
    parser.add_argument("--sequence-length", type=int, default=200)
    parser.add_argument("--rho", type=float, default=0.9)
    parser.add_argument("--query-start", type=int, default=120)
    parser.add_argument("--query-stride", type=int, default=4)
    parser.add_argument("--maximum-offset", type=int, default=60)
    parser.add_argument("--histogram-bins", type=int, default=18)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "experiment_outputs/four_layer_programmed_path_pair_distribution"
        ),
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if args.query_start < args.maximum_offset:
        raise ValueError("query_start must be at least maximum_offset")
    if args.lag > args.maximum_offset:
        raise ValueError("maximum_offset must include lag and lag - 1")
    if args.quick:
        args.n_sequences = min(args.n_sequences, 8)

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    inputs, _, sampled_lags = make_dataset_lagset(
        args.n_sequences,
        args.sequence_length,
        args.rho,
        [args.lag],
        seed=2_400_000 + args.lag,
    )
    if not torch.all(sampled_lags == args.lag):
        raise RuntimeError("dataset returned an unexpected lag")
    inputs = inputs.to(device=device, dtype=torch.float64)
    query_positions = torch.arange(
        args.query_start,
        args.sequence_length,
        args.query_stride,
        device=device,
    )

    conditions: tuple[int | None, ...] = (None,) + args.programmed_offsets
    results = []
    for programmed_offset in conditions:
        print(f"analyzing {condition_label(programmed_offset)}")
        results.append(
            analyze_condition(
                model,
                inputs,
                programmed_offset,
                args.lag,
                query_positions,
                args.maximum_offset,
            )
        )

    for result in results:
        if result["residual_reconstruction_error"] > 1e-10:
            raise RuntimeError("residual path reconstruction failed")
        if result["maximum_score_reconstruction_error"] > 1e-9:
            raise RuntimeError("path-pair score reconstruction failed")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_rows, curve_rows = result_rows(results, args.lag)
    save_csv(args.output_dir / "path_pair_summary.csv", pair_rows)
    save_csv(args.output_dir / "path_pair_offset_curves.csv", curve_rows)

    figures = [
        plot_clean_histograms(
            results[0],
            args.lag,
            args.histogram_bins,
            args.output_dir / "clean_path_pair_histograms.png",
        ),
        plot_programmed_histograms(
            results,
            args.histogram_bins,
            args.output_dir / "programmed_path_pair_histograms.png",
        ),
        plot_programmed_heatmaps(
            results,
            args.output_dir / "programmed_path_pair_heatmaps.png",
        ),
        plot_programmed_margin_heatmaps(
            results,
            args.lag,
            args.output_dir / "programmed_margin_heatmaps.png",
        ),
    ]

    metadata = {
        "checkpoint": str(args.checkpoint),
        "lag": args.lag,
        "correct_offset": args.lag - 1,
        "programmed_layer1_offsets": list(args.programmed_offsets),
        "n_sequences": args.n_sequences,
        "query_positions_per_sequence": len(query_positions),
        "path_labels": results[0]["labels"],
        "path_bit_convention": (
            "bits are Layer1, Layer2, Layer3; 0 means residual/identity branch "
            "and 1 means that layer's frozen attention-OV write branch"
        ),
        "histogram_observation": (
            "one mean signed contribution per query-path/key-path pair; "
            "8 x 8 = 64 observations"
        ),
        "conditions": [
            {
                "condition": result["condition"],
                "programmed_offset": result["programmed_offset"],
                "final_modal_peak": result["modal_peak"],
                "final_correct_rate": result["correct_rate"],
                "residual_reconstruction_error": result[
                    "residual_reconstruction_error"
                ],
                "maximum_score_reconstruction_error": result[
                    "maximum_score_reconstruction_error"
                ],
            }
            for result in results
        ],
    }
    with (args.output_dir / "run_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)

    clean_pairs = [row for row in pair_rows if row["condition"] == "clean"]
    print("\nLargest clean contributions at D=39:")
    for row in sorted(
        clean_pairs,
        key=lambda item: abs(item["mean_contribution_at_correct_D39"]),
        reverse=True,
    )[:10]:
        print(
            f"  q{row['query_path']} x k{row['key_path']}: "
            f"{row['mean_contribution_at_correct_D39']:+.4f}"
        )
    print("\nCondition peaks:")
    for result in results:
        print(
            f"  {result['condition']}: modal D={result['modal_peak']}, "
            f"correct-rate={result['correct_rate']:.1%}"
        )

    if args.no_show:
        for figure in figures:
            plt.close(figure)
    else:
        plt.show()
    print(f"\nSaved results to {args.output_dir}")


if __name__ == "__main__":
    main()
