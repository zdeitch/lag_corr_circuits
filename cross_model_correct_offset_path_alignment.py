"""Compare correct-offset path-pair matrices across lags and model depths.

For each attention-only model and lag, freeze the observed attention matrices,
expand the residual entering the architectural final head into all earlier-layer
identity/write paths, and average the exact path-pair contributions to the raw
QK score at the correct offset ``lag - 1``.  The experiment then measures how
well the complete path-pair matrix aligns between two programmed data lags.

The 5L checkpoint is intentionally excluded by default because its architectural
final head is not the lag-locating head.

Usage:
    python cross_model_correct_offset_path_alignment.py --quick --no-show
    python cross_model_correct_offset_path_alignment.py --no-show
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
from scipy.stats import spearmanr

from cross_model_raw_head1_path_test import (
    MODEL_SPECS,
    load_attention_only_model,
    parse_int_tuple,
)
from four_layer_attention_path_analysis import expand_prefinal_paths
from util import make_dataset_lagset


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def final_peak_rate(
    attention: torch.Tensor,
    query_positions: torch.Tensor,
    correct_offset: int,
    maximum_offset: int,
) -> tuple[int, int]:
    offsets = torch.arange(maximum_offset + 1, device=attention.device)
    rows = attention[
        :,
        query_positions[:, None],
        query_positions[:, None] - offsets[None, :],
    ]
    peaks = rows.mean(dim=1)[:, 1:].argmax(dim=1) + 1
    return int((peaks == correct_offset).sum().item()), len(peaks)


@torch.inference_mode()
def matrix_for_model_lag(
    model,
    lag: int,
    n_sequences: int,
    batch_size: int,
    sequence_length: int,
    rho: float,
    query_start: int,
    query_stride: int,
    maximum_offset: int,
    device: torch.device,
) -> tuple[list[str], dict[str, np.ndarray | float]]:
    inputs, _, sampled_lags = make_dataset_lagset(
        n_sequences,
        sequence_length,
        rho,
        [lag],
        seed=600_000 + 1_000 * model.n_layers + lag,
    )
    if not torch.all(sampled_lags == lag):
        raise RuntimeError("dataset returned an unexpected lag")

    correct_offset = lag - 1
    query_positions = torch.arange(
        query_start,
        sequence_length,
        query_stride,
        device=device,
    )
    labels: list[str] | None = None
    correct_sum: torch.Tensor | None = None
    plus_one_sum: torch.Tensor | None = None
    observation_count = 0
    correct_peak_count = 0
    peak_count = 0
    maximum_residual_error = 0.0
    maximum_score_error = 0.0

    for start in range(0, n_sequences, batch_size):
        batch = inputs[start : start + batch_size].to(
            device=device,
            dtype=torch.float64,
        )
        _, attentions, post_attention, post_mlp = model(batch)
        if any(not torch.equal(a, b) for a, b in zip(post_attention, post_mlp)):
            raise RuntimeError("checkpoint is not behaving as attention-only")

        clean_correct, clean_count = final_peak_rate(
            attentions[-1],
            query_positions,
            correct_offset,
            maximum_offset,
        )
        correct_peak_count += clean_correct
        peak_count += clean_count

        embedding = model.W_r(batch.unsqueeze(-1))
        path_dict = expand_prefinal_paths(
            embedding,
            attentions,
            model,
            post_attention[: model.n_layers - 1],
        )
        current_labels = list(path_dict)
        if labels is None:
            labels = current_labels
            path_count = len(labels)
            correct_sum = torch.zeros(
                path_count, path_count, dtype=torch.float64, device=device
            )
            plus_one_sum = torch.zeros_like(correct_sum)
        elif labels != current_labels:
            raise RuntimeError("path labels changed between batches")

        path_components = torch.stack(list(path_dict.values()))
        final_input = post_attention[-2]
        residual_error = (
            (path_components.sum(dim=0) - final_input).norm()
            / final_input.norm().clamp_min(1e-30)
        )
        maximum_residual_error = max(
            maximum_residual_error, float(residual_error.item())
        )

        query_matrix, key_matrix, _, _ = model.layers[-1][0]
        positions = torch.arange(sequence_length, device=device).view(1, 1, -1)
        path_queries = model.apply_rope(
            path_components @ query_matrix,
            positions,
        )
        path_keys = model.apply_rope(
            path_components @ key_matrix,
            positions,
        )
        full_queries = model.apply_rope(
            final_input @ query_matrix,
            positions.squeeze(0),
        )
        full_keys = model.apply_rope(
            final_input @ key_matrix,
            positions.squeeze(0),
        )

        path_query = path_queries[:, :, query_positions, :]
        full_query = full_queries[:, query_positions, :]
        for offset, accumulator in (
            (correct_offset, correct_sum),
            (correct_offset + 1, plus_one_sum),
        ):
            assert accumulator is not None
            path_key = path_keys[:, :, query_positions - offset, :]
            pair_scores = torch.einsum(
                "pbnh,qbnh->bnpq",
                path_query,
                path_key,
            ) / math.sqrt(model.d_head)
            accumulator += pair_scores.sum(dim=(0, 1))

            direct_scores = (
                full_query * full_keys[:, query_positions - offset, :]
            ).sum(dim=-1) / math.sqrt(model.d_head)
            score_error = (
                pair_scores.sum(dim=(-1, -2)) - direct_scores
            ).abs().max()
            maximum_score_error = max(
                maximum_score_error, float(score_error.item())
            )

        observation_count += len(batch) * len(query_positions)

    assert labels is not None and correct_sum is not None and plus_one_sum is not None
    correct = correct_sum / observation_count
    plus_one = plus_one_sum / observation_count
    return labels, {
        "correct": correct.cpu().numpy(),
        "correct_minus_plus_one": (correct - plus_one).cpu().numpy(),
        "final_correct_peak_rate": correct_peak_count / peak_count,
        "maximum_residual_relative_error": maximum_residual_error,
        "maximum_raw_score_error": maximum_score_error,
    }


def overlap_fraction(first: np.ndarray, second: np.ndarray, count: int) -> float:
    count = min(count, first.size)
    first_top = set(np.argsort(np.abs(first.ravel()))[-count:])
    second_top = set(np.argsort(np.abs(second.ravel()))[-count:])
    return len(first_top & second_top) / count


def alignment_metrics(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    first_flat = first.ravel()
    second_flat = second.ravel()
    cosine = float(
        first_flat @ second_flat
        / (np.linalg.norm(first_flat) * np.linalg.norm(second_flat))
    )
    one_percent = max(8, int(round(0.01 * first.size)))
    return {
        "pearson": float(np.corrcoef(first_flat, second_flat)[0, 1]),
        "cosine": cosine,
        "spearman": float(spearmanr(first_flat, second_flat).statistic),
        "sign_agreement": float(np.mean(np.sign(first_flat) == np.sign(second_flat))),
        "top8_overlap": overlap_fraction(first, second, 8),
        "top1pct_overlap": overlap_fraction(first, second, one_percent),
    }


def plot_model_alignment(
    model_name: str,
    lags: tuple[int, int],
    labels: list[str],
    matrices: list[np.ndarray],
    correlation: float,
    output_path: Path,
) -> plt.Figure:
    limit = max(float(np.abs(matrix).max()) for matrix in matrices)
    figure, axes = plt.subplots(
        1, 2, figsize=(12, 5.4), constrained_layout=True
    )
    image = None
    for axis, lag, matrix in zip(axes, lags, matrices):
        image = axis.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit)
        axis.set_xticks(range(len(labels)))
        axis.set_yticks(range(len(labels)))
        if len(labels) <= 16:
            axis.set_xticklabels(labels, rotation=90, fontsize=7)
            axis.set_yticklabels(labels, fontsize=7)
        else:
            step = max(1, len(labels) // 8)
            visible = [label if index % step == 0 else "" for index, label in enumerate(labels)]
            axis.set_xticklabels(visible, rotation=90, fontsize=6)
            axis.set_yticklabels(visible, fontsize=6)
        axis.set_xlabel("key path")
        axis.set_ylabel("query path")
        axis.set_title(
            f"Lag {lag}, correct D={lag - 1}\nscore sum={matrix.sum():+.2f}"
        )
    if image is not None:
        figure.colorbar(
            image,
            ax=axes,
            label="mean raw-score contribution",
            shrink=0.84,
        )
    figure.suptitle(
        f"{model_name}: correct-offset path-pair alignment; Pearson r={correlation:.3f}"
    )
    figure.savefig(output_path, dpi=180)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="4L,6L,7L")
    parser.add_argument("--lags", type=parse_int_tuple, default=(30, 40))
    parser.add_argument("--n-sequences", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=200)
    parser.add_argument("--rho", type=float, default=0.9)
    parser.add_argument("--query-start", type=int, default=120)
    parser.add_argument("--query-stride", type=int, default=4)
    parser.add_argument("--maximum-offset", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment_outputs/cross_model_correct_offset_path_alignment"),
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if len(args.lags) != 2:
        raise ValueError("this alignment test requires exactly two lags")
    selected_models = tuple(part.strip() for part in args.models.split(",") if part.strip())
    if "5L" in selected_models:
        raise ValueError(
            "5L is excluded: its architectural final head is not the lag locator"
        )
    unknown = set(selected_models) - set(MODEL_SPECS)
    if unknown:
        raise ValueError(f"unknown models: {sorted(unknown)}")
    if args.quick:
        args.n_sequences = min(args.n_sequences, 8)

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    arrays: dict[str, np.ndarray] = {}
    figures: list[plt.Figure] = []
    metadata: dict[str, object] = {
        "lags": list(args.lags),
        "n_sequences_per_model_lag": args.n_sequences,
        "batch_size": args.batch_size,
        "models": {},
    }

    for model_name in selected_models:
        n_layers, checkpoint = MODEL_SPECS[model_name]
        print(f"{model_name}: loading {checkpoint}")
        model = load_attention_only_model(n_layers, checkpoint, device)
        labels: list[str] | None = None
        results = []
        for lag in args.lags:
            print(f"  lag {lag}: expanding {2 ** (n_layers - 1)} paths")
            current_labels, result = matrix_for_model_lag(
                model,
                lag,
                args.n_sequences,
                args.batch_size,
                args.sequence_length,
                args.rho,
                args.query_start,
                args.query_stride,
                args.maximum_offset,
                device,
            )
            if labels is None:
                labels = current_labels
            elif labels != current_labels:
                raise RuntimeError("path labels changed between lags")
            results.append(result)
            arrays[f"{model_name}_lag{lag}_correct"] = result["correct"]
            arrays[f"{model_name}_lag{lag}_margin"] = result[
                "correct_minus_plus_one"
            ]

        assert labels is not None
        raw_metrics = alignment_metrics(results[0]["correct"], results[1]["correct"])
        margin_metrics = alignment_metrics(
            results[0]["correct_minus_plus_one"],
            results[1]["correct_minus_plus_one"],
        )
        rows.append(
            {
                "model": model_name,
                "n_paths": len(labels),
                "n_path_pairs": len(labels) ** 2,
                "lag_a": args.lags[0],
                "lag_b": args.lags[1],
                "final_correct_rate_lag_a": results[0]["final_correct_peak_rate"],
                "final_correct_rate_lag_b": results[1]["final_correct_peak_rate"],
                **{f"raw_{key}": value for key, value in raw_metrics.items()},
                **{f"margin_{key}": value for key, value in margin_metrics.items()},
                "maximum_residual_relative_error": max(
                    result["maximum_residual_relative_error"] for result in results
                ),
                "maximum_raw_score_error": max(
                    result["maximum_raw_score_error"] for result in results
                ),
            }
        )
        figures.append(
            plot_model_alignment(
                model_name,
                args.lags,
                labels,
                [results[0]["correct"], results[1]["correct"]],
                raw_metrics["pearson"],
                args.output_dir / f"{model_name}_correct_offset_alignment.png",
            )
        )
        metadata["models"][model_name] = {
            "checkpoint": str(checkpoint),
            "n_layers": n_layers,
            "n_paths": len(labels),
            "path_labels": labels,
        }
        del model

    save_csv(args.output_dir / "alignment_summary.csv", rows)
    np.savez_compressed(args.output_dir / "correct_offset_matrices.npz", **arrays)
    with (args.output_dir / "run_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)

    for row in rows:
        print(
            f"{row['model']}: paths={row['n_paths']}, "
            f"raw r={row['raw_pearson']:.4f}, "
            f"margin r={row['margin_pearson']:.4f}, "
            f"top1% overlap={row['raw_top1pct_overlap']:.1%}, "
            f"final correct=({row['final_correct_rate_lag_a']:.1%}, "
            f"{row['final_correct_rate_lag_b']:.1%})"
        )

    if args.no_show:
        for figure in figures:
            plt.close(figure)
    else:
        plt.show()
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
