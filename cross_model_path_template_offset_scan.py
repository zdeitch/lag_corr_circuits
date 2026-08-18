"""Scan a correct-offset path-pair fingerprint across all target offsets.

For each usable attention-only model, this script computes the exact frozen-
attention path-pair contribution matrix at every offset for two data lags.  The
reference is lag A's matrix at its correct offset.  We compare it against lag
B's matrix at every candidate offset, both as a raw contribution matrix and as
a local offset margin ``matrix(D) - matrix(D + 1)``.

The 5L checkpoint is excluded because its architectural final head is not the
lag-locating head.
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


def correlations(reference: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Pearson correlation of one path matrix with every candidate matrix."""
    first = reference.reshape(-1)
    second = candidates.reshape(candidates.shape[0], -1)
    first = first - first.mean()
    second = second - second.mean(axis=1, keepdims=True)
    numerator = second @ first
    denominator = np.linalg.norm(second, axis=1) * np.linalg.norm(first)
    return numerator / denominator


def final_correct_count(
    attention: torch.Tensor,
    query_positions: torch.Tensor,
    correct_offset: int,
    maximum_offset: int,
) -> tuple[int, int]:
    offsets = torch.arange(maximum_offset + 1, device=attention.device)
    values = attention[
        :,
        query_positions[:, None],
        query_positions[:, None] - offsets[None, :],
    ]
    peaks = values.mean(dim=1)[:, 1:].argmax(dim=1) + 1
    return int((peaks == correct_offset).sum().item()), len(peaks)


@torch.inference_mode()
def contribution_curve(
    model,
    lag: int,
    n_sequences: int,
    batch_size: int,
    sequence_length: int,
    rho: float,
    query_start: int,
    query_stride: int,
    maximum_offset: int,
    offset_chunk_size: int,
    device: torch.device,
) -> tuple[list[str], np.ndarray, dict[str, float]]:
    inputs, _, sampled_lags = make_dataset_lagset(
        n_sequences,
        sequence_length,
        rho,
        [lag],
        seed=700_000 + 1_000 * model.n_layers + lag,
    )
    if not torch.all(sampled_lags == lag):
        raise RuntimeError("dataset returned an unexpected lag")

    query_positions = torch.arange(
        query_start,
        sequence_length,
        query_stride,
        device=device,
    )
    offsets = torch.arange(1, maximum_offset + 1, device=device)
    labels: list[str] | None = None
    score_sum: torch.Tensor | None = None
    observation_count = 0
    final_correct = 0
    final_count = 0
    maximum_residual_error = 0.0
    maximum_aggregate_score_error = 0.0

    for batch_start in range(0, n_sequences, batch_size):
        batch = inputs[batch_start : batch_start + batch_size].to(
            device=device,
            dtype=torch.float64,
        )
        _, attentions, post_attention, post_mlp = model(batch)
        if any(not torch.equal(a, b) for a, b in zip(post_attention, post_mlp)):
            raise RuntimeError("checkpoint is not behaving as attention-only")

        correct, count = final_correct_count(
            attentions[-1],
            query_positions,
            lag - 1,
            maximum_offset,
        )
        final_correct += correct
        final_count += count

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
            score_sum = torch.zeros(
                maximum_offset,
                path_count,
                path_count,
                dtype=torch.float64,
                device=device,
            )
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
        path_queries = model.apply_rope(path_components @ query_matrix, positions)
        path_keys = model.apply_rope(path_components @ key_matrix, positions)
        full_queries = model.apply_rope(
            final_input @ query_matrix,
            positions.squeeze(0),
        )
        full_keys = model.apply_rope(
            final_input @ key_matrix,
            positions.squeeze(0),
        )
        query_paths = path_queries[:, :, query_positions, :]
        full_query = full_queries[:, query_positions, :]

        assert score_sum is not None
        for offset_start in range(0, maximum_offset, offset_chunk_size):
            offset_stop = min(maximum_offset, offset_start + offset_chunk_size)
            chunk_offsets = offsets[offset_start:offset_stop]
            key_paths = path_keys[
                :,
                :,
                query_positions[:, None] - chunk_offsets[None, :],
                :,
            ]
            chunk_scores = torch.einsum(
                "pbnh,qbndh->dpq",
                query_paths,
                key_paths,
            ) / math.sqrt(model.d_head)
            score_sum[offset_start:offset_stop] += chunk_scores

            full_key = full_keys[
                :,
                query_positions[:, None] - chunk_offsets[None, :],
                :,
            ]
            direct_sum = (
                full_query[:, :, None, :] * full_key
            ).sum(dim=-1).sum(dim=(0, 1)) / math.sqrt(model.d_head)
            aggregate_error = (
                chunk_scores.sum(dim=(-1, -2)) - direct_sum
            ).abs().max()
            maximum_aggregate_score_error = max(
                maximum_aggregate_score_error,
                float(aggregate_error.item()),
            )

        observation_count += len(batch) * len(query_positions)

    assert labels is not None and score_sum is not None
    return labels, (score_sum / observation_count).cpu().numpy(), {
        "final_correct_rate": final_correct / final_count,
        "maximum_residual_relative_error": maximum_residual_error,
        "maximum_aggregate_score_error": maximum_aggregate_score_error,
    }


def best_outside_window(
    offsets: np.ndarray,
    values: np.ndarray,
    center: int,
    half_width: int = 3,
) -> tuple[int, float]:
    allowed = np.abs(offsets - center) > half_width
    allowed_indices = np.where(allowed)[0]
    best_index = allowed_indices[np.nanargmax(values[allowed])]
    return int(offsets[best_index]), float(values[best_index])


def plot_scan(
    model_name: str,
    reference_lag: int,
    target_lag: int,
    offsets: np.ndarray,
    raw_similarity: np.ndarray,
    margin_similarity: np.ndarray,
    output_path: Path,
) -> plt.Figure:
    correct_offset = target_lag - 1
    margin_offsets = offsets[:-1]
    figure, axes = plt.subplots(
        2, 1, figsize=(11, 7.5), sharex=True, constrained_layout=True
    )
    for axis, current_offsets, values, title in (
        (axes[0], offsets, raw_similarity, "Raw path-pair contribution matrix"),
        (
            axes[1],
            margin_offsets,
            margin_similarity,
            "Local margin matrix: contribution(D) − contribution(D+1)",
        ),
    ):
        peak_index = int(np.nanargmax(values))
        axis.plot(current_offsets, values, linewidth=1.6)
        axis.axvline(
            correct_offset,
            color="red",
            linestyle="--",
            label="correct offset",
        )
        axis.scatter(
            [current_offsets[peak_index]],
            [values[peak_index]],
            color="black",
            zorder=3,
            label=f"highest similarity: D={current_offsets[peak_index]}",
        )
        axis.set_ylabel("cellwise Pearson correlation")
        axis.set_title(title)
        axis.grid(alpha=0.2)
        axis.legend()
    axes[1].set_xlabel("candidate offset in target-lag sequences")
    axes[1].set_ylim(-1.0, 1.05)
    figure.suptitle(
        f"{model_name}: lag {reference_lag} correct-offset fingerprint scanned "
        f"across lag {target_lag} offsets"
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
    parser.add_argument("--offset-chunk-size", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment_outputs/cross_model_path_template_offset_scan"),
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
        raise ValueError("exactly two lags are required")
    selected_models = tuple(part.strip() for part in args.models.split(",") if part.strip())
    if "5L" in selected_models:
        raise ValueError("5L architectural final head is not the lag locator")
    unknown = set(selected_models) - set(MODEL_SPECS)
    if unknown:
        raise ValueError(f"unknown models: {sorted(unknown)}")
    if args.quick:
        args.n_sequences = min(args.n_sequences, 8)

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    offsets = np.arange(1, args.maximum_offset + 1)
    reference_lag, target_lag = args.lags

    summary_rows = []
    offset_rows = []
    arrays: dict[str, np.ndarray] = {"offsets": offsets}
    metadata: dict[str, object] = {
        "lags": list(args.lags),
        "n_sequences_per_model_lag": args.n_sequences,
        "query_stride": args.query_stride,
        "models": {},
    }
    figures = []

    for model_name in selected_models:
        n_layers, checkpoint = MODEL_SPECS[model_name]
        print(f"{model_name}: loading {checkpoint}")
        model = load_attention_only_model(n_layers, checkpoint, device)
        curves = []
        diagnostics = []
        labels = None
        for lag in args.lags:
            print(f"  lag {lag}: scanning offsets 1-{args.maximum_offset}")
            current_labels, curve, diagnostic = contribution_curve(
                model,
                lag,
                args.n_sequences,
                args.batch_size,
                args.sequence_length,
                args.rho,
                args.query_start,
                args.query_stride,
                args.maximum_offset,
                args.offset_chunk_size,
                device,
            )
            if labels is None:
                labels = current_labels
            elif labels != current_labels:
                raise RuntimeError("path labels changed between lags")
            curves.append(curve)
            diagnostics.append(diagnostic)
            arrays[f"{model_name}_lag{lag}_curve"] = curve

        assert labels is not None
        reference_index = reference_lag - 2
        reference_raw = curves[0][reference_index]
        reference_margin = curves[0][reference_index] - curves[0][reference_index + 1]
        raw_similarity = correlations(reference_raw, curves[1])
        target_margins = curves[1][:-1] - curves[1][1:]
        margin_similarity = correlations(reference_margin, target_margins)
        arrays[f"{model_name}_raw_similarity"] = raw_similarity
        arrays[f"{model_name}_margin_similarity"] = margin_similarity

        correct_offset = target_lag - 1
        raw_peak_index = int(np.nanargmax(raw_similarity))
        margin_peak_index = int(np.nanargmax(margin_similarity))
        raw_secondary_offset, raw_secondary_value = best_outside_window(
            offsets, raw_similarity, correct_offset
        )
        margin_secondary_offset, margin_secondary_value = best_outside_window(
            offsets[:-1], margin_similarity, correct_offset
        )
        summary_rows.append(
            {
                "model": model_name,
                "n_paths": len(labels),
                "n_path_pairs": len(labels) ** 2,
                "reference_lag": reference_lag,
                "target_lag": target_lag,
                "target_correct_offset": correct_offset,
                "raw_peak_offset": int(offsets[raw_peak_index]),
                "raw_peak_correlation": float(raw_similarity[raw_peak_index]),
                "raw_secondary_offset": raw_secondary_offset,
                "raw_secondary_correlation": raw_secondary_value,
                "margin_peak_offset": int(offsets[:-1][margin_peak_index]),
                "margin_peak_correlation": float(margin_similarity[margin_peak_index]),
                "margin_secondary_offset": margin_secondary_offset,
                "margin_secondary_correlation": margin_secondary_value,
                "final_correct_rate_reference_lag": diagnostics[0]["final_correct_rate"],
                "final_correct_rate_target_lag": diagnostics[1]["final_correct_rate"],
                "maximum_residual_relative_error": max(
                    item["maximum_residual_relative_error"] for item in diagnostics
                ),
                "maximum_aggregate_score_error": max(
                    item["maximum_aggregate_score_error"] for item in diagnostics
                ),
            }
        )
        for offset_index, offset in enumerate(offsets):
            offset_rows.append(
                {
                    "model": model_name,
                    "candidate_offset": int(offset),
                    "raw_pearson": float(raw_similarity[offset_index]),
                    "margin_pearson": (
                        float(margin_similarity[offset_index])
                        if offset_index < len(margin_similarity)
                        else ""
                    ),
                }
            )
        figures.append(
            plot_scan(
                model_name,
                reference_lag,
                target_lag,
                offsets,
                raw_similarity,
                margin_similarity,
                args.output_dir / f"{model_name}_template_offset_scan.png",
            )
        )
        metadata["models"][model_name] = {
            "checkpoint": str(checkpoint),
            "n_layers": n_layers,
            "path_labels": labels,
        }
        del model

    save_csv(args.output_dir / "scan_summary.csv", summary_rows)
    save_csv(args.output_dir / "similarity_by_offset.csv", offset_rows)
    np.savez_compressed(args.output_dir / "path_pair_offset_curves.npz", **arrays)
    with (args.output_dir / "run_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)

    for row in summary_rows:
        print(
            f"{row['model']}: raw peak D={row['raw_peak_offset']} "
            f"r={row['raw_peak_correlation']:.4f}; "
            f"margin peak D={row['margin_peak_offset']} "
            f"r={row['margin_peak_correlation']:.4f}; "
            f"secondary margin D={row['margin_secondary_offset']} "
            f"r={row['margin_secondary_correlation']:.4f}"
        )

    if args.no_show:
        for figure in figures:
            plt.close(figure)
    else:
        plt.show()
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
