"""Measure the sizes and selectivity of all paths in the 4L path expansion.

The residual entering Layer 4 is decomposed into eight frozen-attention paths
through Layers 1--3.  This script reports:

* RMS residual norm of each path;
* RMS Q and K readout norm of each path in the final head;
* mean absolute raw-score contribution of each path pair over offsets;
* mean signed contribution at the correct offset;
* correct-minus-nearby-wrong contribution margin.
* correct-minus-(correct+1) contribution, which distinguishes the required
  retrieval offset ``lag - 1`` from the raw correlation offset ``lag``.

These measurements distinguish a path pair that is simply large from one that
specifically helps the final head select the correct lag.

Usage:
    python four_layer_path_pair_magnitude_analysis.py --quick --no-show
    python four_layer_path_pair_magnitude_analysis.py --no-show
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

from four_layer_attention_path_analysis import expand_prefinal_paths, load_model
from util import make_dataset_lagset


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


@torch.inference_mode()
def collect_measurements(
    model,
    lags: tuple[int, ...],
    wrong_deltas: tuple[int, ...],
    n_sequences: int,
    sequence_length: int,
    rho: float,
    query_start: int,
    query_stride: int,
    maximum_offset: int,
    device: torch.device,
) -> tuple[list[str], dict[str, np.ndarray], float, float]:
    labels: list[str] | None = None
    n_paths = 2 ** (model.n_layers - 1)

    residual_square_sum = torch.zeros(n_paths, dtype=torch.float64, device=device)
    query_square_sum = torch.zeros_like(residual_square_sum)
    key_square_sum = torch.zeros_like(residual_square_sum)
    path_vector_count = 0

    score_sum = torch.zeros(n_paths, n_paths, dtype=torch.float64, device=device)
    score_abs_sum = torch.zeros_like(score_sum)
    all_score_count = 0
    correct_sum = torch.zeros_like(score_sum)
    correct_plus_one_sum = torch.zeros_like(score_sum)
    wrong_sum = torch.zeros_like(score_sum)
    correct_count = 0
    correct_plus_one_count = 0
    wrong_count = 0

    maximum_residual_error = 0.0
    maximum_score_error = 0.0

    for lag in lags:
        print(f"lag {lag}")
        inputs, _, sampled_lags = make_dataset_lagset(
            n_sequences,
            sequence_length,
            rho,
            [lag],
            seed=810_000 + lag,
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
        current_labels = list(path_dict)
        if labels is None:
            labels = current_labels
        elif labels != current_labels:
            raise RuntimeError("path labels changed between lags")

        paths = torch.stack(list(path_dict.values()))
        final_input = post_attention[-2]
        residual_error = (
            (paths.sum(dim=0) - final_input).norm()
            / final_input.norm().clamp_min(1e-30)
        )
        maximum_residual_error = max(
            maximum_residual_error, float(residual_error.item())
        )

        query_matrix, key_matrix, _, _ = model.layers[-1][0]
        positions = torch.arange(sequence_length, device=device).view(1, 1, -1)
        path_queries = model.apply_rope(paths @ query_matrix, positions)
        path_keys = model.apply_rope(paths @ key_matrix, positions)
        full_queries = model.apply_rope(
            final_input @ query_matrix, positions.squeeze(0)
        )
        full_keys = model.apply_rope(
            final_input @ key_matrix, positions.squeeze(0)
        )

        query_positions = torch.arange(
            query_start,
            sequence_length,
            query_stride,
            device=device,
        )
        path_at_queries = paths[:, :, query_positions, :]
        query_at_positions = path_queries[:, :, query_positions, :]
        residual_square_sum += path_at_queries.square().sum(dim=(1, 2, 3))
        query_square_sum += query_at_positions.square().sum(dim=(1, 2, 3))
        key_at_queries = path_keys[:, :, query_positions, :]
        key_square_sum += key_at_queries.square().sum(dim=(1, 2, 3))
        path_vector_count += n_sequences * len(query_positions)

        correct_offset = lag - 1
        wrong_offsets = {
            correct_offset + delta
            for delta in wrong_deltas
            if 1 <= correct_offset + delta <= maximum_offset
        }

        full_query_at_positions = full_queries[:, query_positions, :]
        for offset in range(1, maximum_offset + 1):
            path_key_at_offset = path_keys[:, :, query_positions - offset, :]
            pair_scores = torch.einsum(
                "pbnd,qbnd->bnpq",
                query_at_positions,
                path_key_at_offset,
            ) / math.sqrt(model.d_head)
            score_sum += pair_scores.sum(dim=(0, 1))
            score_abs_sum += pair_scores.abs().sum(dim=(0, 1))
            all_score_count += n_sequences * len(query_positions)

            direct_scores = (
                full_query_at_positions
                * full_keys[:, query_positions - offset, :]
            ).sum(dim=-1) / math.sqrt(model.d_head)
            score_error = (
                pair_scores.sum(dim=(-1, -2)) - direct_scores
            ).abs().max()
            maximum_score_error = max(
                maximum_score_error, float(score_error.item())
            )

            if offset == correct_offset:
                correct_sum += pair_scores.sum(dim=(0, 1))
                correct_count += n_sequences * len(query_positions)
            elif offset == correct_offset + 1:
                correct_plus_one_sum += pair_scores.sum(dim=(0, 1))
                correct_plus_one_count += n_sequences * len(query_positions)
            elif offset in wrong_offsets:
                wrong_sum += pair_scores.sum(dim=(0, 1))
                wrong_count += n_sequences * len(query_positions)

    assert labels is not None
    residual_rms = torch.sqrt(residual_square_sum / path_vector_count)
    query_rms = torch.sqrt(query_square_sum / path_vector_count)
    key_rms = torch.sqrt(key_square_sum / path_vector_count)
    mean_score = score_sum / all_score_count
    mean_abs_score = score_abs_sum / all_score_count
    mean_correct_score = correct_sum / correct_count
    mean_correct_plus_one_score = correct_plus_one_sum / correct_plus_one_count
    mean_wrong_score = wrong_sum / wrong_count
    mean_margin = mean_correct_score - mean_wrong_score
    mean_correct_minus_plus_one = (
        mean_correct_score - mean_correct_plus_one_score
    )

    arrays = {
        "residual_rms_norm": residual_rms.cpu().numpy(),
        "query_rms_norm": query_rms.cpu().numpy(),
        "key_rms_norm": key_rms.cpu().numpy(),
        "mean_score_all_offsets": mean_score.cpu().numpy(),
        "mean_abs_score_all_offsets": mean_abs_score.cpu().numpy(),
        "mean_correct_score": mean_correct_score.cpu().numpy(),
        "mean_correct_plus_one_score": mean_correct_plus_one_score.cpu().numpy(),
        "mean_nearby_wrong_score": mean_wrong_score.cpu().numpy(),
        "mean_correct_minus_wrong_margin": mean_margin.cpu().numpy(),
        "mean_correct_minus_plus_one_margin": mean_correct_minus_plus_one.cpu().numpy(),
    }
    return labels, arrays, maximum_residual_error, maximum_score_error


def heatmap(
    axis: plt.Axes,
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    diverging: bool = False,
) -> None:
    if diverging:
        limit = float(np.max(np.abs(matrix)))
        image = axis.imshow(
            matrix,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            aspect="auto",
        )
    else:
        image = axis.imshow(matrix, cmap="viridis", aspect="auto")
    axis.set_xticks(range(len(labels)), labels)
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("key path")
    axis.set_ylabel("query path")
    axis.set_title(title)
    plt.colorbar(image, ax=axis, shrink=0.8)


def make_plot(
    labels: list[str],
    arrays: dict[str, np.ndarray],
    output_path: Path,
) -> plt.Figure:
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    positions = np.arange(len(labels))
    for axis, key, title in (
        (axes[0, 0], "residual_rms_norm", "Path size in residual space"),
        (axes[0, 1], "query_rms_norm", "Path size after final Q read"),
        (axes[0, 2], "key_rms_norm", "Path size after final K read"),
    ):
        axis.bar(positions, arrays[key])
        axis.set_xticks(positions, labels, rotation=45)
        axis.set_ylabel("RMS vector norm")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.3)

    heatmap(
        axes[1, 0],
        arrays["mean_abs_score_all_offsets"],
        labels,
        "Typical absolute raw-score contribution",
    )
    heatmap(
        axes[1, 1],
        arrays["mean_correct_score"],
        labels,
        "Signed score at the correct offset",
        diverging=True,
    )
    heatmap(
        axes[1, 2],
        arrays["mean_correct_minus_plus_one_margin"],
        labels,
        "Offset lag−1 score minus offset lag score",
        diverging=True,
    )
    figure.suptitle(
        "4-layer path expansion: size is different from lag selectivity"
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
    parser.add_argument(
        "--wrong-deltas",
        type=parse_int_tuple,
        default=(-8, -4, -2, 2, 4, 8),
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
        default=Path("/tmp/four_layer_path_pair_magnitudes"),
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
    if args.quick:
        args.n_sequences = min(args.n_sequences, 16)

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint, device)

    labels, arrays, residual_error, score_error = collect_measurements(
        model=model,
        lags=args.lags,
        wrong_deltas=args.wrong_deltas,
        n_sequences=args.n_sequences,
        sequence_length=args.sequence_length,
        rho=args.rho,
        query_start=args.query_start,
        query_stride=args.query_stride,
        maximum_offset=args.maximum_offset,
        device=device,
    )

    path_rows = [
        {
            "path": label,
            "residual_rms_norm": float(arrays["residual_rms_norm"][index]),
            "final_query_rms_norm": float(arrays["query_rms_norm"][index]),
            "final_key_rms_norm": float(arrays["key_rms_norm"][index]),
        }
        for index, label in enumerate(labels)
    ]
    pair_rows = []
    for query_index, query_label in enumerate(labels):
        for key_index, key_label in enumerate(labels):
            pair_rows.append(
                {
                    "query_path": query_label,
                    "key_path": key_label,
                    "mean_signed_score_all_offsets": float(
                        arrays["mean_score_all_offsets"][query_index, key_index]
                    ),
                    "mean_absolute_score_all_offsets": float(
                        arrays["mean_abs_score_all_offsets"][query_index, key_index]
                    ),
                    "mean_correct_score": float(
                        arrays["mean_correct_score"][query_index, key_index]
                    ),
                    "mean_correct_plus_one_score": float(
                        arrays["mean_correct_plus_one_score"][query_index, key_index]
                    ),
                    "mean_nearby_wrong_score": float(
                        arrays["mean_nearby_wrong_score"][query_index, key_index]
                    ),
                    "mean_correct_minus_wrong_margin": float(
                        arrays["mean_correct_minus_wrong_margin"][query_index, key_index]
                    ),
                    "mean_correct_minus_plus_one_margin": float(
                        arrays["mean_correct_minus_plus_one_margin"][query_index, key_index]
                    ),
                }
            )

    save_csv(args.output_dir / "path_readout_magnitudes.csv", path_rows)
    save_csv(args.output_dir / "path_pair_score_magnitudes.csv", pair_rows)
    np.savez_compressed(args.output_dir / "magnitude_arrays.npz", **arrays)
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "lags": list(args.lags),
                "n_sequences_per_lag": args.n_sequences,
                "wrong_deltas": list(args.wrong_deltas),
                "path_labels": labels,
                "path_bit_meaning": (
                    "bits correspond to Layers 1, 2, 3; "
                    "0=identity and 1=frozen attention/OV write"
                ),
                "maximum_residual_reconstruction_error": residual_error,
                "maximum_raw_score_reconstruction_error": score_error,
            },
            handle,
            indent=2,
        )

    figure = make_plot(
        labels,
        arrays,
        args.output_dir / "path_pair_magnitudes.png",
    )
    if args.no_show:
        plt.close(figure)
    else:
        plt.show()

    print("\nPath readout magnitudes:")
    for row in path_rows:
        print(
            f"  {row['path']}: residual={row['residual_rms_norm']:.3f}, "
            f"Q={row['final_query_rms_norm']:.3f}, "
            f"K={row['final_key_rms_norm']:.3f}"
        )

    print("\nLargest typical absolute pair scores:")
    for row in sorted(
        pair_rows,
        key=lambda item: item["mean_absolute_score_all_offsets"],
        reverse=True,
    )[:12]:
        print(
            f"  q{row['query_path']} × k{row['key_path']}: "
            f"abs={row['mean_absolute_score_all_offsets']:.3f}, "
            f"correct={row['mean_correct_score']:+.3f}, "
            f"correct-vs-lag={row['mean_correct_minus_plus_one_margin']:+.3f}"
        )

    print("\nLargest lag−1-minus-lag pair margins:")
    for row in sorted(
        pair_rows,
        key=lambda item: abs(item["mean_correct_minus_plus_one_margin"]),
        reverse=True,
    )[:12]:
        print(
            f"  q{row['query_path']} × k{row['key_path']}: "
            f"lag−1 minus lag={row['mean_correct_minus_plus_one_margin']:+.3f}, "
            f"abs={row['mean_absolute_score_all_offsets']:.3f}"
        )
    print(f"\nSaved results to {args.output_dir}")


if __name__ == "__main__":
    main()
