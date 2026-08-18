"""Causally shift one earlier attention pattern in the 4L model.

For each sequence, this script takes the clean attention matrix of Layer 1, 2,
or 3 and moves all of its key-axis mass one position nearer or farther.  This
preserves the observed pattern's shape much better than replacing it with a
hard stripe.  The patched layer uses the shifted matrix; all later layers are
then recomputed normally.

Measurements include final-attention retrieval, final raw-score margin,
prediction MSE, and the peak of the q000 x k100 path-pair contribution.

Usage:
    python four_layer_earlier_attention_shift_patch.py --quick --no-show
    python four_layer_earlier_attention_shift_patch.py --no-show
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
import torch.nn.functional as functional

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


def shift_attention_key_axis(
    attention: torch.Tensor,
    offset_shift: int,
) -> torch.Tensor:
    """Move attention mass so positive shift means a larger past offset."""
    if offset_shift == 0:
        return attention.clone()
    _, sequence_length, _ = attention.shape
    shifted = torch.zeros_like(attention)
    if offset_shift > 0:
        shifted[..., : sequence_length - offset_shift] = attention[
            ..., offset_shift:
        ]
    else:
        amount = -offset_shift
        shifted[..., amount:] = attention[..., : sequence_length - amount]

    index = torch.arange(sequence_length, device=attention.device)
    future_mask = index.unsqueeze(0) > index.unsqueeze(1)
    shifted = shifted.masked_fill(future_mask, 0)
    row_sums = shifted.sum(dim=-1, keepdim=True)
    zero_rows = row_sums.squeeze(-1) <= 1e-30
    shifted = shifted / row_sums.clamp_min(1e-30)
    if zero_rows.any():
        batch_index, query_index = torch.where(zero_rows)
        shifted[batch_index, query_index, query_index] = 1.0
    return shifted


@torch.inference_mode()
def forward_with_attention_patch(
    model,
    inputs: torch.Tensor,
    patch_layer: int,
    patched_attention: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """Patch one earlier layer and recompute all downstream attention."""
    batch_size, sequence_length = inputs.shape
    positions = torch.arange(sequence_length, device=inputs.device).unsqueeze(0)
    positions = positions.expand(batch_size, sequence_length)
    index = torch.arange(sequence_length, device=inputs.device)
    causal_mask = index.unsqueeze(0) > index.unsqueeze(1)

    residual = model.W_r(inputs.unsqueeze(-1))
    attentions: list[torch.Tensor] = []
    post_attention: list[torch.Tensor] = []
    final_raw_scores: torch.Tensor | None = None

    for layer_index in range(model.n_layers):
        query_matrix, key_matrix, value_matrix, output_matrix = model.layers[
            layer_index
        ][0]
        if layer_index == patch_layer:
            attention = patched_attention
            write = (attention @ (residual @ value_matrix)) @ output_matrix
        else:
            query = model.apply_rope(residual @ query_matrix, positions)
            key = model.apply_rope(residual @ key_matrix, positions)
            raw_scores = query @ key.transpose(-2, -1) / math.sqrt(model.d_head)
            masked_scores = raw_scores.masked_fill(causal_mask, float("-inf"))
            attention = functional.softmax(masked_scores, dim=-1)
            write = (attention @ (residual @ value_matrix)) @ output_matrix
            if layer_index == model.n_layers - 1:
                final_raw_scores = raw_scores
        attentions.append(attention)
        residual = residual + write
        post_attention.append(residual)

    if final_raw_scores is None:
        raise RuntimeError("final raw scores were not collected")
    prediction = model.W_U(residual).squeeze(-1)
    return prediction, attentions, post_attention, final_raw_scores


def profile_from_square_matrix(
    matrix: torch.Tensor,
    query_positions: torch.Tensor,
    maximum_offset: int,
) -> torch.Tensor:
    offsets = torch.arange(1, maximum_offset + 1, device=matrix.device)
    values = matrix[
        :,
        query_positions[:, None],
        query_positions[:, None] - offsets[None, :],
    ]
    return values.mean(dim=1)


@torch.inference_mode()
def raw_head1_pair_peaks(
    model,
    embedding: torch.Tensor,
    attentions: list[torch.Tensor],
    post_attention: list[torch.Tensor],
    query_positions: torch.Tensor,
    maximum_offset: int,
) -> torch.Tensor:
    """Return each sequence's peak offset for q000 x k100."""
    paths = expand_prefinal_paths(
        embedding,
        attentions,
        model,
        post_attention[: model.n_layers - 1],
    )
    query_component = paths["000"]
    key_component = paths["100"]
    query_matrix, key_matrix, _, _ = model.layers[-1][0]
    positions = torch.arange(embedding.shape[1], device=embedding.device).unsqueeze(0)
    query = model.apply_rope(query_component @ query_matrix, positions)
    key = model.apply_rope(key_component @ key_matrix, positions)
    curves = torch.empty(
        embedding.shape[0],
        maximum_offset,
        dtype=embedding.dtype,
        device=embedding.device,
    )
    for offset in range(1, maximum_offset + 1):
        scores = (
            query[:, query_positions, :]
            * key[:, query_positions - offset, :]
        ).sum(dim=-1) / math.sqrt(model.d_head)
        curves[:, offset - 1] = scores.mean(dim=1)
    return curves.argmax(dim=1) + 1


@torch.inference_mode()
def evaluate_condition(
    model,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    clean_attention: torch.Tensor,
    lag: int,
    patch_layer: int,
    offset_shift: int,
    burn_in: int,
    query_positions: torch.Tensor,
    maximum_offset: int,
) -> dict:
    patched_attention = shift_attention_key_axis(clean_attention, offset_shift)
    prediction, attentions, post_attention, final_raw_scores = (
        forward_with_attention_patch(
            model,
            inputs,
            patch_layer,
            patched_attention,
        )
    )
    final_profiles = profile_from_square_matrix(
        attentions[-1], query_positions, maximum_offset
    )
    final_peaks = final_profiles.argmax(dim=1) + 1
    raw_profiles = profile_from_square_matrix(
        final_raw_scores, query_positions, maximum_offset
    )
    correct_offset = lag - 1
    raw_correct_minus_lag = (
        raw_profiles[:, correct_offset - 1] - raw_profiles[:, lag - 1]
    )
    start = lag + burn_in
    sequence_mse = (
        prediction[:, start:] - targets[:, start:]
    ).square().mean(dim=1)
    pair_peaks = raw_head1_pair_peaks(
        model,
        model.W_r(inputs.unsqueeze(-1)),
        attentions,
        post_attention,
        query_positions,
        maximum_offset,
    )
    return {
        "final_peaks": final_peaks.cpu().numpy(),
        "raw_correct_minus_lag": raw_correct_minus_lag.cpu().numpy(),
        "sequence_mse": sequence_mse.cpu().numpy(),
        "raw_head1_pair_peaks": pair_peaks.cpu().numpy(),
    }


def aggregate_results(
    observations: list[dict],
    patch_layers: tuple[int, ...],
    shifts: tuple[int, ...],
) -> list[dict]:
    rows = []
    for patch_layer in patch_layers:
        for shift in shifts:
            selected = [
                row
                for row in observations
                if row["patch_layer"] == patch_layer and row["offset_shift"] == shift
            ]
            final_errors = np.concatenate(
                [row["final_peaks"] - (row["lag"] - 1) for row in selected]
            )
            pair_errors = np.concatenate(
                [row["raw_head1_pair_peaks"] - (row["lag"] - 1) for row in selected]
            )
            raw_margins = np.concatenate(
                [row["raw_correct_minus_lag"] for row in selected]
            )
            sequence_mse = np.concatenate(
                [row["sequence_mse"] for row in selected]
            )
            final_mode = Counter(final_errors.tolist()).most_common(1)[0][0]
            pair_mode = Counter(pair_errors.tolist()).most_common(1)[0][0]
            rows.append(
                {
                    "patch_layer": patch_layer + 1,
                    "offset_shift": shift,
                    "final_exact_correct_rate": float(np.mean(final_errors == 0)),
                    "final_modal_offset_error": int(final_mode),
                    "mean_final_raw_lag_minus_one_vs_lag_margin": float(
                        raw_margins.mean()
                    ),
                    "mean_prediction_mse": float(sequence_mse.mean()),
                    "raw_head1_pair_exact_correct_rate": float(
                        np.mean(pair_errors == 0)
                    ),
                    "raw_head1_pair_modal_offset_error": int(pair_mode),
                }
            )
    return rows


def plot_results(
    aggregate_rows: list[dict],
    patch_layers: tuple[int, ...],
    shifts: tuple[int, ...],
    output_path: Path,
) -> plt.Figure:
    metrics = (
        ("final_exact_correct_rate", "Final attention exact-correct rate", "viridis", 0, 1),
        ("mean_prediction_mse", "Prediction MSE", "magma", None, None),
        (
            "mean_final_raw_lag_minus_one_vs_lag_margin",
            "Final raw score: lag−1 minus lag",
            "RdBu_r",
            None,
            None,
        ),
        (
            "raw_head1_pair_modal_offset_error",
            "q000×k100 modal peak error",
            "coolwarm",
            None,
            None,
        ),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for axis, (metric, title, cmap, fixed_min, fixed_max) in zip(axes.flat, metrics):
        matrix = np.asarray(
            [
                [
                    next(
                        row[metric]
                        for row in aggregate_rows
                        if row["patch_layer"] == layer + 1
                        and row["offset_shift"] == shift
                    )
                    for shift in shifts
                ]
                for layer in patch_layers
            ]
        )
        if cmap in ("RdBu_r", "coolwarm"):
            limit = float(np.max(np.abs(matrix)))
            vmin, vmax = -limit, limit
        else:
            vmin, vmax = fixed_min, fixed_max
        image = axis.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                text = f"{value:.1%}" if metric.endswith("rate") else f"{value:+.3f}"
                axis.text(column_index, row_index, text, ha="center", va="center")
        axis.set_xticks(range(len(shifts)), [f"{shift:+d}" for shift in shifts])
        axis.set_yticks(
            range(len(patch_layers)),
            [f"Layer {layer + 1}" for layer in patch_layers],
        )
        axis.set_xlabel("attention-offset shift")
        axis.set_ylabel("patched layer")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=0.8)
    figure.suptitle(
        "4L causal test: shift one earlier attention pattern and recompute downstream"
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
        "--lags", type=parse_int_tuple, default=(25, 30, 35, 40, 45, 50)
    )
    parser.add_argument(
        "--patch-layers",
        type=parse_int_tuple,
        default=(1, 2, 3),
        help="one-indexed earlier layers",
    )
    parser.add_argument(
        "--shifts", type=parse_int_tuple, default=(-1, 0, 1)
    )
    parser.add_argument("--n-sequences", type=int, default=96)
    parser.add_argument("--sequence-length", type=int, default=200)
    parser.add_argument("--rho", type=float, default=0.9)
    parser.add_argument("--burn-in", type=int, default=30)
    parser.add_argument("--query-start", type=int, default=120)
    parser.add_argument("--query-stride", type=int, default=4)
    parser.add_argument("--maximum-offset", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/four_layer_earlier_attention_shift_patch"),
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    patch_layers = tuple(layer - 1 for layer in args.patch_layers)
    if any(layer < 0 or layer >= 3 for layer in patch_layers):
        raise ValueError("patch layers must be one-indexed Layers 1, 2, or 3")
    if args.query_start < args.maximum_offset:
        raise ValueError("query_start must be at least maximum_offset")
    if args.quick:
        args.n_sequences = min(args.n_sequences, 16)

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint, device)
    query_positions = torch.arange(
        args.query_start,
        args.sequence_length,
        args.query_stride,
        device=device,
    )
    observations: list[dict] = []

    for lag in args.lags:
        print(f"lag {lag}")
        inputs, targets, sampled_lags = make_dataset_lagset(
            args.n_sequences,
            args.sequence_length,
            args.rho,
            [lag],
            seed=970_000 + lag,
        )
        if not torch.all(sampled_lags == lag):
            raise RuntimeError("dataset returned an unexpected lag")
        inputs = inputs.to(device=device, dtype=torch.float64)
        targets = targets.to(device=device, dtype=torch.float64)
        with torch.inference_mode():
            _, clean_attentions, _, _ = model(inputs)

        for patch_layer in patch_layers:
            for shift in args.shifts:
                print(f"  Layer {patch_layer + 1}, shift {shift:+d}")
                result = evaluate_condition(
                    model=model,
                    inputs=inputs,
                    targets=targets,
                    clean_attention=clean_attentions[patch_layer],
                    lag=lag,
                    patch_layer=patch_layer,
                    offset_shift=shift,
                    burn_in=args.burn_in,
                    query_positions=query_positions,
                    maximum_offset=args.maximum_offset,
                )
                result.update(
                    {
                        "lag": lag,
                        "patch_layer": patch_layer,
                        "offset_shift": shift,
                    }
                )
                observations.append(result)

    aggregate_rows = aggregate_results(
        observations,
        patch_layers,
        args.shifts,
    )
    save_csv(args.output_dir / "aggregate_shift_patch_results.csv", aggregate_rows)
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "lags": list(args.lags),
                "n_sequences_per_lag": args.n_sequences,
                "patch_layers": list(args.patch_layers),
                "attention_offset_shifts": list(args.shifts),
                "intervention": (
                    "shift each sequence's clean attention matrix along the key axis, "
                    "renormalize rows, and recompute all downstream layers"
                ),
                "aggregate": aggregate_rows,
            },
            handle,
            indent=2,
        )

    figure = plot_results(
        aggregate_rows,
        patch_layers,
        args.shifts,
        args.output_dir / "earlier_attention_shift_patch.png",
    )
    if args.no_show:
        plt.close(figure)
    else:
        plt.show()

    print("\nAggregate results:")
    for row in aggregate_rows:
        print(
            f"  L{row['patch_layer']} shift {row['offset_shift']:+d}: "
            f"final-correct={row['final_exact_correct_rate']:.1%}, "
            f"final mode error={row['final_modal_offset_error']:+d}, "
            f"raw margin={row['mean_final_raw_lag_minus_one_vs_lag_margin']:+.3f}, "
            f"MSE={row['mean_prediction_mse']:.3f}, "
            f"q000×k100 mode error={row['raw_head1_pair_modal_offset_error']:+d}"
        )
    print(f"\nSaved results to {args.output_dir}")


if __name__ == "__main__":
    main()
