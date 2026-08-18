"""Program absolute offsets into earlier heads and measure the locator head.

This is a full transfer-function experiment rather than a local +/-1 shift.
For each attention-only checkpoint, replace one earlier layer's attention with
an exact causal stripe at every requested offset, recompute all downstream
layers, and record where the model's lag-locating attention head peaks.

The lag-locating target is the architectural final layer except in the 5-layer
checkpoint, where Layer 4 is the programmable lag head and Layer 5 is a fixed
offset-1 head.  Architectural-final measurements are still saved for every
model.

Usage:
    python full_earlier_offset_programming.py --quick --no-show
    python full_earlier_offset_programming.py --no-show
    python full_earlier_offset_programming.py --models 4L --n-sequences 64
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
import torch.nn.functional as functional

from cross_model_raw_head1_path_test import (
    MODEL_SPECS,
    attention_profiles,
    load_attention_only_model,
    parse_int_tuple,
)
from util import make_dataset_lagset


# One-indexed layer number of the head that actually locates lag - 1.
LOCATOR_LAYER = {"4L": 4, "5L": 4, "6L": 6, "7L": 7}


def parse_offset_spec(text: str) -> tuple[int, ...]:
    """Parse either ``0:50`` (inclusive) or ``0,1,5,10``."""
    text = text.strip()
    if ":" in text:
        parts = tuple(int(part.strip()) for part in text.split(":"))
        if len(parts) not in (2, 3):
            raise argparse.ArgumentTypeError("range must be start:stop[:step]")
        start, stop = parts[:2]
        step = parts[2] if len(parts) == 3 else 1
        if step <= 0 or stop < start:
            raise argparse.ArgumentTypeError("offset range must increase")
        return tuple(range(start, stop + 1, step))
    return parse_int_tuple(text)


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixed_offset_attention(
    batch_size: int,
    sequence_length: int,
    offsets: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return one exact causal attention stripe per batch item.

    ``offsets[b] == D`` makes query row ``t`` attend to key ``t-D``.  In the
    first D rows, where that key does not exist, the row attends to itself.
    """
    if offsets.shape != (batch_size,):
        raise ValueError("offsets must contain one value per batch item")
    queries = torch.arange(sequence_length, device=device).unsqueeze(0)
    keys = queries - offsets.unsqueeze(1)
    keys = torch.where(keys >= 0, keys, queries).long()
    attention = torch.zeros(
        batch_size,
        sequence_length,
        sequence_length,
        dtype=dtype,
        device=device,
    )
    attention.scatter_(2, keys.unsqueeze(-1), 1.0)
    return attention


@torch.inference_mode()
def forward_programmed_attention(
    model,
    inputs: torch.Tensor,
    patch_layer: int,
    programmed_attention: torch.Tensor,
    locator_layer: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Patch one layer and return prediction, locator A, and final A."""
    batch_size, sequence_length = inputs.shape
    positions = torch.arange(sequence_length, device=inputs.device).unsqueeze(0)
    positions = positions.expand(batch_size, sequence_length)
    index = torch.arange(sequence_length, device=inputs.device)
    causal_mask = index.unsqueeze(0) > index.unsqueeze(1)
    residual = model.W_r(inputs.unsqueeze(-1))
    locator_attention = None
    final_attention = None

    for layer_index in range(model.n_layers):
        query_matrix, key_matrix, value_matrix, output_matrix = model.layers[
            layer_index
        ][0]
        if layer_index == patch_layer:
            attention = programmed_attention
        else:
            query = model.apply_rope(residual @ query_matrix, positions)
            key = model.apply_rope(residual @ key_matrix, positions)
            scores = query @ key.transpose(-2, -1) / math.sqrt(model.d_head)
            attention = functional.softmax(
                scores.masked_fill(causal_mask, float("-inf")), dim=-1
            )
        write = (attention @ (residual @ value_matrix)) @ output_matrix
        residual = residual + write
        if layer_index == locator_layer:
            locator_attention = attention
        if layer_index == model.n_layers - 1:
            final_attention = attention

    if locator_attention is None or final_attention is None:
        raise RuntimeError("requested attention layer was not collected")
    prediction = model.W_U(residual).squeeze(-1)
    return prediction, locator_attention, final_attention


def modal_value(values: torch.Tensor) -> int:
    return int(torch.bincount(values.long()).argmax().item())


def profile_peaks(
    attention: torch.Tensor,
    query_positions: torch.Tensor,
    maximum_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    profiles = attention_profiles(
        attention, query_positions, maximum_offset
    )
    return profiles, profiles.argmax(dim=1)


def total_variation(
    patched: torch.Tensor,
    clean: torch.Tensor,
    query_positions: torch.Tensor,
) -> torch.Tensor:
    return 0.5 * (
        patched[:, query_positions, :] - clean[:, query_positions, :]
    ).abs().sum(dim=-1).mean(dim=1)


@torch.inference_mode()
def run_model_lag(
    model,
    model_name: str,
    lag: int,
    programmed_offsets: tuple[int, ...],
    n_sequences: int,
    sequence_length: int,
    rho: float,
    query_start: int,
    query_stride: int,
    maximum_offset: int,
    burn_in: int,
    condition_chunk_size: int,
    device: torch.device,
) -> tuple[list[dict], list[dict], dict]:
    inputs, targets, sampled_lags = make_dataset_lagset(
        n_sequences,
        sequence_length,
        rho,
        [lag],
        seed=2_060_000 + 1_000 * model.n_layers + lag,
    )
    if not torch.all(sampled_lags == lag):
        raise RuntimeError("dataset returned an unexpected lag")
    inputs = inputs.to(device=device, dtype=torch.float64)
    targets = targets.to(device=device, dtype=torch.float64)
    clean_prediction, clean_attentions, _, _ = model(inputs)

    locator_index = LOCATOR_LAYER[model_name] - 1
    clean_locator = clean_attentions[locator_index]
    clean_final = clean_attentions[-1]
    query_positions = torch.arange(
        query_start, sequence_length, query_stride, device=device
    )
    clean_locator_profiles, clean_locator_peaks = profile_peaks(
        clean_locator, query_positions, maximum_offset
    )
    clean_final_profiles, clean_final_peaks = profile_peaks(
        clean_final, query_positions, maximum_offset
    )
    correct_offset = lag - 1
    mse_start = min(lag + burn_in, sequence_length - 1)
    clean_mse = (
        clean_prediction[:, mse_start:] - targets[:, mse_start:]
    ).square().mean(dim=1)

    condition_rows: list[dict] = []
    histogram_rows: list[dict] = []
    patch_layers = range(locator_index)

    for patch_layer in patch_layers:
        print(f"    program Layer {patch_layer + 1}")
        clean_patch_profiles, clean_patch_peaks = profile_peaks(
            clean_attentions[patch_layer], query_positions, maximum_offset
        )
        clean_patch_modal_offset = modal_value(clean_patch_peaks)
        for chunk_start in range(0, len(programmed_offsets), condition_chunk_size):
            chunk = programmed_offsets[
                chunk_start : chunk_start + condition_chunk_size
            ]
            n_conditions = len(chunk)
            repeated_inputs = inputs.repeat(n_conditions, 1)
            repeated_targets = targets.repeat(n_conditions, 1)
            batch_offsets = torch.tensor(
                chunk, device=device, dtype=torch.long
            ).repeat_interleave(n_sequences)
            programmed_attention = fixed_offset_attention(
                n_conditions * n_sequences,
                sequence_length,
                batch_offsets,
                inputs.dtype,
                device,
            )
            prediction, locator_attention, final_attention = (
                forward_programmed_attention(
                    model,
                    repeated_inputs,
                    patch_layer,
                    programmed_attention,
                    locator_index,
                )
            )
            locator_profiles, locator_peaks = profile_peaks(
                locator_attention, query_positions, maximum_offset
            )
            final_profiles, final_peaks = profile_peaks(
                final_attention, query_positions, maximum_offset
            )
            locator_attention = locator_attention.reshape(
                n_conditions, n_sequences, sequence_length, sequence_length
            )
            final_attention = final_attention.reshape(
                n_conditions, n_sequences, sequence_length, sequence_length
            )
            prediction = prediction.reshape(
                n_conditions, n_sequences, sequence_length
            )
            locator_profiles = locator_profiles.reshape(
                n_conditions, n_sequences, maximum_offset + 1
            )
            locator_peaks = locator_peaks.reshape(n_conditions, n_sequences)
            final_profiles = final_profiles.reshape(
                n_conditions, n_sequences, maximum_offset + 1
            )
            final_peaks = final_peaks.reshape(n_conditions, n_sequences)

            for condition_index, programmed_offset in enumerate(chunk):
                condition_locator = locator_attention[condition_index]
                condition_final = final_attention[condition_index]
                condition_prediction = prediction[condition_index]
                locator_tv = total_variation(
                    condition_locator, clean_locator, query_positions
                )
                final_tv = total_variation(
                    condition_final, clean_final, query_positions
                )
                sequence_mse = (
                    condition_prediction[:, mse_start:]
                    - repeated_targets.reshape(
                        n_conditions, n_sequences, sequence_length
                    )[condition_index, :, mse_start:]
                ).square().mean(dim=1)
                these_locator_peaks = locator_peaks[condition_index]
                these_final_peaks = final_peaks[condition_index]
                locator_correct_mass = locator_profiles[
                    condition_index, :, correct_offset
                ]
                final_correct_mass = final_profiles[
                    condition_index, :, correct_offset
                ]

                condition_rows.append(
                    {
                        "model": model_name,
                        "lag": lag,
                        "locator_layer": locator_index + 1,
                        "patch_layer": patch_layer + 1,
                        "clean_patch_layer_modal_offset": clean_patch_modal_offset,
                        "programmed_offset": programmed_offset,
                        "n_sequences": n_sequences,
                        "locator_modal_peak": modal_value(these_locator_peaks),
                        "locator_peak_mean": float(
                            these_locator_peaks.double().mean().item()
                        ),
                        "locator_peak_sd": float(
                            these_locator_peaks.double().std(unbiased=False).item()
                        ),
                        "locator_correct_rate": float(
                            (these_locator_peaks == correct_offset).double().mean().item()
                        ),
                        "locator_clean_peak_retention_rate": float(
                            (
                                these_locator_peaks == clean_locator_peaks
                            ).double().mean().item()
                        ),
                        "mean_locator_correct_mass": float(
                            locator_correct_mass.mean().item()
                        ),
                        "mean_locator_matrix_total_variation": float(
                            locator_tv.mean().item()
                        ),
                        "architectural_final_modal_peak": modal_value(
                            these_final_peaks
                        ),
                        "architectural_final_correct_rate": float(
                            (these_final_peaks == correct_offset).double().mean().item()
                        ),
                        "mean_architectural_final_correct_mass": float(
                            final_correct_mass.mean().item()
                        ),
                        "mean_architectural_final_matrix_total_variation": float(
                            final_tv.mean().item()
                        ),
                        "mean_prediction_mse": float(sequence_mse.mean().item()),
                        "mean_prediction_mse_change": float(
                            (sequence_mse - clean_mse).mean().item()
                        ),
                    }
                )
                counts = torch.bincount(
                    these_locator_peaks.long(), minlength=maximum_offset + 1
                )
                for output_offset in range(maximum_offset + 1):
                    histogram_rows.append(
                        {
                            "model": model_name,
                            "lag": lag,
                            "locator_layer": locator_index + 1,
                            "patch_layer": patch_layer + 1,
                            "clean_patch_layer_modal_offset": clean_patch_modal_offset,
                            "programmed_offset": programmed_offset,
                            "locator_output_offset": output_offset,
                            "sequence_count": int(counts[output_offset].item()),
                            "sequence_rate": float(
                                counts[output_offset].item() / n_sequences
                            ),
                        }
                    )

    baseline = {
        "model": model_name,
        "lag": lag,
        "locator_layer": locator_index + 1,
        "correct_offset": correct_offset,
        "clean_locator_modal_peak": modal_value(clean_locator_peaks),
        "clean_locator_correct_rate": float(
            (clean_locator_peaks == correct_offset).double().mean().item()
        ),
        "clean_locator_correct_mass": float(
            clean_locator_profiles[:, correct_offset].mean().item()
        ),
        "clean_architectural_final_modal_peak": modal_value(clean_final_peaks),
        "clean_architectural_final_correct_rate": float(
            (clean_final_peaks == correct_offset).double().mean().item()
        ),
        "clean_architectural_final_correct_mass": float(
            clean_final_profiles[:, correct_offset].mean().item()
        ),
        "clean_prediction_mse": float(clean_mse.mean().item()),
    }
    return condition_rows, histogram_rows, baseline


def plot_transfer_heatmaps(
    model_name: str,
    lag: int,
    locator_layer: int,
    programmed_offsets: tuple[int, ...],
    histogram_rows: list[dict],
    maximum_offset: int,
    output_path: Path,
) -> plt.Figure:
    patch_layers = sorted({row["patch_layer"] for row in histogram_rows})
    n_columns = min(3, len(patch_layers))
    n_rows = math.ceil(len(patch_layers) / n_columns)
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(5.2 * n_columns, 4.1 * n_rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes).reshape(-1)
    image = None
    for axis, patch_layer in zip(axes_array, patch_layers):
        selected = [
            row for row in histogram_rows if row["patch_layer"] == patch_layer
        ]
        lookup = {
            (row["programmed_offset"], row["locator_output_offset"]): row[
                "sequence_rate"
            ]
            for row in selected
        }
        matrix = np.asarray(
            [
                [
                    lookup[(input_offset, output_offset)]
                    for input_offset in programmed_offsets
                ]
                for output_offset in range(maximum_offset + 1)
            ]
        )
        image = axis.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(
                programmed_offsets[0] - 0.5,
                programmed_offsets[-1] + 0.5,
                -0.5,
                maximum_offset + 0.5,
            ),
            cmap="magma",
            vmin=0,
            vmax=1,
        )
        axis.axhline(lag - 1, color="cyan", linestyle="--", linewidth=1.2)
        natural_offset = selected[0]["clean_patch_layer_modal_offset"]
        axis.axvline(
            natural_offset, color="white", linestyle=":", linewidth=1.2
        )
        axis.set_title(f"Program Layer {patch_layer}")
        axis.set_xlabel("programmed earlier-layer offset")
        axis.set_ylabel(f"Layer {locator_layer} peak offset")
    for axis in axes_array[len(patch_layers) :]:
        axis.remove()
    if image is not None:
        figure.colorbar(image, ax=axes_array[: len(patch_layers)], label="fraction of sequences")
    figure.suptitle(
        f"{model_name}, lag {lag}: earlier-offset → locator-offset transfer\n"
        f"cyan = correct locator offset {lag - 1}; white = layer's learned offset"
    )
    figure.savefig(output_path, dpi=180)
    return figure


def plot_metric_curves(
    model_name: str,
    lag: int,
    locator_layer: int,
    condition_rows: list[dict],
    output_path: Path,
) -> plt.Figure:
    patch_layers = sorted({row["patch_layer"] for row in condition_rows})
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    metrics = (
        ("locator_correct_rate", "Locator peak at lag−1", (0, 1)),
        (
            "mean_locator_matrix_total_variation",
            "Change in full locator attention matrix",
            (0, 1),
        ),
        ("mean_locator_correct_mass", "Attention mass at lag−1", (0, 1)),
        ("mean_prediction_mse", "Prediction MSE", None),
    )
    for axis, (metric, title, limits) in zip(axes.flat, metrics):
        for patch_layer in patch_layers:
            rows = sorted(
                (
                    row
                    for row in condition_rows
                    if row["patch_layer"] == patch_layer
                ),
                key=lambda row: row["programmed_offset"],
            )
            axis.plot(
                [row["programmed_offset"] for row in rows],
                [row[metric] for row in rows],
                marker=".",
                linewidth=1.3,
                label=f"Layer {patch_layer}",
            )
            natural_offset = rows[0]["clean_patch_layer_modal_offset"]
            natural_row = next(
                (
                    row
                    for row in rows
                    if row["programmed_offset"] == natural_offset
                ),
                None,
            )
            if natural_row is not None:
                axis.scatter(
                    [natural_offset],
                    [natural_row[metric]],
                    marker="*",
                    s=95,
                    edgecolor="black",
                    linewidth=0.6,
                    zorder=5,
                )
        axis.set_title(title)
        axis.set_xlabel("programmed earlier-layer offset")
        if limits is not None:
            axis.set_ylim(*limits)
        axis.grid(alpha=0.2)
    axes[0, 0].legend(ncol=2)
    figure.suptitle(
        f"{model_name}, lag {lag}: effect on Layer {locator_layer} and prediction\n"
        "stars = each patched layer's learned offset"
    )
    figure.savefig(output_path, dpi=180)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="4L,5L,6L,7L")
    parser.add_argument("--lags", type=parse_int_tuple, default=(40,))
    parser.add_argument(
        "--programmed-offsets", type=parse_offset_spec, default=tuple(range(51))
    )
    parser.add_argument("--n-sequences", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=140)
    parser.add_argument("--rho", type=float, default=0.9)
    parser.add_argument("--query-start", type=int, default=80)
    parser.add_argument("--query-stride", type=int, default=3)
    parser.add_argument("--maximum-offset", type=int, default=60)
    parser.add_argument("--burn-in", type=int, default=30)
    parser.add_argument("--condition-chunk-size", type=int, default=6)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/full_earlier_offset_programming"),
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
    if any(offset < 0 for offset in args.programmed_offsets):
        raise ValueError("programmed offsets must be nonnegative")
    if max(args.programmed_offsets) > args.maximum_offset:
        raise ValueError("maximum_offset must include every programmed offset")
    if args.query_start < args.maximum_offset:
        raise ValueError("query_start must be at least maximum_offset")
    if args.quick:
        args.n_sequences = min(args.n_sequences, 4)
        args.programmed_offsets = tuple(
            offset
            for offset in (0, 1, 2, 5, 10, 20, 30, 39, 40, 50)
            if offset <= args.maximum_offset
        )

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_condition_rows: list[dict] = []
    all_histogram_rows: list[dict] = []
    baselines: list[dict] = []
    figures: list[plt.Figure] = []

    for model_name in selected_models:
        n_layers, checkpoint = MODEL_SPECS[model_name]
        print(f"\n{model_name}: {checkpoint}")
        model = load_attention_only_model(n_layers, checkpoint, device)
        for lag in args.lags:
            print(f"  lag {lag}")
            condition_rows, histogram_rows, baseline = run_model_lag(
                model=model,
                model_name=model_name,
                lag=lag,
                programmed_offsets=args.programmed_offsets,
                n_sequences=args.n_sequences,
                sequence_length=args.sequence_length,
                rho=args.rho,
                query_start=args.query_start,
                query_stride=args.query_stride,
                maximum_offset=args.maximum_offset,
                burn_in=args.burn_in,
                condition_chunk_size=args.condition_chunk_size,
                device=device,
            )
            all_condition_rows.extend(condition_rows)
            all_histogram_rows.extend(histogram_rows)
            baselines.append(baseline)
            figures.append(
                plot_transfer_heatmaps(
                    model_name,
                    lag,
                    LOCATOR_LAYER[model_name],
                    args.programmed_offsets,
                    histogram_rows,
                    args.maximum_offset,
                    args.output_dir
                    / f"{model_name}_lag{lag}_offset_transfer_heatmaps.png",
                )
            )
            figures.append(
                plot_metric_curves(
                    model_name,
                    lag,
                    LOCATOR_LAYER[model_name],
                    condition_rows,
                    args.output_dir / f"{model_name}_lag{lag}_metrics.png",
                )
            )

    save_csv(args.output_dir / "condition_results.csv", all_condition_rows)
    save_csv(args.output_dir / "output_peak_histograms.csv", all_histogram_rows)
    save_csv(args.output_dir / "clean_baselines.csv", baselines)
    with (args.output_dir / "run_metadata.json").open("w") as handle:
        json.dump(
            {
                "models": list(selected_models),
                "lags": list(args.lags),
                "programmed_offsets": list(args.programmed_offsets),
                "n_sequences_per_lag": args.n_sequences,
                "sequence_length": args.sequence_length,
                "locator_layers": LOCATOR_LAYER,
                "intervention": (
                    "replace one earlier attention matrix with an exact causal "
                    "one-hot stripe at the programmed offset"
                ),
                "boundary_rule": (
                    "queries earlier than the programmed offset attend to self"
                ),
                "baselines": baselines,
            },
            handle,
            indent=2,
        )

    if args.no_show:
        for figure in figures:
            plt.close(figure)
    else:
        plt.show()

    print(f"\nSaved results to {args.output_dir}")


if __name__ == "__main__":
    main()
