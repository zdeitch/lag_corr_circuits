"""Visualize actual Layer-4 attention rows after programming earlier stripes.

For a lag-40 dataset, replace Layer 1, 2, or 3 attention with exact causal
one-hot stripes at selected offsets, recompute the rest of the 4L model, and
display Layer 4 in offset coordinates:

    horizontal axis: query position minus key position
    vertical axis:   query row
    color:           softmax attention mass

Each earlier layer gets two pages: one representative sequence and the mean
over all sequences.  All panels use the same linear 0--1 color scale.

Usage:
    python four_layer_programmed_attention_rows.py --quick --no-show
    python four_layer_programmed_attention_rows.py --no-show
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

from four_layer_attention_path_analysis import load_model
from four_layer_earlier_attention_shift_patch import forward_with_attention_patch
from full_earlier_offset_programming import fixed_offset_attention, parse_offset_spec
from util import make_dataset_lagset


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def attention_in_offset_coordinates(
    attention: torch.Tensor,
    query_positions: torch.Tensor,
    maximum_offset: int,
) -> torch.Tensor:
    offsets = torch.arange(maximum_offset + 1, device=attention.device)
    return attention[
        :,
        query_positions[:, None],
        query_positions[:, None] - offsets[None, :],
    ]


def select_representative_sequence(
    clean_final_attention: torch.Tensor,
    query_positions: torch.Tensor,
    maximum_offset: int,
    correct_offset: int,
) -> int:
    rows = attention_in_offset_coordinates(
        clean_final_attention, query_positions, maximum_offset
    )
    profiles = rows.mean(dim=1)
    peaks = profiles[:, 1:].argmax(dim=1) + 1
    candidates = torch.where(peaks == correct_offset)[0]
    if len(candidates) == 0:
        candidates = torch.arange(len(profiles), device=profiles.device)
    masses = profiles[candidates, correct_offset]
    median = masses.median()
    local_index = (masses - median).abs().argmin()
    return int(candidates[local_index].item())


@torch.inference_mode()
def collect_conditions(
    model,
    inputs: torch.Tensor,
    patch_layer: int,
    programmed_offsets: tuple[int, ...],
    query_positions: torch.Tensor,
    maximum_offset: int,
    correct_offset: int,
    representative_index: int,
) -> list[dict]:
    _, clean_attentions, _, _ = model(inputs)
    conditions: list[tuple[str, int | None, torch.Tensor]] = [
        ("clean", None, clean_attentions[-1])
    ]

    for programmed_offset in programmed_offsets:
        offsets = torch.full(
            (inputs.shape[0],),
            programmed_offset,
            dtype=torch.long,
            device=inputs.device,
        )
        stripe = fixed_offset_attention(
            inputs.shape[0],
            inputs.shape[1],
            offsets,
            inputs.dtype,
            inputs.device,
        )
        _, attentions, _, _ = forward_with_attention_patch(
            model,
            inputs,
            patch_layer=patch_layer,
            patched_attention=stripe,
        )
        conditions.append(
            (f"program D={programmed_offset}", programmed_offset, attentions[-1])
        )

    results: list[dict] = []
    for label, programmed_offset, final_attention in conditions:
        row_distributions = attention_in_offset_coordinates(
            final_attention, query_positions, maximum_offset
        )
        profiles = row_distributions.mean(dim=1)
        sequence_peaks = profiles[:, 1:].argmax(dim=1) + 1
        peak_counts = torch.bincount(
            sequence_peaks, minlength=maximum_offset + 1
        )
        modal_peak = int(peak_counts.argmax().item())
        representative_rows = row_distributions[representative_index]
        representative_profile = representative_rows.mean(dim=0)
        representative_peak = int(
            representative_profile[1:].argmax().item() + 1
        )
        visible_mass = row_distributions.sum(dim=-1)
        row_entropy = -(
            final_attention[:, query_positions, :]
            * final_attention[:, query_positions, :].clamp_min(1e-300).log()
        ).sum(dim=-1)
        results.append(
            {
                "label": label,
                "programmed_offset": programmed_offset,
                "representative_rows": representative_rows.cpu().numpy(),
                "mean_rows": row_distributions.mean(dim=0).cpu().numpy(),
                "representative_peak": representative_peak,
                "modal_sequence_peak": modal_peak,
                "correct_peak_rate": float(
                    (sequence_peaks == correct_offset).double().mean().item()
                ),
                "mean_attention_mass_at_correct_offset": float(
                    profiles[:, correct_offset].mean().item()
                ),
                "mean_visible_mass": float(visible_mass.mean().item()),
                "mean_row_entropy": float(row_entropy.mean().item()),
            }
        )
    return results


def plot_attention_pages(
    results: list[dict],
    patch_layer: int,
    query_positions: torch.Tensor,
    maximum_offset: int,
    correct_offset: int,
    array_key: str,
    page_label: str,
    output_path: Path,
) -> plt.Figure:
    n_columns = 4
    n_rows = math.ceil(len(results) / n_columns)
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(4.9 * n_columns, 3.9 * n_rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    flat_axes = np.atleast_1d(axes).reshape(-1)
    image = None
    first_query = int(query_positions[0].item())
    last_query = int(query_positions[-1].item())
    for axis, result in zip(flat_axes, results):
        matrix = result[array_key]
        image = axis.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(-0.5, maximum_offset + 0.5, first_query - 0.5, last_query + 0.5),
            cmap="magma",
            vmin=0,
            vmax=1,
        )
        axis.axvline(correct_offset, color="cyan", linestyle="--", linewidth=1.1)
        displayed_peak = (
            result["representative_peak"]
            if array_key == "representative_rows"
            else result["modal_sequence_peak"]
        )
        peak_label = (
            "representative peak"
            if array_key == "representative_rows"
            else "modal sequence peak"
        )
        axis.set_title(
            f"{result['label']}\n{peak_label} D={displayed_peak}; "
            f"batch correct={result['correct_peak_rate']:.0%}"
        )
        axis.set_xlabel("query − key offset")
        axis.set_ylabel("final-layer query row")
    for axis in flat_axes[len(results) :]:
        axis.remove()
    if image is not None:
        figure.colorbar(
            image,
            ax=flat_axes[: len(results)],
            label="final-layer softmax attention mass",
            shrink=0.85,
        )
    figure.suptitle(
        f"4L lag 40: program Layer {patch_layer + 1}, inspect Layer 4 rows\n"
        f"{page_label}; linear color scale; cyan = correct offset {correct_offset}"
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
    parser.add_argument("--lag", type=int, default=40)
    parser.add_argument(
        "--programmed-offsets",
        type=parse_offset_spec,
        default=(0, 1, 2, 10, 21, 23, 41),
    )
    parser.add_argument("--n-sequences", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=200)
    parser.add_argument("--rho", type=float, default=0.9)
    parser.add_argument("--query-start", type=int, default=120)
    parser.add_argument("--maximum-offset", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "experiment_outputs/four_layer_programmed_attention_rows"
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
    if args.lag - 1 > args.maximum_offset:
        raise ValueError("maximum_offset must include lag - 1")
    if max(args.programmed_offsets) > args.maximum_offset:
        raise ValueError("maximum_offset must include programmed offsets")
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
        seed=2_500_000 + args.lag,
    )
    if not torch.all(sampled_lags == args.lag):
        raise RuntimeError("dataset returned an unexpected lag")
    inputs = inputs.to(device=device, dtype=torch.float64)
    query_positions = torch.arange(
        args.query_start, args.sequence_length, device=device
    )
    _, clean_attentions, _, _ = model(inputs)
    representative_index = select_representative_sequence(
        clean_attentions[-1],
        query_positions,
        args.maximum_offset,
        args.lag - 1,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures: list[plt.Figure] = []
    summary_rows: list[dict] = []
    arrays: dict[str, np.ndarray] = {}
    for patch_layer in range(3):
        print(f"programming Layer {patch_layer + 1}")
        results = collect_conditions(
            model,
            inputs,
            patch_layer,
            args.programmed_offsets,
            query_positions,
            args.maximum_offset,
            args.lag - 1,
            representative_index,
        )
        figures.append(
            plot_attention_pages(
                results,
                patch_layer,
                query_positions,
                args.maximum_offset,
                args.lag - 1,
                "representative_rows",
                f"representative clean sequence index {representative_index}",
                args.output_dir
                / f"program_layer{patch_layer + 1}_representative_rows.png",
            )
        )
        figures.append(
            plot_attention_pages(
                results,
                patch_layer,
                query_positions,
                args.maximum_offset,
                args.lag - 1,
                "mean_rows",
                f"mean at each query row over {args.n_sequences} sequences",
                args.output_dir / f"program_layer{patch_layer + 1}_mean_rows.png",
            )
        )
        for condition_index, result in enumerate(results):
            safe_name = "clean" if result["programmed_offset"] is None else (
                f"D{result['programmed_offset']}"
            )
            arrays[
                f"layer{patch_layer + 1}_{safe_name}_representative"
            ] = result["representative_rows"]
            arrays[f"layer{patch_layer + 1}_{safe_name}_mean"] = result[
                "mean_rows"
            ]
            summary_rows.append(
                {
                    "patch_layer": patch_layer + 1,
                    "condition": result["label"],
                    "programmed_offset": result["programmed_offset"],
                    "representative_sequence_index": representative_index,
                    "representative_final_peak": result[
                        "representative_peak"
                    ],
                    "modal_sequence_final_peak": result[
                        "modal_sequence_peak"
                    ],
                    "correct_final_peak_rate": result["correct_peak_rate"],
                    "mean_attention_mass_at_correct_offset": result[
                        "mean_attention_mass_at_correct_offset"
                    ],
                    "mean_mass_visible_in_offsets_0_to_maximum": result[
                        "mean_visible_mass"
                    ],
                    "mean_full_row_entropy": result["mean_row_entropy"],
                }
            )

    save_csv(args.output_dir / "condition_summary.csv", summary_rows)
    np.savez_compressed(args.output_dir / "attention_row_arrays.npz", **arrays)
    with (args.output_dir / "run_metadata.json").open("w") as handle:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "lag": args.lag,
                "correct_offset": args.lag - 1,
                "programmed_offsets": list(args.programmed_offsets),
                "n_sequences": args.n_sequences,
                "sequence_length": args.sequence_length,
                "query_rows": [args.query_start, args.sequence_length - 1],
                "maximum_displayed_offset": args.maximum_offset,
                "representative_sequence_index": representative_index,
                "color_scale": "linear attention mass from 0 to 1",
            },
            handle,
            indent=2,
        )

    print(f"representative sequence index: {representative_index}")
    for row in summary_rows:
        print(
            f"L{row['patch_layer']} {row['condition']}: "
            f"modal D={row['modal_sequence_final_peak']}, "
            f"correct={row['correct_final_peak_rate']:.1%}, "
            f"mass@39={row['mean_attention_mass_at_correct_offset']:.3f}, "
            f"entropy={row['mean_full_row_entropy']:.3f}"
        )
    if args.no_show:
        for figure in figures:
            plt.close(figure)
    else:
        plt.show()
    print(f"\nSaved results to {args.output_dir}")


if __name__ == "__main__":
    main()
