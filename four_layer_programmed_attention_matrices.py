"""Show ordinary Layer-4 query-by-key attention matrices after earlier patches.

The model and interventions match ``four_layer_programmed_attention_rows.py``,
but these figures keep the original matrix coordinates:

    horizontal axis: key position
    vertical axis:   query position

A fixed query-minus-key offset therefore appears as a diagonal stripe.

Usage:
    python four_layer_programmed_attention_matrices.py --quick --no-show
    python four_layer_programmed_attention_matrices.py --no-show
"""

from __future__ import annotations

import argparse
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


def offset_profile(
    attention: torch.Tensor,
    query_start: int,
    maximum_offset: int,
) -> torch.Tensor:
    """Average each sequence's attention mass at each query-minus-key offset."""
    sequence_length = attention.shape[-1]
    query_positions = torch.arange(
        query_start, sequence_length, device=attention.device
    )
    offsets = torch.arange(maximum_offset + 1, device=attention.device)
    rows = attention[
        :,
        query_positions[:, None],
        query_positions[:, None] - offsets[None, :],
    ]
    return rows.mean(dim=1)


def select_representative(
    clean_attention: torch.Tensor,
    query_start: int,
    maximum_offset: int,
    correct_offset: int,
) -> int:
    profile = offset_profile(clean_attention, query_start, maximum_offset)
    peaks = profile[:, 1:].argmax(dim=1) + 1
    candidates = torch.where(peaks == correct_offset)[0]
    if candidates.numel() == 0:
        candidates = torch.arange(len(profile), device=profile.device)
    masses = profile[candidates, correct_offset]
    local_index = (masses - masses.median()).abs().argmin()
    return int(candidates[local_index].item())


@torch.inference_mode()
def collect_conditions(
    model,
    inputs: torch.Tensor,
    patch_layer: int,
    programmed_offsets: tuple[int, ...],
    representative_index: int,
    query_start: int,
    maximum_offset: int,
    correct_offset: int,
) -> list[dict]:
    _, clean_attentions, _, _ = model(inputs)
    conditions: list[tuple[str, torch.Tensor]] = [("clean", clean_attentions[-1])]

    for programmed_offset in programmed_offsets:
        offsets = torch.full(
            (inputs.shape[0],), programmed_offset,
            dtype=torch.long, device=inputs.device,
        )
        stripe = fixed_offset_attention(
            inputs.shape[0], inputs.shape[1], offsets,
            inputs.dtype, inputs.device,
        )
        _, attentions, _, _ = forward_with_attention_patch(
            model,
            inputs,
            patch_layer=patch_layer,
            patched_attention=stripe,
        )
        conditions.append((f"program D={programmed_offset}", attentions[-1]))

    results = []
    for label, final_attention in conditions:
        profiles = offset_profile(final_attention, query_start, maximum_offset)
        peaks = profiles[:, 1:].argmax(dim=1) + 1
        modal_peak = int(torch.bincount(peaks).argmax().item())
        representative_peak = int(profiles[representative_index, 1:].argmax().item() + 1)
        results.append(
            {
                "label": label,
                "representative": final_attention[representative_index].cpu().numpy(),
                "mean": final_attention.mean(dim=0).cpu().numpy(),
                "representative_peak": representative_peak,
                "modal_peak": modal_peak,
                "correct_rate": float((peaks == correct_offset).double().mean().item()),
            }
        )
    return results


def plot_page(
    results: list[dict],
    patch_layer: int,
    sequence_length: int,
    correct_offset: int,
    matrix_key: str,
    page_label: str,
    output_path: Path,
) -> plt.Figure:
    columns = 4
    rows = math.ceil(len(results) / columns)
    figure, axes = plt.subplots(
        rows, columns,
        figsize=(4.8 * columns, 4.6 * rows),
        sharex=True, sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).reshape(-1)
    image = None
    stripe_queries = np.arange(correct_offset, sequence_length)
    stripe_keys = stripe_queries - correct_offset

    for axis, result in zip(axes, results):
        image = axis.imshow(
            result[matrix_key],
            origin="lower",
            aspect="equal",
            interpolation="nearest",
            cmap="magma",
            vmin=0,
            vmax=1,
            extent=(-0.5, sequence_length - 0.5, -0.5, sequence_length - 0.5),
        )
        axis.plot(
            stripe_keys, stripe_queries,
            color="cyan", linestyle="--", linewidth=1.0,
        )
        displayed_peak = (
            result["representative_peak"]
            if matrix_key == "representative"
            else result["modal_peak"]
        )
        peak_label = "representative" if matrix_key == "representative" else "modal"
        axis.set_title(
            f"{result['label']}\n{peak_label} final D={displayed_peak}; "
            f"batch correct={result['correct_rate']:.0%}"
        )
        axis.set_xlabel("key position")
        axis.set_ylabel("query position")

    for axis in axes[len(results):]:
        axis.remove()
    if image is not None:
        figure.colorbar(
            image,
            ax=axes[:len(results)],
            label="Layer-4 softmax attention mass",
            shrink=0.82,
        )
    figure.suptitle(
        f"4L lag 40: program Layer {patch_layer + 1}, inspect ordinary Layer-4 matrix\n"
        f"{page_label}; cyan = correct diagonal D={correct_offset}"
    )
    figure.savefig(output_path, dpi=180)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("models/attn_d64_4L_int_ext.pt"),
    )
    parser.add_argument("--lag", type=int, default=40)
    parser.add_argument(
        "--programmed-offsets", type=parse_offset_spec,
        default=(0, 1, 2, 10, 21, 23, 41),
    )
    parser.add_argument("--n-sequences", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=200)
    parser.add_argument("--rho", type=float, default=0.9)
    parser.add_argument("--query-start", type=int, default=120)
    parser.add_argument("--maximum-offset", type=int, default=100)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("experiment_outputs/four_layer_programmed_attention_matrices"),
    )
    parser.add_argument(
        "--device", choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.query_start < args.maximum_offset:
        raise ValueError("query-start must be at least maximum-offset")
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

    _, clean_attentions, _, _ = model(inputs)
    representative_index = select_representative(
        clean_attentions[-1],
        args.query_start,
        args.maximum_offset,
        args.lag - 1,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    figures = []
    for patch_layer in range(3):
        print(f"programming Layer {patch_layer + 1}")
        results = collect_conditions(
            model,
            inputs,
            patch_layer,
            args.programmed_offsets,
            representative_index,
            args.query_start,
            args.maximum_offset,
            args.lag - 1,
        )
        figures.append(
            plot_page(
                results,
                patch_layer,
                args.sequence_length,
                args.lag - 1,
                "representative",
                f"representative clean sequence index {representative_index}",
                args.output_dir / f"program_layer{patch_layer + 1}_representative_matrices.png",
            )
        )
        figures.append(
            plot_page(
                results,
                patch_layer,
                args.sequence_length,
                args.lag - 1,
                "mean",
                f"mean over {args.n_sequences} sequences",
                args.output_dir / f"program_layer{patch_layer + 1}_mean_matrices.png",
            )
        )

    with (args.output_dir / "run_metadata.json").open("w") as handle:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "lag": args.lag,
                "correct_offset": args.lag - 1,
                "programmed_offsets": list(args.programmed_offsets),
                "n_sequences": args.n_sequences,
                "sequence_length": args.sequence_length,
                "representative_sequence_index": representative_index,
                "axes": {"x": "key position", "y": "query position"},
                "color_scale": "linear softmax attention mass from 0 to 1",
            },
            handle,
            indent=2,
        )

    if args.no_show:
        for figure in figures:
            plt.close(figure)
    else:
        plt.show()
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
