"""Test whether a correct-offset path-pair fingerprint is offset-specific.

This consumes the full path-pair curves saved by
``four_layer_multilag_path_pair_curves.py``.  It takes the lag-A matrix at its
correct offset as a reference and compares it with lag B's matrix at every
candidate offset.  It also compares local offset margins:

    contribution(D) - contribution(D + 1)

The margin comparison removes much of the stable path-magnitude background and
asks whether the same path pairs distinguish the correct offset from its next
neighbor.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def similarity(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    first = reference.ravel()
    second = candidate.ravel()
    pearson = float(np.corrcoef(first, second)[0, 1])
    cosine = float(
        first @ second / (np.linalg.norm(first) * np.linalg.norm(second))
    )
    return pearson, cosine


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--curves",
        type=Path,
        default=Path(
            "experiment_outputs/four_layer_correct_offset_path_pair_comparison/"
            "path_pair_curves.npz"
        ),
    )
    parser.add_argument("--reference-lag", type=int, default=30)
    parser.add_argument("--target-lag", type=int, default=40)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "experiment_outputs/four_layer_correct_offset_path_pair_comparison"
        ),
    )
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    data = np.load(args.curves)
    offsets = data["offsets"]
    reference_curves = data[f"lag_{args.reference_lag}_mean"]
    target_curves = data[f"lag_{args.target_lag}_mean"]
    reference_offset = args.reference_lag - 1
    target_offset = args.target_lag - 1
    reference_index = int(np.where(offsets == reference_offset)[0][0])
    reference_raw = reference_curves[..., reference_index]
    reference_margin = (
        reference_curves[..., reference_index]
        - reference_curves[..., reference_index + 1]
    )

    rows = []
    for offset_index, offset in enumerate(offsets):
        raw_pearson, raw_cosine = similarity(
            reference_raw,
            target_curves[..., offset_index],
        )
        row = {
            "candidate_offset": int(offset),
            "raw_pearson": raw_pearson,
            "raw_cosine": raw_cosine,
            "margin_pearson": "",
            "margin_cosine": "",
        }
        if offset_index + 1 < len(offsets):
            target_margin = (
                target_curves[..., offset_index]
                - target_curves[..., offset_index + 1]
            )
            margin_pearson, margin_cosine = similarity(
                reference_margin,
                target_margin,
            )
            row["margin_pearson"] = margin_pearson
            row["margin_cosine"] = margin_cosine
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(args.output_dir / "template_similarity_by_offset.csv", rows)

    raw = np.asarray([row["raw_pearson"] for row in rows])
    margin_offsets = offsets[:-1]
    margin = np.asarray([float(row["margin_pearson"]) for row in rows[:-1]])
    raw_peak = int(offsets[np.nanargmax(raw)])
    margin_peak = int(margin_offsets[np.nanargmax(margin)])

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11, 7.5),
        sharex=True,
        constrained_layout=True,
    )
    axes[0].plot(offsets, raw, linewidth=1.6)
    axes[0].axvline(target_offset, color="red", linestyle="--", label="correct offset")
    axes[0].scatter(
        [raw_peak], [raw[raw_peak - int(offsets[0])]],
        color="black", zorder=3,
        label=f"highest similarity: D={raw_peak}",
    )
    axes[0].set_ylabel("cellwise Pearson correlation")
    axes[0].set_title("Raw path-pair contribution matrix")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].plot(margin_offsets, margin, linewidth=1.6)
    axes[1].axvline(target_offset, color="red", linestyle="--", label="correct offset")
    axes[1].scatter(
        [margin_peak], [margin[margin_peak - int(margin_offsets[0])]],
        color="black", zorder=3,
        label=f"highest similarity: D={margin_peak}",
    )
    axes[1].set_xlabel("candidate offset in target-lag sequences")
    axes[1].set_ylabel("cellwise Pearson correlation")
    axes[1].set_title("Local margin matrix: contribution(D) − contribution(D+1)")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    axes[1].set_ylim(-1.0, 1.05)

    figure.suptitle(
        f"Lag {args.reference_lag} correct-offset fingerprint (D={reference_offset}) "
        f"scanned across lag {args.target_lag} offsets"
    )
    output_path = args.output_dir / "template_similarity_offset_scan.png"
    figure.savefig(output_path, dpi=180)
    if args.no_show:
        plt.close(figure)
    else:
        plt.show()

    correct_raw = raw[int(np.where(offsets == target_offset)[0][0])]
    correct_margin = margin[int(np.where(margin_offsets == target_offset)[0][0])]
    print(
        f"raw peak D={raw_peak}, r={np.nanmax(raw):.4f}; "
        f"correct D={target_offset}, r={correct_raw:.4f}"
    )
    print(
        f"margin peak D={margin_peak}, r={np.nanmax(margin):.4f}; "
        f"correct D={target_offset}, r={correct_margin:.4f}"
    )
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
