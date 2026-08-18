"""Compare path-level delay decoding across programmable attention-only models.

The experiment applies the held-out ridge-probe analysis from
``four_layer_path_delay_decoding.py`` to the usable 4L, 6L, and 7L models.  For
every pre-final residual path, it compares three notions of path delay:

1. Naive subset sum: add the late-position mean-profile peak offset of every
   earlier layer whose write branch appears in the path.
2. Greedy routed delay: starting from a late position, follow the actual
   row-wise attention argmax backward through the selected layers.  This
   accounts for the fact that a composed path uses earlier heads at earlier
   intermediate positions, where their preferred offset can differ.
3. Decoded delay: the strongest held-out path-only R^2 near the naive subset
   prediction when predicting x[t-d] from path(t).  Incremental R^2 beyond
   x[t] is retained as a control but is not used to locate the peak, because
   the lag-40 autocorrelation makes the current-value baseline unusually strong
   at delays 40 and 80.

The 5L checkpoint is deliberately excluded: its architectural final head is
not the lag locator, so its pre-final path basis is not comparable to the other
models' final-locator input.

Usage:
    python cross_model_path_delay_decoding.py --no-show
    python cross_model_path_delay_decoding.py --quick --no-show
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch

from cross_model_raw_head1_path_test import (
    MODEL_SPECS,
    load_attention_only_model,
)
from four_layer_path_delay_decoding import (
    collect_path_examples,
    expected_subset_delays,
    make_balanced_inputs,
    parse_int_tuple,
    regression_metrics,
    ridge_predict,
)


DEFAULT_MODELS = ("4L", "6L", "7L")


def parse_models(text: str) -> tuple[str, ...]:
    models = tuple(part.strip() for part in text.split(",") if part.strip())
    unknown = [model for model in models if model not in MODEL_SPECS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown model(s): {unknown}")
    if "5L" in models:
        raise argparse.ArgumentTypeError(
            "5L is excluded because its architectural final head is not the lag locator"
        )
    if not models:
        raise argparse.ArgumentTypeError("expected at least one model")
    return models


@torch.no_grad()
def greedy_routed_delay_modes(
    model,
    inputs: torch.Tensor,
    query_start: int,
    query_stride: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, int]:
    """Follow each selected head's row argmax backward through every path."""
    sequence_length = inputs.shape[1]
    query_positions = torch.arange(
        query_start,
        sequence_length,
        query_stride,
        device=device,
    )
    labels = [
        "".join(bits)
        for bits in itertools.product("01", repeat=model.n_layers - 1)
    ]
    counts = {
        label: torch.zeros(sequence_length, dtype=torch.int64)
        for label in labels
    }

    for batch_start in range(0, len(inputs), batch_size):
        batch = inputs[batch_start : batch_start + batch_size].to(
            device=device,
            dtype=torch.float64,
        )
        _, attentions, _, _ = model(batch)
        batch_indices = torch.arange(len(batch), device=device)[:, None]
        destinations = query_positions[None, :].expand(len(batch), -1)

        for label in labels:
            current_positions = destinations.clone()
            for layer_index in range(model.n_layers - 2, -1, -1):
                if label[layer_index] == "0":
                    continue
                rows = attentions[layer_index][batch_indices, current_positions]
                current_positions = rows.argmax(dim=-1)
            routed_delays = (destinations - current_positions).reshape(-1).cpu()
            counts[label] += torch.bincount(
                routed_delays,
                minlength=sequence_length,
            )

    return {label: int(values.argmax().item()) for label, values in counts.items()}


def local_peak(
    values: np.ndarray,
    center: int,
    radius: int,
) -> tuple[int, float]:
    start = max(0, center - radius)
    stop = min(len(values), center + radius + 1)
    if start >= stop:
        return -1, float("nan")
    local_index = int(np.nanargmax(values[start:stop]))
    index = start + local_index
    return index, float(values[index])


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def decode_one_model(
    model_name: str,
    model,
    train_inputs: torch.Tensor,
    test_inputs: torch.Tensor,
    lags: tuple[int, ...],
    max_delay: int,
    query_start: int,
    query_stride: int,
    batch_size: int,
    ridge_strength: float,
    local_radius: int,
    route_sequences: int,
    device: torch.device,
) -> dict:
    train_features, train_targets, train_current, train_diagnostics = (
        collect_path_examples(
            model,
            train_inputs,
            max_delay,
            query_start,
            query_stride,
            batch_size,
            device,
        )
    )
    test_features, test_targets, test_current, test_diagnostics = (
        collect_path_examples(
            model,
            test_inputs,
            max_delay,
            query_start,
            query_stride,
            batch_size,
            device,
        )
    )
    path_labels = train_diagnostics["path_labels"]
    if path_labels != test_diagnostics["path_labels"]:
        raise RuntimeError(f"{model_name}: train/test path labels differ")

    baseline_predictions = ridge_predict(
        train_current,
        train_targets,
        test_current,
        ridge_strength,
    )
    baseline_r2, baseline_correlation = regression_metrics(
        test_targets,
        baseline_predictions,
    )

    path_r2 = np.empty((len(path_labels), max_delay + 1))
    path_correlation = np.empty_like(path_r2)
    incremental_r2 = np.empty_like(path_r2)
    augmented_r2 = np.empty_like(path_r2)

    for path_index, label in enumerate(path_labels):
        path_predictions = ridge_predict(
            train_features[label],
            train_targets,
            test_features[label],
            ridge_strength,
        )
        path_r2[path_index], path_correlation[path_index] = regression_metrics(
            test_targets,
            path_predictions,
        )

        augmented_train = np.concatenate(
            [train_current, train_features[label]], axis=1
        )
        augmented_test = np.concatenate(
            [test_current, test_features[label]], axis=1
        )
        augmented_predictions = ridge_predict(
            augmented_train,
            train_targets,
            augmented_test,
            ridge_strength,
        )
        augmented_r2[path_index], _ = regression_metrics(
            test_targets,
            augmented_predictions,
        )
        incremental_r2[path_index] = augmented_r2[path_index] - baseline_r2

    early_peaks = train_diagnostics["early_attention_mean_profile_peaks"]
    naive_delays = expected_subset_delays(path_labels, early_peaks)
    greedy_delays = greedy_routed_delay_modes(
        model,
        train_inputs[: min(route_sequences, len(train_inputs))],
        query_start,
        query_stride,
        batch_size,
        device,
    )

    summaries = []
    delay_rows = []
    for path_index, label in enumerate(path_labels):
        naive = naive_delays[label]
        decoded_delay, decoded_value = local_peak(
            path_r2[path_index],
            naive,
            local_radius,
        )
        global_delay = int(np.nanargmax(path_r2[path_index]))
        summaries.append(
            {
                "model": model_name,
                "path": label,
                "naive_subset_sum_delay": naive,
                "greedy_routed_delay": greedy_delays[label],
                "decoded_local_peak_delay": decoded_delay,
                "decoded_local_peak_path_r2": decoded_value,
                "decoded_local_peak_incremental_r2": float(
                    incremental_r2[path_index, decoded_delay]
                ),
                "path_r2_at_naive_delay": float(path_r2[path_index, naive]),
                "decoded_peak_advantage_over_naive_r2": float(
                    decoded_value - path_r2[path_index, naive]
                ),
                "decoded_global_peak_delay": global_delay,
                "decoded_global_peak_path_r2": float(
                    path_r2[path_index, global_delay]
                ),
                "decoded_minus_naive": decoded_delay - naive,
                "decoded_minus_greedy": decoded_delay - greedy_delays[label],
            }
        )
        for delay in range(max_delay + 1):
            delay_rows.append(
                {
                    "model": model_name,
                    "path": label,
                    "delay": delay,
                    "path_only_r2": float(path_r2[path_index, delay]),
                    "path_only_pearson_r": float(
                        path_correlation[path_index, delay]
                    ),
                    "current_only_r2": float(baseline_r2[delay]),
                    "current_only_pearson_r": float(
                        baseline_correlation[delay]
                    ),
                    "incremental_r2_over_current": float(
                        incremental_r2[path_index, delay]
                    ),
                }
            )

    return {
        "model": model_name,
        "lags": list(lags),
        "path_labels": path_labels,
        "early_attention_peaks": early_peaks,
        "naive_delays": naive_delays,
        "greedy_delays": greedy_delays,
        "path_r2": path_r2,
        "path_correlation": path_correlation,
        "incremental_r2": incremental_r2,
        "baseline_r2": baseline_r2,
        "baseline_correlation": baseline_correlation,
        "summaries": summaries,
        "delay_rows": delay_rows,
        "train_diagnostics": train_diagnostics,
        "test_diagnostics": test_diagnostics,
    }


def symmetric_norm(arrays: list[np.ndarray], minimum_span: float = 0.01):
    extent = max(
        minimum_span,
        max(float(np.nanmax(np.abs(array))) for array in arrays),
    )
    return TwoSlopeNorm(vmin=-extent, vcenter=0.0, vmax=extent)


def plot_model_heatmap(
    result: dict,
    path_norm: TwoSlopeNorm,
    incremental_norm: TwoSlopeNorm,
    output_path: Path,
) -> None:
    labels = result["path_labels"]
    figure_height = max(5.0, 0.24 * len(labels) + 2.0)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(19, figure_height),
        constrained_layout=True,
    )
    panels = (
        (result["path_r2"], path_norm, "Path-only decoding", r"held-out $R^2$"),
        (
            result["incremental_r2"],
            incremental_norm,
            "Additional decoding beyond x[t]",
            r"held-out incremental $R^2$",
        ),
    )
    for axis, (values, norm, title, colorbar_label) in zip(axes, panels):
        image = axis.imshow(
            values,
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu_r",
            norm=norm,
            extent=(-0.5, values.shape[1] - 0.5, len(labels) - 0.5, -0.5),
        )
        for row_index, label in enumerate(labels):
            naive = result["naive_delays"][label]
            greedy = result["greedy_delays"][label]
            if 0 <= naive < values.shape[1]:
                axis.scatter(
                    naive,
                    row_index,
                    marker="o",
                    s=28,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=1.0,
                )
            if 0 <= greedy < values.shape[1]:
                axis.scatter(
                    greedy,
                    row_index,
                    marker="x",
                    s=24,
                    color="black",
                    linewidths=1.0,
                )
        axis.set_yticks(np.arange(len(labels)), labels)
        axis.set_xlabel("prior-value delay d in x[t-d]")
        axis.set_ylabel("residual path")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, label=colorbar_label)
    peaks = ", ".join(str(value) for value in result["early_attention_peaks"])
    figure.suptitle(
        f"{result['model']} path-delay decoding\n"
        f"early-head late-position peaks: [{peaks}]; "
        "circle = naive sum, × = routed argmax delay"
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_cross_model_summary(
    results: list[dict],
    minimum_path_r2: float,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    all_rows = [row for result in results for row in result["summaries"]]
    qualifying = [
        row
        for row in all_rows
        if row["path"].count("1") > 0
        and row["decoded_local_peak_path_r2"] >= minimum_path_r2
    ]
    maximum_delay = max(
        max(
            row["naive_subset_sum_delay"],
            row["greedy_routed_delay"],
            row["decoded_local_peak_delay"],
        )
        for row in qualifying
    )

    for result, color in zip(results, colors):
        rows = [row for row in qualifying if row["model"] == result["model"]]
        sizes = [35 + 160 * row["decoded_local_peak_path_r2"] for row in rows]
        axes[0].scatter(
            [row["naive_subset_sum_delay"] for row in rows],
            [row["decoded_local_peak_delay"] for row in rows],
            s=sizes,
            alpha=0.68,
            color=color,
            label=result["model"],
        )
        axes[1].scatter(
            [row["greedy_routed_delay"] for row in rows],
            [row["decoded_local_peak_delay"] for row in rows],
            s=sizes,
            alpha=0.68,
            color=color,
            label=result["model"],
        )

    for axis, xlabel in zip(
        axes[:2],
        ("naive subset-sum delay", "greedy routed delay"),
    ):
        axis.plot([0, maximum_delay], [0, maximum_delay], color="black", linewidth=1)
        axis.set_xlim(-1, maximum_delay + 1)
        axis.set_ylim(-1, maximum_delay + 1)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("strongest nearby decoded delay")
        axis.grid(alpha=0.2)
    axes[0].set_title("Do fixed head offsets add?")
    axes[1].set_title("Does position-aware routing explain deviations?")
    axes[0].legend()

    error_values = np.arange(-5, 6)
    width = 0.8 / len(results)
    for model_index, (result, color) in enumerate(zip(results, colors)):
        rows = [row for row in qualifying if row["model"] == result["model"]]
        errors = np.asarray([row["decoded_minus_naive"] for row in rows])
        fractions = np.asarray([(errors == value).mean() for value in error_values])
        positions = error_values + (model_index - (len(results) - 1) / 2) * width
        axes[2].bar(
            positions,
            fractions,
            width=width,
            color=color,
            alpha=0.75,
            label=result["model"],
        )
    axes[2].set_xlabel("decoded delay − naive subset sum")
    axes[2].set_ylabel("fraction of informative paths")
    axes[2].set_xticks(error_values)
    axes[2].set_title(
        f"Offset-error distribution (path-only R² ≥ {minimum_path_r2:g})"
    )
    axes[2].grid(axis="y", alpha=0.2)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=parse_models, default=DEFAULT_MODELS)
    parser.add_argument("--lags", type=parse_int_tuple, default=(40,))
    parser.add_argument("--n-train-per-lag", type=int, default=96)
    parser.add_argument("--n-test-per-lag", type=int, default=96)
    parser.add_argument("--sequence-length", type=int, default=200)
    parser.add_argument("--rho", type=float, default=0.9)
    parser.add_argument("--max-delay", type=int, default=100)
    parser.add_argument("--query-start", type=int, default=120)
    parser.add_argument("--query-stride", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--route-sequences", type=int, default=32)
    parser.add_argument("--ridge-strength", type=float, default=1e-4)
    parser.add_argument("--local-radius", type=int, default=4)
    parser.add_argument("--minimum-path-r2", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=98117)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment_outputs/cross_model_path_delay_decoding"),
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.n_train_per_lag = min(args.n_train_per_lag, 16)
        args.n_test_per_lag = min(args.n_test_per_lag, 16)
        args.route_sequences = min(args.route_sequences, 8)
        args.query_stride = max(args.query_stride, 4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for model_index, model_name in enumerate(args.models):
        n_layers, checkpoint = MODEL_SPECS[model_name]
        print(f"\nRunning {model_name}: {checkpoint}")
        model = load_attention_only_model(n_layers, checkpoint, device)
        train_inputs = make_balanced_inputs(
            args.lags,
            args.n_train_per_lag,
            args.sequence_length,
            args.rho,
            args.seed + 100_000 * model_index,
        )
        test_inputs = make_balanced_inputs(
            args.lags,
            args.n_test_per_lag,
            args.sequence_length,
            args.rho,
            args.seed + 1_000_000 + 100_000 * model_index,
        )
        result = decode_one_model(
            model_name,
            model,
            train_inputs,
            test_inputs,
            args.lags,
            args.max_delay,
            args.query_start,
            args.query_stride,
            args.batch_size,
            args.ridge_strength,
            args.local_radius,
            args.route_sequences,
            device,
        )
        results.append(result)
        print(
            f"  early peaks={result['early_attention_peaks']}; "
            f"paths={len(result['path_labels'])}; "
            "reconstruction error="
            f"{max(result['train_diagnostics']['maximum_path_reconstruction_relative_error'], result['test_diagnostics']['maximum_path_reconstruction_relative_error']):.3e}"
        )
        del model, train_inputs, test_inputs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    common_path_norm = symmetric_norm([result["path_r2"] for result in results])
    common_incremental_norm = symmetric_norm(
        [result["incremental_r2"] for result in results]
    )
    for result in results:
        plot_model_heatmap(
            result,
            common_path_norm,
            common_incremental_norm,
            args.output_dir / f"{result['model']}_incremental_delay_decoding.png",
        )
    plot_cross_model_summary(
        results,
        args.minimum_path_r2,
        args.output_dir / "cross_model_delay_summary.png",
    )

    summary_rows = [row for result in results for row in result["summaries"]]
    delay_rows = [row for result in results for row in result["delay_rows"]]
    save_csv(args.output_dir / "path_delay_summary.csv", summary_rows)
    save_csv(args.output_dir / "path_delay_curves.csv", delay_rows)

    arrays = {}
    json_models = {}
    for result in results:
        name = result["model"]
        arrays[f"{name}_labels"] = np.asarray(result["path_labels"])
        arrays[f"{name}_path_r2"] = result["path_r2"]
        arrays[f"{name}_incremental_r2"] = result["incremental_r2"]
        arrays[f"{name}_baseline_r2"] = result["baseline_r2"]
        json_models[name] = {
            "path_labels": result["path_labels"],
            "early_attention_peaks": result["early_attention_peaks"],
            "naive_delays": result["naive_delays"],
            "greedy_delays": result["greedy_delays"],
            "train_diagnostics": result["train_diagnostics"],
            "test_diagnostics": result["test_diagnostics"],
        }
    np.savez_compressed(args.output_dir / "path_delay_arrays.npz", **arrays)

    qualifying = [
        row
        for row in summary_rows
        if row["path"].count("1") > 0
        and row["decoded_local_peak_path_r2"] >= args.minimum_path_r2
    ]
    aggregate = {}
    for model_name in args.models:
        rows = [row for row in qualifying if row["model"] == model_name]
        naive_errors = np.asarray([row["decoded_minus_naive"] for row in rows])
        greedy_errors = np.asarray([row["decoded_minus_greedy"] for row in rows])
        aggregate[model_name] = {
            "n_informative_paths": len(rows),
            "fraction_exact_naive": float(np.mean(naive_errors == 0)),
            "fraction_within_one_naive": float(np.mean(np.abs(naive_errors) <= 1)),
            "mean_absolute_naive_error": float(np.mean(np.abs(naive_errors))),
            "fraction_exact_greedy": float(np.mean(greedy_errors == 0)),
            "fraction_within_one_greedy": float(np.mean(np.abs(greedy_errors) <= 1)),
            "mean_absolute_greedy_error": float(np.mean(np.abs(greedy_errors))),
            "naive_error_counts": {
                str(error): int(np.sum(naive_errors == error))
                for error in sorted(set(naive_errors.tolist()))
            },
        }

    metadata = {
        "models": list(args.models),
        "excluded_model": "5L: architectural final head is not the lag locator",
        "lags": list(args.lags),
        "n_train_sequences_per_lag": args.n_train_per_lag,
        "n_test_sequences_per_lag": args.n_test_per_lag,
        "sequence_length": args.sequence_length,
        "max_delay": args.max_delay,
        "query_start": args.query_start,
        "query_stride": args.query_stride,
        "ridge_strength_fraction_of_n_train": args.ridge_strength,
        "local_peak_radius": args.local_radius,
        "informative_path_threshold_path_only_r2": args.minimum_path_r2,
        "aggregate": aggregate,
        "model_details": json_models,
    }
    with (args.output_dir / "run_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)

    print("\nCross-model offset agreement among informative paths:")
    for model_name in args.models:
        item = aggregate[model_name]
        print(
            f"  {model_name}: n={item['n_informative_paths']}; "
            f"naive exact={item['fraction_exact_naive']:.1%}, "
            f"within1={item['fraction_within_one_naive']:.1%}, "
            f"MAE={item['mean_absolute_naive_error']:.2f}; "
            f"routed exact={item['fraction_exact_greedy']:.1%}, "
            f"within1={item['fraction_within_one_greedy']:.1%}, "
            f"MAE={item['mean_absolute_greedy_error']:.2f}"
        )
    print(f"\nWrote results to {args.output_dir}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
