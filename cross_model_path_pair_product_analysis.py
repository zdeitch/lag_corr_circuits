"""Test whether final-head path pairs act like lagged scalar products.

For the sharp 6L and 7L attention-only models, each frozen-attention residual
path has an effective delay equal to the sum of its selected earlier-head
offsets.  This script asks whether a final-head path-pair contribution is well
approximated by a weighted product of the corresponding delayed input scalars.

For every model and data lag:

* decompose the residual entering the final layer into exact paths;
* compute every path-pair raw-logit contribution at D=L-1 and D=L;
* fit contribution ~= intercept + slope * delayed_scalar_product on training
  sequences and report held-out R^2 on disjoint sequences;
* group the correct-minus-adjacent-wrong contribution by
  key_delay - query_delay.

At the correct attention offset D=L-1, path pairs with delay difference +1
compare original scalars separated by L.  At the adjacent wrong offset D=L,
same-delay path pairs compare original scalars separated by L.

Usage:
    python cross_model_path_pair_product_analysis.py --no-show
    python cross_model_path_pair_product_analysis.py --quick --no-show
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch

from cross_model_raw_head1_path_test import (
    MODEL_SPECS,
    load_attention_only_model,
)
from four_layer_attention_path_analysis import expand_prefinal_paths
from four_layer_path_delay_decoding import (
    expected_subset_delays,
    make_balanced_inputs,
    parse_int_tuple,
)


DEFAULT_MODELS = ("6L", "7L")


def parse_models(text: str) -> tuple[str, ...]:
    models = tuple(part.strip() for part in text.split(",") if part.strip())
    allowed = {"6L", "7L"}
    unknown = [model for model in models if model not in allowed]
    if unknown or not models:
        raise argparse.ArgumentTypeError(
            f"choose one or more of {sorted(allowed)}; received {models}"
        )
    return models


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def infer_early_attention_peaks(
    model,
    inputs: torch.Tensor,
    query_start: int,
    query_stride: int,
    maximum_offset: int,
    batch_size: int,
    device: torch.device,
) -> list[int]:
    query_positions = torch.arange(
        max(query_start, maximum_offset),
        inputs.shape[1],
        query_stride,
        device=device,
    )
    offsets = torch.arange(maximum_offset + 1, device=device)
    profile_sum = torch.zeros(
        model.n_layers - 1,
        maximum_offset + 1,
        dtype=torch.float64,
        device=device,
    )
    count = 0
    for start in range(0, len(inputs), batch_size):
        batch = inputs[start : start + batch_size].to(
            device=device,
            dtype=torch.float64,
        )
        _, attentions, _, _ = model(batch)
        for layer_index in range(model.n_layers - 1):
            values = attentions[layer_index][
                :,
                query_positions[:, None],
                query_positions[:, None] - offsets[None, :],
            ]
            profile_sum[layer_index] += values.sum(dim=(0, 1))
        count += len(batch) * len(query_positions)
    return (profile_sum / count).argmax(dim=1).cpu().tolist()


def empty_stats(path_count: int) -> dict[str, np.ndarray | int]:
    shape = (path_count, path_count)
    return {
        "n": 0,
        "sum_x": np.zeros(shape),
        "sum_y": np.zeros(shape),
        "sum_x2": np.zeros(shape),
        "sum_y2": np.zeros(shape),
        "sum_xy": np.zeros(shape),
    }


def update_stats(
    stats: dict[str, np.ndarray | int],
    predictor: torch.Tensor,
    contribution: torch.Tensor,
) -> None:
    # Both tensors: batch x sampled-query-position x query-path x key-path.
    stats["n"] = int(stats["n"]) + predictor.shape[0] * predictor.shape[1]
    reduce_dims = (0, 1)
    stats["sum_x"] += predictor.sum(dim=reduce_dims).cpu().numpy()
    stats["sum_y"] += contribution.sum(dim=reduce_dims).cpu().numpy()
    stats["sum_x2"] += predictor.square().sum(dim=reduce_dims).cpu().numpy()
    stats["sum_y2"] += contribution.square().sum(dim=reduce_dims).cpu().numpy()
    stats["sum_xy"] += (predictor * contribution).sum(dim=reduce_dims).cpu().numpy()


def fit_scalar_product_regression(
    train: dict[str, np.ndarray | int],
    test: dict[str, np.ndarray | int],
) -> dict[str, np.ndarray]:
    n_train = int(train["n"])
    mean_x = train["sum_x"] / n_train
    mean_y = train["sum_y"] / n_train
    centered_x2 = train["sum_x2"] - np.square(train["sum_x"]) / n_train
    centered_xy = train["sum_xy"] - train["sum_x"] * train["sum_y"] / n_train
    slope = centered_xy / np.maximum(centered_x2, 1e-30)
    intercept = mean_y - slope * mean_x

    n_test = int(test["n"])
    sse = (
        test["sum_y2"]
        - 2 * intercept * test["sum_y"]
        - 2 * slope * test["sum_xy"]
        + n_test * np.square(intercept)
        + 2 * intercept * slope * test["sum_x"]
        + np.square(slope) * test["sum_x2"]
    )
    sst = test["sum_y2"] - np.square(test["sum_y"]) / n_test
    heldout_r2 = 1.0 - sse / np.maximum(sst, 1e-30)

    centered_test_x2 = test["sum_x2"] - np.square(test["sum_x"]) / n_test
    centered_test_y2 = test["sum_y2"] - np.square(test["sum_y"]) / n_test
    centered_test_xy = (
        test["sum_xy"] - test["sum_x"] * test["sum_y"] / n_test
    )
    heldout_correlation = centered_test_xy / np.sqrt(
        np.maximum(centered_test_x2 * centered_test_y2, 1e-30)
    )
    return {
        "slope": slope,
        "intercept": intercept,
        "heldout_r2": heldout_r2,
        "heldout_correlation": heldout_correlation,
    }


@torch.no_grad()
def collect_split(
    model,
    inputs: torch.Tensor,
    lag: int,
    path_labels: list[str],
    path_delays: np.ndarray,
    query_start: int,
    query_stride: int,
    batch_size: int,
    device: torch.device,
    retain_sequence_margins: bool,
) -> dict:
    correct_offset = lag - 1
    wrong_offset = lag
    largest_required_history = wrong_offset + int(path_delays.max())
    first_query = max(query_start, largest_required_history)
    query_positions = torch.arange(
        first_query,
        inputs.shape[1],
        query_stride,
        device=device,
    )
    if len(query_positions) == 0:
        raise ValueError(
            f"no query positions remain for lag={lag}, max path delay={path_delays.max()}"
        )

    path_delay_tensor = torch.tensor(path_delays, device=device)
    stats = {
        correct_offset: empty_stats(len(path_labels)),
        wrong_offset: empty_stats(len(path_labels)),
    }
    sequence_correct = []
    sequence_wrong = []
    maximum_residual_error = 0.0
    maximum_logit_error = 0.0

    for start in range(0, len(inputs), batch_size):
        batch = inputs[start : start + batch_size].to(
            device=device,
            dtype=torch.float64,
        )
        _, attentions, post_attention, post_mlp = model(batch)
        if any(not torch.equal(a, b) for a, b in zip(post_attention, post_mlp)):
            raise RuntimeError("checkpoint is not attention-only")
        embedding = model.W_r(batch.unsqueeze(-1))
        path_dict = expand_prefinal_paths(
            embedding,
            attentions,
            model,
            post_attention[: model.n_layers - 1],
        )
        if list(path_dict) != path_labels:
            raise RuntimeError("path labels changed between batches")
        paths = torch.stack(list(path_dict.values()))
        final_input = post_attention[-2]
        residual_error = (
            (paths.sum(dim=0) - final_input).norm()
            / final_input.norm().clamp_min(1e-30)
        )
        maximum_residual_error = max(
            maximum_residual_error,
            float(residual_error.item()),
        )

        query_matrix, key_matrix, _, _ = model.layers[-1][0]
        positions = torch.arange(inputs.shape[1], device=device).view(1, 1, -1)
        path_queries = model.apply_rope(paths @ query_matrix, positions)
        path_keys = model.apply_rope(paths @ key_matrix, positions)
        full_queries = model.apply_rope(
            final_input @ query_matrix,
            positions.squeeze(0),
        )
        full_keys = model.apply_rope(
            final_input @ key_matrix,
            positions.squeeze(0),
        )
        selected_queries = path_queries[:, :, query_positions, :]

        contribution_by_offset = {}
        for offset in (correct_offset, wrong_offset):
            selected_keys = path_keys[:, :, query_positions - offset, :]
            contribution = torch.einsum(
                "pbnh,qbnh->bnpq",
                selected_queries,
                selected_keys,
            ) / math.sqrt(model.d_head)
            contribution_by_offset[offset] = contribution

            query_source_indices = (
                query_positions[:, None] - path_delay_tensor[None, :]
            )
            key_source_indices = (
                query_positions[:, None]
                - offset
                - path_delay_tensor[None, :]
            )
            query_values = batch[:, query_source_indices]
            key_values = batch[:, key_source_indices]
            scalar_products = query_values.unsqueeze(-1) * key_values.unsqueeze(-2)
            update_stats(stats[offset], scalar_products, contribution)

            direct = (
                full_queries[:, query_positions, :]
                * full_keys[:, query_positions - offset, :]
            ).sum(dim=-1) / math.sqrt(model.d_head)
            logit_error = (contribution.sum(dim=(-1, -2)) - direct).abs().max()
            maximum_logit_error = max(
                maximum_logit_error,
                float(logit_error.item()),
            )

        if retain_sequence_margins:
            sequence_correct.append(
                contribution_by_offset[correct_offset].mean(dim=1).cpu().numpy()
            )
            sequence_wrong.append(
                contribution_by_offset[wrong_offset].mean(dim=1).cpu().numpy()
            )

    output = {
        "stats": stats,
        "n_query_positions_per_sequence": len(query_positions),
        "first_query_position": int(query_positions[0].item()),
        "maximum_residual_reconstruction_error": maximum_residual_error,
        "maximum_raw_logit_reconstruction_error": maximum_logit_error,
    }
    if retain_sequence_margins:
        output["sequence_correct"] = np.concatenate(sequence_correct, axis=0)
        output["sequence_wrong"] = np.concatenate(sequence_wrong, axis=0)
    return output


def path_pair_rows(
    model_name: str,
    lag: int,
    path_labels: list[str],
    path_delays: np.ndarray,
    train_result: dict,
    test_result: dict,
) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    correct_offset = lag - 1
    wrong_offset = lag
    correct_fit = fit_scalar_product_regression(
        train_result["stats"][correct_offset],
        test_result["stats"][correct_offset],
    )
    wrong_fit = fit_scalar_product_regression(
        train_result["stats"][wrong_offset],
        test_result["stats"][wrong_offset],
    )
    sequence_correct = test_result["sequence_correct"]
    sequence_wrong = test_result["sequence_wrong"]
    sequence_margin = sequence_correct - sequence_wrong
    mean_correct = sequence_correct.mean(axis=0)
    mean_wrong = sequence_wrong.mean(axis=0)
    mean_margin = sequence_margin.mean(axis=0)

    rows = []
    for query_index, query_label in enumerate(path_labels):
        for key_index, key_label in enumerate(path_labels):
            query_delay = int(path_delays[query_index])
            key_delay = int(path_delays[key_index])
            rows.append(
                {
                    "model": model_name,
                    "lag": lag,
                    "correct_offset": correct_offset,
                    "wrong_offset": wrong_offset,
                    "query_path": query_label,
                    "key_path": key_label,
                    "query_delay": query_delay,
                    "key_delay": key_delay,
                    "delay_difference_key_minus_query": key_delay - query_delay,
                    "source_separation_at_correct": (
                        correct_offset + key_delay - query_delay
                    ),
                    "mean_correct_contribution": float(
                        mean_correct[query_index, key_index]
                    ),
                    "mean_wrong_contribution": float(
                        mean_wrong[query_index, key_index]
                    ),
                    "mean_correct_minus_wrong_contribution": float(
                        mean_margin[query_index, key_index]
                    ),
                    "sd_margin_across_sequences": float(
                        sequence_margin[:, query_index, key_index].std(ddof=1)
                    ),
                    "correct_product_slope": float(
                        correct_fit["slope"][query_index, key_index]
                    ),
                    "correct_product_heldout_r2": float(
                        correct_fit["heldout_r2"][query_index, key_index]
                    ),
                    "correct_product_heldout_correlation": float(
                        correct_fit["heldout_correlation"][query_index, key_index]
                    ),
                    "wrong_product_slope": float(
                        wrong_fit["slope"][query_index, key_index]
                    ),
                    "wrong_product_heldout_r2": float(
                        wrong_fit["heldout_r2"][query_index, key_index]
                    ),
                }
            )
    return rows, mean_margin, correct_fit["heldout_r2"], sequence_margin


def grouped_rows(
    model_name: str,
    lag: int,
    path_delays: np.ndarray,
    mean_margin: np.ndarray,
    product_r2: np.ndarray,
    sequence_margin: np.ndarray,
) -> list[dict]:
    differences = path_delays[None, :] - path_delays[:, None]
    rows = []
    for difference in range(int(differences.min()), int(differences.max()) + 1):
        mask = differences == difference
        if not mask.any():
            continue
        sequence_totals = sequence_margin[:, mask].sum(axis=1)
        values = mean_margin[mask]
        fidelity = product_r2[mask]
        rows.append(
            {
                "model": model_name,
                "lag": lag,
                "delay_difference_key_minus_query": difference,
                "n_path_pairs": int(mask.sum()),
                "summed_mean_margin": float(values.sum()),
                "mean_margin_per_pair": float(values.mean()),
                "summed_absolute_mean_margin": float(np.abs(values).sum()),
                "sequence_total_margin_sd": float(sequence_totals.std(ddof=1)),
                "sequence_total_margin_p05": float(
                    np.quantile(sequence_totals, 0.05)
                ),
                "sequence_total_margin_p95": float(
                    np.quantile(sequence_totals, 0.95)
                ),
                "median_correct_product_heldout_r2": float(np.median(fidelity)),
                "mean_correct_product_heldout_r2": float(np.mean(fidelity)),
                "fraction_pairs_product_r2_above_0_5": float(
                    np.mean(fidelity > 0.5)
                ),
            }
        )
    return rows


def plot_grouped_results(
    models: tuple[str, ...],
    lags: tuple[int, ...],
    rows: list[dict],
    difference_limit: int,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        len(models),
        3,
        figsize=(17, 4.2 * len(models)),
        constrained_layout=True,
        squeeze=False,
    )
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(lags)))
    metric_specs = (
        ("summed_mean_margin", "summed margin", "Total contribution to locator"),
        (
            "mean_margin_per_pair",
            "mean margin per pair",
            "Average pair contribution",
        ),
        (
            "median_correct_product_heldout_r2",
            r"median held-out $R^2$",
            "Weighted-product fidelity at correct offset",
        ),
    )
    for model_index, model_name in enumerate(models):
        for column, (metric, ylabel, title) in enumerate(metric_specs):
            axis = axes[model_index, column]
            lag_curves = []
            common_differences = None
            for lag, color in zip(lags, colors):
                selected = sorted(
                    (
                        row
                        for row in rows
                        if row["model"] == model_name and row["lag"] == lag
                    ),
                    key=lambda row: row["delay_difference_key_minus_query"],
                )
                differences = np.asarray(
                    [row["delay_difference_key_minus_query"] for row in selected]
                )
                values = np.asarray([row[metric] for row in selected])
                keep = np.abs(differences) <= difference_limit
                differences = differences[keep]
                values = values[keep]
                axis.plot(
                    differences,
                    values,
                    marker="o",
                    markersize=3,
                    linewidth=1,
                    alpha=0.65,
                    color=color,
                    label=f"lag {lag}",
                )
                lag_curves.append(values)
                common_differences = differences
            if lag_curves and all(len(curve) == len(lag_curves[0]) for curve in lag_curves):
                axis.plot(
                    common_differences,
                    np.mean(lag_curves, axis=0),
                    color="black",
                    linewidth=2,
                    label="mean across lags",
                )
            axis.axvline(1, color="tab:red", linestyle="--", linewidth=1.2)
            axis.axvline(0, color="0.45", linestyle=":", linewidth=1.1)
            axis.axhline(0, color="0.7", linewidth=0.8)
            axis.set_xlim(-difference_limit, difference_limit)
            axis.set_xlabel("key-path delay − query-path delay")
            axis.set_ylabel(ylabel)
            axis.set_title(f"{model_name}: {title}")
            axis.grid(alpha=0.2)
            if column == 0:
                axis.legend(fontsize=8)
    figure.suptitle(
        "Path-pair evidence grouped by effective delay difference\n"
        "red dashed: +1 pairs match the true lag at D=L−1; gray dotted: "
        "same-delay pairs match it at D=L"
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_lag_matrix(
    model_name: str,
    lag: int,
    labels: list[str],
    delays: np.ndarray,
    margin: np.ndarray,
    product_r2: np.ndarray,
    output_path: Path,
) -> None:
    order = np.asarray(sorted(range(len(labels)), key=lambda index: (delays[index], labels[index])))
    ordered_delays = delays[order]
    ordered_margin = margin[np.ix_(order, order)]
    ordered_r2 = product_r2[np.ix_(order, order)]
    difference = ordered_delays[None, :] - ordered_delays[:, None]

    margin_extent = max(float(np.abs(ordered_margin).max()), 1e-6)
    figure, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
    margin_image = axes[0].imshow(
        ordered_margin,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-margin_extent, vcenter=0, vmax=margin_extent),
        interpolation="nearest",
        aspect="equal",
    )
    fidelity_image = axes[1].imshow(
        ordered_r2,
        cmap="viridis",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="equal",
    )
    for axis in axes:
        axis.contour(difference == 1, levels=[0.5], colors=["red"], linewidths=0.9)
        axis.contour(
            difference == 0,
            levels=[0.5],
            colors=["white"],
            linewidths=0.7,
            linestyles="dotted",
        )
        tick_step = max(1, len(labels) // 8)
        ticks = np.arange(0, len(labels), tick_step)
        axis.set_xticks(ticks, ordered_delays[ticks])
        axis.set_yticks(ticks, ordered_delays[ticks])
        axis.set_xlabel("key path, ordered by decoded delay")
        axis.set_ylabel("query path, ordered by decoded delay")
    axes[0].set_title("Correct D=L−1 minus adjacent wrong D=L contribution")
    axes[1].set_title("Held-out weighted-scalar-product fidelity")
    figure.colorbar(margin_image, ax=axes[0], label="mean raw-score margin")
    figure.colorbar(fidelity_image, ax=axes[1], label=r"held-out $R^2$")
    figure.suptitle(
        f"{model_name}, data lag {lag}: red contour = delay difference +1; "
        "white dotted = difference 0"
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=parse_models, default=DEFAULT_MODELS)
    parser.add_argument("--lags", type=parse_int_tuple, default=(30, 40, 50))
    parser.add_argument("--n-train-per-lag", type=int, default=64)
    parser.add_argument("--n-test-per-lag", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=200)
    parser.add_argument("--rho", type=float, default=0.9)
    parser.add_argument("--query-start", type=int, default=120)
    parser.add_argument("--query-stride", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--peak-sequences", type=int, default=24)
    parser.add_argument("--maximum-head-offset", type=int, default=100)
    parser.add_argument("--difference-limit", type=int, default=12)
    parser.add_argument("--seed", type=int, default=431921)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment_outputs/cross_model_path_pair_products"),
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.n_train_per_lag = min(args.n_train_per_lag, 12)
        args.n_test_per_lag = min(args.n_test_per_lag, 12)
        args.peak_sequences = min(args.peak_sequences, 8)
        args.query_stride = max(args.query_stride, 8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_pair_rows = []
    all_group_rows = []
    matrix_outputs = {}
    diagnostics = {}

    for model_index, model_name in enumerate(args.models):
        n_layers, checkpoint = MODEL_SPECS[model_name]
        print(f"\nRunning {model_name}: {checkpoint}")
        model = load_attention_only_model(n_layers, checkpoint, device)

        peak_inputs = make_balanced_inputs(
            args.lags,
            args.peak_sequences,
            args.sequence_length,
            args.rho,
            args.seed + 100_000 * model_index,
        )
        early_peaks = infer_early_attention_peaks(
            model,
            peak_inputs,
            args.query_start,
            args.query_stride,
            args.maximum_head_offset,
            args.batch_size,
            device,
        )
        labels = [
            format(index, f"0{n_layers - 1}b")
            for index in range(2 ** (n_layers - 1))
        ]
        delay_map = expected_subset_delays(labels, early_peaks)
        path_delays = np.asarray([delay_map[label] for label in labels])
        print(f"  early-head peaks: {early_peaks}")
        print(f"  paths: {len(labels)}, maximum composed delay: {path_delays.max()}")

        model_diagnostics = {
            "checkpoint": str(checkpoint),
            "early_attention_peaks": early_peaks,
            "path_delays": delay_map,
            "lags": {},
        }
        for lag_index, lag in enumerate(args.lags):
            train_inputs = make_balanced_inputs(
                (lag,),
                args.n_train_per_lag,
                args.sequence_length,
                args.rho,
                args.seed + 1_000_000 * model_index + 10_000 * lag_index,
            )
            test_inputs = make_balanced_inputs(
                (lag,),
                args.n_test_per_lag,
                args.sequence_length,
                args.rho,
                args.seed
                + 50_000_000
                + 1_000_000 * model_index
                + 10_000 * lag_index,
            )
            train_result = collect_split(
                model,
                train_inputs,
                lag,
                labels,
                path_delays,
                args.query_start,
                args.query_stride,
                args.batch_size,
                device,
                retain_sequence_margins=False,
            )
            test_result = collect_split(
                model,
                test_inputs,
                lag,
                labels,
                path_delays,
                args.query_start,
                args.query_stride,
                args.batch_size,
                device,
                retain_sequence_margins=True,
            )
            pair_rows, margin, product_r2, sequence_margin = path_pair_rows(
                model_name,
                lag,
                labels,
                path_delays,
                train_result,
                test_result,
            )
            group_rows = grouped_rows(
                model_name,
                lag,
                path_delays,
                margin,
                product_r2,
                sequence_margin,
            )
            all_pair_rows.extend(pair_rows)
            all_group_rows.extend(group_rows)
            matrix_outputs[(model_name, lag)] = (margin, product_r2, labels, path_delays)
            model_diagnostics["lags"][str(lag)] = {
                "n_train_sequences": args.n_train_per_lag,
                "n_test_sequences": args.n_test_per_lag,
                "n_query_positions_per_sequence": test_result[
                    "n_query_positions_per_sequence"
                ],
                "first_query_position": test_result["first_query_position"],
                "maximum_residual_reconstruction_error": max(
                    train_result["maximum_residual_reconstruction_error"],
                    test_result["maximum_residual_reconstruction_error"],
                ),
                "maximum_raw_logit_reconstruction_error": max(
                    train_result["maximum_raw_logit_reconstruction_error"],
                    test_result["maximum_raw_logit_reconstruction_error"],
                ),
            }
            plus_one = next(
                row
                for row in group_rows
                if row["delay_difference_key_minus_query"] == 1
            )
            zero = next(
                row
                for row in group_rows
                if row["delay_difference_key_minus_query"] == 0
            )
            print(
                f"  lag {lag}: +1 summed margin={plus_one['summed_mean_margin']:+.3f}, "
                f"diff0={zero['summed_mean_margin']:+.3f}, "
                f"+1 median product R2={plus_one['median_correct_product_heldout_r2']:+.3f}"
            )
        diagnostics[model_name] = model_diagnostics
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_csv(args.output_dir / "path_pair_products.csv", all_pair_rows)
    save_csv(args.output_dir / "delay_difference_groups.csv", all_group_rows)
    with (args.output_dir / "run_metadata.json").open("w") as handle:
        json.dump(
            {
                "models": list(args.models),
                "lags": list(args.lags),
                "rho": args.rho,
                "sequence_length": args.sequence_length,
                "n_train_sequences_per_lag": args.n_train_per_lag,
                "n_test_sequences_per_lag": args.n_test_per_lag,
                "query_start": args.query_start,
                "query_stride": args.query_stride,
                "diagnostics": diagnostics,
            },
            handle,
            indent=2,
        )

    plot_grouped_results(
        args.models,
        args.lags,
        all_group_rows,
        args.difference_limit,
        args.output_dir / "delay_difference_summary.png",
    )
    for (model_name, lag), (margin, product_r2, labels, delays) in matrix_outputs.items():
        plot_lag_matrix(
            model_name,
            lag,
            labels,
            delays,
            margin,
            product_r2,
            args.output_dir / f"{model_name}_lag{lag}_path_pair_matrix.png",
        )

    print(f"\nWrote results to {args.output_dir}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
