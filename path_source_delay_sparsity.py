"""Resolve frozen-attention residual paths into exact source delays.

For an attention-only model, cache the attention matrices from an observed
forward pass.  Conditional on those matrices, every pre-final residual path is
linear.  The scalar-data part of a path can therefore be written exactly as a
sum over original source positions.

This script:

1. builds the exact source-position routing distribution for every path;
2. verifies that it reconstructs the existing vector-valued path expansion;
3. keeps the smallest number of sources accounting for 90%, 95%, and 99% of
   routing mass and measures the resulting path-vector error;
4. separates the leading source-delay terms inside every path and regresses
   each term against the particular x[t-d] from which it was constructed;
5. includes a routing-weight-normalized sanity check and a complete-path probe;
   and
6. compares literal routing mass with observational delay decoding.

Only the data-dependent part of W_r is source-resolved.  W_r's constant bias
has no originating timeseries position and is deliberately excluded.

Examples:
    python path_source_delay_sparsity.py --models 4L --lags 40 --no-show
    python path_source_delay_sparsity.py --models 4L,6L,7L --quick --no-show
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch

from cross_model_raw_head1_path_test import (
    MODEL_SPECS,
    load_attention_only_model,
)
from util import make_dataset_lagset


DEFAULT_MODELS = ("4L", "6L", "7L")


def parse_models(text: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in text.split(",") if part.strip())
    unknown = [name for name in names if name not in MODEL_SPECS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown model(s): {unknown}")
    if "5L" in names:
        raise argparse.ArgumentTypeError(
            "5L is excluded because its architectural final head is not the "
            "lag-programmable locator"
        )
    if not names:
        raise argparse.ArgumentTypeError("expected at least one model")
    return names


def parse_int_tuple(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def parse_float_tuple(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values or any(not 0.0 < value < 1.0 for value in values):
        raise argparse.ArgumentTypeError("thresholds must lie strictly between 0 and 1")
    if tuple(sorted(values)) != values:
        raise argparse.ArgumentTypeError("thresholds must be in ascending order")
    return values


def labels_for_model(model) -> list[str]:
    return [
        "".join(bits)
        for bits in itertools.product("01", repeat=model.n_layers - 1)
    ]


def data_embedding(model, inputs: torch.Tensor) -> torch.Tensor:
    """W_r's scalar-dependent component, excluding its constant bias."""
    direction = model.W_r.weight[:, 0]
    return inputs.unsqueeze(-1) * direction


def path_direction(model, label: str) -> torch.Tensor:
    """Residual-space direction for one scalar-data path (row convention)."""
    direction = model.W_r.weight[:, 0]
    for layer_index, bit in enumerate(label):
        if bit == "0":
            continue
        _, _, value_matrix, output_matrix = model.layers[layer_index][0]
        direction = (direction @ value_matrix) @ output_matrix
    return direction


def direct_data_path(
    model,
    inputs: torch.Tensor,
    attentions: list[torch.Tensor],
    label: str,
) -> torch.Tensor:
    """Construct one data path using the existing vector-valued operations."""
    component = data_embedding(model, inputs)
    for layer_index, bit in enumerate(label):
        if bit == "0":
            continue
        _, _, value_matrix, output_matrix = model.layers[layer_index][0]
        component = (
            attentions[layer_index] @ (component @ value_matrix)
        ) @ output_matrix
    return component


def source_routing_rows(
    attentions: list[torch.Tensor],
    label: str,
    query_positions: torch.Tensor,
) -> torch.Tensor:
    """Exact source weights for selected destination rows of one path.

    If the path selects Layers 2 and 3, its complete sequence-routing matrix is
    A3 @ A2.  We only need selected destination rows, so we start with one-hot
    destination rows and multiply backward through the selected matrices.
    """
    batch_size, sequence_length, _ = attentions[0].shape
    n_queries = len(query_positions)
    routing = torch.zeros(
        batch_size,
        n_queries,
        sequence_length,
        dtype=attentions[0].dtype,
        device=attentions[0].device,
    )
    routing[:, torch.arange(n_queries, device=routing.device), query_positions] = 1.0

    for layer_index in range(len(label) - 1, -1, -1):
        if label[layer_index] == "1":
            routing = torch.bmm(routing, attentions[layer_index])
    return routing


def make_inputs(
    lag: int,
    n_sequences: int,
    sequence_length: int,
    rho: float,
    seed: int,
) -> torch.Tensor:
    inputs, _, sampled_lags = make_dataset_lagset(
        n_sequences,
        sequence_length,
        rho,
        [lag],
        seed=seed,
    )
    if not torch.all(sampled_lags == lag):
        raise RuntimeError("dataset returned an unexpected lag")
    return inputs


def quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability))


@torch.inference_mode()
def collect_one_split(
    model,
    inputs: torch.Tensor,
    lag: int,
    burn_in: int,
    query_stride: int,
    maximum_probe_delay: int,
    thresholds: tuple[float, ...],
    batch_size: int,
    device: torch.device,
    collect_routing_statistics: bool,
) -> dict:
    """Collect path amplitudes, probe targets, and optional routing statistics."""
    sequence_length = inputs.shape[1]
    first_query = max(lag + burn_in, maximum_probe_delay)
    query_positions = torch.arange(
        first_query,
        sequence_length,
        query_stride,
        device=device,
    )
    if len(query_positions) == 0:
        raise ValueError(
            "no query positions remain; reduce burn-in/max probe delay or "
            "increase sequence length"
        )

    labels = labels_for_model(model)
    directions = {label: path_direction(model, label) for label in labels}
    amplitudes: dict[str, list[np.ndarray]] = {label: [] for label in labels}
    component_amplitudes: dict[str, list[np.ndarray]] = {
        label: [] for label in labels
    }
    component_routing_weights: dict[str, list[np.ndarray]] = {
        label: [] for label in labels
    }
    target_chunks: list[np.ndarray] = []

    profile_sums = {
        label: torch.zeros(sequence_length, dtype=torch.float64)
        for label in labels
    }
    profile_observations = Counter()
    support_chunks: dict[str, dict[float, list[np.ndarray]]] = {
        label: {threshold: [] for threshold in thresholds} for label in labels
    }
    top_mass_chunks: dict[str, list[np.ndarray]] = {label: [] for label in labels}
    dominant_delay_counts = {
        label: torch.zeros(sequence_length, dtype=torch.int64) for label in labels
    }
    exact_amplitude_square = Counter()
    sparse_error_square = {
        label: Counter() for label in labels
    }
    maximum_source_reconstruction_error = 0.0
    maximum_routing_row_sum_error = 0.0

    delays = torch.arange(maximum_probe_delay + 1, device=device)
    target_indices = query_positions[:, None] - delays[None, :]

    for batch_start in range(0, len(inputs), batch_size):
        batch = inputs[batch_start : batch_start + batch_size].to(
            device=device,
            dtype=torch.float64,
        )
        _, attentions, post_attention, post_mlp = model(batch)
        if any(
            not torch.equal(post_attn, post_mlp)
            for post_attn, post_mlp in zip(post_attention, post_mlp)
        ):
            raise RuntimeError("checkpoint is not behaving as attention-only")

        targets = batch[:, target_indices]
        target_chunks.append(
            targets.reshape(-1, maximum_probe_delay + 1).cpu().numpy()
        )
        source_values = batch[:, None, :].expand(-1, len(query_positions), -1)

        for label in labels:
            routing = source_routing_rows(attentions, label, query_positions)
            row_sum_error = (routing.sum(dim=-1) - 1.0).abs().max()
            maximum_routing_row_sum_error = max(
                maximum_routing_row_sum_error,
                float(row_sum_error.item()),
            )

            amplitude = (routing * source_values).sum(dim=-1)
            amplitudes[label].append(amplitude.reshape(-1).cpu().numpy())

            # Keep every fixed-delay source term needed by the probes.  Entry
            # d is exactly routing[t, t-d] * x[t-d].  Multiplying this scalar
            # by the path direction gives the corresponding 64-D component.
            delayed_source_indices = target_indices[None, :, :].expand(
                batch.shape[0], -1, -1
            )
            delayed_routing = routing.gather(
                dim=-1,
                index=delayed_source_indices,
            )
            delayed_components = delayed_routing * targets
            component_amplitudes[label].append(
                delayed_components.reshape(
                    -1, maximum_probe_delay + 1
                ).cpu().numpy()
            )
            component_routing_weights[label].append(
                delayed_routing.reshape(
                    -1, maximum_probe_delay + 1
                ).cpu().numpy()
            )

            # This is the central exactness check: source weights times source
            # values and the path direction must reproduce the existing path.
            direct = direct_data_path(model, batch, attentions, label)[
                :, query_positions, :
            ]
            reconstructed = amplitude.unsqueeze(-1) * directions[label]
            relative_error = (
                (reconstructed - direct).norm()
                / direct.norm().clamp_min(1e-30)
            )
            maximum_source_reconstruction_error = max(
                maximum_source_reconstruction_error,
                float(relative_error.item()),
            )

            if not collect_routing_statistics:
                continue

            batch_count = batch.shape[0] * len(query_positions)
            profile_observations[label] += batch_count

            source_indices = torch.arange(sequence_length, device=device)
            delay_by_source = query_positions[:, None] - source_indices[None, :]
            valid = delay_by_source >= 0
            flat_delays = delay_by_source[None, :, :].expand(
                batch.shape[0], -1, -1
            )[valid[None, :, :].expand(batch.shape[0], -1, -1)]
            flat_weights = routing[
                valid[None, :, :].expand(batch.shape[0], -1, -1)
            ]
            profile_sums[label].scatter_add_(
                0,
                flat_delays.cpu(),
                flat_weights.cpu(),
            )

            sorted_weights, sorted_sources = routing.sort(dim=-1, descending=True)
            cumulative_mass = sorted_weights.cumsum(dim=-1)
            sorted_values = source_values.gather(dim=-1, index=sorted_sources)
            cumulative_amplitude = (sorted_weights * sorted_values).cumsum(dim=-1)
            top_mass_chunks[label].append(
                sorted_weights[..., 0].reshape(-1).cpu().numpy()
            )

            dominant_sources = sorted_sources[..., 0]
            dominant_delays = (
                query_positions[None, :] - dominant_sources
            ).reshape(-1).cpu()
            dominant_delay_counts[label] += torch.bincount(
                dominant_delays,
                minlength=sequence_length,
            )

            exact_amplitude_square[label] += float(amplitude.square().sum().item())
            for threshold in thresholds:
                counts = (cumulative_mass < threshold).sum(dim=-1) + 1
                counts = counts.clamp_max(sequence_length)
                support_chunks[label][threshold].append(
                    counts.reshape(-1).cpu().numpy()
                )
                selected_index = (counts - 1).unsqueeze(-1)
                sparse_amplitude = cumulative_amplitude.gather(
                    dim=-1,
                    index=selected_index,
                ).squeeze(-1)
                sparse_error_square[label][threshold] += float(
                    (sparse_amplitude - amplitude).square().sum().item()
                )

    result = {
        "labels": labels,
        "query_positions": query_positions.cpu().numpy(),
        "amplitudes": {
            label: np.concatenate(chunks) for label, chunks in amplitudes.items()
        },
        "component_amplitudes": {
            label: np.concatenate(chunks)
            for label, chunks in component_amplitudes.items()
        },
        "component_routing_weights": {
            label: np.concatenate(chunks)
            for label, chunks in component_routing_weights.items()
        },
        "targets": np.concatenate(target_chunks, axis=0),
        "directions": {
            label: directions[label].detach().cpu().numpy() for label in labels
        },
        "maximum_source_reconstruction_relative_error": (
            maximum_source_reconstruction_error
        ),
        "maximum_routing_row_sum_error": maximum_routing_row_sum_error,
    }

    if collect_routing_statistics:
        result.update(
            {
                "routing_profiles": {
                    label: (
                        profile_sums[label] / profile_observations[label]
                    ).numpy()
                    for label in labels
                },
                "supports": {
                    label: {
                        threshold: np.concatenate(chunks)
                        for threshold, chunks in by_threshold.items()
                    }
                    for label, by_threshold in support_chunks.items()
                },
                "top_masses": {
                    label: np.concatenate(chunks)
                    for label, chunks in top_mass_chunks.items()
                },
                "dominant_delay_counts": {
                    label: counts.numpy()
                    for label, counts in dominant_delay_counts.items()
                },
                "sparse_relative_rmse": {
                    label: {
                        threshold: float(
                            np.sqrt(
                                sparse_error_square[label][threshold]
                                / max(exact_amplitude_square[label], 1e-30)
                            )
                        )
                        for threshold in thresholds
                    }
                    for label in labels
                },
            }
        )
    return result


def ridge_decode_delays(
    train_amplitude: np.ndarray,
    train_targets: np.ndarray,
    test_amplitude: np.ndarray,
    test_targets: np.ndarray,
    ridge_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode every x[t-d] from the complete scalar amplitude of one path."""
    mean = float(train_amplitude.mean())
    scale = float(train_amplitude.std())
    if scale < 1e-12:
        return (
            np.full(train_targets.shape[1], np.nan),
            np.full(train_targets.shape[1], np.nan),
        )
    x_train = ((train_amplitude - mean) / scale)[:, None]
    x_test = ((test_amplitude - mean) / scale)[:, None]
    target_mean = train_targets.mean(axis=0, keepdims=True)
    centered_targets = train_targets - target_mean
    penalty = ridge_strength * len(x_train)
    coefficients = (x_train.T @ centered_targets) / (
        float((x_train.T @ x_train).item()) + penalty
    )
    predictions = x_test @ coefficients + target_mean

    residual_ss = np.square(test_targets - predictions).sum(axis=0)
    centered_test = test_targets - test_targets.mean(axis=0, keepdims=True)
    total_ss = np.square(centered_test).sum(axis=0)
    r2 = 1.0 - residual_ss / np.maximum(total_ss, 1e-30)

    centered_predictions = predictions - predictions.mean(axis=0, keepdims=True)
    covariance = (centered_test * centered_predictions).sum(axis=0)
    denominator = np.sqrt(
        np.square(centered_test).sum(axis=0)
        * np.square(centered_predictions).sum(axis=0)
    )
    correlation = covariance / np.maximum(denominator, 1e-30)
    return r2, correlation


def probe_one_source_component(
    train_component_amplitude: np.ndarray,
    train_routing_weight: np.ndarray,
    train_targets: np.ndarray,
    test_component_amplitude: np.ndarray,
    test_routing_weight: np.ndarray,
    test_targets: np.ndarray,
    source_delay: int,
    ridge_strength: float,
    minimum_routing_weight: float,
) -> dict:
    """Probe one literal path/delay term before and after weight normalization."""
    raw_r2, raw_correlation = ridge_decode_delays(
        train_component_amplitude,
        train_targets,
        test_component_amplitude,
        test_targets,
        ridge_strength,
    )
    finite = np.flatnonzero(np.isfinite(raw_r2))
    raw_best_delay = int(finite[np.argmax(raw_r2[finite])]) if finite.size else -1

    train_mask = train_routing_weight >= minimum_routing_weight
    test_mask = test_routing_weight >= minimum_routing_weight
    normalized_r2 = np.full(train_targets.shape[1], np.nan)
    normalized_correlation = np.full(train_targets.shape[1], np.nan)
    if train_mask.sum() >= 10 and test_mask.sum() >= 10:
        normalized_train = (
            train_component_amplitude[train_mask]
            / train_routing_weight[train_mask]
        )
        normalized_test = (
            test_component_amplitude[test_mask]
            / test_routing_weight[test_mask]
        )
        normalized_r2, normalized_correlation = ridge_decode_delays(
            normalized_train,
            train_targets[train_mask],
            normalized_test,
            test_targets[test_mask],
            ridge_strength,
        )

    return {
        "raw_r2_curve": raw_r2,
        "raw_correlation_curve": raw_correlation,
        "raw_matching_r2": float(raw_r2[source_delay]),
        "raw_matching_correlation": float(raw_correlation[source_delay]),
        "raw_best_delay": raw_best_delay,
        "raw_best_r2": (
            float(raw_r2[raw_best_delay]) if raw_best_delay >= 0 else float("nan")
        ),
        "normalized_matching_r2": float(normalized_r2[source_delay]),
        "normalized_matching_correlation": float(
            normalized_correlation[source_delay]
        ),
        "n_train_normalized": int(train_mask.sum()),
        "n_test_normalized": int(test_mask.sum()),
    }


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def centered_norm(values: np.ndarray, minimum_span: float = 0.1) -> TwoSlopeNorm:
    finite = values[np.isfinite(values)]
    extent = max(float(np.abs(finite).max()) if finite.size else 0.0, minimum_span)
    return TwoSlopeNorm(vmin=-extent, vcenter=0.0, vmax=extent)


def plot_pages(
    model_name: str,
    lag: int,
    labels: list[str],
    routing_profiles: dict[str, np.ndarray],
    probe_r2: dict[str, np.ndarray],
    supports: dict[str, dict[float, np.ndarray]],
    thresholds: tuple[float, ...],
    maximum_probe_delay: int,
    paths_per_page: int,
    output_dir: Path,
) -> list[Path]:
    output_paths = []
    n_pages = (len(labels) + paths_per_page - 1) // paths_per_page
    for page in range(n_pages):
        page_labels = labels[page * paths_per_page : (page + 1) * paths_per_page]
        routing = np.stack([routing_profiles[label] for label in page_labels])
        decoding = np.stack([probe_r2[label] for label in page_labels])
        support_medians = np.array(
            [
                [np.median(supports[label][threshold]) for threshold in thresholds]
                for label in page_labels
            ]
        )

        height = max(7.0, 0.42 * len(page_labels) + 4.5)
        figure, axes = plt.subplots(
            1,
            3,
            figsize=(18, height),
            gridspec_kw={"width_ratios": (1.6, 1.6, 0.62)},
            constrained_layout=True,
        )
        routing_image = axes[0].imshow(
            routing,
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=0.0,
            vmax=max(0.05, float(routing.max())),
            extent=(-0.5, routing.shape[1] - 0.5, len(page_labels) - 0.5, -0.5),
        )
        axes[0].set_title("Literal frozen-attention routing mass")
        axes[0].set_xlabel("source delay d in x[t-d]")
        axes[0].set_ylabel("residual path")
        axes[0].set_yticks(np.arange(len(page_labels)), page_labels)
        figure.colorbar(routing_image, ax=axes[0], label="mean routing mass")

        decode_image = axes[1].imshow(
            decoding,
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu_r",
            norm=centered_norm(decoding),
            extent=(
                -0.5,
                maximum_probe_delay + 0.5,
                len(page_labels) - 0.5,
                -0.5,
            ),
        )
        axes[1].set_title("Held-out decoding from the complete data path")
        axes[1].set_xlabel("target delay d in x[t-d]")
        axes[1].set_yticks(np.arange(len(page_labels)), page_labels)
        figure.colorbar(decode_image, ax=axes[1], label="held-out R²")

        support_image = axes[2].imshow(
            support_medians,
            aspect="auto",
            interpolation="nearest",
            cmap="magma",
            vmin=1,
            vmax=max(2, float(support_medians.max())),
        )
        axes[2].set_title("Sparse support")
        axes[2].set_xticks(
            np.arange(len(thresholds)),
            [f"{100 * threshold:.0f}%" for threshold in thresholds],
        )
        axes[2].set_xlabel("routing mass retained")
        axes[2].set_yticks(np.arange(len(page_labels)), page_labels)
        for row_index in range(len(page_labels)):
            for column_index in range(len(thresholds)):
                value = support_medians[row_index, column_index]
                axes[2].text(
                    column_index,
                    row_index,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.55 * support_medians.max() else "black",
                    fontsize=8,
                )
        figure.colorbar(support_image, ax=axes[2], label="median source count")

        figure.suptitle(
            f"{model_name} attention-only · lag {lag} · exact path source delays\n"
            f"page {page + 1}/{n_pages}; routing is exact conditional on the "
            "observed attention matrices",
            fontsize=15,
        )
        output_path = output_dir / (
            f"{model_name}_lag{lag}_source_delays_page{page + 1}.png"
        )
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        output_paths.append(output_path)
    return output_paths


def plot_support_cdf(
    datasets: list[dict],
    thresholds: tuple[float, ...],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        1,
        len(thresholds),
        figsize=(5.5 * len(thresholds), 4.8),
        constrained_layout=True,
        squeeze=False,
    )
    for threshold_index, threshold in enumerate(thresholds):
        axis = axes[0, threshold_index]
        for dataset in datasets:
            values = np.concatenate(
                [
                    dataset["supports"][label][threshold]
                    for label in dataset["labels"]
                ]
            )
            maximum = int(values.max())
            source_counts = np.arange(1, maximum + 1)
            cdf = np.array([(values <= count).mean() for count in source_counts])
            axis.step(
                source_counts,
                cdf,
                where="post",
                linewidth=1.8,
                label=f"{dataset['model']} · lag {dataset['lag']}",
            )
        axis.set_title(f"Retain {100 * threshold:.0f}% routing mass")
        axis.set_xlabel("number of original source terms retained")
        axis.set_ylabel("fraction of path rows")
        axis.set_ylim(0.0, 1.01)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(
        "How sparse are the exact source-delay mixtures?\n"
        "Each observation is one residual path at one sequence position",
        fontsize=15,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_source_component_pages(
    model_name: str,
    lag: int,
    labels: list[str],
    component_results: dict[str, list[dict]],
    paths_per_page: int,
    output_dir: Path,
) -> list[Path]:
    """Show the matching-value regression for each selected source term."""
    output_paths = []
    n_pages = (len(labels) + paths_per_page - 1) // paths_per_page
    n_components = max(len(component_results[label]) for label in labels)
    for page in range(n_pages):
        page_labels = labels[page * paths_per_page : (page + 1) * paths_per_page]
        values = np.full((len(page_labels), n_components), np.nan)
        for row_index, label in enumerate(page_labels):
            for column_index, result in enumerate(component_results[label]):
                values[row_index, column_index] = result["raw_matching_r2"]

        height = max(5.5, 0.48 * len(page_labels) + 2.8)
        figure, axis = plt.subplots(
            figsize=(2.2 * n_components + 5.0, height),
            constrained_layout=True,
        )
        image = axis.imshow(
            values,
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu_r",
            norm=centered_norm(values),
        )
        axis.set_title(
            "Regress each source-resolved component against its own x[t-d]"
        )
        axis.set_xlabel("source component rank within this path")
        axis.set_ylabel("residual path")
        axis.set_xticks(
            np.arange(n_components),
            [f"source {index + 1}" for index in range(n_components)],
        )
        axis.set_yticks(np.arange(len(page_labels)), page_labels)
        for row_index, label in enumerate(page_labels):
            for column_index, result in enumerate(component_results[label]):
                value = result["raw_matching_r2"]
                axis.text(
                    column_index,
                    row_index,
                    f"d={result['source_delay']}\n"
                    f"mass={result['mean_routing_mass']:.2f}\n"
                    f"R²={value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if np.isfinite(value) and abs(value) > 0.55 else "black",
                )
        figure.colorbar(image, ax=axis, label="held-out matching-delay R²")
        figure.suptitle(
            f"{model_name} attention-only · lag {lag} · individual source terms\n"
            "The normalized control divides out the known routing weight and "
            "is reported in the CSV",
            fontsize=14,
        )
        output_path = output_dir / (
            f"{model_name}_lag{lag}_source_component_probes_page{page + 1}.png"
        )
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        output_paths.append(output_path)
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=parse_models, default=DEFAULT_MODELS)
    parser.add_argument("--lags", type=parse_int_tuple, default=(40,))
    parser.add_argument("--n-train-per-lag", type=int, default=32)
    parser.add_argument("--n-test-per-lag", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=200)
    parser.add_argument("--rho", type=float, default=0.9)
    parser.add_argument("--burn-in", type=int, default=30)
    parser.add_argument("--query-stride", type=int, default=2)
    parser.add_argument("--maximum-probe-delay", type=int, default=100)
    parser.add_argument(
        "--thresholds",
        type=parse_float_tuple,
        default=(0.90, 0.95, 0.99),
    )
    parser.add_argument("--ridge-strength", type=float, default=1e-6)
    parser.add_argument(
        "--component-top-k",
        type=int,
        default=4,
        help="number of leading fixed-delay source terms to probe per path",
    )
    parser.add_argument(
        "--minimum-component-routing",
        type=float,
        default=1e-6,
        help="minimum routing weight used by the divide-by-weight sanity check",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--paths-per-page", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment_outputs/path_source_delay_sparsity"),
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.n_train_per_lag = min(args.n_train_per_lag, 8)
        args.n_test_per_lag = min(args.n_test_per_lag, 8)
        args.query_stride = max(args.query_stride, 4)

    if args.maximum_probe_delay >= args.sequence_length:
        raise ValueError("maximum probe delay must be below sequence length")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    sparsity_rows: list[dict] = []
    routing_rows: list[dict] = []
    probe_rows: list[dict] = []
    source_component_rows: list[dict] = []
    source_component_curve_rows: list[dict] = []
    model_summary_rows: list[dict] = []
    plot_datasets: list[dict] = []
    diagnostics: dict[str, dict] = {}

    for model_name in args.models:
        n_layers, checkpoint = MODEL_SPECS[model_name]
        print(f"loading {model_name}: {checkpoint}")
        model = load_attention_only_model(n_layers, checkpoint, device)

        for lag_index, lag in enumerate(args.lags):
            print(f"  lag {lag}: collecting train/test paths")
            train_inputs = make_inputs(
                lag,
                args.n_train_per_lag,
                args.sequence_length,
                args.rho,
                args.seed + 100_000 * lag_index + 1_000 * n_layers,
            )
            test_inputs = make_inputs(
                lag,
                args.n_test_per_lag,
                args.sequence_length,
                args.rho,
                args.seed + 500_000 + 100_000 * lag_index + 1_000 * n_layers,
            )
            train = collect_one_split(
                model,
                train_inputs,
                lag,
                args.burn_in,
                args.query_stride,
                args.maximum_probe_delay,
                args.thresholds,
                args.batch_size,
                device,
                collect_routing_statistics=True,
            )
            test = collect_one_split(
                model,
                test_inputs,
                lag,
                args.burn_in,
                args.query_stride,
                args.maximum_probe_delay,
                args.thresholds,
                args.batch_size,
                device,
                collect_routing_statistics=True,
            )
            if train["labels"] != test["labels"]:
                raise RuntimeError("train/test path labels differ")

            labels = test["labels"]
            probe_r2: dict[str, np.ndarray] = {}
            source_component_results: dict[str, list[dict]] = {
                label: [] for label in labels
            }
            for label in labels:
                r2, correlation = ridge_decode_delays(
                    train["amplitudes"][label],
                    train["targets"],
                    test["amplitudes"][label],
                    test["targets"],
                    args.ridge_strength,
                )
                probe_r2[label] = r2
                for delay in range(args.maximum_probe_delay + 1):
                    probe_rows.append(
                        {
                            "model": model_name,
                            "lag": lag,
                            "path": label,
                            "write_count": label.count("1"),
                            "target_delay": delay,
                            "heldout_r2": float(r2[delay]),
                            "heldout_correlation": float(correlation[delay]),
                        }
                    )

                # Select source delays using train-only routing profiles.  Each
                # selected feature is one literal term alpha_d(t)*x[t-d], not
                # the recombined path amplitude.
                train_profile = train["routing_profiles"][label][
                    : args.maximum_probe_delay + 1
                ]
                ordered_delays = np.argsort(train_profile)[::-1]
                selected_delays = [
                    int(delay)
                    for delay in ordered_delays
                    if train_profile[delay] > 1e-12
                ][: args.component_top_k]
                cumulative_selected_mass = 0.0
                for source_rank, source_delay in enumerate(selected_delays, start=1):
                    mean_routing_mass = float(train_profile[source_delay])
                    cumulative_selected_mass += mean_routing_mass
                    result = probe_one_source_component(
                        train["component_amplitudes"][label][:, source_delay],
                        train["component_routing_weights"][label][
                            :, source_delay
                        ],
                        train["targets"],
                        test["component_amplitudes"][label][:, source_delay],
                        test["component_routing_weights"][label][
                            :, source_delay
                        ],
                        test["targets"],
                        source_delay,
                        args.ridge_strength,
                        args.minimum_component_routing,
                    )
                    result.update(
                        {
                            "source_rank": source_rank,
                            "source_delay": source_delay,
                            "mean_routing_mass": mean_routing_mass,
                            "cumulative_selected_mass": cumulative_selected_mass,
                        }
                    )
                    source_component_results[label].append(result)
                    source_component_rows.append(
                        {
                            "model": model_name,
                            "lag": lag,
                            "path": label,
                            "write_count": label.count("1"),
                            "source_rank": source_rank,
                            "source_delay": source_delay,
                            "train_mean_routing_mass": mean_routing_mass,
                            "train_cumulative_selected_mass": (
                                cumulative_selected_mass
                            ),
                            "raw_matching_heldout_r2": result[
                                "raw_matching_r2"
                            ],
                            "raw_matching_heldout_correlation": result[
                                "raw_matching_correlation"
                            ],
                            "raw_best_decoded_delay": result["raw_best_delay"],
                            "raw_best_heldout_r2": result["raw_best_r2"],
                            "normalized_matching_heldout_r2": result[
                                "normalized_matching_r2"
                            ],
                            "normalized_matching_heldout_correlation": result[
                                "normalized_matching_correlation"
                            ],
                            "n_train_normalized": result[
                                "n_train_normalized"
                            ],
                            "n_test_normalized": result["n_test_normalized"],
                        }
                    )
                    for target_delay in range(args.maximum_probe_delay + 1):
                        source_component_curve_rows.append(
                            {
                                "model": model_name,
                                "lag": lag,
                                "path": label,
                                "source_rank": source_rank,
                                "source_delay": source_delay,
                                "target_delay": target_delay,
                                "raw_heldout_r2": float(
                                    result["raw_r2_curve"][target_delay]
                                ),
                                "raw_heldout_correlation": float(
                                    result["raw_correlation_curve"][target_delay]
                                ),
                            }
                        )

            for label in labels:
                profile = test["routing_profiles"][label]
                counts = test["dominant_delay_counts"][label]
                mode_delay = int(counts.argmax())
                mode_rate = float(counts.max() / counts.sum())
                top_masses = test["top_masses"][label]
                direction_norm = float(np.linalg.norm(test["directions"][label]))

                base_row = {
                    "model": model_name,
                    "lag": lag,
                    "path": label,
                    "write_count": label.count("1"),
                    "direction_norm": direction_norm,
                    "dominant_delay_mode": mode_delay,
                    "dominant_delay_mode_rate": mode_rate,
                    "mean_top1_routing_mass": float(top_masses.mean()),
                    "p05_top1_routing_mass": quantile(top_masses, 0.05),
                    "median_top1_routing_mass": quantile(top_masses, 0.50),
                    "p95_top1_routing_mass": quantile(top_masses, 0.95),
                }
                for threshold in args.thresholds:
                    support = test["supports"][label][threshold]
                    prefix = f"support_{int(round(100 * threshold))}"
                    base_row.update(
                        {
                            f"{prefix}_mean": float(support.mean()),
                            f"{prefix}_median": quantile(support, 0.50),
                            f"{prefix}_p05": quantile(support, 0.05),
                            f"{prefix}_p95": quantile(support, 0.95),
                            f"{prefix}_fraction_le_1": float((support <= 1).mean()),
                            f"{prefix}_fraction_le_2": float((support <= 2).mean()),
                            f"{prefix}_fraction_le_4": float((support <= 4).mean()),
                            f"{prefix}_path_relative_rmse": test[
                                "sparse_relative_rmse"
                            ][label][threshold],
                        }
                    )
                sparsity_rows.append(base_row)

                for delay, mass in enumerate(profile):
                    routing_rows.append(
                        {
                            "model": model_name,
                            "lag": lag,
                            "path": label,
                            "write_count": label.count("1"),
                            "source_delay": delay,
                            "mean_routing_mass": float(mass),
                        }
                    )

            all_supports = {
                threshold: np.concatenate(
                    [test["supports"][label][threshold] for label in labels]
                )
                for threshold in args.thresholds
            }
            summary_row = {
                "model": model_name,
                "lag": lag,
                "n_paths": len(labels),
                "n_test_sequences": args.n_test_per_lag,
                "n_query_positions_per_sequence": len(test["query_positions"]),
            }
            for threshold, support in all_supports.items():
                prefix = f"support_{int(round(100 * threshold))}"
                summary_row.update(
                    {
                        f"{prefix}_median": quantile(support, 0.50),
                        f"{prefix}_p95": quantile(support, 0.95),
                        f"{prefix}_fraction_le_1": float((support <= 1).mean()),
                        f"{prefix}_fraction_le_2": float((support <= 2).mean()),
                        f"{prefix}_fraction_le_4": float((support <= 4).mean()),
                    }
                )
            model_summary_rows.append(summary_row)

            diagnostic_key = f"{model_name}_lag{lag}"
            diagnostics[diagnostic_key] = {
                "checkpoint": str(checkpoint),
                "first_query": int(test["query_positions"][0]),
                "last_query": int(test["query_positions"][-1]),
                "maximum_train_source_reconstruction_relative_error": train[
                    "maximum_source_reconstruction_relative_error"
                ],
                "maximum_test_source_reconstruction_relative_error": test[
                    "maximum_source_reconstruction_relative_error"
                ],
                "maximum_train_routing_row_sum_error": train[
                    "maximum_routing_row_sum_error"
                ],
                "maximum_test_routing_row_sum_error": test[
                    "maximum_routing_row_sum_error"
                ],
            }
            plot_datasets.append(
                {
                    "model": model_name,
                    "lag": lag,
                    "labels": labels,
                    "routing_profiles": test["routing_profiles"],
                    "supports": test["supports"],
                }
            )
            plot_pages(
                model_name,
                lag,
                labels,
                test["routing_profiles"],
                probe_r2,
                test["supports"],
                args.thresholds,
                args.maximum_probe_delay,
                args.paths_per_page,
                args.output_dir,
            )
            plot_source_component_pages(
                model_name,
                lag,
                labels,
                source_component_results,
                args.paths_per_page,
                args.output_dir,
            )

    save_csv(args.output_dir / "path_sparsity_summary.csv", sparsity_rows)
    save_csv(args.output_dir / "routing_profiles.csv", routing_rows)
    save_csv(args.output_dir / "delay_probe_results.csv", probe_rows)
    save_csv(
        args.output_dir / "source_component_probe_summary.csv",
        source_component_rows,
    )
    save_csv(
        args.output_dir / "source_component_probe_curves.csv",
        source_component_curve_rows,
    )
    save_csv(args.output_dir / "model_summary.csv", model_summary_rows)
    plot_support_cdf(
        plot_datasets,
        args.thresholds,
        args.output_dir / "support_count_cdf.png",
    )
    run_summary = {
        "models": list(args.models),
        "lags": list(args.lags),
        "n_train_per_lag": args.n_train_per_lag,
        "n_test_per_lag": args.n_test_per_lag,
        "sequence_length": args.sequence_length,
        "rho": args.rho,
        "burn_in": args.burn_in,
        "query_stride": args.query_stride,
        "maximum_probe_delay": args.maximum_probe_delay,
        "thresholds": list(args.thresholds),
        "ridge_strength": args.ridge_strength,
        "component_top_k": args.component_top_k,
        "minimum_component_routing": args.minimum_component_routing,
        "seed": args.seed,
        "source_resolution": (
            "scalar-dependent W_r component only; constant W_r bias excluded"
        ),
        "diagnostics": diagnostics,
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2)
    )

    print("\nAggregate sparse-support summary")
    for row in model_summary_rows:
        print(f"{row['model']} lag {row['lag']} ({row['n_paths']} paths)")
        for threshold in args.thresholds:
            prefix = f"support_{int(round(100 * threshold))}"
            print(
                f"  {100 * threshold:.0f}% mass: median={row[prefix + '_median']:.0f}, "
                f"p95={row[prefix + '_p95']:.0f}, "
                f"<=2 sources={100 * row[prefix + '_fraction_le_2']:.1f}%"
            )
    print(f"\nsaved outputs to {args.output_dir}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
