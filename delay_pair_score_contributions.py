"""Test whether +1 delay pairs dominate the final-head locator score.

Two nested decompositions are evaluated on held-out sequences:

1. Hard path pairs.  Give each frozen-attention residual path its dominant
   source delay, ignoring the rest of its routing distribution.  Mark a pair
   when key_delay - query_delay == 1.
2. Source-component pairs.  Split every path into fixed-delay source terms,
   retaining enough train-set routing mass to reach a chosen threshold.  Mark
   a component pair when its literal key delay minus query delay is one.

At the correct final offset D=L-1, a marked pair compares original values whose
source positions are separated by the data lag L.  We compare marked and
unmarked contributions to both the raw correct-offset score and the
correct-minus-nearby-wrong locator margin.  Sequence means are the statistical
units.

The hard/full analysis exactly sums to the model's real pre-softmax score.  The
hard/data and source-component analyses isolate the scalar-data portion of the
residual; the latter is a measured sparse approximation whose retained score
is checked against the complete data-path score.

Usage:
    python delay_pair_score_contributions.py --models 4L --quick --no-show
    python delay_pair_score_contributions.py --models 4L,6L,7L --no-show
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel
import torch

from cross_model_raw_head1_path_test import MODEL_SPECS, load_attention_only_model
from four_layer_attention_path_analysis import expand_prefinal_paths
from path_source_delay_sparsity import (
    collect_one_split,
    labels_for_model,
    make_inputs,
    parse_float_tuple,
    parse_int_tuple,
    parse_models,
    path_direction,
    source_routing_rows,
)


DEFAULT_MODELS = ("4L", "6L", "7L")


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_source_delays(
    train_profiles: dict[str, np.ndarray],
    labels: list[str],
    maximum_delay: int,
    retained_mass: float,
    maximum_components_per_path: int,
) -> tuple[dict[str, list[int]], dict[str, float]]:
    selected: dict[str, list[int]] = {}
    retained: dict[str, float] = {}
    for label in labels:
        profile = train_profiles[label][: maximum_delay + 1]
        order = np.argsort(profile)[::-1]
        delays = []
        cumulative = 0.0
        for delay in order:
            mass = float(profile[delay])
            if mass <= 1e-15:
                continue
            delays.append(int(delay))
            cumulative += mass
            if cumulative >= retained_mass or len(delays) >= maximum_components_per_path:
                break
        selected[label] = delays
        retained[label] = cumulative
    return selected, retained


def path_geometry(model, directions: torch.Tensor, offset: int) -> torch.Tensor:
    """Final QK/RoPE bilinear coefficient between data-path directions."""
    query_matrix, key_matrix, _, _ = model.layers[-1][0]
    query_positions = torch.full(
        (len(directions),), offset, dtype=torch.long, device=directions.device
    )
    key_positions = torch.zeros_like(query_positions)
    query = model.apply_rope(directions @ query_matrix, query_positions)
    key = model.apply_rope(directions @ key_matrix, key_positions)
    return query @ key.T / math.sqrt(model.d_head)


def sequence_group_stats_from_pair_tensor(
    pair_scores: torch.Tensor,
    plus_mask: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Group one explicit B x Q x N x N pair tensor by a fixed mask."""
    sequence_matrix = pair_scores.mean(dim=1)
    other_mask = ~plus_mask
    plus_total = sequence_matrix[:, plus_mask].sum(dim=1)
    other_total = sequence_matrix[:, other_mask].sum(dim=1)
    return {
        "plus_pair_mean": sequence_matrix[:, plus_mask].mean(dim=1).cpu().numpy(),
        "other_pair_mean": sequence_matrix[:, other_mask].mean(dim=1).cpu().numpy(),
        "plus_total": plus_total.cpu().numpy(),
        "other_total": other_total.cpu().numpy(),
        "total": (plus_total + other_total).cpu().numpy(),
    }


def sequence_group_stats_from_amplitudes(
    query_amplitudes: torch.Tensor,
    key_amplitudes: torch.Tensor,
    geometry: torch.Tensor,
    plus_mask: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Group factorized pair contributions without constructing BxQxNxN."""
    batch_size, n_queries, n_components = query_amplitudes.shape
    query_flat = query_amplitudes.reshape(-1, n_components)
    key_flat = key_amplitudes.reshape(-1, n_components)
    plus_geometry = geometry * plus_mask
    plus_scores = ((query_flat @ plus_geometry) * key_flat).sum(dim=-1)
    total_scores = ((query_flat @ geometry) * key_flat).sum(dim=-1)
    plus_total = plus_scores.reshape(batch_size, n_queries).mean(dim=1)
    total = total_scores.reshape(batch_size, n_queries).mean(dim=1)
    other_total = total - plus_total
    plus_count = int(plus_mask.sum().item())
    other_count = plus_mask.numel() - plus_count
    return {
        "plus_pair_mean": (plus_total / plus_count).cpu().numpy(),
        "other_pair_mean": (other_total / other_count).cpu().numpy(),
        "plus_total": plus_total.cpu().numpy(),
        "other_total": other_total.cpu().numpy(),
        "total": total.cpu().numpy(),
    }


def add_stats(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> dict:
    return {key: first[key] + second[key] for key in first}


def divide_stats(values: dict[str, np.ndarray], divisor: float) -> dict:
    return {key: value / divisor for key, value in values.items()}


def subtract_stats(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> dict:
    return {key: first[key] - second[key] for key in first}


def append_sequence_stats(
    destination: dict[str, dict[str, list[np.ndarray]]],
    analysis: str,
    metric: str,
    values: dict[str, np.ndarray],
) -> None:
    for key, array in values.items():
        destination[analysis][f"{metric}_{key}"].append(array)


def initialize_sequence_store(analyses: tuple[str, ...]) -> dict:
    keys = (
        "correct_plus_pair_mean",
        "correct_other_pair_mean",
        "correct_plus_total",
        "correct_other_total",
        "correct_total",
        "margin_plus_pair_mean",
        "margin_other_pair_mean",
        "margin_plus_total",
        "margin_other_total",
        "margin_total",
    )
    return {analysis: {key: [] for key in keys} for analysis in analyses}


def source_amplitudes_for_positions(
    model,
    batch: torch.Tensor,
    attentions: list[torch.Tensor],
    labels: list[str],
    selected_delays: dict[str, list[int]],
    union_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int, str]]]:
    """Return complete data-path amplitudes and selected source components."""
    batch_size = len(batch)
    data_amplitudes = torch.empty(
        batch_size,
        len(union_positions),
        len(labels),
        dtype=batch.dtype,
        device=batch.device,
    )
    component_metadata: list[tuple[int, int, str]] = []
    for path_index, label in enumerate(labels):
        for delay in selected_delays[label]:
            component_metadata.append((path_index, delay, f"{label}:d{delay}"))
    component_amplitudes = torch.empty(
        batch_size,
        len(union_positions),
        len(component_metadata),
        dtype=batch.dtype,
        device=batch.device,
    )

    component_start = 0
    source_values = batch[:, None, :].expand(-1, len(union_positions), -1)
    for path_index, label in enumerate(labels):
        routing = source_routing_rows(attentions, label, union_positions)
        data_amplitudes[:, :, path_index] = (routing * source_values).sum(dim=-1)
        delays = selected_delays[label]
        if delays:
            delay_tensor = torch.tensor(delays, device=batch.device)
            source_indices = union_positions[:, None] - delay_tensor[None, :]
            if int(source_indices.min().item()) < 0:
                raise RuntimeError("a selected source delay precedes position zero")
            gather_indices = source_indices[None, :, :].expand(batch_size, -1, -1)
            weights = routing.gather(dim=-1, index=gather_indices)
            values = batch[:, source_indices]
            count = len(delays)
            component_amplitudes[
                :, :, component_start : component_start + count
            ] = weights * values
            component_start += count
    return data_amplitudes, component_amplitudes, component_metadata


def summarize_paired_groups(
    plus: np.ndarray,
    other: np.ndarray,
) -> dict[str, float]:
    difference = plus - other
    n = len(difference)
    standard_deviation = float(difference.std(ddof=1)) if n > 1 else 0.0
    standard_error = standard_deviation / math.sqrt(max(n, 1))
    test = ttest_rel(plus, other)
    return {
        "n_sequences": n,
        "plus_mean": float(plus.mean()),
        "other_mean": float(other.mean()),
        "mean_difference": float(difference.mean()),
        "difference_ci95_low": float(difference.mean() - 1.96 * standard_error),
        "difference_ci95_high": float(difference.mean() + 1.96 * standard_error),
        "paired_cohen_d": float(
            difference.mean() / standard_deviation
            if standard_deviation > 1e-30
            else np.nan
        ),
        "plus_greater_sequence_rate": float((difference > 0).mean()),
        "paired_t_pvalue": float(test.pvalue),
    }


def ranked_pair_rows(
    model_name: str,
    lag: int,
    analysis: str,
    metric: str,
    matrix: np.ndarray,
    labels: list[str],
    delays: np.ndarray,
    plus_mask: np.ndarray,
) -> list[dict]:
    order = np.argsort(matrix.ravel())[::-1]
    rows = []
    for rank, flat_index in enumerate(order, start=1):
        query_index, key_index = np.unravel_index(flat_index, matrix.shape)
        rows.append(
            {
                "model": model_name,
                "lag": lag,
                "analysis": analysis,
                "metric": metric,
                "rank_by_signed_contribution": rank,
                "query_item": labels[query_index],
                "key_item": labels[key_index],
                "query_delay": int(delays[query_index]),
                "key_delay": int(delays[key_index]),
                "delay_difference_key_minus_query": int(
                    delays[key_index] - delays[query_index]
                ),
                "is_plus_one": bool(plus_mask[query_index, key_index]),
                "mean_contribution": float(matrix[query_index, key_index]),
            }
        )
    return rows


def plot_group_comparison(
    model_name: str,
    lag: int,
    sequence_arrays: dict[str, dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    analyses = ("hard_full", "hard_data", "source_components")
    titles = (
        "Exact full residual paths",
        "Complete scalar-data paths",
        "Sparse source-delay components",
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for column, (analysis, title) in enumerate(zip(analyses, titles)):
        for row, metric in enumerate(("correct", "margin")):
            plus = sequence_arrays[analysis][f"{metric}_plus_pair_mean"]
            other = sequence_arrays[analysis][f"{metric}_other_pair_mean"]
            axis = axes[row, column]
            axis.boxplot(
                [plus, other],
                tick_labels=["+1 pairs", "other pairs"],
                showfliers=False,
                widths=0.55,
            )
            for sequence_index in range(len(plus)):
                axis.plot(
                    [1, 2],
                    [plus[sequence_index], other[sequence_index]],
                    color="0.75",
                    linewidth=0.6,
                    alpha=0.6,
                )
            difference = plus - other
            axis.axhline(0.0, color="0.5", linewidth=0.8)
            axis.set_title(
                f"{title}\nmean paired difference={difference.mean():+.3g}"
            )
            axis.set_ylabel(
                "raw-score contribution per pair"
                if metric == "correct"
                else "correct-minus-wrong contribution per pair"
            )
            axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        f"{model_name} · lag {lag}: do key-delay − query-delay = +1 pairs dominate?",
        fontsize=15,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_pair_matrices(
    model_name: str,
    lag: int,
    matrices: dict[str, dict[str, np.ndarray]],
    masks: dict[str, np.ndarray],
    labels: dict[str, list[str]],
    output_path: Path,
) -> None:
    analyses = ("hard_full", "source_components")
    figure, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    for row, analysis in enumerate(analyses):
        for column, metric in enumerate(("correct", "margin")):
            matrix = matrices[analysis][metric]
            limit = max(float(np.abs(matrix).max()), 1e-12)
            axis = axes[row, column]
            image = axis.imshow(
                matrix,
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                aspect="auto",
                interpolation="nearest",
            )
            plus_coordinates = np.argwhere(masks[analysis])
            if len(plus_coordinates) <= 300:
                axis.scatter(
                    plus_coordinates[:, 1],
                    plus_coordinates[:, 0],
                    s=7 if matrix.shape[0] > 32 else 20,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=0.4,
                )
            axis.set_title(
                ("correct-offset contribution" if metric == "correct" else "correct-minus-wrong contribution")
                + f"\n+1 cells={int(masks[analysis].sum())}/{matrix.size}"
            )
            axis.set_xlabel("key path/component")
            axis.set_ylabel("query path/component")
            if matrix.shape[0] <= 16:
                axis.set_xticks(np.arange(matrix.shape[0]), labels[analysis], rotation=90)
                axis.set_yticks(np.arange(matrix.shape[0]), labels[analysis])
            figure.colorbar(image, ax=axis, shrink=0.82)
    figure.suptitle(
        f"{model_name} · lag {lag}: pair contributions; outlined cells are +1 delay pairs",
        fontsize=15,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


@torch.inference_mode()
def analyze_model_lag(
    model_name: str,
    model,
    lag: int,
    train_inputs: torch.Tensor,
    test_inputs: torch.Tensor,
    burn_in: int,
    query_stride: int,
    maximum_source_delay: int,
    source_mass_threshold: float,
    maximum_components_per_path: int,
    wrong_deltas: tuple[int, ...],
    batch_size: int,
    device: torch.device,
) -> dict:
    train = collect_one_split(
        model,
        train_inputs,
        lag,
        burn_in,
        query_stride,
        maximum_source_delay,
        (source_mass_threshold,),
        batch_size,
        device,
        collect_routing_statistics=True,
    )
    labels = train["labels"]
    hard_delays = np.array(
        [int(train["routing_profiles"][label].argmax()) for label in labels]
    )
    selected_delays, retained_mass = select_source_delays(
        train["routing_profiles"],
        labels,
        maximum_source_delay,
        source_mass_threshold,
        maximum_components_per_path,
    )
    maximum_selected_delay = max(max(values) for values in selected_delays.values())
    correct_offset = lag - 1
    wrong_offsets = tuple(
        sorted(
            {
                correct_offset + delta
                for delta in wrong_deltas
                if correct_offset + delta >= 1
            }
        )
    )
    offsets = (correct_offset,) + wrong_offsets
    first_query = max(
        lag + burn_in,
        max(offsets) + maximum_selected_delay,
    )
    query_positions = torch.arange(
        first_query,
        test_inputs.shape[1],
        query_stride,
        device=device,
    )
    if len(query_positions) == 0:
        raise ValueError("no query positions remain after source-history requirements")
    union_positions = torch.unique(
        torch.cat(
            [query_positions]
            + [query_positions - offset for offset in offsets]
        ),
        sorted=True,
    )
    union_lookup = {
        int(position): index for index, position in enumerate(union_positions.tolist())
    }
    query_union_indices = torch.tensor(
        [union_lookup[int(position)] for position in query_positions.tolist()],
        device=device,
    )
    key_union_indices = {
        offset: torch.tensor(
            [
                union_lookup[int(position - offset)]
                for position in query_positions.tolist()
            ],
            device=device,
        )
        for offset in offsets
    }

    directions = torch.stack([path_direction(model, label) for label in labels])
    hard_mask_np = hard_delays[None, :] - hard_delays[:, None] == 1
    hard_mask = torch.tensor(hard_mask_np, device=device)
    analyses = ("hard_full", "hard_data", "source_components")
    sequence_store = initialize_sequence_store(analyses)
    matrix_sums: dict[str, dict[str, torch.Tensor | None]] = {
        analysis: {"correct": None, "wrong": None} for analysis in analyses
    }
    observation_count = 0
    maximum_full_score_error = 0.0
    maximum_data_score_error = 0.0
    maximum_source_score_error = 0.0
    maximum_residual_error = 0.0
    component_metadata_reference = None

    for batch_start in range(0, len(test_inputs), batch_size):
        batch = test_inputs[batch_start : batch_start + batch_size].to(
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
        if list(path_dict) != labels:
            raise RuntimeError("path labels changed")
        full_paths = torch.stack(list(path_dict.values()))
        final_input = post_attention[-2]
        residual_error = (
            (full_paths.sum(dim=0) - final_input).norm()
            / final_input.norm().clamp_min(1e-30)
        )
        maximum_residual_error = max(maximum_residual_error, float(residual_error))

        data_amplitudes, component_amplitudes, component_metadata = (
            source_amplitudes_for_positions(
                model,
                batch,
                attentions,
                labels,
                selected_delays,
                union_positions,
            )
        )
        if component_metadata_reference is None:
            component_metadata_reference = component_metadata
        elif component_metadata_reference != component_metadata:
            raise RuntimeError("component metadata changed")
        component_path_indices = torch.tensor(
            [item[0] for item in component_metadata], device=device
        )
        component_delays_np = np.array([item[1] for item in component_metadata])
        component_mask_np = (
            component_delays_np[None, :] - component_delays_np[:, None] == 1
        )
        component_mask = torch.tensor(component_mask_np, device=device)

        positions = torch.arange(test_inputs.shape[1], device=device).view(1, 1, -1)
        query_matrix, key_matrix, _, _ = model.layers[-1][0]
        path_queries = model.apply_rope(full_paths @ query_matrix, positions)
        path_keys = model.apply_rope(full_paths @ key_matrix, positions)
        full_queries = model.apply_rope(
            final_input @ query_matrix, positions.squeeze(0)
        )
        full_keys = model.apply_rope(final_input @ key_matrix, positions.squeeze(0))
        selected_path_queries = path_queries[:, :, query_positions, :]
        selected_full_queries = full_queries[:, query_positions, :]
        query_data = data_amplitudes[:, query_union_indices, :]
        query_components = component_amplitudes[:, query_union_indices, :]

        correct_stats: dict[str, dict] = {}
        wrong_stats_sum: dict[str, dict] = {}
        wrong_matrix_batch: dict[str, torch.Tensor | None] = {
            analysis: None for analysis in analyses
        }
        correct_matrix_batch: dict[str, torch.Tensor] = {}

        for offset in offsets:
            geometry = path_geometry(model, directions, offset)
            component_geometry = geometry[
                component_path_indices[:, None], component_path_indices[None, :]
            ]
            key_data = data_amplitudes[:, key_union_indices[offset], :]
            key_components = component_amplitudes[:, key_union_indices[offset], :]

            full_pair = torch.einsum(
                "pbqh,rbqh->bqpr",
                selected_path_queries,
                path_keys[:, :, query_positions - offset, :],
            ) / math.sqrt(model.d_head)
            full_matrix = full_pair.sum(dim=(0, 1))
            full_stats = sequence_group_stats_from_pair_tensor(full_pair, hard_mask)

            data_flat_query = query_data.reshape(-1, len(labels))
            data_flat_key = key_data.reshape(-1, len(labels))
            data_matrix = (data_flat_query.T @ data_flat_key) * geometry
            data_stats = sequence_group_stats_from_amplitudes(
                query_data, key_data, geometry, hard_mask
            )

            component_flat_query = query_components.reshape(
                -1, query_components.shape[-1]
            )
            component_flat_key = key_components.reshape(-1, key_components.shape[-1])
            component_matrix = (
                component_flat_query.T @ component_flat_key
            ) * component_geometry
            component_stats = sequence_group_stats_from_amplitudes(
                query_components,
                key_components,
                component_geometry,
                component_mask,
            )

            direct_full = (
                selected_full_queries
                * full_keys[:, query_positions - offset, :]
            ).sum(dim=-1) / math.sqrt(model.d_head)
            maximum_full_score_error = max(
                maximum_full_score_error,
                float((full_pair.sum(dim=(-1, -2)) - direct_full).abs().max()),
            )
            query_data_residual = torch.einsum(
                "bqp,pm->bqm", query_data, directions
            )
            key_data_residual = torch.einsum(
                "bqp,pm->bqm", key_data, directions
            )
            query_position_batch = query_positions[None, :].expand(
                len(batch), -1
            )
            key_position_batch = (query_positions - offset)[None, :].expand(
                len(batch), -1
            )
            direct_data = (
                model.apply_rope(
                    query_data_residual @ query_matrix,
                    query_position_batch,
                )
                * model.apply_rope(
                    key_data_residual @ key_matrix,
                    key_position_batch,
                )
            ).sum(dim=-1) / math.sqrt(model.d_head)
            factorized_data = (
                (data_flat_query @ geometry) * data_flat_key
            ).sum(dim=-1).reshape(len(batch), len(query_positions))
            maximum_data_score_error = max(
                maximum_data_score_error,
                float((factorized_data - direct_data).abs().max()),
            )
            selected_by_path_query = torch.zeros_like(query_data)
            selected_by_path_key = torch.zeros_like(key_data)
            selected_by_path_query.scatter_add_(
                -1,
                component_path_indices.view(1, 1, -1).expand(
                    len(batch), len(query_positions), -1
                ),
                query_components,
            )
            selected_by_path_key.scatter_add_(
                -1,
                component_path_indices.view(1, 1, -1).expand(
                    len(batch), len(query_positions), -1
                ),
                key_components,
            )
            selected_stats = sequence_group_stats_from_amplitudes(
                selected_by_path_query,
                selected_by_path_key,
                geometry,
                hard_mask,
            )
            selected_query_residual = torch.einsum(
                "bqp,pm->bqm", selected_by_path_query, directions
            )
            selected_key_residual = torch.einsum(
                "bqp,pm->bqm", selected_by_path_key, directions
            )
            direct_selected = (
                model.apply_rope(
                    selected_query_residual @ query_matrix,
                    query_position_batch,
                )
                * model.apply_rope(
                    selected_key_residual @ key_matrix,
                    key_position_batch,
                )
            ).sum(dim=-1) / math.sqrt(model.d_head)
            factorized_components = (
                (component_flat_query @ component_geometry) * component_flat_key
            ).sum(dim=-1).reshape(len(batch), len(query_positions))
            maximum_source_score_error = max(
                maximum_source_score_error,
                float((factorized_components - direct_selected).abs().max()),
                float(
                    np.max(np.abs(component_stats["total"] - selected_stats["total"]))
                ),
            )

            current_stats = {
                "hard_full": full_stats,
                "hard_data": data_stats,
                "source_components": component_stats,
            }
            current_matrices = {
                "hard_full": full_matrix,
                "hard_data": data_matrix,
                "source_components": component_matrix,
            }
            if offset == correct_offset:
                correct_stats = current_stats
                correct_matrix_batch = current_matrices
            else:
                for analysis in analyses:
                    if analysis not in wrong_stats_sum:
                        wrong_stats_sum[analysis] = current_stats[analysis]
                        wrong_matrix_batch[analysis] = current_matrices[analysis]
                    else:
                        wrong_stats_sum[analysis] = add_stats(
                            wrong_stats_sum[analysis], current_stats[analysis]
                        )
                        assert wrong_matrix_batch[analysis] is not None
                        wrong_matrix_batch[analysis] = (
                            wrong_matrix_batch[analysis] + current_matrices[analysis]
                        )

        for analysis in analyses:
            wrong_average = divide_stats(wrong_stats_sum[analysis], len(wrong_offsets))
            margin = subtract_stats(correct_stats[analysis], wrong_average)
            append_sequence_stats(
                sequence_store, analysis, "correct", correct_stats[analysis]
            )
            append_sequence_stats(sequence_store, analysis, "margin", margin)
            if matrix_sums[analysis]["correct"] is None:
                matrix_sums[analysis]["correct"] = correct_matrix_batch[analysis]
                assert wrong_matrix_batch[analysis] is not None
                matrix_sums[analysis]["wrong"] = (
                    wrong_matrix_batch[analysis] / len(wrong_offsets)
                )
            else:
                matrix_sums[analysis]["correct"] = (
                    matrix_sums[analysis]["correct"] + correct_matrix_batch[analysis]
                )
                assert wrong_matrix_batch[analysis] is not None
                matrix_sums[analysis]["wrong"] = (
                    matrix_sums[analysis]["wrong"]
                    + wrong_matrix_batch[analysis] / len(wrong_offsets)
                )
        observation_count += len(batch) * len(query_positions)

    sequence_arrays = {
        analysis: {
            key: np.concatenate(chunks)
            for key, chunks in sequence_store[analysis].items()
        }
        for analysis in analyses
    }
    matrices = {}
    for analysis in analyses:
        correct = matrix_sums[analysis]["correct"].cpu().numpy() / observation_count
        wrong = matrix_sums[analysis]["wrong"].cpu().numpy() / observation_count
        matrices[analysis] = {"correct": correct, "margin": correct - wrong}

    source_score_approximation = {}
    for metric in ("correct", "margin"):
        exact = sequence_arrays["hard_data"][f"{metric}_total"]
        approximate = sequence_arrays["source_components"][f"{metric}_total"]
        error = approximate - exact
        source_score_approximation[metric] = {
            "relative_rmse": float(
                np.sqrt(np.mean(np.square(error)))
                / max(np.sqrt(np.mean(np.square(exact))), 1e-30)
            ),
            "pearson": float(np.corrcoef(exact, approximate)[0, 1]),
            "exact_mean": float(exact.mean()),
            "approximate_mean": float(approximate.mean()),
            "mean_error": float(error.mean()),
        }

    assert component_metadata_reference is not None
    component_labels = [item[2] for item in component_metadata_reference]
    component_delays = np.array([item[1] for item in component_metadata_reference])
    component_mask_np = component_delays[None, :] - component_delays[:, None] == 1
    return {
        "labels": labels,
        "hard_delays": hard_delays,
        "hard_mask": hard_mask_np,
        "selected_delays": selected_delays,
        "retained_mass": retained_mass,
        "component_labels": component_labels,
        "component_delays": component_delays,
        "component_mask": component_mask_np,
        "sequence_arrays": sequence_arrays,
        "matrices": matrices,
        "source_score_approximation": source_score_approximation,
        "query_first": int(query_positions[0]),
        "query_last": int(query_positions[-1]),
        "n_query_positions": len(query_positions),
        "wrong_offsets": wrong_offsets,
        "maximum_residual_relative_error": maximum_residual_error,
        "maximum_full_score_absolute_error": maximum_full_score_error,
        "maximum_data_score_absolute_error": maximum_data_score_error,
        "maximum_source_score_absolute_error": maximum_source_score_error,
    }


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
    parser.add_argument("--maximum-source-delay", type=int, default=100)
    parser.add_argument("--source-mass-threshold", type=float, default=0.90)
    parser.add_argument("--maximum-components-per-path", type=int, default=32)
    parser.add_argument("--wrong-deltas", type=parse_int_tuple, default=(-3, -2, -1, 1, 2, 3))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment_outputs/delay_pair_score_contributions"),
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.n_train_per_lag = min(args.n_train_per_lag, 8)
        args.n_test_per_lag = min(args.n_test_per_lag, 8)
        args.query_stride = max(args.query_stride, 4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    group_rows = []
    ranked_rows = []
    path_delay_rows = []
    diagnostics = {}
    for model_name in args.models:
        n_layers, checkpoint = MODEL_SPECS[model_name]
        model = load_attention_only_model(n_layers, checkpoint, device)
        for lag_index, lag in enumerate(args.lags):
            print(f"{model_name}, lag {lag}")
            train_inputs = make_inputs(
                lag,
                args.n_train_per_lag,
                args.sequence_length,
                args.rho,
                args.seed + 10_000 * n_layers + 1_000 * lag_index,
            )
            test_inputs = make_inputs(
                lag,
                args.n_test_per_lag,
                args.sequence_length,
                args.rho,
                args.seed + 500_000 + 10_000 * n_layers + 1_000 * lag_index,
            )
            result = analyze_model_lag(
                model_name,
                model,
                lag,
                train_inputs,
                test_inputs,
                args.burn_in,
                args.query_stride,
                args.maximum_source_delay,
                args.source_mass_threshold,
                args.maximum_components_per_path,
                args.wrong_deltas,
                args.batch_size,
                device,
            )
            masks = {
                "hard_full": result["hard_mask"],
                "hard_data": result["hard_mask"],
                "source_components": result["component_mask"],
            }
            item_labels = {
                "hard_full": result["labels"],
                "hard_data": result["labels"],
                "source_components": result["component_labels"],
            }
            item_delays = {
                "hard_full": result["hard_delays"],
                "hard_data": result["hard_delays"],
                "source_components": result["component_delays"],
            }
            for analysis in ("hard_full", "hard_data", "source_components"):
                arrays = result["sequence_arrays"][analysis]
                for metric in ("correct", "margin"):
                    summary = summarize_paired_groups(
                        arrays[f"{metric}_plus_pair_mean"],
                        arrays[f"{metric}_other_pair_mean"],
                    )
                    matrix = result["matrices"][analysis][metric]
                    mask = masks[analysis]
                    ranking = np.argsort(matrix.ravel())[::-1]
                    top_count = min(10, matrix.size)
                    top_plus_rate = float(mask.ravel()[ranking[:top_count]].mean())
                    summary.update(
                        {
                            "model": model_name,
                            "lag": lag,
                            "analysis": analysis,
                            "metric": metric,
                            "n_items": matrix.shape[0],
                            "n_pairs": matrix.size,
                            "n_plus_one_pairs": int(mask.sum()),
                            "plus_one_pair_fraction": float(mask.mean()),
                            "top10_plus_one_fraction": top_plus_rate,
                            "plus_total_mean": float(
                                arrays[f"{metric}_plus_total"].mean()
                            ),
                            "other_total_mean": float(
                                arrays[f"{metric}_other_total"].mean()
                            ),
                            "complete_total_mean": float(
                                arrays[f"{metric}_total"].mean()
                            ),
                            "source_vs_complete_data_relative_rmse": (
                                result["source_score_approximation"][metric][
                                    "relative_rmse"
                                ]
                                if analysis == "source_components"
                                else float("nan")
                            ),
                            "source_vs_complete_data_pearson": (
                                result["source_score_approximation"][metric][
                                    "pearson"
                                ]
                                if analysis == "source_components"
                                else float("nan")
                            ),
                        }
                    )
                    group_rows.append(summary)
                    ranked_rows.extend(
                        ranked_pair_rows(
                            model_name,
                            lag,
                            analysis,
                            metric,
                            matrix,
                            item_labels[analysis],
                            item_delays[analysis],
                            mask,
                        )
                    )

            for path_index, label in enumerate(result["labels"]):
                path_delay_rows.append(
                    {
                        "model": model_name,
                        "lag": lag,
                        "path": label,
                        "hard_dominant_delay": int(result["hard_delays"][path_index]),
                        "selected_source_delays": ",".join(
                            str(value) for value in result["selected_delays"][label]
                        ),
                        "selected_train_mean_routing_mass": result["retained_mass"][label],
                    }
                )

            diagnostics[f"{model_name}_lag{lag}"] = {
                "checkpoint": str(checkpoint),
                "query_first": result["query_first"],
                "query_last": result["query_last"],
                "n_query_positions": result["n_query_positions"],
                "wrong_offsets": list(result["wrong_offsets"]),
                "n_paths": len(result["labels"]),
                "n_source_components": len(result["component_labels"]),
                "maximum_residual_relative_error": result[
                    "maximum_residual_relative_error"
                ],
                "maximum_full_score_absolute_error": result[
                    "maximum_full_score_absolute_error"
                ],
                "maximum_data_score_absolute_error": result[
                    "maximum_data_score_absolute_error"
                ],
                "maximum_source_score_absolute_error": result[
                    "maximum_source_score_absolute_error"
                ],
                "source_score_approximation": result[
                    "source_score_approximation"
                ],
            }
            plot_group_comparison(
                model_name,
                lag,
                result["sequence_arrays"],
                args.output_dir / f"{model_name}_lag{lag}_group_comparison.png",
            )
            plot_pair_matrices(
                model_name,
                lag,
                result["matrices"],
                masks,
                item_labels,
                args.output_dir / f"{model_name}_lag{lag}_pair_matrices.png",
            )

    save_csv(args.output_dir / "group_comparison.csv", group_rows)
    save_csv(args.output_dir / "ranked_pair_contributions.csv", ranked_rows)
    save_csv(args.output_dir / "path_delay_definitions.csv", path_delay_rows)
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "models": list(args.models),
                "lags": list(args.lags),
                "n_train_per_lag": args.n_train_per_lag,
                "n_test_per_lag": args.n_test_per_lag,
                "sequence_length": args.sequence_length,
                "rho": args.rho,
                "burn_in": args.burn_in,
                "query_stride": args.query_stride,
                "maximum_source_delay": args.maximum_source_delay,
                "source_mass_threshold": args.source_mass_threshold,
                "maximum_components_per_path": args.maximum_components_per_path,
                "wrong_deltas": list(args.wrong_deltas),
                "seed": args.seed,
                "diagnostics": diagnostics,
            },
            indent=2,
        )
    )

    print("\nGroup comparison")
    for row in group_rows:
        if row["metric"] != "margin":
            continue
        print(
            f"{row['model']} {row['analysis']}: +1={row['plus_mean']:+.4g}, "
            f"other={row['other_mean']:+.4g}, diff={row['mean_difference']:+.4g}, "
            f"d={row['paired_cohen_d']:+.2f}, p={row['paired_t_pvalue']:.2g}, "
            f"top10 +1={100 * row['top10_plus_one_fraction']:.0f}%"
        )
    print(f"saved outputs to {args.output_dir}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
