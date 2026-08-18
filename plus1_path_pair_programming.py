"""Causally program the final locator with the +1 source-delay score channel.

The residual entering Layer 4 is expanded through the frozen attention
patterns of Layers 1--3.  Its scalar-data portion is then resolved exactly by
original source position.  For a final query/key pair, this gives many
bilinear score terms.  This script isolates the terms for which

    key source delay - query source delay == +1.

At final-head offset D=L-1, those terms compare original sequence values that
are L positions apart.  The intervention moves their *observed score field*
from the natural offset L-1 to a requested offset M-1:

    1. subtract the +1 contribution at the natural offset;
    2. replace the +1 contribution at the target offset with the natural one;
    3. optionally multiply the transplanted contribution by a gain;
    4. rerun final-layer softmax, OV, residual addition, and W_U.

This is a final-logit/edge intervention conditional on the clean earlier-layer
activations.  It tests whether the +1 channel is sufficient to relocate the
final attention stripe and recover the programmed-lag MSE grid.  It does not
by itself intervene on how Layers 1--3 constructed the source components.

An offset-difference-zero channel is transplanted as a magnitude-matched
negative control.  A hard one-hot final-attention stripe is also evaluated as
the established upper-bound programming intervention.

Usage:
    python plus1_path_pair_programming.py --quick --no-show
    python plus1_path_pair_programming.py --no-show
    python plus1_path_pair_programming.py --lags 10,20,40,60,80 --gains 1,4,8,16
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from four_layer_attention_path_analysis import load_model
from path_source_delay_sparsity import labels_for_model, path_direction
from util import make_dataset_lagset


def parse_int_tuple(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def parse_float_tuple(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("gains must be positive")
    return values


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def all_path_routings(
    attentions: list[torch.Tensor],
    n_prefinal_layers: int,
) -> OrderedDict[str, torch.Tensor]:
    """Exact frozen-attention source routing matrix for every residual path."""
    batch_size, sequence_length, _ = attentions[0].shape
    identity = torch.eye(
        sequence_length,
        dtype=attentions[0].dtype,
        device=attentions[0].device,
    ).expand(batch_size, -1, -1)
    paths: OrderedDict[str, torch.Tensor] = OrderedDict({"": identity})
    for layer_index in range(n_prefinal_layers):
        expanded: OrderedDict[str, torch.Tensor] = OrderedDict()
        for label, routing in paths.items():
            expanded[label + "0"] = routing
            expanded[label + "1"] = torch.bmm(
                attentions[layer_index], routing
            )
        paths = expanded
    return paths


def path_geometry(
    model,
    directions: torch.Tensor,
    offset: int,
) -> torch.Tensor:
    """Final QK/RoPE bilinear coefficient between data-path directions."""
    query_matrix, key_matrix, _, _ = model.layers[-1][0]
    query_positions = torch.full(
        (len(directions),),
        offset,
        dtype=torch.long,
        device=directions.device,
    )
    key_positions = torch.zeros_like(query_positions)
    queries = model.apply_rope(directions @ query_matrix, query_positions)
    keys = model.apply_rope(directions @ key_matrix, key_positions)
    return queries @ keys.T / math.sqrt(model.d_head)


def source_delay_difference_channel(
    model,
    inputs: torch.Tensor,
    routings: torch.Tensor,
    directions: torch.Tensor,
    offset: int,
    delay_difference: int,
) -> torch.Tensor:
    """Score terms with key_delay - query_delay == delay_difference.

    The result has shape batch x sequence.  Entry [b, q] is this channel's
    contribution to the final logit comparing query q with key q-offset.
    Entries without enough causal history are zero.
    """
    if delay_difference < 0:
        raise ValueError("this experiment currently expects a nonnegative difference")
    batch_size, sequence_length = inputs.shape
    result = torch.zeros(
        batch_size,
        sequence_length,
        dtype=inputs.dtype,
        device=inputs.device,
    )
    source_separation = offset + delay_difference
    if source_separation >= sequence_length or offset >= sequence_length:
        return result

    query_positions = torch.arange(
        offset, sequence_length, device=inputs.device
    )
    query_sources = torch.arange(
        source_separation, sequence_length, device=inputs.device
    )
    key_positions = query_positions - offset
    key_sources = query_sources - source_separation

    # path x batch x query-position x source-position
    query_routing = routings[:, :, query_positions, :]
    query_routing = query_routing[:, :, :, query_sources]
    key_routing = routings[:, :, key_positions, :]
    key_routing = key_routing[:, :, :, key_sources]

    query_values = inputs[:, query_sources]
    key_values = inputs[:, key_sources]
    query_amplitudes = query_routing * query_values[None, :, None, :]
    key_amplitudes = key_routing * key_values[None, :, None, :]
    geometry = path_geometry(model, directions, offset)
    result[:, query_positions] = torch.einsum(
        "pbqs,pr,rbqs->bq",
        query_amplitudes,
        geometry,
        key_amplitudes,
    )
    return result


def explicit_channel_check(
    geometry: torch.Tensor,
    inputs: torch.Tensor,
    routings: torch.Tensor,
    vectorized: torch.Tensor,
    offset: int,
    delay_difference: int,
) -> float:
    """Small direct sum used to independently check the vectorized formula."""
    batch_index = 0
    sequence_length = inputs.shape[1]
    first_query = max(offset + delay_difference, offset)
    candidate_queries = sorted(
        set(
            query
            for query in (
                first_query,
                (first_query + sequence_length - 1) // 2,
                sequence_length - 1,
            )
            if 0 <= query < sequence_length
        )
    )
    maximum_error = 0.0
    for query in candidate_queries:
        key = query - offset
        total = torch.zeros((), dtype=inputs.dtype, device=inputs.device)
        for query_delay in range(query + 1):
            key_delay = query_delay + delay_difference
            key_source = key - key_delay
            if key_source < 0:
                continue
            query_source = query - query_delay
            query_amplitude = (
                routings[:, batch_index, query, query_source]
                * inputs[batch_index, query_source]
            )
            key_amplitude = (
                routings[:, batch_index, key, key_source]
                * inputs[batch_index, key_source]
            )
            total = total + query_amplitude @ geometry @ key_amplitude
        error = abs(float(total.item() - vectorized[batch_index, query].item()))
        maximum_error = max(maximum_error, error)
    return maximum_error


def final_layer_quantities(model, residual: torch.Tensor) -> dict[str, torch.Tensor]:
    batch_size, sequence_length, _ = residual.shape
    positions = torch.arange(sequence_length, device=residual.device)
    positions = positions.unsqueeze(0).expand(batch_size, -1)
    query_matrix, key_matrix, value_matrix, output_matrix = model.layers[-1][0]
    queries = model.apply_rope(residual @ query_matrix, positions)
    keys = model.apply_rope(residual @ key_matrix, positions)
    logits = queries @ keys.transpose(-2, -1) / math.sqrt(model.d_head)
    index = torch.arange(sequence_length, device=residual.device)
    causal_mask = index.unsqueeze(0) > index.unsqueeze(1)
    logits = logits.masked_fill(causal_mask, -torch.inf)
    attention = F.softmax(logits, dim=-1)
    values = residual @ value_matrix
    write = (attention @ values) @ output_matrix
    prediction = model.W_U(residual + write).squeeze(-1)
    return {
        "logits": logits,
        "attention": attention,
        "values": values,
        "prediction": prediction,
    }


def prediction_from_logits(
    model,
    residual: torch.Tensor,
    values: torch.Tensor,
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    attention = F.softmax(logits, dim=-1)
    _, _, _, output_matrix = model.layers[-1][0]
    write = (attention @ values) @ output_matrix
    prediction = model.W_U(residual + write).squeeze(-1)
    return prediction, attention


def stripe_attention(
    batch_size: int,
    sequence_length: int,
    programmed_lag: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    offset = programmed_lag - 1
    query = torch.arange(sequence_length, device=device)
    key = (query - offset).clamp_min(0)
    attention = torch.zeros(
        batch_size,
        sequence_length,
        sequence_length,
        dtype=dtype,
        device=device,
    )
    attention[:, query, key] = 1.0
    return attention


def prediction_from_attention(
    model,
    residual: torch.Tensor,
    values: torch.Tensor,
    attention: torch.Tensor,
) -> torch.Tensor:
    _, _, _, output_matrix = model.layers[-1][0]
    write = (attention @ values) @ output_matrix
    return model.W_U(residual + write).squeeze(-1)


def replace_channel_at_target(
    clean_logits: torch.Tensor,
    donor_channel: torch.Tensor,
    target_channel: torch.Tensor,
    donor_offset: int,
    target_offset: int,
    gain: float,
    inserted_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Remove the donor stripe and insert its score at the target stripe."""
    patched = clean_logits.clone()
    sequence_length = clean_logits.shape[-1]
    if inserted_scale is None:
        inserted_scale = torch.ones(
            clean_logits.shape[0],
            dtype=clean_logits.dtype,
            device=clean_logits.device,
        )

    if donor_offset == target_offset:
        query = torch.arange(donor_offset, sequence_length, device=patched.device)
        key = query - donor_offset
        replacement = gain * inserted_scale[:, None] * donor_channel[:, query]
        patched[:, query, key] += replacement - target_channel[:, query]
        return patched

    donor_query = torch.arange(
        donor_offset, sequence_length, device=patched.device
    )
    donor_key = donor_query - donor_offset
    patched[:, donor_query, donor_key] -= donor_channel[:, donor_query]

    target_query = torch.arange(
        target_offset, sequence_length, device=patched.device
    )
    target_key = target_query - target_offset
    replacement = gain * inserted_scale[:, None] * donor_channel[:, target_query]
    patched[:, target_query, target_key] += (
        replacement - target_channel[:, target_query]
    )
    return patched


def ablate_natural_channel(
    clean_logits: torch.Tensor,
    channel: torch.Tensor,
    offset: int,
) -> torch.Tensor:
    patched = clean_logits.clone()
    query = torch.arange(offset, clean_logits.shape[-1], device=clean_logits.device)
    key = query - offset
    patched[:, query, key] -= channel[:, query]
    return patched


def per_sequence_mse(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    true_lag: int,
    programmed_lag: int,
    burn_in: int,
) -> torch.Tensor:
    start = max(true_lag, programmed_lag) + burn_in
    return (prediction[:, start:] - targets[:, start:]).square().mean(dim=1)


def attention_metrics(
    attention: torch.Tensor,
    true_lag: int,
    programmed_lag: int,
    burn_in: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    start = max(true_lag, programmed_lag) + burn_in
    query = torch.arange(start, attention.shape[-1], device=attention.device)
    target_key = query - (programmed_lag - 1)
    target_mass = attention[:, query, target_key].mean(dim=1)
    row_wins = (
        attention[:, query, :].argmax(dim=-1) == target_key[None, :]
    ).double().mean(dim=1)
    return target_mass, row_wins


def rms_match_scale(
    reference: torch.Tensor,
    control: torch.Tensor,
    start: int,
) -> torch.Tensor:
    reference_rms = reference[:, start:].square().mean(dim=1).sqrt()
    control_rms = control[:, start:].square().mean(dim=1).sqrt()
    return reference_rms / control_rms.clamp_min(1e-12)


def summarize(values: torch.Tensor) -> dict[str, float]:
    return {
        "mean": float(values.mean().item()),
        "median": float(values.median().item()),
        "p05": float(torch.quantile(values, 0.05).item()),
        "p95": float(torch.quantile(values, 0.95).item()),
    }


def plot_grids(
    lags: tuple[int, ...],
    grids: dict[str, np.ndarray],
    title: str,
    output_path: Path,
) -> plt.Figure:
    columns = min(3, len(grids))
    rows = math.ceil(len(grids) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.2 * columns, 4.6 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    axes_flat = axes.reshape(-1)
    all_values = np.concatenate([grid.ravel() for grid in grids.values()])
    low = float(np.nanmin(all_values))
    high = float(np.nanpercentile(all_values, 95))
    if high <= low:
        high = low + 1e-12
    image = None
    if len(lags) <= 20:
        tick_indices = np.arange(len(lags))
    else:
        tick_indices = np.unique(
            np.linspace(0, len(lags) - 1, 11, dtype=int)
        )
    tick_labels = [lags[index] for index in tick_indices]
    for axis, (label, grid) in zip(axes_flat, grids.items()):
        image = axis.imshow(
            grid,
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=low,
            vmax=high,
        )
        # The image still contains one measured cell per integer lag.  Only the
        # text ticks are thinned when the grid is dense.
        axis.set_xticks(tick_indices, tick_labels, rotation=60)
        axis.set_yticks(tick_indices, tick_labels)
        axis.set_xlabel("programmed lag")
        axis.set_ylabel("true data lag")
        diagonal = np.diag(grid)
        axis.set_title(
            f"{label}\ndiagonal mean={diagonal.mean():.3f}"
        )
    for axis in axes_flat[len(grids):]:
        axis.remove()
    if image is not None:
        figure.colorbar(
            image,
            ax=axes_flat[: len(grids)].tolist(),
            label="mean next-value MSE",
            shrink=0.82,
        )
    figure.suptitle(title)
    figure.savefig(output_path, dpi=180)
    return figure


def plot_single_dense_grid(
    lags: tuple[int, ...],
    grid: np.ndarray,
    title: str,
    output_path: Path,
) -> plt.Figure:
    """Large single-panel rendering for an integer-by-integer lag grid."""
    figure, axis = plt.subplots(figsize=(10.5, 9), constrained_layout=True)
    high = float(np.nanpercentile(grid, 95))
    low = float(np.nanmin(grid))
    image = axis.imshow(
        grid,
        origin="lower",
        aspect="equal",
        cmap="magma",
        vmin=low,
        vmax=max(high, low + 1e-12),
        interpolation="nearest",
    )
    if len(lags) <= 25:
        tick_indices = np.arange(len(lags))
    else:
        tick_indices = np.unique(np.linspace(0, len(lags) - 1, 11, dtype=int))
    tick_labels = [lags[index] for index in tick_indices]
    axis.set_xticks(tick_indices, tick_labels)
    axis.set_yticks(tick_indices, tick_labels)
    axis.set_xlabel("programmed lag")
    axis.set_ylabel("true data lag")
    axis.set_title(f"{title}\nmean diagonal MSE={np.diag(grid).mean():.4f}")
    figure.colorbar(image, ax=axis, label="mean next-value MSE", shrink=0.86)
    figure.savefig(output_path, dpi=220)
    return figure


@torch.inference_mode()
def run(args: argparse.Namespace) -> tuple[list[dict], list[dict], dict]:
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    if model.n_layers != 4:
        raise ValueError("this first causal grid is intentionally the 4L model")
    lags = args.lags
    labels = labels_for_model(model)
    directions = torch.stack([path_direction(model, label) for label in labels])

    grid_rows: list[dict] = []
    ablation_rows: list[dict] = []
    diagnostics: dict[str, float | int | list] = {
        "path_labels": labels,
        "n_paths": len(labels),
        "source_delay_difference_intervened_on": 1,
        "control_source_delay_difference": 0,
        "maximum_clean_attention_error": 0.0,
        "maximum_clean_prediction_error": 0.0,
        "maximum_path_routing_reconstruction_error": 0.0,
        "maximum_vectorized_channel_error": 0.0,
    }

    for true_lag in lags:
        print(f"true lag {true_lag}")
        inputs, targets, sampled_lags = make_dataset_lagset(
            args.n_sequences,
            args.sequence_length,
            args.rho,
            [true_lag],
            seed=args.seed + true_lag,
        )
        if not torch.all(sampled_lags == true_lag):
            raise RuntimeError("dataset returned an unexpected lag")
        inputs = inputs.to(device=device, dtype=torch.float64)
        targets = targets.to(device=device, dtype=torch.float64)

        cell_store: dict[tuple[str, float | None, int], dict[str, list[torch.Tensor]]] = {}
        ablation_store: dict[str, list[torch.Tensor]] = {
            "clean": [],
            "plus1_natural_ablation": [],
            "difference0_natural_ablation": [],
        }
        natural_offset = true_lag - 1

        for batch_start in range(0, args.n_sequences, args.batch_size):
            batch = inputs[batch_start : batch_start + args.batch_size]
            batch_targets = targets[batch_start : batch_start + args.batch_size]
            clean_prediction, attentions, post_attention, post_mlp = model(batch)
            if any(not torch.equal(a, b) for a, b in zip(post_attention, post_mlp)):
                raise RuntimeError("checkpoint is not attention-only")
            final_input = post_attention[-2]
            final = final_layer_quantities(model, final_input)
            diagnostics["maximum_clean_attention_error"] = max(
                float(diagnostics["maximum_clean_attention_error"]),
                float((final["attention"] - attentions[-1]).abs().max().item()),
            )
            diagnostics["maximum_clean_prediction_error"] = max(
                float(diagnostics["maximum_clean_prediction_error"]),
                float((final["prediction"] - clean_prediction).abs().max().item()),
            )

            routing_dict = all_path_routings(
                attentions, model.n_layers - 1
            )
            if list(routing_dict) != labels:
                raise RuntimeError("path-routing labels do not match path directions")
            routings = torch.stack(list(routing_dict.values()))

            data_embedding = batch.unsqueeze(-1) * model.W_r.weight[:, 0]
            reconstructed_data = torch.zeros_like(data_embedding)
            for path_index, label in enumerate(labels):
                amplitudes = torch.bmm(routings[path_index], batch.unsqueeze(-1))
                reconstructed_data += amplitudes * directions[path_index]
            direct_data_paths = data_embedding
            paths = OrderedDict({"": direct_data_paths})
            for layer_index in range(model.n_layers - 1):
                expanded: OrderedDict[str, torch.Tensor] = OrderedDict()
                _, _, value_matrix, output_matrix = model.layers[layer_index][0]
                for label, component in paths.items():
                    expanded[label + "0"] = component
                    expanded[label + "1"] = (
                        attentions[layer_index] @ (component @ value_matrix)
                    ) @ output_matrix
                paths = expanded
            direct_data = torch.stack(list(paths.values())).sum(dim=0)
            routing_error = (
                (reconstructed_data - direct_data).norm()
                / direct_data.norm().clamp_min(1e-30)
            )
            diagnostics["maximum_path_routing_reconstruction_error"] = max(
                float(diagnostics["maximum_path_routing_reconstruction_error"]),
                float(routing_error.item()),
            )

            needed_offsets = tuple(sorted({lag - 1 for lag in lags}))
            plus_channels: dict[int, torch.Tensor] = {}
            control_channels: dict[int, torch.Tensor] = {}
            for offset in needed_offsets:
                plus_channels[offset] = source_delay_difference_channel(
                    model, batch, routings, directions, offset, 1
                )
                if args.with_control:
                    control_channels[offset] = source_delay_difference_channel(
                        model, batch, routings, directions, offset, 0
                    )

            geometry = path_geometry(model, directions, natural_offset)
            explicit_error = explicit_channel_check(
                geometry,
                batch,
                routings,
                plus_channels[natural_offset],
                natural_offset,
                1,
            )
            diagnostics["maximum_vectorized_channel_error"] = max(
                float(diagnostics["maximum_vectorized_channel_error"]),
                explicit_error,
            )

            clean_mse = per_sequence_mse(
                final["prediction"], batch_targets, true_lag, true_lag, args.burn_in
            )
            ablation_store["clean"].append(clean_mse.cpu())
            plus_ablated_logits = ablate_natural_channel(
                final["logits"], plus_channels[natural_offset], natural_offset
            )
            plus_ablated_prediction, _ = prediction_from_logits(
                model, final_input, final["values"], plus_ablated_logits
            )
            ablation_store["plus1_natural_ablation"].append(
                per_sequence_mse(
                    plus_ablated_prediction,
                    batch_targets,
                    true_lag,
                    true_lag,
                    args.burn_in,
                ).cpu()
            )
            if args.with_control:
                control_ablated_logits = ablate_natural_channel(
                    final["logits"], control_channels[natural_offset], natural_offset
                )
                control_ablated_prediction, _ = prediction_from_logits(
                    model, final_input, final["values"], control_ablated_logits
                )
                ablation_store["difference0_natural_ablation"].append(
                    per_sequence_mse(
                        control_ablated_prediction,
                        batch_targets,
                        true_lag,
                        true_lag,
                        args.burn_in,
                    ).cpu()
                )

            for programmed_lag in lags:
                target_offset = programmed_lag - 1
                evaluation_start = max(true_lag, programmed_lag) + args.burn_in
                control_scale = None
                if args.with_control:
                    control_scale = rms_match_scale(
                        plus_channels[natural_offset],
                        control_channels[natural_offset],
                        evaluation_start,
                    )

                clean_cell_mse = per_sequence_mse(
                    final["prediction"],
                    batch_targets,
                    true_lag,
                    programmed_lag,
                    args.burn_in,
                )
                clean_mass, clean_wins = attention_metrics(
                    final["attention"], true_lag, programmed_lag, args.burn_in
                )
                key = ("clean", None, programmed_lag)
                store = cell_store.setdefault(
                    key, {"mse": [], "mass": [], "wins": []}
                )
                store["mse"].append(clean_cell_mse.cpu())
                store["mass"].append(clean_mass.cpu())
                store["wins"].append(clean_wins.cpu())

                hard_attention = stripe_attention(
                    len(batch),
                    args.sequence_length,
                    programmed_lag,
                    batch.dtype,
                    batch.device,
                )
                hard_prediction = prediction_from_attention(
                    model, final_input, final["values"], hard_attention
                )
                hard_mse = per_sequence_mse(
                    hard_prediction,
                    batch_targets,
                    true_lag,
                    programmed_lag,
                    args.burn_in,
                )
                hard_mass, hard_wins = attention_metrics(
                    hard_attention, true_lag, programmed_lag, args.burn_in
                )
                key = ("hard_stripe", None, programmed_lag)
                store = cell_store.setdefault(
                    key, {"mse": [], "mass": [], "wins": []}
                )
                store["mse"].append(hard_mse.cpu())
                store["mass"].append(hard_mass.cpu())
                store["wins"].append(hard_wins.cpu())

                for gain in args.gains:
                    patched_logits = replace_channel_at_target(
                        final["logits"],
                        plus_channels[natural_offset],
                        plus_channels[target_offset],
                        natural_offset,
                        target_offset,
                        gain,
                    )
                    prediction, attention = prediction_from_logits(
                        model, final_input, final["values"], patched_logits
                    )
                    mse = per_sequence_mse(
                        prediction,
                        batch_targets,
                        true_lag,
                        programmed_lag,
                        args.burn_in,
                    )
                    mass, wins = attention_metrics(
                        attention, true_lag, programmed_lag, args.burn_in
                    )
                    key = ("plus1", gain, programmed_lag)
                    store = cell_store.setdefault(
                        key, {"mse": [], "mass": [], "wins": []}
                    )
                    store["mse"].append(mse.cpu())
                    store["mass"].append(mass.cpu())
                    store["wins"].append(wins.cpu())

                    if args.with_control:
                        control_logits = replace_channel_at_target(
                            final["logits"],
                            control_channels[natural_offset],
                            control_channels[target_offset],
                            natural_offset,
                            target_offset,
                            gain,
                            inserted_scale=control_scale,
                        )
                        control_prediction, control_attention = prediction_from_logits(
                            model, final_input, final["values"], control_logits
                        )
                        control_mse = per_sequence_mse(
                            control_prediction,
                            batch_targets,
                            true_lag,
                            programmed_lag,
                            args.burn_in,
                        )
                        control_mass, control_wins = attention_metrics(
                            control_attention,
                            true_lag,
                            programmed_lag,
                            args.burn_in,
                        )
                        key = ("difference0_control", gain, programmed_lag)
                        store = cell_store.setdefault(
                            key, {"mse": [], "mass": [], "wins": []}
                        )
                        store["mse"].append(control_mse.cpu())
                        store["mass"].append(control_mass.cpu())
                        store["wins"].append(control_wins.cpu())

        for condition, chunks in ablation_store.items():
            if not chunks:
                continue
            values = torch.cat(chunks)
            stats = summarize(values)
            ablation_rows.append(
                {
                    "true_lag": true_lag,
                    "condition": condition,
                    "mse_mean": stats["mean"],
                    "mse_median": stats["median"],
                    "mse_p05": stats["p05"],
                    "mse_p95": stats["p95"],
                    "n_sequences": len(values),
                }
            )

        for (condition, gain, programmed_lag), measurements in cell_store.items():
            mse = torch.cat(measurements["mse"])
            mass = torch.cat(measurements["mass"])
            wins = torch.cat(measurements["wins"])
            mse_stats = summarize(mse)
            grid_rows.append(
                {
                    "true_lag": true_lag,
                    "programmed_lag": programmed_lag,
                    "condition": condition,
                    "gain": "" if gain is None else gain,
                    "mse_mean": mse_stats["mean"],
                    "mse_median": mse_stats["median"],
                    "mse_p05": mse_stats["p05"],
                    "mse_p95": mse_stats["p95"],
                    "target_attention_mass_mean": float(mass.mean().item()),
                    "target_row_top1_rate_mean": float(wins.mean().item()),
                    "n_sequences": len(mse),
                }
            )

    return grid_rows, ablation_rows, diagnostics


def grid_from_rows(
    rows: list[dict],
    lags: tuple[int, ...],
    condition: str,
    gain: float | None,
    value: str,
) -> np.ndarray:
    lookup = {}
    for row in rows:
        row_gain = None if row["gain"] == "" else float(row["gain"])
        if row["condition"] == condition and row_gain == gain:
            lookup[(int(row["true_lag"]), int(row["programmed_lag"]))] = float(
                row[value]
            )
    return np.asarray(
        [[lookup[(true_lag, programmed_lag)] for programmed_lag in lags]
         for true_lag in lags],
        dtype=float,
    )


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
        default=(7, 10, 13, 19, 29, 40, 50, 60, 70, 80, 90, 100, 110, 120),
    )
    parser.add_argument(
        "--dense-lag-range",
        type=parse_int_tuple,
        default=None,
        metavar="MIN,MAX",
        help=(
            "replace --lags with every integer lag from MIN through MAX; "
            "for example, --dense-lag-range 1,100"
        ),
    )
    parser.add_argument(
        "--gains", type=parse_float_tuple, default=(1.0, 4.0, 8.0, 16.0)
    )
    parser.add_argument("--n-sequences", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=240)
    parser.add_argument("--rho", type=float, default=0.9)
    parser.add_argument("--burn-in", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment_outputs/plus1_path_pair_programming"),
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--no-control",
        dest="with_control",
        action="store_false",
        help="skip the magnitude-matched delay-difference-zero control",
    )
    parser.set_defaults(with_control=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if args.dense_lag_range is not None:
        if len(args.dense_lag_range) != 2:
            raise ValueError("--dense-lag-range requires exactly MIN,MAX")
        minimum_lag, maximum_lag = args.dense_lag_range
        if maximum_lag < minimum_lag:
            raise ValueError("dense lag maximum must be at least its minimum")
        args.lags = tuple(range(minimum_lag, maximum_lag + 1))

    if any(lag < 1 for lag in args.lags):
        raise ValueError("lags must be positive")
    if max(args.lags) + args.burn_in >= args.sequence_length:
        raise ValueError("sequence length must exceed maximum lag plus burn-in")
    if args.quick:
        default_quick = (10, 40, 70, 100)
        args.lags = tuple(lag for lag in default_quick if lag in args.lags)
        if len(args.lags) < 2:
            args.lags = tuple(sorted(set((args.lags[0],)))) if args.lags else (10, 40)
        args.n_sequences = min(args.n_sequences, 4)
        args.batch_size = min(args.batch_size, 2)

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={args.device}; checkpoint={args.checkpoint}")
    print(f"lags={args.lags}; gains={args.gains}; n={args.n_sequences}")

    grid_rows, ablation_rows, diagnostics = run(args)
    save_csv(args.output_dir / "programming_grid.csv", grid_rows)
    save_csv(args.output_dir / "natural_offset_ablation.csv", ablation_rows)

    primary_grids = {
        "clean": grid_from_rows(grid_rows, args.lags, "clean", None, "mse_mean"),
        "hard one-hot stripe": grid_from_rows(
            grid_rows, args.lags, "hard_stripe", None, "mse_mean"
        ),
    }
    for gain in args.gains:
        primary_grids[f"+1 transfer, gain {gain:g}"] = grid_from_rows(
            grid_rows, args.lags, "plus1", gain, "mse_mean"
        )
    figure = plot_grids(
        args.lags,
        primary_grids,
        "4L final-head programming with exact +1 source-delay score terms",
        args.output_dir / "plus1_programming_mse_grids.png",
    )
    if args.no_show:
        plt.close(figure)

    strongest_gain = max(args.gains)
    dense_grid = grid_from_rows(
        grid_rows, args.lags, "plus1", strongest_gain, "mse_mean"
    )
    dense_figure = plot_single_dense_grid(
        args.lags,
        dense_grid,
        f"Exact +1 source-delay programming, gain {strongest_gain:g}",
        args.output_dir / "plus1_programming_dense_grid.png",
    )
    if args.no_show:
        plt.close(dense_figure)

    if args.with_control:
        control_grids = {}
        for gain in args.gains:
            control_grids[f"+1 channel, gain {gain:g}"] = grid_from_rows(
                grid_rows, args.lags, "plus1", gain, "mse_mean"
            )
            control_grids[f"difference-0 control, gain {gain:g}"] = grid_from_rows(
                grid_rows,
                args.lags,
                "difference0_control",
                gain,
                "mse_mean",
            )
        control_figure = plot_grids(
            args.lags,
            control_grids,
            "Specificity control: transfer a magnitude-matched non-+1 channel",
            args.output_dir / "plus1_vs_control_mse_grids.png",
        )
        if args.no_show:
            plt.close(control_figure)

    summary_conditions = []
    for condition, gain in [
        ("clean", None),
        ("hard_stripe", None),
        *(("plus1", gain) for gain in args.gains),
        *(("difference0_control", gain) for gain in args.gains if args.with_control),
    ]:
        mse_grid = grid_from_rows(
            grid_rows, args.lags, condition, gain, "mse_mean"
        )
        mass_grid = grid_from_rows(
            grid_rows,
            args.lags,
            condition,
            gain,
            "target_attention_mass_mean",
        )
        win_grid = grid_from_rows(
            grid_rows,
            args.lags,
            condition,
            gain,
            "target_row_top1_rate_mean",
        )
        off_diagonal = ~np.eye(len(args.lags), dtype=bool)
        summary_conditions.append(
            {
                "condition": condition,
                "gain": gain,
                "diagonal_mse_mean": float(np.diag(mse_grid).mean()),
                "off_diagonal_mse_mean": float(mse_grid[off_diagonal].mean()),
                "diagonal_target_mass_mean": float(np.diag(mass_grid).mean()),
                "off_diagonal_target_mass_mean": float(mass_grid[off_diagonal].mean()),
                "diagonal_target_row_top1_rate": float(np.diag(win_grid).mean()),
                "off_diagonal_target_row_top1_rate": float(win_grid[off_diagonal].mean()),
            }
        )

    summary = {
        "checkpoint": str(args.checkpoint),
        "lags": list(args.lags),
        "gains": list(args.gains),
        "n_sequences_per_true_lag": args.n_sequences,
        "sequence_length": args.sequence_length,
        "rho": args.rho,
        "burn_in": args.burn_in,
        "known_lag_mse_floor": 1 - args.rho**2,
        "intervention": (
            "remove exact data-derived +1 source-delay contribution at natural "
            "offset; replace target-offset +1 contribution with the natural "
            "score vector times gain; rerun final softmax, OV, and W_U"
        ),
        "causal_scope": (
            "final-head logit/edge intervention conditional on clean Layers 1-3; "
            "not a residual-component intervention"
        ),
        "diagnostics": diagnostics,
        "condition_summaries": summary_conditions,
    }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    print("\nCondition summary:")
    for row in summary_conditions:
        gain = "" if row["gain"] is None else f", gain={row['gain']:g}"
        print(
            f"  {row['condition']}{gain}: diagonal MSE="
            f"{row['diagonal_mse_mean']:.4f}, off-diagonal MSE="
            f"{row['off_diagonal_mse_mean']:.4f}, off-diagonal target mass="
            f"{row['off_diagonal_target_mass_mean']:.3f}, target top1="
            f"{row['off_diagonal_target_row_top1_rate']:.1%}"
        )
    print("\nDiagnostics:")
    for key, value in diagnostics.items():
        if key != "path_labels":
            print(f"  {key}: {value}")
    print(f"\nSaved outputs to {args.output_dir.resolve()}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
