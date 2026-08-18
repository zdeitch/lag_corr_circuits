"""
Exact frozen-attention path decomposition of individual residual vectors.

For an N-layer attention-only model, this script expands the residual entering
the final layer through the N-1 earlier residual blocks. Each earlier block is

    identity + cached_attention_write

so a 7-layer model has 2**6 = 64 pre-final paths.

The cached attention matrices come from the clean forward pass. With those
matrices frozen, the value/output computation is linear and the path sum
exactly reconstructs the residual entering the final layer.

The decomposition is run three ways:

    full:      bias + scalar input direction
    data:      scalar input direction only
    bias:      embedding bias only

This separates transformation of the original one-dimensional data signal
from constant bias contributions.

Usage:
    python residual_path_contribution_norms.py
    python residual_path_contribution_norms.py --lag 40 --seed 0
    python residual_path_contribution_norms.py --positions 199,160,120
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

from model_class import AutocorrRoPE
from util import make_dataset_lagset


def parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def load_model(
    checkpoint: Path,
    n_layers: int,
    d_model: int,
    d_head: int,
    device: torch.device,
) -> AutocorrRoPE:
    model = AutocorrRoPE(
        d_model=d_model,
        d_head=d_head,
        n_layers=n_layers,
        n_heads=1,
        use_mlp=False,
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    assert model.n_heads == 1
    assert not model.use_mlp
    return model


def attention_write(
    component: torch.Tensor,
    attention: torch.Tensor,
    head_parameters: torch.nn.ParameterList,
) -> torch.Tensor:
    """Apply a frozen attention matrix and the head's V/O transformation."""
    _, _, value_matrix, output_matrix = head_parameters
    return (attention @ (component @ value_matrix)) @ output_matrix


def expand_paths(
    initial_component: torch.Tensor,
    attentions: list[torch.Tensor],
    model: AutocorrRoPE,
    n_expanded_layers: int,
    expected_residuals: list[torch.Tensor] | None = None,
) -> OrderedDict[str, torch.Tensor]:
    """
    Expand identity/write choices through the earlier attention layers.

    Bit convention, left to right:
        0 = identity route at that layer
        1 = frozen attention + OV write route at that layer
    """
    paths: OrderedDict[str, torch.Tensor] = OrderedDict({"": initial_component})

    for layer_index in range(n_expanded_layers):
        expanded: OrderedDict[str, torch.Tensor] = OrderedDict()
        attention = attentions[layer_index]
        head_parameters = model.layers[layer_index][0]

        for bits, component in paths.items():
            expanded[bits + "0"] = component
            expanded[bits + "1"] = attention_write(
                component,
                attention,
                head_parameters,
            )

        paths = expanded

        if expected_residuals is not None:
            reconstructed = torch.stack(list(paths.values()), dim=0).sum(dim=0)
            expected = expected_residuals[layer_index]
            relative_error = (
                (reconstructed - expected).norm()
                / expected.norm().clamp_min(1e-30)
            )
            if float(relative_error.item()) > 1e-10:
                raise RuntimeError(
                    f"path reconstruction failed after layer {layer_index + 1}: "
                    f"relative error={float(relative_error.item()):.3e}"
                )

    return paths


def final_attention_logits(
    model: AutocorrRoPE,
    residual: torch.Tensor,
) -> torch.Tensor:
    """Return final-head pre-softmax logits for one sequence."""
    _, sequence_length, _ = residual.shape
    positions = torch.arange(
        sequence_length,
        device=residual.device,
    ).unsqueeze(0)
    query_positions = positions.expand(residual.shape[0], sequence_length)

    query_matrix, key_matrix, _, _ = model.layers[-1][0]
    query = model.apply_rope(residual @ query_matrix, query_positions)
    key = model.apply_rope(residual @ key_matrix, query_positions)
    logits = query @ key.transpose(-2, -1) / math.sqrt(model.d_head)

    indices = torch.arange(sequence_length, device=residual.device)
    causal_mask = indices.unsqueeze(0) > indices.unsqueeze(1)
    return logits.masked_fill(causal_mask, float("-inf"))


def path_statistics(
    labels: list[str],
    full_paths: torch.Tensor,
    data_paths: torch.Tensor,
    bias_paths: torch.Tensor,
    residual: torch.Tensor,
) -> list[dict[str, float | int | str]]:
    residual_norm = residual.norm().clamp_min(1e-30)
    residual_norm_squared = residual.square().sum().clamp_min(1e-30)
    rows: list[dict[str, float | int | str]] = []

    for index, bits in enumerate(labels):
        full = full_paths[index]
        data = data_paths[index]
        bias = bias_paths[index]
        full_norm = full.norm()

        rows.append(
            {
                "path_index": index,
                "path_bits": bits,
                "write_count": bits.count("1"),
                "full_norm": float(full_norm.item()),
                "data_norm": float(data.norm().item()),
                "bias_norm": float(bias.norm().item()),
                "cosine_with_residual": float(
                    ((full @ residual) / (full_norm * residual_norm).clamp_min(1e-30))
                    .item()
                ),
                # These signed shares sum to one across all paths.
                "signed_alignment_share": float(
                    ((full @ residual) / residual_norm_squared).item()
                ),
                "data_alignment_share": float(
                    ((data @ residual) / residual_norm_squared).item()
                ),
                "bias_alignment_share": float(
                    ((bias @ residual) / residual_norm_squared).item()
                ),
            }
        )
    return rows


def aggregate_by_write_count(
    labels: list[str],
    full_paths: torch.Tensor,
    data_paths: torch.Tensor,
    bias_paths: torch.Tensor,
    residual: torch.Tensor,
) -> list[dict[str, float | int]]:
    residual_norm_squared = residual.square().sum().clamp_min(1e-30)
    rows: list[dict[str, float | int]] = []

    for write_count in range(len(labels[0]) + 1):
        indices = [
            index for index, bits in enumerate(labels) if bits.count("1") == write_count
        ]
        full_group = full_paths[indices]
        data_group = data_paths[indices]
        bias_group = bias_paths[indices]

        full_sum = full_group.sum(dim=0)
        data_sum = data_group.sum(dim=0)
        bias_sum = bias_group.sum(dim=0)

        rows.append(
            {
                "write_count": write_count,
                "n_paths": len(indices),
                # Root-sum-square summarizes individual path sizes without
                # pretending the non-orthogonal path energies are additive.
                "full_norm_rss": float(
                    full_group.square().sum(dim=1).sum().sqrt().item()
                ),
                "data_norm_rss": float(
                    data_group.square().sum(dim=1).sum().sqrt().item()
                ),
                "bias_norm_rss": float(
                    bias_group.square().sum(dim=1).sum().sqrt().item()
                ),
                # Norm of the actual vector obtained after summing the group.
                "full_group_sum_norm": float(full_sum.norm().item()),
                "data_group_sum_norm": float(data_sum.norm().item()),
                "bias_group_sum_norm": float(bias_sum.norm().item()),
                "signed_alignment_share": float(
                    ((full_sum @ residual) / residual_norm_squared).item()
                ),
                "data_alignment_share": float(
                    ((data_sum @ residual) / residual_norm_squared).item()
                ),
                "bias_alignment_share": float(
                    ((bias_sum @ residual) / residual_norm_squared).item()
                ),
            }
        )

    return rows


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/sweep_d64_7L_attn_lag1_50_extended.pt"),
    )
    parser.add_argument("--n-layers", type=int, default=7)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-head", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=200)
    parser.add_argument("--lag", type=int, default=40)
    parser.add_argument("--rho", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--query-position", type=int, default=199)
    parser.add_argument(
        "--positions",
        type=parse_int_list,
        default=None,
        help=(
            "positions to inspect; default uses query, correct key, and "
            "highest-logit wrong key"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/residual_path_contribution_norms_7L"),
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    if args.n_layers < 2:
        raise ValueError("model must have at least two layers")
    if args.query_position >= args.sequence_length:
        raise ValueError("query position must be inside the sequence")
    if args.lag - 1 > args.query_position:
        raise ValueError("correct key does not exist at this query position")

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(
        args.checkpoint,
        args.n_layers,
        args.d_model,
        args.d_head,
        device,
    ).double()
    inputs, _, sampled_lags = make_dataset_lagset(
        1,
        args.sequence_length,
        args.rho,
        [args.lag],
        seed=args.seed,
    )
    inputs = inputs.to(device=device, dtype=torch.float64)
    assert int(sampled_lags[0].item()) == args.lag

    with torch.inference_mode():
        _, attentions, post_attention, post_mlp = model(inputs)
        embedding = model.W_r(inputs.unsqueeze(-1))

    for after_attention, after_mlp in zip(post_attention, post_mlp):
        if not torch.equal(after_attention, after_mlp):
            raise RuntimeError("model is not behaving as attention-only")

    n_expanded_layers = model.n_layers - 1
    expected_path_count = 2**n_expanded_layers

    input_direction = model.W_r.weight[:, 0]
    input_bias = model.W_r.bias
    data_embedding = inputs.unsqueeze(-1) * input_direction.view(1, 1, -1)
    bias_embedding = input_bias.view(1, 1, -1).expand_as(data_embedding)

    with torch.inference_mode():
        full_path_dict = expand_paths(
            embedding,
            attentions,
            model,
            n_expanded_layers,
            expected_residuals=post_attention[:n_expanded_layers],
        )
        data_path_dict = expand_paths(
            data_embedding,
            attentions,
            model,
            n_expanded_layers,
        )
        bias_path_dict = expand_paths(
            bias_embedding,
            attentions,
            model,
            n_expanded_layers,
        )

    labels = list(full_path_dict.keys())
    if len(labels) != expected_path_count:
        raise RuntimeError("unexpected number of expanded paths")
    if labels != list(data_path_dict.keys()) or labels != list(bias_path_dict.keys()):
        raise RuntimeError("path labels differ between decompositions")

    full_components = torch.stack(list(full_path_dict.values()), dim=0)
    data_components = torch.stack(list(data_path_dict.values()), dim=0)
    bias_components = torch.stack(list(bias_path_dict.values()), dim=0)
    final_input = post_attention[n_expanded_layers - 1]

    pathwise_split_error = (
        full_components - data_components - bias_components
    ).abs().max()
    reconstruction_error = (
        full_components.sum(dim=0) - final_input
    ).norm() / final_input.norm().clamp_min(1e-30)

    if float(pathwise_split_error.item()) > 1e-5:
        raise RuntimeError("full path does not equal data path plus bias path")
    if float(reconstruction_error.item()) > 1e-5:
        raise RuntimeError("path sum does not reconstruct final-layer input")

    correct_offset = args.lag - 1
    correct_key = args.query_position - correct_offset
    logits = final_attention_logits(model, final_input)[0, args.query_position].clone()
    wrong_logits = logits.clone()
    wrong_logits[correct_key] = float("-inf")
    hardest_wrong_key = int(wrong_logits.argmax().item())
    hardest_wrong_offset = args.query_position - hardest_wrong_key

    if args.positions is None:
        positions = [
            args.query_position,
            correct_key,
            hardest_wrong_key,
        ]
        position_names = [
            "query",
            "correct_key",
            "hardest_wrong_key",
        ]
    else:
        positions = args.positions
        position_names = [f"position_{position}" for position in positions]

    for position in positions:
        if not 0 <= position < args.sequence_length:
            raise ValueError(f"position {position} is outside the sequence")

    all_path_rows: list[dict] = []
    all_group_rows: list[dict] = []
    per_position = {}

    for name, position in zip(position_names, positions):
        full_at_position = full_components[:, 0, position, :]
        data_at_position = data_components[:, 0, position, :]
        bias_at_position = bias_components[:, 0, position, :]
        residual = final_input[0, position]

        position_reconstruction_error = (
            full_at_position.sum(dim=0) - residual
        ).norm() / residual.norm().clamp_min(1e-30)

        path_rows = path_statistics(
            labels,
            full_at_position,
            data_at_position,
            bias_at_position,
            residual,
        )
        group_rows = aggregate_by_write_count(
            labels,
            full_at_position,
            data_at_position,
            bias_at_position,
            residual,
        )

        for row in path_rows:
            all_path_rows.append(
                {
                    "position_name": name,
                    "position": position,
                    **row,
                }
            )
        for row in group_rows:
            all_group_rows.append(
                {
                    "position_name": name,
                    "position": position,
                    **row,
                }
            )

        per_position[name] = {
            "position": position,
            "residual_norm": float(residual.norm().item()),
            "reconstruction_relative_error": float(
                position_reconstruction_error.item()
            ),
            "sum_signed_alignment_share": float(
                sum(float(row["signed_alignment_share"]) for row in path_rows)
            ),
            "sum_data_alignment_share": float(
                sum(float(row["data_alignment_share"]) for row in path_rows)
            ),
            "sum_bias_alignment_share": float(
                sum(float(row["bias_alignment_share"]) for row in path_rows)
            ),
        }

    save_csv(output_dir / "path_contributions.csv", all_path_rows)
    save_csv(output_dir / "path_contributions_by_write_count.csv", all_group_rows)

    summary = {
        "checkpoint": str(args.checkpoint),
        "lag": args.lag,
        "seed": args.seed,
        "query_position": args.query_position,
        "correct_key": correct_key,
        "correct_offset": correct_offset,
        "hardest_wrong_key": hardest_wrong_key,
        "hardest_wrong_offset": hardest_wrong_offset,
        "correct_logit": float(logits[correct_key].item()),
        "hardest_wrong_logit": float(logits[hardest_wrong_key].item()),
        "n_expanded_layers": n_expanded_layers,
        "n_paths": len(labels),
        "global_reconstruction_relative_error": float(
            reconstruction_error.item()
        ),
        "pathwise_data_plus_bias_max_error": float(
            pathwise_split_error.item()
        ),
        "positions": per_position,
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    # Individual path norms.
    figure, axes = plt.subplots(
        len(positions),
        1,
        figsize=(18, 4.2 * len(positions)),
        squeeze=False,
    )
    path_indices = np.arange(len(labels))
    write_counts = np.asarray([bits.count("1") for bits in labels])
    color_map = plt.get_cmap("viridis", n_expanded_layers + 1)

    for axis, name, position in zip(axes[:, 0], position_names, positions):
        selected = [
            row for row in all_path_rows if row["position_name"] == name
        ]
        data_norms = np.asarray([float(row["data_norm"]) for row in selected])
        full_norms = np.asarray([float(row["full_norm"]) for row in selected])

        axis.bar(
            path_indices,
            data_norms,
            color=[color_map(count) for count in write_counts],
            alpha=0.85,
            label="data-dependent path norm",
        )
        axis.plot(
            path_indices,
            full_norms,
            color="black",
            linewidth=1.0,
            marker=".",
            markersize=3,
            label="full path norm",
        )
        axis.set_title(f"{name}: position {position}")
        axis.set_xlabel("path index (binary labels in CSV)")
        axis.set_ylabel("residual-vector norm")
        axis.grid(axis="y", alpha=0.3)
        axis.legend(fontsize=8)

    figure.suptitle(
        "Frozen-attention path contribution norms",
        y=1.005,
    )
    figure.tight_layout()
    figure.savefig(output_dir / "individual_path_norms.png", dpi=180)
    plt.close(figure)

    # Aggregates by number of attention-write branches.
    figure, axes = plt.subplots(
        len(positions),
        2,
        figsize=(15, 4.2 * len(positions)),
        squeeze=False,
    )

    for row_axes, name, position in zip(axes, position_names, positions):
        selected = [
            row for row in all_group_rows if row["position_name"] == name
        ]
        counts = np.asarray([int(row["write_count"]) for row in selected])
        data_rss = np.asarray([float(row["data_norm_rss"]) for row in selected])
        data_sum_norm = np.asarray(
            [float(row["data_group_sum_norm"]) for row in selected]
        )
        full_sum_norm = np.asarray(
            [float(row["full_group_sum_norm"]) for row in selected]
        )
        alignment = np.asarray(
            [float(row["signed_alignment_share"]) for row in selected]
        )

        row_axes[0].plot(
            counts,
            data_rss,
            marker="o",
            label="RSS of data-path norms",
        )
        row_axes[0].plot(
            counts,
            data_sum_norm,
            marker="o",
            label="norm of summed data paths",
        )
        row_axes[0].plot(
            counts,
            full_sum_norm,
            marker="o",
            label="norm of summed full paths",
        )
        row_axes[0].set_title(f"{name}: position {position}")
        row_axes[0].set_xlabel("number of attention-write branches")
        row_axes[0].set_ylabel("norm")
        row_axes[0].grid(alpha=0.3)
        row_axes[0].legend(fontsize=8)

        row_axes[1].bar(counts, alignment, color="tab:purple")
        row_axes[1].axhline(0.0, color="black", linewidth=0.7)
        row_axes[1].set_title("Signed contribution along full residual")
        row_axes[1].set_xlabel("number of attention-write branches")
        row_axes[1].set_ylabel("share; groups sum to 1")
        row_axes[1].grid(axis="y", alpha=0.3)

    figure.suptitle(
        "Path norms aggregated by write count",
        y=1.005,
    )
    figure.tight_layout()
    figure.savefig(output_dir / "path_norms_by_write_count.png", dpi=180)
    plt.close(figure)

    print(f"checkpoint: {args.checkpoint}")
    print(
        f"lag={args.lag}, query={args.query_position}, "
        f"correct key={correct_key} (D={correct_offset}), "
        f"hardest wrong key={hardest_wrong_key} (D={hardest_wrong_offset})"
    )
    print(
        f"correct logit={float(logits[correct_key].item()):+.6f}, "
        f"hardest wrong logit={float(logits[hardest_wrong_key].item()):+.6f}"
    )
    print(
        f"expanded {n_expanded_layers} layers into {len(labels)} paths"
    )
    print(
        f"global reconstruction relative error: "
        f"{float(reconstruction_error.item()):.3e}"
    )
    print(
        f"max pathwise full-(data+bias) error: "
        f"{float(pathwise_split_error.item()):.3e}"
    )

    for name in position_names:
        position_summary = per_position[name]
        print(
            f"{name:>18s} position={position_summary['position']:3d} "
            f"residual_norm={position_summary['residual_norm']:.6f} "
            f"reconstruction_error="
            f"{position_summary['reconstruction_relative_error']:.3e} "
            f"alignment_sum="
            f"{position_summary['sum_signed_alignment_share']:.6f}"
        )

    print(f"saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
