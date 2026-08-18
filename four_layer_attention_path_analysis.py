"""Verify lag-adaptive retrieval and expand its final-head margin into paths.

The checkpoint is a four-layer, one-head, attention-only AutocorrRoPE model.
For the score computed by layer 4, the residuals entering that layer have
passed through three earlier residual blocks.  Freezing the clean attention
patterns makes each block exactly ``identity + attention/OV write``, yielding
2**3 = 8 paths on each side of the final query/key score.

This script performs two experiments:

1. Across several programmed data lags, measure prediction quality and whether
   the final attention profile peaks at the required offset ``lag - 1``.
2. At one selected lag, exactly decompose the final head's
   correct-minus-nearby-wrong pre-softmax logit margin into an 8 x 8 matrix of
   query-path/key-path contributions.

The path analysis is an exact decomposition of the observed forward pass with
the earlier attention matrices frozen.  It is not a counterfactual claim that
those attention matrices would stay fixed after an intervention.

Usage:
    python four_layer_attention_path_analysis.py --quick
    python four_layer_attention_path_analysis.py
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


def parse_int_tuple(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def load_model(checkpoint: Path, device: torch.device) -> AutocorrRoPE:
    model = AutocorrRoPE(
        d_model=64,
        d_head=64,
        n_layers=4,
        n_heads=1,
        use_mlp=False,
    ).to(device=device, dtype=torch.float64)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def frozen_attention_write(
    component: torch.Tensor,
    attention: torch.Tensor,
    head_parameters: torch.nn.ParameterList,
) -> torch.Tensor:
    """Apply one observed attention pattern and that head's OV map."""
    _, _, value_matrix, output_matrix = head_parameters
    return (attention @ (component @ value_matrix)) @ output_matrix


def expand_prefinal_paths(
    embedding: torch.Tensor,
    attentions: list[torch.Tensor],
    model: AutocorrRoPE,
    expected_post_attention: list[torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Return the eight paths through layers 1--3 of the four-layer model."""
    paths: OrderedDict[str, torch.Tensor] = OrderedDict({"": embedding})

    for layer_index in range(model.n_layers - 1):
        expanded: OrderedDict[str, torch.Tensor] = OrderedDict()
        for bits, component in paths.items():
            expanded[bits + "0"] = component
            expanded[bits + "1"] = frozen_attention_write(
                component,
                attentions[layer_index],
                model.layers[layer_index][0],
            )
        paths = expanded

        reconstruction = torch.stack(list(paths.values())).sum(dim=0)
        expected = expected_post_attention[layer_index]
        error = (reconstruction - expected).norm() / expected.norm().clamp_min(1e-30)
        if float(error.item()) > 1e-10:
            raise RuntimeError(
                f"path reconstruction failed after layer {layer_index + 1}: "
                f"relative error={float(error.item()):.3e}"
            )

    return paths


def attention_profiles(
    attention: torch.Tensor,
    query_start: int,
    max_offset: int,
) -> torch.Tensor:
    """Per-sequence mean attention mass at each query-minus-key offset."""
    _, sequence_length, _ = attention.shape
    first_query = max(query_start, max_offset)
    query_positions = torch.arange(
        first_query, sequence_length, device=attention.device
    )
    offsets = torch.arange(max_offset + 1, device=attention.device)
    values = attention[
        :,
        query_positions[:, None],
        query_positions[:, None] - offsets[None, :],
    ]
    return values.mean(dim=1)


def prediction_metrics(
    prediction: torch.Tensor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    lag: int,
    burn_in: int,
) -> tuple[float, float]:
    """Prediction/true-lag correlation and next-token MSE after warm-up."""
    sequence_length = inputs.shape[1]
    start = lag + burn_in
    predicted = prediction[:, start:].reshape(-1).cpu().numpy()
    lagged = inputs[
        :, start - lag + 1 : sequence_length - lag + 1
    ].reshape(-1).cpu().numpy()
    correlation = float(np.corrcoef(predicted, lagged)[0, 1])
    mse = float((prediction[:, start:] - targets[:, start:]).square().mean().item())
    return correlation, mse


@torch.inference_mode()
def verify_lag_adaptation(
    model: AutocorrRoPE,
    lags: tuple[int, ...],
    n_sequences: int,
    sequence_length: int,
    rho: float,
    burn_in: int,
    query_start: int,
    max_offset: int,
    device: torch.device,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []

    for lag in lags:
        if lag - 1 > max_offset:
            raise ValueError(
                f"lag {lag} requires offset {lag - 1}, above max_offset={max_offset}"
            )
        inputs, targets, sampled_lags = make_dataset_lagset(
            n_sequences,
            sequence_length,
            rho,
            [lag],
            seed=10_000 + lag,
        )
        inputs = inputs.to(device=device, dtype=torch.float64)
        targets = targets.to(device=device, dtype=torch.float64)
        if not torch.all(sampled_lags == lag):
            raise RuntimeError("dataset returned an unexpected lag")

        prediction, attentions, post_attention, post_mlp = model(inputs)
        if any(not torch.equal(a, b) for a, b in zip(post_attention, post_mlp)):
            raise RuntimeError("checkpoint is not behaving as attention-only")

        profiles = attention_profiles(attentions[-1], query_start, max_offset)
        # Include offset zero only when lag=1; otherwise self-attention is not a
        # candidate for the required lagged retrieval.
        candidate_start = 0 if lag == 1 else 1
        candidate_profiles = profiles[:, candidate_start:]
        top_offsets = candidate_profiles.argmax(dim=1) + candidate_start
        correct_offset = lag - 1
        target_mass = profiles[:, correct_offset]
        wrong_profiles = profiles.clone()
        wrong_profiles[:, correct_offset] = -torch.inf
        if candidate_start == 1:
            wrong_profiles[:, 0] = -torch.inf
        hardest_wrong_mass = wrong_profiles.max(dim=1).values

        correlation, mse = prediction_metrics(
            prediction, inputs, targets, lag, burn_in
        )
        rows.append(
            {
                "lag": lag,
                "target_offset": correct_offset,
                "prediction_correlation": correlation,
                "prediction_mse": mse,
                "exact_top_offset_rate": float(
                    (top_offsets == correct_offset).double().mean().item()
                ),
                "within_one_top_offset_rate": float(
                    ((top_offsets - correct_offset).abs() <= 1).double().mean().item()
                ),
                "mean_top_offset": float(top_offsets.double().mean().item()),
                "sd_top_offset": float(top_offsets.double().std(unbiased=True).item()),
                "mean_target_attention_mass": float(target_mass.mean().item()),
                "mean_target_vs_hardest_wrong_mass_margin": float(
                    (target_mass - hardest_wrong_mass).mean().item()
                ),
            }
        )

    return rows


def save_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_lag_adaptation(rows: list[dict], output_path: Path) -> None:
    lag = np.asarray([row["lag"] for row in rows])
    correlation = np.asarray([row["prediction_correlation"] for row in rows])
    exact = np.asarray([row["exact_top_offset_rate"] for row in rows])
    within_one = np.asarray([row["within_one_top_offset_rate"] for row in rows])
    target_mass = np.asarray([row["mean_target_attention_mass"] for row in rows])
    mass_margin = np.asarray(
        [row["mean_target_vs_hardest_wrong_mass_margin"] for row in rows]
    )

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(lag, correlation, marker="o")
    axes[0].axvline(50, color="gray", linestyle=":", label="training maximum")
    axes[0].set_ylabel("prediction correlation with true lagged value")
    axes[0].legend(fontsize=8)

    axes[1].plot(lag, exact, marker="o", label="exact offset")
    axes[1].plot(lag, within_one, marker="o", label="within ±1")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_ylabel("fraction of sequences")
    axes[1].legend(fontsize=8)

    axes[2].plot(lag, target_mass, marker="o", label="target mass")
    axes[2].plot(lag, mass_margin, marker="o", label="target − hardest wrong")
    axes[2].axhline(0, color="gray", linewidth=1)
    axes[2].set_ylabel("mean final-attention mass")
    axes[2].legend(fontsize=8)

    for axis in axes:
        axis.set_xlabel("data lag")
        axis.grid(alpha=0.3)
    figure.suptitle("Four-layer attention-only model: lag-adaptive final retrieval")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


@torch.inference_mode()
def path_pair_margin_analysis(
    model: AutocorrRoPE,
    lag: int,
    wrong_deltas: tuple[int, ...],
    n_sequences: int,
    sequence_length: int,
    rho: float,
    query_start: int,
    query_stride: int,
    device: torch.device,
) -> tuple[list[dict], dict, np.ndarray, list[str]]:
    correct_offset = lag - 1
    wrong_offsets = tuple(
        correct_offset + delta
        for delta in wrong_deltas
        if correct_offset + delta >= 1
    )
    if not wrong_offsets:
        raise ValueError("no valid wrong offsets")
    maximum_offset = max((correct_offset,) + wrong_offsets)
    if query_start < maximum_offset:
        raise ValueError(
            f"query_start={query_start} is below sampled offset {maximum_offset}"
        )

    inputs, _, sampled_lags = make_dataset_lagset(
        n_sequences,
        sequence_length,
        rho,
        [lag],
        seed=200_000 + lag,
    )
    inputs = inputs.to(device=device, dtype=torch.float64)
    if not torch.all(sampled_lags == lag):
        raise RuntimeError("dataset returned an unexpected lag")

    _, attentions, post_attention, post_mlp = model(inputs)
    if any(not torch.equal(a, b) for a, b in zip(post_attention, post_mlp)):
        raise RuntimeError("checkpoint is not behaving as attention-only")

    embedding = model.W_r(inputs.unsqueeze(-1))
    path_dict = expand_prefinal_paths(
        embedding,
        attentions,
        model,
        post_attention[: model.n_layers - 1],
    )
    labels = list(path_dict)
    path_components = torch.stack(list(path_dict.values()))
    final_input = post_attention[-2]
    reconstruction_error = (
        (path_components.sum(dim=0) - final_input).norm()
        / final_input.norm().clamp_min(1e-30)
    )

    query_matrix, key_matrix, _, _ = model.layers[-1][0]
    positions = torch.arange(sequence_length, device=device).view(1, 1, -1)
    path_queries = model.apply_rope(path_components @ query_matrix, positions)
    path_keys = model.apply_rope(path_components @ key_matrix, positions)
    full_queries = model.apply_rope(
        final_input @ query_matrix, positions.squeeze(0)
    )
    full_keys = model.apply_rope(final_input @ key_matrix, positions.squeeze(0))

    query_positions = torch.arange(
        query_start, sequence_length, query_stride, device=device
    )

    contribution_by_offset: dict[int, torch.Tensor] = {}
    maximum_pair_logit_error = 0.0
    for offset in (correct_offset,) + wrong_offsets:
        path_q = path_queries[:, :, query_positions, :]
        path_k = path_keys[:, :, query_positions - offset, :]
        # batch x query-position x query-path x key-path
        pair_contributions = torch.einsum(
            "pbnh,qbnh->bnpq", path_q, path_k
        ) / math.sqrt(model.d_head)
        contribution_by_offset[offset] = pair_contributions.mean(dim=1)

        direct_logits = (
            full_queries[:, query_positions, :]
            * full_keys[:, query_positions - offset, :]
        ).sum(dim=-1) / math.sqrt(model.d_head)
        pair_error = (
            pair_contributions.sum(dim=(-1, -2)) - direct_logits
        ).abs().max()
        maximum_pair_logit_error = max(
            maximum_pair_logit_error, float(pair_error.item())
        )

    sequence_matrices = contribution_by_offset[correct_offset] - torch.stack(
        [contribution_by_offset[offset] for offset in wrong_offsets]
    ).mean(dim=0)
    sequence_margins = sequence_matrices.sum(dim=(-1, -2))
    mean_matrix = sequence_matrices.mean(dim=0)
    sd_matrix = sequence_matrices.std(dim=0, unbiased=True)
    se_matrix = sd_matrix / math.sqrt(n_sequences)

    singular_values = torch.linalg.svdvals(mean_matrix)
    singular_energy = singular_values.square()
    cumulative_energy = singular_energy.cumsum(dim=0) / singular_energy.sum()
    rank90 = int((cumulative_energy < 0.90).sum().item()) + 1
    rank99 = int((cumulative_energy < 0.99).sum().item()) + 1

    rows: list[dict] = []
    for query_path, query_label in enumerate(labels):
        for key_path, key_label in enumerate(labels):
            values = sequence_matrices[:, query_path, key_path]
            mean = float(values.mean().item())
            sd = float(values.std(unbiased=True).item())
            same_sign = values >= 0 if mean >= 0 else values <= 0
            rows.append(
                {
                    "query_path": query_label,
                    "key_path": key_label,
                    "mean_margin_contribution": mean,
                    "sd_across_sequences": sd,
                    "se_of_mean": sd / math.sqrt(n_sequences),
                    "ci95_low": mean - 1.96 * sd / math.sqrt(n_sequences),
                    "ci95_high": mean + 1.96 * sd / math.sqrt(n_sequences),
                    "same_sign_fraction": float(same_sign.double().mean().item()),
                }
            )

    direct_margin = contribution_by_offset[correct_offset].sum(dim=(-1, -2))
    direct_margin = direct_margin - torch.stack(
        [
            contribution_by_offset[offset].sum(dim=(-1, -2))
            for offset in wrong_offsets
        ]
    ).mean(dim=0)
    margin_reconstruction_error = (sequence_margins - direct_margin).abs().max()

    mean_margin = float(sequence_margins.mean().item())
    summary = {
        "lag": lag,
        "correct_offset": correct_offset,
        "wrong_offsets": list(wrong_offsets),
        "n_sequences": n_sequences,
        "n_query_positions_per_sequence": len(query_positions),
        "path_labels": labels,
        "path_bit_meaning": "0=identity, 1=frozen attention/OV write in layers 1--3",
        "n_paths_per_side": len(labels),
        "n_path_pairs": len(labels) ** 2,
        "final_input_reconstruction_relative_error": float(
            reconstruction_error.item()
        ),
        "maximum_single_pair_logit_reconstruction_error": maximum_pair_logit_error,
        "maximum_margin_reconstruction_error": float(
            margin_reconstruction_error.item()
        ),
        "mean_correct_minus_wrong_logit_margin": mean_margin,
        "sd_sequence_margin": float(sequence_margins.std(unbiased=True).item()),
        "fraction_sequences_positive_margin": float(
            (sequence_margins > 0).double().mean().item()
        ),
        "mean_matrix_sum": float(mean_matrix.sum().item()),
        "mean_matrix_absolute_sum": float(mean_matrix.abs().sum().item()),
        "cancellation_ratio_abs_sum_over_abs_total": float(
            mean_matrix.abs().sum().item() / max(abs(mean_margin), 1e-30)
        ),
        "mean_path_matrix_singular_values": [
            float(value) for value in singular_values.cpu()
        ],
        "mean_path_matrix_rank90": rank90,
        "mean_path_matrix_rank99": rank99,
    }
    return rows, summary, mean_matrix.cpu().numpy(), labels


def plot_path_matrix(
    matrix: np.ndarray,
    labels: list[str],
    lag: int,
    wrong_offsets: list[int],
    output_path: Path,
) -> None:
    maximum = max(float(np.abs(matrix).max()), 1e-12)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    image = axes[0].imshow(
        matrix,
        cmap="RdBu_r",
        vmin=-maximum,
        vmax=maximum,
        aspect="equal",
    )
    axes[0].set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].set_xlabel("key path")
    axes[0].set_ylabel("query path")
    axes[0].set_title("Mean signed contribution to correct-minus-wrong logit")
    figure.colorbar(image, ax=axes[0], label="pre-softmax attention logits")

    query_totals = matrix.sum(axis=1)
    key_totals = matrix.sum(axis=0)
    positions = np.arange(len(labels))
    width = 0.38
    axes[1].bar(positions - width / 2, query_totals, width, label="query-path total")
    axes[1].bar(positions + width / 2, key_totals, width, label="key-path total")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(positions, labels, rotation=45, ha="right")
    axes[1].set_ylabel("summed logit-margin contribution")
    axes[1].set_title("Path totals")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    figure.suptitle(
        f"4L attention-only path decomposition: lag {lag}, "
        f"wrong offsets {wrong_offsets}\n"
        "path bits: 0=identity, 1=attention/OV write in layers 1--3"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


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
        default=(5, 10, 20, 30, 40, 50, 52, 55, 58, 70),
    )
    parser.add_argument("--path-lag", type=int, default=40)
    parser.add_argument(
        "--wrong-deltas",
        type=parse_int_tuple,
        default=(-8, -4, -2, 2, 4, 8),
    )
    parser.add_argument("--n-sequences", type=int, default=96)
    parser.add_argument("--path-n-sequences", type=int, default=96)
    parser.add_argument("--sequence-length", type=int, default=200)
    parser.add_argument("--rho", type=float, default=0.9)
    parser.add_argument("--burn-in", type=int, default=30)
    parser.add_argument("--query-start", type=int, default=120)
    parser.add_argument("--profile-max-offset", type=int, default=100)
    parser.add_argument("--path-query-start", type=int, default=140)
    parser.add_argument("--path-query-stride", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/four_layer_attention_path_analysis"),
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.n_sequences = min(args.n_sequences, 24)
        args.path_n_sequences = min(args.path_n_sequences, 24)

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint, device)

    adaptation_rows = verify_lag_adaptation(
        model=model,
        lags=args.lags,
        n_sequences=args.n_sequences,
        sequence_length=args.sequence_length,
        rho=args.rho,
        burn_in=args.burn_in,
        query_start=args.query_start,
        max_offset=args.profile_max_offset,
        device=device,
    )
    save_rows(args.output_dir / "lag_adaptation.csv", adaptation_rows)
    plot_lag_adaptation(
        adaptation_rows,
        args.output_dir / "lag_adaptation.png",
    )

    path_rows, path_summary, path_matrix, labels = path_pair_margin_analysis(
        model=model,
        lag=args.path_lag,
        wrong_deltas=args.wrong_deltas,
        n_sequences=args.path_n_sequences,
        sequence_length=args.sequence_length,
        rho=args.rho,
        query_start=args.path_query_start,
        query_stride=args.path_query_stride,
        device=device,
    )
    save_rows(args.output_dir / "path_pair_margin.csv", path_rows)
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "device": str(device),
                "lag_adaptation": adaptation_rows,
                "path_analysis": path_summary,
            },
            handle,
            indent=2,
        )
    plot_path_matrix(
        path_matrix,
        labels,
        args.path_lag,
        path_summary["wrong_offsets"],
        args.output_dir / "path_pair_margin.png",
    )

    print("\nLag adaptation:")
    print(
        " lag  target  pred-corr  exact-hit  ±1-hit  target-mass  mass-margin"
    )
    for row in adaptation_rows:
        print(
            f" {row['lag']:>3}  {row['target_offset']:>6}  "
            f"{row['prediction_correlation']:>9.3f}  "
            f"{row['exact_top_offset_rate']:>9.3f}  "
            f"{row['within_one_top_offset_rate']:>6.3f}  "
            f"{row['mean_target_attention_mass']:>11.3f}  "
            f"{row['mean_target_vs_hardest_wrong_mass_margin']:>11.3f}"
        )

    print("\nPath-pair margin decomposition:")
    for key in (
        "n_paths_per_side",
        "n_path_pairs",
        "mean_correct_minus_wrong_logit_margin",
        "fraction_sequences_positive_margin",
        "cancellation_ratio_abs_sum_over_abs_total",
        "mean_path_matrix_rank90",
        "mean_path_matrix_rank99",
        "final_input_reconstruction_relative_error",
        "maximum_single_pair_logit_reconstruction_error",
    ):
        print(f"  {key}: {path_summary[key]}")

    strongest = sorted(
        path_rows,
        key=lambda row: abs(row["mean_margin_contribution"]),
        reverse=True,
    )[:10]
    print("\nTen strongest mean path-pair contributions:")
    for row in strongest:
        print(
            f"  q={row['query_path']} k={row['key_path']}  "
            f"mean={row['mean_margin_contribution']:+.4f}  "
            f"95% CI=[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}]  "
            f"same-sign={row['same_sign_fraction']:.2f}"
        )
    print(f"\nSaved results to {args.output_dir}")


if __name__ == "__main__":
    main()
