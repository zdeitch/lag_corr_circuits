"""Decode earlier scalar values from each frozen-attention residual path.

For the 4-layer attention-only model, the residual entering the final layer is
expanded into the eight exact paths through Layers 1--3.  A separate linear
ridge probe is then fit for every path to predict all prior scalar inputs
``x[t - d]`` from the path vector at ``t``.  Train and test sequences are
disjoint.

Two held-out measurements are reported:

1. Path-only R^2: how well that path vector predicts each prior value.
2. Incremental R^2 over x[t]: how much a probe using ``[x[t], path(t)]``
   improves over a probe using only the current scalar ``x[t]``.  This helps
   separate transported history from correlations already available in x[t].

The decomposition is exact for the observed forward pass with the earlier
attention matrices frozen.  The probes are observational and do not establish
that the final attention head uses all decodable information.

Usage:
    python four_layer_path_delay_decoding.py
    python four_layer_path_delay_decoding.py --quick
    python four_layer_path_delay_decoding.py --lags 30,40,50 --no-show
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch

from four_layer_attention_path_analysis import expand_prefinal_paths, load_model
from util import make_dataset_lagset


def parse_int_tuple(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def make_balanced_inputs(
    lags: tuple[int, ...],
    n_sequences_per_lag: int,
    sequence_length: int,
    rho: float,
    seed: int,
) -> torch.Tensor:
    """Generate exactly n_sequences_per_lag independently for every lag."""
    chunks = []
    for lag_index, lag in enumerate(lags):
        inputs, _, sampled_lags = make_dataset_lagset(
            n_sequences_per_lag,
            sequence_length,
            rho,
            [lag],
            seed=seed + 10_000 * lag_index + lag,
        )
        if not torch.all(sampled_lags == lag):
            raise RuntimeError("dataset returned an unexpected lag")
        chunks.append(inputs)
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def collect_path_examples(
    model,
    inputs: torch.Tensor,
    max_delay: int,
    query_start: int,
    query_stride: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict]:
    """Collect path vectors and x[t-d] targets without mixing sequences."""
    sequence_length = inputs.shape[1]
    first_query = max(query_start, max_delay)
    query_positions = torch.arange(
        first_query,
        sequence_length,
        query_stride,
        device=device,
    )
    if len(query_positions) == 0:
        raise ValueError("no query positions remain after max_delay/query_start")

    delays = torch.arange(max_delay + 1, device=device)
    feature_chunks: dict[str, list[np.ndarray]] = {}
    target_chunks: list[np.ndarray] = []
    current_chunks: list[np.ndarray] = []
    attention_profile_sums = torch.zeros(
        model.n_layers - 1,
        max_delay + 1,
        dtype=torch.float64,
        device=device,
    )
    attention_profile_count = 0
    maximum_reconstruction_error = 0.0
    labels: list[str] | None = None

    for batch_start in range(0, len(inputs), batch_size):
        batch = inputs[batch_start : batch_start + batch_size].to(
            device=device,
            dtype=torch.float64,
        )
        _, attentions, post_attention, post_mlp = model(batch)
        if any(
            not torch.equal(attn_resid, mlp_resid)
            for attn_resid, mlp_resid in zip(post_attention, post_mlp)
        ):
            raise RuntimeError("checkpoint is not behaving as attention-only")

        embedding = model.W_r(batch.unsqueeze(-1))
        path_dict = expand_prefinal_paths(
            embedding,
            attentions,
            model,
            post_attention[: model.n_layers - 1],
        )
        current_labels = list(path_dict)
        if labels is None:
            labels = current_labels
            feature_chunks = {label: [] for label in labels + ["FULL"]}
        elif labels != current_labels:
            raise RuntimeError("path labels changed between batches")

        path_stack = torch.stack(list(path_dict.values()))
        final_input = post_attention[-2]
        reconstruction_error = (
            (path_stack.sum(dim=0) - final_input).norm()
            / final_input.norm().clamp_min(1e-30)
        )
        maximum_reconstruction_error = max(
            maximum_reconstruction_error,
            float(reconstruction_error.item()),
        )

        for path_index, label in enumerate(labels):
            values = path_stack[path_index, :, query_positions, :]
            feature_chunks[label].append(
                values.reshape(-1, values.shape[-1]).cpu().numpy()
            )
        full_values = final_input[:, query_positions, :]
        feature_chunks["FULL"].append(
            full_values.reshape(-1, full_values.shape[-1]).cpu().numpy()
        )

        target_indices = query_positions[:, None] - delays[None, :]
        targets = batch[:, target_indices]
        target_chunks.append(targets.reshape(-1, max_delay + 1).cpu().numpy())
        current_chunks.append(
            batch[:, query_positions].reshape(-1, 1).cpu().numpy()
        )

        for layer_index in range(model.n_layers - 1):
            attention = attentions[layer_index]
            offset_values = attention[
                :,
                query_positions[:, None],
                query_positions[:, None] - delays[None, :],
            ]
            attention_profile_sums[layer_index] += offset_values.sum(dim=(0, 1))
        attention_profile_count += batch.shape[0] * len(query_positions)

    if labels is None:
        raise RuntimeError("no examples were collected")

    features = {
        label: np.concatenate(chunks, axis=0)
        for label, chunks in feature_chunks.items()
    }
    targets = np.concatenate(target_chunks, axis=0)
    current = np.concatenate(current_chunks, axis=0)
    attention_profiles = (
        attention_profile_sums / attention_profile_count
    ).cpu().numpy()
    early_attention_peaks = attention_profiles.argmax(axis=1).astype(int)

    diagnostics = {
        "path_labels": labels,
        "n_positions_per_sequence": int(len(query_positions)),
        "first_query_position": int(query_positions[0].item()),
        "last_query_position": int(query_positions[-1].item()),
        "maximum_path_reconstruction_relative_error": maximum_reconstruction_error,
        "early_attention_mean_profile_peaks": early_attention_peaks.tolist(),
        "early_attention_profiles": attention_profiles.tolist(),
    }
    return features, targets, current, diagnostics


def standardize_from_train(
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale[scale < 1e-12] = 1.0
    return (train - mean) / scale, (test - mean) / scale


def ridge_predict(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    ridge_strength: float,
) -> np.ndarray:
    """Fit one multi-output ridge regression using train-only normalization."""
    x_train, x_test = standardize_from_train(train_features, test_features)
    target_mean = train_targets.mean(axis=0, keepdims=True)
    centered_targets = train_targets - target_mean

    penalty = ridge_strength * len(x_train)
    gram = x_train.T @ x_train
    rhs = x_train.T @ centered_targets
    coefficients = np.linalg.solve(
        gram + penalty * np.eye(gram.shape[0]),
        rhs,
    )
    return x_test @ coefficients + target_mean


def regression_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    residual_ss = np.square(targets - predictions).sum(axis=0)
    centered_targets = targets - targets.mean(axis=0, keepdims=True)
    total_ss = np.square(centered_targets).sum(axis=0)
    r2 = 1.0 - residual_ss / np.maximum(total_ss, 1e-30)

    centered_predictions = predictions - predictions.mean(axis=0, keepdims=True)
    covariance = (centered_targets * centered_predictions).sum(axis=0)
    denominator = np.sqrt(
        np.square(centered_targets).sum(axis=0)
        * np.square(centered_predictions).sum(axis=0)
    )
    correlation = covariance / np.maximum(denominator, 1e-30)
    return r2, correlation


def expected_subset_delays(
    labels: list[str],
    early_attention_peaks: list[int],
) -> dict[str, int]:
    return {
        label: int(
            sum(
                int(bit == "1") * early_attention_peaks[layer_index]
                for layer_index, bit in enumerate(label)
            )
        )
        for label in labels
    }


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def centered_norm(values: np.ndarray, minimum_span: float = 0.01) -> TwoSlopeNorm:
    extent = max(float(np.nanmax(np.abs(values))), minimum_span)
    return TwoSlopeNorm(vmin=-extent, vcenter=0.0, vmax=extent)


def plot_results(
    labels: list[str],
    path_r2: np.ndarray,
    incremental_r2: np.ndarray,
    baseline_r2: np.ndarray,
    expected_delays: dict[str, int],
    lags: tuple[int, ...],
    output_path: Path,
) -> None:
    delays = np.arange(path_r2.shape[1])
    figure = plt.figure(figsize=(15, 9), constrained_layout=True)
    grid = figure.add_gridspec(3, 1, height_ratios=(1.15, 1.15, 0.65))
    axes = [figure.add_subplot(grid[index, 0]) for index in range(3)]

    image_r2 = axes[0].imshow(
        path_r2,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        norm=centered_norm(path_r2, minimum_span=0.1),
        extent=(-0.5, len(delays) - 0.5, len(labels) - 0.5, -0.5),
    )
    axes[0].set_title("Held-out path-only linear decoding")
    axes[0].set_ylabel("residual path")
    axes[0].set_yticks(np.arange(len(labels)), labels)
    figure.colorbar(image_r2, ax=axes[0], label=r"held-out $R^2$")

    image_incremental = axes[1].imshow(
        incremental_r2,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        norm=centered_norm(incremental_r2, minimum_span=0.01),
        extent=(-0.5, len(delays) - 0.5, len(labels) - 0.5, -0.5),
    )
    axes[1].set_title(
        "Additional decoding beyond the current scalar x[t] "
        "([x[t], path] probe minus x[t]-only probe)"
    )
    axes[1].set_ylabel("residual path")
    axes[1].set_yticks(np.arange(len(labels)), labels)
    figure.colorbar(image_incremental, ax=axes[1], label=r"incremental $R^2$")

    for row_index, label in enumerate(labels):
        expected = expected_delays.get(label)
        if expected is None or not 0 <= expected < len(delays):
            continue
        for axis in axes[:2]:
            axis.scatter(
                expected,
                row_index,
                marker="o",
                s=34,
                facecolors="none",
                edgecolors="black",
                linewidths=1.2,
            )

    axes[2].plot(delays, baseline_r2, color="black", linewidth=1.8)
    axes[2].axhline(0.0, color="0.6", linewidth=0.8)
    axes[2].set_title(
        "Baseline: predict each prior value using only the current scalar x[t]"
    )
    axes[2].set_xlabel("prior-value delay d in x[t-d]")
    axes[2].set_ylabel(r"held-out $R^2$")
    axes[2].grid(alpha=0.25)

    for axis in axes[:2]:
        axis.set_xlabel("prior-value delay d in x[t-d]")
        axis.set_xlim(-0.5, len(delays) - 0.5)

    lag_text = ", ".join(str(lag) for lag in lags)
    figure.suptitle(
        "What earlier scalar values are linearly decodable from each path?\n"
        f"4-layer attention-only model; data lag(s): {lag_text}; "
        "circles show subset sums of observed early-head peak offsets",
        fontsize=15,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/attn_d64_4L_int_ext.pt"),
    )
    parser.add_argument("--lags", type=parse_int_tuple, default=(40,))
    parser.add_argument("--n-train-per-lag", type=int, default=192)
    parser.add_argument("--n-test-per-lag", type=int, default=192)
    parser.add_argument("--sequence-length", type=int, default=200)
    parser.add_argument("--rho", type=float, default=0.9)
    parser.add_argument("--max-delay", type=int, default=100)
    parser.add_argument("--query-start", type=int, default=120)
    parser.add_argument("--query-stride", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--ridge-strength",
        type=float,
        default=1e-4,
        help="ridge penalty divided by number of training observations",
    )
    parser.add_argument("--seed", type=int, default=76123)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment_outputs/four_layer_path_delay_decoding"),
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.n_train_per_lag = min(args.n_train_per_lag, 24)
        args.n_test_per_lag = min(args.n_test_per_lag, 24)
        args.query_stride = max(args.query_stride, 4)

    if args.max_delay >= args.sequence_length:
        raise ValueError("max_delay must be below sequence_length")
    if args.ridge_strength < 0:
        raise ValueError("ridge_strength must be nonnegative")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_inputs = make_balanced_inputs(
        args.lags,
        args.n_train_per_lag,
        args.sequence_length,
        args.rho,
        args.seed,
    )
    test_inputs = make_balanced_inputs(
        args.lags,
        args.n_test_per_lag,
        args.sequence_length,
        args.rho,
        args.seed + 1_000_000,
    )

    train_features, train_targets, train_current, train_diagnostics = (
        collect_path_examples(
            model,
            train_inputs,
            args.max_delay,
            args.query_start,
            args.query_stride,
            args.batch_size,
            device,
        )
    )
    test_features, test_targets, test_current, test_diagnostics = (
        collect_path_examples(
            model,
            test_inputs,
            args.max_delay,
            args.query_start,
            args.query_stride,
            args.batch_size,
            device,
        )
    )

    labels = train_diagnostics["path_labels"] + ["FULL"]
    if labels != test_diagnostics["path_labels"] + ["FULL"]:
        raise RuntimeError("train/test path labels differ")

    baseline_predictions = ridge_predict(
        train_current,
        train_targets,
        test_current,
        args.ridge_strength,
    )
    baseline_r2, baseline_correlation = regression_metrics(
        test_targets,
        baseline_predictions,
    )

    path_r2 = np.empty((len(labels), args.max_delay + 1))
    path_correlation = np.empty_like(path_r2)
    augmented_r2 = np.empty_like(path_r2)
    augmented_correlation = np.empty_like(path_r2)
    rows: list[dict] = []

    for label_index, label in enumerate(labels):
        path_predictions = ridge_predict(
            train_features[label],
            train_targets,
            test_features[label],
            args.ridge_strength,
        )
        path_r2[label_index], path_correlation[label_index] = regression_metrics(
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
            args.ridge_strength,
        )
        (
            augmented_r2[label_index],
            augmented_correlation[label_index],
        ) = regression_metrics(test_targets, augmented_predictions)

        for delay in range(args.max_delay + 1):
            rows.append(
                {
                    "path": label,
                    "delay": delay,
                    "path_only_r2": float(path_r2[label_index, delay]),
                    "path_only_pearson_r": float(
                        path_correlation[label_index, delay]
                    ),
                    "current_only_r2": float(baseline_r2[delay]),
                    "current_only_pearson_r": float(
                        baseline_correlation[delay]
                    ),
                    "current_plus_path_r2": float(
                        augmented_r2[label_index, delay]
                    ),
                    "current_plus_path_pearson_r": float(
                        augmented_correlation[label_index, delay]
                    ),
                    "incremental_r2_over_current": float(
                        augmented_r2[label_index, delay] - baseline_r2[delay]
                    ),
                }
            )

    incremental_r2 = augmented_r2 - baseline_r2[None, :]
    early_peaks = train_diagnostics["early_attention_mean_profile_peaks"]
    expected_delays = expected_subset_delays(
        train_diagnostics["path_labels"],
        early_peaks,
    )

    per_path_summary = {}
    for label_index, label in enumerate(labels):
        best_path_delay = int(np.nanargmax(path_r2[label_index]))
        best_incremental_delay = int(np.nanargmax(incremental_r2[label_index]))
        per_path_summary[label] = {
            "expected_subset_sum_delay": expected_delays.get(label),
            "best_path_only_r2_delay": best_path_delay,
            "best_path_only_r2": float(path_r2[label_index, best_path_delay]),
            "best_incremental_r2_delay": best_incremental_delay,
            "best_incremental_r2": float(
                incremental_r2[label_index, best_incremental_delay]
            ),
        }

    summary = {
        "experiment": "held-out linear decoding of prior scalar values from paths",
        "checkpoint": str(args.checkpoint),
        "lags": list(args.lags),
        "rho": args.rho,
        "sequence_length": args.sequence_length,
        "max_delay": args.max_delay,
        "query_start": args.query_start,
        "query_stride": args.query_stride,
        "n_train_sequences_per_lag": args.n_train_per_lag,
        "n_test_sequences_per_lag": args.n_test_per_lag,
        "n_train_observations": int(len(train_targets)),
        "n_test_observations": int(len(test_targets)),
        "ridge_strength_fraction_of_n_train": args.ridge_strength,
        "actual_ridge_penalty": args.ridge_strength * len(train_targets),
        "train_test_sequences_disjoint": True,
        "early_attention_mean_profile_peaks": early_peaks,
        "expected_subset_sum_delays": expected_delays,
        "train_diagnostics": train_diagnostics,
        "test_diagnostics": test_diagnostics,
        "per_path": per_path_summary,
    }

    csv_path = args.output_dir / "path_delay_decoding.csv"
    json_path = args.output_dir / "path_delay_decoding_summary.json"
    npz_path = args.output_dir / "path_delay_decoding_arrays.npz"
    plot_path = args.output_dir / "path_delay_decoding.png"
    save_csv(csv_path, rows)
    with json_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    np.savez_compressed(
        npz_path,
        labels=np.asarray(labels),
        delays=np.arange(args.max_delay + 1),
        path_r2=path_r2,
        path_correlation=path_correlation,
        current_only_r2=baseline_r2,
        current_only_correlation=baseline_correlation,
        current_plus_path_r2=augmented_r2,
        current_plus_path_correlation=augmented_correlation,
        incremental_r2_over_current=incremental_r2,
    )
    plot_results(
        labels,
        path_r2,
        incremental_r2,
        baseline_r2,
        expected_delays,
        args.lags,
        plot_path,
    )

    print(f"device: {device}")
    print(f"train observations: {len(train_targets):,}")
    print(f"test observations:  {len(test_targets):,}")
    print(f"early-head mean-profile peaks: {early_peaks}")
    print(
        "maximum path reconstruction error: "
        f"{max(train_diagnostics['maximum_path_reconstruction_relative_error'], test_diagnostics['maximum_path_reconstruction_relative_error']):.3e}"
    )
    print("\nBest held-out decoding delay by path:")
    for label in labels:
        item = per_path_summary[label]
        print(
            f"  {label:>4}: path-only peak d={item['best_path_only_r2_delay']:3d} "
            f"R2={item['best_path_only_r2']:+.3f}; "
            f"incremental peak d={item['best_incremental_r2_delay']:3d} "
            f"delta-R2={item['best_incremental_r2']:+.3f}; "
            f"subset prediction={item['expected_subset_sum_delay']}"
        )
    print(f"\nWrote {plot_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {npz_path}")

    if not args.no_show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
