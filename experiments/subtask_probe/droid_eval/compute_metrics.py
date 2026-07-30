#!/usr/bin/env python3
"""Phase 3: Compute metrics comparing action conditions against ground truth.

Loads ground truth action chunks and predicted actions from all 3 conditions,
computes L2 distance, cosine similarity, per-dimension MAE, and gripper accuracy.

Usage:
    uv run python experiments/subtask_probe/droid_eval/compute_metrics.py \
        --samples_dir ./.experiments_cache/droid_eval \
        --predictions_dir ./.experiments_cache/droid_eval/predictions \
        --output ./.experiments_cache/droid_eval/results.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from .constants import CONDITION_NAMES, GRIPPER_THRESHOLD, JOINT_NAMES
from .utils import load_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class ConditionMetrics(TypedDict):
    l2_distance: float
    cosine_similarity: float
    per_dim_mae: list[float]
    gripper_accuracy: float
    per_step_l2: list[float]


def compute_frame_metrics(
    ground_truth: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> dict[str, ConditionMetrics]:
    """Compute per-condition metrics for a single frame.

    Args:
        ground_truth: [action_horizon, 8] ground truth action chunk
        predictions: dict mapping condition name → [action_horizon, 8] predicted actions
    """
    results = {}

    for condition_name, pred in predictions.items():
        # Ensure shapes match
        min_horizon = min(ground_truth.shape[0], pred.shape[0])
        gt = ground_truth[:min_horizon]
        pr = pred[:min_horizon]

        # L2 distance (over entire action chunk)
        l2_distance = float(np.linalg.norm(gt - pr))

        # Cosine similarity (flatten both)
        gt_flat = gt.flatten()
        pr_flat = pr.flatten()
        cos_sim = float(
            np.dot(gt_flat, pr_flat) / (np.linalg.norm(gt_flat) * np.linalg.norm(pr_flat) + 1e-10)
        )

        # Per-dimension MAE
        per_dim_mae = np.mean(np.abs(gt - pr), axis=0).tolist()  # [8]

        # Gripper accuracy (binary: open/closed)
        gt_gripper = (gt[:, -1] > GRIPPER_THRESHOLD).astype(int)
        pred_gripper = (pr[:, -1] > GRIPPER_THRESHOLD).astype(int)
        gripper_accuracy = float(np.mean(gt_gripper == pred_gripper))

        # Per-timestep L2 (for trajectory analysis)
        per_step_l2 = np.linalg.norm(gt - pr, axis=-1).tolist()  # [min_horizon]

        results[condition_name] = {
            "l2_distance": l2_distance,
            "cosine_similarity": cos_sim,
            "per_dim_mae": per_dim_mae,
            "gripper_accuracy": gripper_accuracy,
            "per_step_l2": per_step_l2,
        }

    return results


def aggregate_metrics(
    all_frame_metrics: list[dict[str, ConditionMetrics]],
    task_types: list[str],
    episode_progress: list[float],
) -> dict[str, Any]:
    """Aggregate per-frame metrics into summary statistics."""
    results = {}

    for condition in CONDITION_NAMES:
        l2_distances = [fm[condition]["l2_distance"] for fm in all_frame_metrics]
        cos_sims = [fm[condition]["cosine_similarity"] for fm in all_frame_metrics]
        gripper_accs = [fm[condition]["gripper_accuracy"] for fm in all_frame_metrics]
        per_dim_maes = np.array([fm[condition]["per_dim_mae"] for fm in all_frame_metrics])

        results[condition] = {
            "overall": {
                "l2_distance": {
                    "mean": float(np.mean(l2_distances)),
                    "std": float(np.std(l2_distances)),
                },
                "cosine_similarity": {
                    "mean": float(np.mean(cos_sims)),
                    "std": float(np.std(cos_sims)),
                },
                "gripper_accuracy": {
                    "mean": float(np.mean(gripper_accs)),
                    "std": float(np.std(gripper_accs)),
                },
                "per_dim_mae": {
                    name: float(per_dim_maes[:, i].mean())
                    for i, name in enumerate(JOINT_NAMES[: per_dim_maes.shape[1]])
                },
                "n_frames": len(l2_distances),
            },
        }

        # By task type
        for task_type in ["multi_step", "single_step"]:
            mask = [t == task_type for t in task_types]
            if not any(mask):
                continue

            type_l2 = [d for d, m in zip(l2_distances, mask, strict=True) if m]
            type_cos = [d for d, m in zip(cos_sims, mask, strict=True) if m]
            type_gripper = [d for d, m in zip(gripper_accs, mask, strict=True) if m]

            results[condition][f"task_type_{task_type}"] = {
                "l2_distance": {"mean": float(np.mean(type_l2)), "std": float(np.std(type_l2))},
                "cosine_similarity": {
                    "mean": float(np.mean(type_cos)),
                    "std": float(np.std(type_cos)),
                },
                "gripper_accuracy": {
                    "mean": float(np.mean(type_gripper)),
                    "std": float(np.std(type_gripper)),
                },
                "n_frames": len(type_l2),
            }

        # By episode progress (early/middle/late)
        for label, low, high in [
            ("early", 0.0, 0.33),
            ("middle", 0.33, 0.67),
            ("late", 0.67, 1.01),
        ]:
            mask = [low <= p < high for p in episode_progress]
            if not any(mask):
                continue

            prog_l2 = [d for d, m in zip(l2_distances, mask, strict=True) if m]
            prog_cos = [d for d, m in zip(cos_sims, mask, strict=True) if m]

            results[condition][f"progress_{label}"] = {
                "l2_distance": {"mean": float(np.mean(prog_l2)), "std": float(np.std(prog_l2))},
                "cosine_similarity": {
                    "mean": float(np.mean(prog_cos)),
                    "std": float(np.std(prog_cos)),
                },
                "n_frames": len(prog_l2),
            }

    # Pairwise comparisons (paired differences)
    pairwise = {}
    for condition_a, condition_b in [
        ("baseline", "subtask"),
    ]:
        l2_a = np.array([fm[condition_a]["l2_distance"] for fm in all_frame_metrics])
        l2_b = np.array([fm[condition_b]["l2_distance"] for fm in all_frame_metrics])
        diff = l2_a - l2_b  # positive = condition_b is closer to ground truth

        pairwise[f"{condition_a}_vs_{condition_b}"] = {
            "l2_diff_mean": float(np.mean(diff)),
            "l2_diff_std": float(np.std(diff)),
            "pct_b_better": float(np.mean(diff > 0) * 100),
            "pct_a_better": float(np.mean(diff < 0) * 100),
        }

        # Wilcoxon signed-rank test
        try:
            from scipy.stats import wilcoxon

            stat, p_value = wilcoxon(l2_a, l2_b, alternative="two-sided")
            pairwise[f"{condition_a}_vs_{condition_b}"]["wilcoxon_p"] = float(p_value)
            pairwise[f"{condition_a}_vs_{condition_b}"]["wilcoxon_stat"] = float(stat)
        except ImportError:
            logger.warning("scipy not available, skipping Wilcoxon test")
        except ValueError as e:
            logger.warning("Wilcoxon test failed: %s", e)

    results["pairwise"] = pairwise
    return results


def print_summary(results: dict[str, Any]) -> None:
    """Print a human-readable summary table."""
    print("\n" + "=" * 80)
    print("  DROID EVALUATION RESULTS")
    print("=" * 80)

    # Overall metrics table
    print(f"\n  {'Condition':<15} {'L2 Dist':>12} {'Cos Sim':>12} {'Grip Acc':>12} {'N':>6}")
    print(f"  {'-' * 57}")

    for condition in CONDITION_NAMES:
        overall = results[condition]["overall"]
        l2 = overall["l2_distance"]
        cos = overall["cosine_similarity"]
        grip = overall["gripper_accuracy"]
        n = overall["n_frames"]
        print(
            f"  {condition:<15} {l2['mean']:>8.4f}+-{l2['std']:<4.4f}"
            f" {cos['mean']:>8.4f}+-{cos['std']:<4.4f}"
            f" {grip['mean']:>8.4f}+-{grip['std']:<4.4f}"
            f" {n:>6}"
        )

    # Per-dimension MAE
    print("\n  Per-dimension MAE:")
    print(f"  {'Condition':<15}", end="")
    for name in JOINT_NAMES:
        print(f" {name:>8}", end="")
    print()
    print(f"  {'-' * (15 + 8 * len(JOINT_NAMES))}")

    for condition in CONDITION_NAMES:
        per_dim = results[condition]["overall"]["per_dim_mae"]
        print(f"  {condition:<15}", end="")
        for name in JOINT_NAMES:
            if name in per_dim:
                print(f" {per_dim[name]:>8.4f}", end="")
        print()

    # Task type breakdown
    for task_type in ["multi_step", "single_step"]:
        key = f"task_type_{task_type}"
        if key not in results["baseline"]:
            continue

        print(f"\n  {task_type.replace('_', ' ').title()} Tasks:")
        print(f"  {'Condition':<15} {'L2 Dist':>12} {'Cos Sim':>12} {'N':>6}")
        print(f"  {'-' * 45}")

        for condition in CONDITION_NAMES:
            if key not in results[condition]:
                continue
            data = results[condition][key]
            l2 = data["l2_distance"]
            cos = data["cosine_similarity"]
            n = data["n_frames"]
            print(
                f"  {condition:<15} {l2['mean']:>8.4f}+-{l2['std']:<4.4f}"
                f" {cos['mean']:>8.4f}+-{cos['std']:<4.4f}"
                f" {n:>6}"
            )

    # Pairwise comparisons
    print("\n  Pairwise Comparisons:")
    print(f"  {'Comparison':<30} {'L2 Diff':>10} {'% B Better':>12} {'Wilcoxon p':>12}")
    print(f"  {'-' * 64}")

    for comparison, data in results["pairwise"].items():
        p_str = f"{data['wilcoxon_p']:.4f}" if "wilcoxon_p" in data else "N/A"
        print(
            f"  {comparison:<30} {data['l2_diff_mean']:>10.4f}"
            f" {data['pct_b_better']:>11.1f}%"
            f" {p_str:>12}"
        )

    print("\n" + "=" * 80)

    # Interpretation
    pairwise = results["pairwise"]
    comparison = pairwise.get("baseline_vs_subtask", {})

    print("\n  Interpretation:")
    if comparison.get("l2_diff_mean", 0) > 0:
        print("  - Subtask produces actions CLOSER to ground truth than baseline (L2 diff > 0)")
    else:
        print("  - Subtask produces actions FARTHER from ground truth than baseline")

    p = comparison.get("wilcoxon_p", 1.0)
    if p < 0.05:
        print(f"  - Difference is statistically significant (p={p:.4f})")
    else:
        print(f"  - Difference is NOT statistically significant (p={p:.4f})")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute eval metrics")
    parser.add_argument("--samples_dir", type=str, required=True)
    parser.add_argument("--predictions_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    predictions_dir = Path(args.predictions_dir)

    # Load manifests
    manifest = load_manifest(samples_dir)

    with (predictions_dir / "prediction_manifest.json").open() as f:
        pred_manifest = json.load(f)

    # Build episode metadata index
    episode_metadata = {}
    for episode in manifest:
        episode_metadata[episode["episode_id"]] = episode

    # Process all frames
    all_frame_metrics = []
    task_types = []
    episode_progress_values = []

    for pred_entry in pred_manifest:
        episode_id = pred_entry["episode_id"]
        frame_idx = pred_entry["frame_idx"]
        episode = episode_metadata[episode_id]

        # Load ground truth
        frame_file = samples_dir / next(
            f["file"] for f in episode["frames"] if f["frame_idx"] == frame_idx
        )
        frame_data = np.load(frame_file)
        ground_truth = frame_data["ground_truth_actions"]  # [15, 8]

        # Load predictions
        pred_file = predictions_dir / pred_entry["prediction_file"]
        pred_data = np.load(pred_file)

        predictions = {
            "baseline": pred_data["baseline"],
            "subtask": pred_data["subtask"],
        }

        # Compute metrics
        frame_metrics = compute_frame_metrics(ground_truth, predictions)
        all_frame_metrics.append(frame_metrics)
        task_types.append(episode["task_type"])

        # Episode progress: where in the trajectory is this frame?
        progress = frame_idx / max(episode["traj_len"] - 1, 1)
        episode_progress_values.append(progress)

    logger.info("Computed metrics for %d frames", len(all_frame_metrics))

    # Aggregate
    results = aggregate_metrics(all_frame_metrics, task_types, episode_progress_values)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    logger.info("Results saved to %s", output_path)

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
