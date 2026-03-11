"""Visualization utilities for training metrics, learning curves, and design exploration."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(
    fold_histories: List[Dict[str, List[float]]],
    output_dir: Optional[Path] = None,
) -> None:
    """
    Plot training and validation loss across all folds.

    This visualization shows:
    - Training loss per epoch for each fold (convergence behavior)
    - Validation loss per epoch for each fold
    - How loss stabilizes after ~150 epochs with lr=2e-5 and batch_size=32

    Args:
        fold_histories: List of dicts, each with 'train_loss' and 'val_loss' keys
                       (each a list of floats per epoch)
        output_dir: Directory to save the plot. If None, displays but doesn't save.

    Example:
        >>> histories = [
        ...     {'train_loss': [0.5, 0.4, ...], 'val_loss': [0.6, 0.45, ...]},
        ...     {'train_loss': [0.5, 0.4, ...], 'val_loss': [0.6, 0.45, ...]},
        ... ]
        >>> plot_training_curves(histories, Path('plots'))
    """
    n_folds = len(fold_histories)
    n_epochs = len(fold_histories[0]["train_loss"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: All folds overlayed
    ax = axes[0]
    for fold_idx, history in enumerate(fold_histories):
        ax.plot(
            history["train_loss"],
            label=f"Fold {fold_idx + 1} (train)",
            alpha=0.7,
            linestyle="-",
        )
        ax.plot(
            history["val_loss"],
            label=f"Fold {fold_idx + 1} (val)",
            alpha=0.7,
            linestyle="--",
        )

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("MSE Loss", fontsize=11)
    ax.set_title("Per-Fold Training & Validation Loss (200 epochs, lr=2e-5)", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Plot 2: Mean and std across folds
    ax = axes[1]
    train_losses = np.array([h["train_loss"] for h in fold_histories])
    val_losses = np.array([h["val_loss"] for h in fold_histories])

    train_mean = train_losses.mean(axis=0)
    train_std = train_losses.std(axis=0)
    val_mean = val_losses.mean(axis=0)
    val_std = val_losses.std(axis=0)

    epochs = np.arange(n_epochs)
    ax.plot(epochs, train_mean, label="Train (mean ± std)", color="blue", linewidth=2)
    ax.fill_between(
        epochs, train_mean - train_std, train_mean + train_std, alpha=0.2, color="blue"
    )

    ax.plot(epochs, val_mean, label="Val (mean ± std)", color="orange", linewidth=2)
    ax.fill_between(
        epochs, val_mean - val_std, val_mean + val_std, alpha=0.2, color="orange"
    )

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("MSE Loss", fontsize=11)
    ax.set_title(f"Mean Loss Across {n_folds} Folds (with ±1 std band)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / "training_curves.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Training curves saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_learning_curves(
    fold_metrics: List[Dict[str, float]], output_dir: Optional[Path] = None
) -> None:
    """
    Plot learning curves: final MSE, MAE, and R² across folds.

    This shows model generalization and performance per fold, helping identify overfitting.

    Args:
        fold_metrics: List of dicts with 'mse', 'mae', 'r2' keys
        output_dir: Directory to save. If None, displays only.

    Example:
        >>> metrics = [
        ...     {'mse': 0.025, 'mae': 0.10, 'r2': 0.92},
        ...     {'mse': 0.023, 'mae': 0.09, 'r2': 0.93},
        ... ]
        >>> plot_learning_curves(metrics, Path('plots'))
    """
    n_folds = len(fold_metrics)
    fold_indices = np.arange(1, n_folds + 1)

    mses = np.array([m["mse"] for m in fold_metrics])
    maes = np.array([m["mae"] for m in fold_metrics])
    r2s = np.array([m["r2"] for m in fold_metrics])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # MSE
    ax = axes[0]
    ax.bar(fold_indices, mses, color="steelblue", alpha=0.7, edgecolor="black")
    ax.axhline(mses.mean(), color="red", linestyle="--", label=f"Mean: {mses.mean():.4f}")
    ax.set_xlabel("Fold", fontsize=11)
    ax.set_ylabel("MSE", fontsize=11)
    ax.set_title("Mean Squared Error per Fold", fontsize=12)
    ax.set_xticks(fold_indices)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # MAE
    ax = axes[1]
    ax.bar(fold_indices, maes, color="coral", alpha=0.7, edgecolor="black")
    ax.axhline(maes.mean(), color="red", linestyle="--", label=f"Mean: {maes.mean():.4f}")
    ax.set_xlabel("Fold", fontsize=11)
    ax.set_ylabel("MAE", fontsize=11)
    ax.set_title("Mean Absolute Error per Fold", fontsize=12)
    ax.set_xticks(fold_indices)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # R²
    ax = axes[2]
    ax.bar(fold_indices, r2s, color="lightgreen", alpha=0.7, edgecolor="black")
    ax.axhline(r2s.mean(), color="red", linestyle="--", label=f"Mean: {r2s.mean():.4f}")
    ax.set_xlabel("Fold", fontsize=11)
    ax.set_ylabel("R²", fontsize=11)
    ax.set_title("R² Score per Fold", fontsize=12)
    ax.set_xticks(fold_indices)
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / "learning_curves.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Learning curves saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    response_names: Optional[List[str]] = None,
    sample_indices: Optional[List[int]] = None,
    output_dir: Optional[Path] = None,
) -> None:
    """
    Plot predicted vs actual responses for diagnostics.

    Helps visualize model accuracy across the response space. Shows up to 9 response
    variables in a 3x3 grid for quick visual inspection.

    Args:
        y_true: Ground truth responses (n_samples, n_responses)
        y_pred: Model predictions (n_samples, n_responses)
        response_names: Names of response variables (for labels)
        sample_indices: Which samples to include. If None, uses all.
        output_dir: Directory to save. If None, displays only.
    """
    if sample_indices is None:
        sample_indices = np.arange(len(y_true))

    y_true_subset = y_true[sample_indices]
    y_pred_subset = y_pred[sample_indices]
    n_responses = min(9, y_true_subset.shape[1])  # Show up to 9

    if response_names is None:
        response_names = [f"Response {i}" for i in range(y_true_subset.shape[1])]

    n_cols = 3
    n_rows = (n_responses + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(12, 4 * n_rows), squeeze=False
    )
    axes = axes.flatten()

    for i in range(n_responses):
        ax = axes[i]
        ax.scatter(
            y_true_subset[:, i],
            y_pred_subset[:, i],
            alpha=0.6,
            s=20,
            edgecolors="k",
            linewidth=0.5,
        )

        # Add diagonal reference line
        y_min = min(y_true_subset[:, i].min(), y_pred_subset[:, i].min())
        y_max = max(y_true_subset[:, i].max(), y_pred_subset[:, i].max())
        ax.plot([y_min, y_max], [y_min, y_max], "r--", alpha=0.5, linewidth=1)

        ax.set_xlabel("Actual", fontsize=10)
        ax.set_ylabel("Predicted", fontsize=10)
        ax.set_title(response_names[i], fontsize=11)
        ax.grid(True, alpha=0.3)

    # Hide extra subplots
    for i in range(n_responses, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / "predictions_vs_actual.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Predictions vs actual plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_hyperparameter_summary(
    hyperparams: Dict[str, any],
    metrics_summary: Dict[str, float],
    output_dir: Optional[Path] = None,
) -> None:
    """
    Create a summary plot of hyperparameters and overall metrics.

    Useful for documenting the training configuration and final performance.

    Args:
        hyperparams: Dict with hyperparameter names and values
                    (e.g., {'epochs': 200, 'batch_size': 32, 'lr': 2e-5})
        metrics_summary: Dict with final metrics (e.g., {'mean_mse': 0.024, 'mean_mae': 0.095})
        output_dir: Directory to save. If None, displays only.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Hyperparameters as text
    ax1.axis("off")
    hp_text = "Hyperparameters (from notebook trial-and-error):\n\n"
    for key, val in hyperparams.items():
        if isinstance(val, float) and val < 0.001:
            hp_text += f"  • {key}: {val:.2e}\n"
        else:
            hp_text += f"  • {key}: {val}\n"

    ax1.text(0.1, 0.9, hp_text, fontsize=11, verticalalignment="top",
             family="monospace", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Metrics as text
    ax2.axis("off")
    metrics_text = "Final Performance Metrics:\n\n"
    for key, val in metrics_summary.items():
        metrics_text += f"  • {key}: {val:.4f}\n"

    ax2.text(0.1, 0.9, metrics_text, fontsize=11, verticalalignment="top",
             family="monospace", bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5))

    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / "hyperparameter_summary.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Hyperparameter summary saved to {save_path}")
    else:
        plt.show()

    plt.close()
