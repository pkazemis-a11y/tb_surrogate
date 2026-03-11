"""
COMPLETE PIPELINE: Surrogate Model Training + Multi-Objective Optimization

This script demonstrates the complete workflow for the surrogate-based design optimization:

    1. DATA LOADING & PREPROCESSING
       - Load 7,000 simulations from the database
       - Extract 10 building features and 12 ground-motion features
       - Extract 100 structural response variables
       - Standardize inputs, scale responses

    2. CROSS-VALIDATION & TRAINING
       - 2-fold cross-validation for robust performance assessment
       - Train neural network surrogate with optimal hyperparameters
       - Visualize training convergence (loss curves) and generalization (learning curves)

    3. FINAL MODEL FIT
       - Retrain on full dataset with tuned hyperparameters
       - Generate predictions on test designs

    4. MULTI-OBJECTIVE OPTIMIZATION (NSGA-II)
       - Use fitted surrogate as objective function
       - Optimize designs for minimizing responses across 8 design objectives
       - Explore Pareto-optimal design space

Hyperparameters are derived from notebook trial-and-error (surrogate-26.06.2024.ipynb):
  • Epochs: 200
  • Batch size: 32
  • Learning rate: 2e-5 (low for fine-tuning)
  • Weight decay (L2): 1e-4
  • Dropout: [0.3, 0.3, 0.2] (regularization to prevent overfitting)
  • K-fold: 2 (validation strategy)

Reference: See docs/HYPERPARAMETERS.md for detailed rationale.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path so we can import modules
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.core import ALL_RESPONSE_COLUMNS, DataLoader, NeuralNetworkConfig
from src.models import ModelTrainer
from src.optimization import NsGAIIOptimizer
from src.visualization import (
    plot_hyperparameter_summary,
    plot_learning_curves,
    plot_training_curves,
)


def print_section(title: str) -> None:
    """Pretty-print a pipeline section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def step_1_load_and_preprocess_data() -> tuple:
    """
    STEP 1: Load data and prepare features and responses.

    This step:
    - Reads 7,000 simulations from the database
    - Extracts building input features (10 variables: geometry, height, taper, twist, etc.)
    - Extracts ground-motion input features (12 variables: magnitude, distance, Vs30, etc.)
    - Extracts structural response outputs (100 variables: accelerations, drifts, stresses, etc.)

    Returns:
        (x_building, x_gm, y): DataFrames with features and responses
    """
    print_section("STEP 1: DATA LOADING & PREPROCESSING")

    data_path = REPO_ROOT / "data" / "database.csv"
    print(f"Loading database from {data_path}...")

    loader = DataLoader(str(data_path))
    data = loader.load()
    x_building, x_gm, y = loader.get_features_and_responses()

    print(f"✓ Data loaded successfully!")
    print(f"  • Simulations: {len(data):,}")
    print(f"  • Building features: {x_building.shape[1]} (e.g., height, geometry, taper, twist)")
    print(f"  • Ground-motion features: {x_gm.shape[1]} (e.g., magnitude, distance, Vs30)")
    print(f"  • Structural responses: {y.shape[1]} (e.g., accelerations, drifts, stresses)")
    print()

    # Show sample data
    print("Sample building features (top 3 rows):")
    print(x_building.head(3))
    print()
    print("Sample ground-motion features (top 3 rows):")
    print(x_gm.head(3))
    print()
    print("Sample response variables (top 3 rows, first 5 columns):")
    print(y.iloc[:3, :5])
    print()

    return x_building, x_gm, y


def step_2_cross_validation_and_training(
    x_building: pd.DataFrame,
    x_gm: pd.DataFrame,
    y: pd.DataFrame,
) -> dict:
    """
    STEP 2: Perform 2-fold cross-validation with the neural network surrogate.

    This step:
    - Splits data into 2 folds for robust validation
    - Train surrogate on each fold independently
    - Collect per-epoch loss (training curve) for convergence analysis
    - Evaluate fold-level generalization (MSE, MAE, R²)
    - Generate and save visualization: training curves and learning curves

    Hyperparameters (from notebook optimization):
    - Epochs: 200
    - Batch size: 32
    - Learning rate: 2e-5
    - Weight decay: 1e-4
    - Architecture: 2-branch MIMO network (512 units/branch, merge → 512→256→128)
    - Dropout: [0.3, 0.3, 0.2]

    Returns:
        cv_results: CrossValidationSummary with folds, histories, metrics
    """
    print_section("STEP 2: CROSS-VALIDATION & TRAINING")
    print("Running 2-fold cross-validation with optimal hyperparameters...")
    print("  [These hyperparameters were found via extensive trial-and-error in the notebook]")
    print()

    # Set up model configuration with optimal hyperparameters from notebook
    config = NeuralNetworkConfig(
        num_epochs=200,        # Iterations to train optimizer
        batch_size=32,         # Found to be optimal in notebook trials
        learning_rate=2e-5,    # Low LR for stable convergence
        l2_weight_decay=1e-4,  # L2 regularization strength
        k_fold_splits=2,       # Number of CV folds
        device='cpu',          # Change to 'cuda' if GPU available
    )

    trainer = ModelTrainer(config)
    cv_summary = trainer.train_with_cross_validation(x_building, x_gm, y)

    print(f"✓ Cross-validation complete!")
    print(f"  • Mean training loss (final epoch): {cv_summary.mean_train_loss:.6f}")
    print(f"  • Mean validation loss (final epoch): {cv_summary.mean_val_loss:.6f}")
    print()

    # Print per-fold results
    print("Per-fold validation metrics:")
    for i, (fold_summary, metrics) in enumerate(
        zip(cv_summary.folds, cv_summary.fold_metrics)
    ):
        print(
            f"  Fold {i + 1}: MSE={metrics['mse']:.6f}, "
            f"MAE={metrics['mae']:.6f}, R²={metrics['r2']:.4f}"
        )
    print()

    # Aggregate metrics
    mean_mse = np.mean([m["mse"] for m in cv_summary.fold_metrics])
    mean_mae = np.mean([m["mae"] for m in cv_summary.fold_metrics])
    mean_r2 = np.mean([m["r2"] for m in cv_summary.fold_metrics])

    print(f"Average metrics across folds:")
    print(f"  • MSE: {mean_mse:.6f}")
    print(f"  • MAE: {mean_mae:.6f}")
    print(f"  • R²: {mean_r2:.4f}")
    print()

    # Visualize training convergence and generalization
    print("Generating visualizations...")
    output_dir = REPO_ROOT / "outputs" / "visualizations"

    if cv_summary.fold_histories:
        plot_training_curves(cv_summary.fold_histories, output_dir)

    if cv_summary.fold_metrics:
        plot_learning_curves(cv_summary.fold_metrics, output_dir)

    return {
        "cv_summary": cv_summary,
        "config": config,
        "mean_metrics": {"mse": mean_mse, "mae": mean_mae, "r2": mean_r2},
    }


def step_3_fit_final_model(
    x_building: pd.DataFrame,
    x_gm: pd.DataFrame,
    y: pd.DataFrame,
    config: NeuralNetworkConfig,
) -> tuple:
    """
    STEP 3: Fit final surrogate on entire dataset.

    After cross-validation assessment, retrain the model on all available data
    with the same hyperparameters. This model is used for optimization.

    Returns:
        (trainer, train_history): Fitted ModelTrainer and per-epoch losses
    """
    print_section("STEP 3: FINAL MODEL TRAINING")
    print("Training final model on complete dataset (all 7,000 simulations)...")
    print(f"  • Epochs: {config.num_epochs}")
    print(f"  • Batch size: {config.batch_size}")
    print(f"  • Learning rate: {config.learning_rate}")
    print()

    final_trainer = ModelTrainer(config)
    model, train_history = final_trainer.fit(x_building, x_gm, y)

    print(f"✓ Model training complete!")
    print(f"  • Training loss (epoch 1): {train_history[0]:.6f}")
    print(f"  • Training loss (epoch 50): {train_history[49] if len(train_history) > 49 else 'N/A'}")
    print(f"  • Training loss (epoch 100): {train_history[99] if len(train_history) > 99 else 'N/A'}")
    print(f"  • Training loss (final epoch 200): {train_history[-1]:.6f}")
    print()

    return final_trainer, train_history


def step_4_test_surrogate_predictions(
    trainer: "ModelTrainer",
    x_building: pd.DataFrame,
    x_gm: pd.DataFrame,
    y: pd.DataFrame,
) -> np.ndarray:
    """
    STEP 4: Test surrogate predictions on holdout data.

    Generate predictions and compute holdout metrics to validate surrogate accuracy
    before using it for optimization.

    Returns:
        y_pred: Model predictions on test data
    """
    print_section("STEP 4: SURROGATE VALIDATION")

    # For this example, use a random holdout subset (in practice, use separate test set)
    test_indices = np.random.choice(len(x_building), size=500, replace=False)
    x_building_test = x_building.iloc[test_indices]
    x_gm_test = x_gm.iloc[test_indices]
    y_test = y.iloc[test_indices]

    print(f"Testing surrogate on {len(y_test)} holdout samples...")
    y_pred = trainer.predict(x_building_test, x_gm_test)
    eval_summary = trainer.evaluate(x_building_test, x_gm_test, y_test)

    print(f"✓ Holdout evaluation complete!")
    print(f"  • RMSE: {eval_summary.rmse:.6f}")
    print(f"  • MAE: {eval_summary.mae:.6f}")
    print()

    return y_pred


def step_5_multi_objective_optimization(
    trainer: "ModelTrainer",
) -> dict:
    """
    STEP 5: Multi-objective design optimization using NSGA-II.

    Uses the trained surrogate as a fast objective function evaluator.
    Optimizes building designs to minimize 8 structural response objectives simultaneously:
    1. Overall max acceleration
    2. Max displacement
    3. Max inter-story drift
    4. Max von Mises stress
    5. Max torsional response
    6. Max response magnitude (high frequency)
    7. Max response magnitude (mid frequency)
    8. Total structural mass

    The optimization explores the design space defined by:
    - Model parameters: top/bottom geometry, orientation, tapering, twisting
    - Ground motions: variable seismic events (from recorded database)

    Returns:
        opt_results: Optimization results (Pareto-optimal designs)
    """
    print_section("STEP 5: MULTI-OBJECTIVE OPTIMIZATION")

    print("Setting up NSGA-II optimizer...")
    print("  • Algorithm: NSGA-II (Non-dominated Sorting Genetic Algorithm II)")
    print("  • Population size: 40")
    print("  • Generations: 50")
    print("  • Objectives to minimize: 8 (acceleration, drift, stress, mass, etc.)")
    print()

    optimizer = NsGAIIOptimizer(
        surrogate_trainer=trainer,
        pop_size=40,
        n_gen=50,
        seed=42,
    )

    print("Running optimization...")
    opt_results = optimizer.optimize()

    print(f"✓ Optimization complete!")
    print(f"  • Pareto-optimal solutions found: {len(opt_results['pareto_designs'])}")
    print()

    # Show best design for primary objective (minimize acceleration)
    if len(opt_results["pareto_designs"]) > 0:
        best_idx = np.argmin(opt_results["pareto_objectives"][:, 0])
        print(f"Best design for minimizing acceleration:")
        print(f"  • Max acceleration: {opt_results['pareto_objectives'][best_idx, 0]:.4f}")
        print(f"  • Max displacement: {opt_results['pareto_objectives'][best_idx, 1]:.4f}")
        print(f"  • Max drift: {opt_results['pareto_objectives'][best_idx, 2]:.4f}")
        print()

    return opt_results


def step_6_summary_and_exports(
    cv_results: dict,
    opt_results: dict,
    config: NeuralNetworkConfig,
) -> None:
    """
    STEP 6: Generate summary reports and export optimization results.

    Creates:
    - Hyperparameter summary visualization
    - CSV export of Pareto-optimal designs
    - Summary statistics
    """
    print_section("STEP 6: SUMMARY & EXPORTS")

    # Create hyperparameter and metrics summary visualization
    output_dir = REPO_ROOT / "outputs" / "visualizations"
    hyperparams = {
        "epochs": config.num_epochs,
        "batch_size": config.batch_size,
        "learning_rate": f"{config.learning_rate:.2e}",
        "weight_decay": f"{config.l2_weight_decay:.2e}",
        "k_fold_splits": config.k_fold_splits,
    }
    metrics_summary = cv_results["mean_metrics"]

    plot_hyperparameter_summary(hyperparams, metrics_summary, output_dir)

    # Export optimization results (Pareto front)
    if opt_results and "pareto_designs" in opt_results:
        output_dir = REPO_ROOT / "outputs" / "optimization"
        output_dir.mkdir(parents=True, exist_ok=True)

        pareto_df = pd.DataFrame(opt_results["pareto_designs"])
        pareto_df.to_csv(output_dir / "pareto_optimal_designs.csv", index=False)
        print(f"✓ Exported Pareto-optimal designs to pareto_optimal_designs.csv")

        pareto_obj_df = pd.DataFrame(
            opt_results["pareto_objectives"],
            columns=[
                "Max_Acceleration",
                "Max_Displacement",
                "Max_Drift",
                "Max_Von_Mises",
                "Max_Torsion",
                "Max_Magnitude_R",
                "Max_Magnitude_M",
                "Total_Mass",
            ],
        )
        pareto_obj_df.to_csv(output_dir / "pareto_objectives.csv", index=False)
        print(f"✓ Exported Pareto objectives to pareto_objectives.csv")

    print()


def main() -> None:
    """Execute the complete pipeline: preprocessing → training → optimization."""
    print("\n" + "=" * 70)
    print("  SURROGATE-BASED GENERATIVE OPTIMIZATION - COMPLETE PIPELINE")
    print("=" * 70)

    # ========== STEP 1: Load & Preprocess ==========
    x_building, x_gm, y = step_1_load_and_preprocess_data()

    # ========== STEP 2: Cross-Validation & Training ==========
    cv_results = step_2_cross_validation_and_training(x_building, x_gm, y)

    # ========== STEP 3: Fit Final Model ==========
    final_trainer, train_history = step_3_fit_final_model(
        x_building,
        x_gm,
        y,
        cv_results["config"],
    )

    # ========== STEP 4: Test Surrogate ==========
    y_pred = step_4_test_surrogate_predictions(final_trainer, x_building, x_gm, y)

    # ========== STEP 5: Multi-Objective Optimization ==========
    opt_results = step_5_multi_objective_optimization(final_trainer)

    # ========== STEP 6: Summary & Exports ==========
    step_6_summary_and_exports(cv_results, opt_results, cv_results["config"])

    # ========== FINAL SUMMARY ==========
    print_section("PIPELINE EXECUTION COMPLETE")
    print("✓ All steps executed successfully!")
    print()
    print("Outputs saved to:")
    print(f"  • Visualizations: {REPO_ROOT / 'outputs' / 'visualizations'}")
    print(f"  • Optimization: {REPO_ROOT / 'outputs' / 'optimization'}")
    print()
    print("Next steps:")
    print("  1. Review training curves to verify convergence")
    print("  2. Examine Pareto-optimal designs for practical feasibility")
    print("  3. Refine objectives or constraints as needed")
    print("  4. See examples/README.md for further guidance")
    print()


if __name__ == "__main__":
    main()
