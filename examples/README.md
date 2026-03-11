# Examples

Run examples from the repository root:

```bash
poetry run python examples/load_data.py
poetry run python examples/complete_pipeline.py
```

## load_data.py

**Purpose**: Quick data inspection and contract verification

- Load 7,000 simulations from `data/database.csv`
- Inspect 10 building features, 12 ground-motion features, 91 responses
- Verify data shapes and column names
- Useful as a sanity check before running training

**Runtime**: < 1 second

## complete_pipeline.py

**Purpose**: Full 6-step surrogate-based optimization workflow

The complete pipeline demonstrates the **entire workflow** from raw data to optimized designs:

### Step 1: Data Loading & Preprocessing
- Load 7,000 simulations and extract inputs/outputs
- Display data shapes and sample values
- Verify all required columns are present
- Standardize the input space and scale the response space for stable learning
- Establish the preprocessing stage that every later step depends on

### Step 2: 2-Fold Cross-Validation & Training
- Train neural network surrogate with **optimal hyperparameters**:
  - Epochs: 100
  - Batch size: 32
  - Learning rate: 2e-5
  - L2 regularization: 1e-4
  - Dropout: [0.3, 0.3, 0.2]
- Collect per-epoch loss history (training curves)
- Compute fold-level metrics: MSE, MAE, R²
- Generate visualizations:
  - `training_curves.png` – Per-fold train/val loss over 100 epochs
  - `learning_curves.png` – MSE, MAE, R² across folds

### Step 3: Final Model Training
- Retrain model on **full dataset** (all 7,000 simulations)
- Use same hyperparameters as cross-validation
- Display training progress at key epochs (1, 50, 100)
- Produce the predictor needed for downstream design search

### Step 4: Surrogate Validation
- Generate predictions on 500 holdout samples
- Compute RMSE and MAE to verify surrogate accuracy
- Ensure model generalizes before optimization

### Step 5: Multi-Objective Optimization (NSGA-II)
- Use the trained surrogate as a **fast objective function**
- Run genetic algorithm with:
  - Population size: 100 designs
  - Generations: 10
  - Objectives: 8 responses (acceleration, drift, stress, mass, etc.)
- Extract Pareto-optimal designs (best trade-offs)

This ordering is intentional: if preprocessing has not been fitted, training is invalid; if training has not produced a usable predictor, optimization is invalid.

### Step 6: Summary & Exports
- Generate hyperparameter summary visualization
- Export Pareto-optimal designs to `pareto_optimal_designs.csv`
- Export predicted objectives to `pareto_objectives.csv`

### Output Files

All outputs stored in `outputs/` subdirectory:

**Visualizations** (`outputs/visualizations/`):
- `training_curves.png` – Loss convergence (per-fold and mean ± std)
- `learning_curves.png` – Generalization metrics (MSE, MAE, R²)
- `hyperparameter_summary.png` – Config and performance summary

**Optimization Results** (`outputs/optimization/`):
- `pareto_optimal_designs.csv` – Design variables for each Pareto solution
- `pareto_objectives.csv` – Predicted responses for each solution

**Runtime**: ~15-30 minutes (varies by hardware; 5-10 min on GPU)

## What Makes This Example Worth Showing?

This is not just a short demo. It shows the full logic of the workflow:

1. **Preprocessing is explicit** instead of being hidden inside training.
2. **Validation comes before final fitting** so the predictor is checked before it is used.
3. **Optimization comes last** because it depends on a working prediction stage.
4. **Diagnostics are saved** so convergence and generalization can be inspected.
5. **Outputs are exportable** for further analysis outside the script.

## Hyperparameter Justification

All hyperparameters (epochs, batch size, learning rate, etc.) were selected through repeated comparative tuning. For detailed rationale, see:

**[docs/HYPERPARAMETERS.md](../docs/HYPERPARAMETERS.md)**

Key insights:
- **Epochs = 100**: Validation loss plateaus; further training risks overfitting
- **Batch size = 32**: Tuned for 7,000-sample dataset (tested 16, 32, 64, 128)
- **Learning rate = 2e-5**: Very low for fine-tuned convergence (tested 1e-5, 5e-5, 1e-4)
- **Weight decay = 1e-4**: L2 regularization prevents overfitting without underfitting
- **Dropout = [0.3, 0.3, 0.2]**: Graduated rates (aggressive early, moderate late)

## Customization

### Modify hyperparameters

Common tuning directions:

- increase epochs if validation loss still trends downward
- reduce learning rate if optimization becomes unstable
- increase regularization if training error drops much faster than validation error
- increase fold count if you need a stronger estimate of generalization

### Modify optimization objectives

Common tuning directions:

- increase the population when you want a richer Pareto front
- increase generations when the search has not converged
- reduce the number of objectives when decision-making becomes too diffuse

### Use a different predictor

Any alternative approach still needs to preserve the same pipeline logic:
- preprocessing first
- supervised learning second
- validation before optimization
- full response prediction during search

## Related Documentation

- [GETTING_STARTED.md](../GETTING_STARTED.md) – Full walkthrough
- [docs/HYPERPARAMETERS.md](../docs/HYPERPARAMETERS.md) – Why these values?
- [docs/METHODOLOGY.md](../docs/METHODOLOGY.md) – Technical background
- [docs/DATASET.md](../docs/DATASET.md) – Feature/response definitions

## Notes

- Examples are **self-contained** and use only the public package API
- The repository includes **source code**, not trained weights (see GETTING_STARTED)
- Both examples are designed to **run without modification** on the included dataset