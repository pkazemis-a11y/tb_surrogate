# Examples

Run examples from the repository root:

```bash
poetry run python examples/load_data.py
poetry run python examples/complete_pipeline.py
```

## load_data.py

**Purpose**: Quick data inspection and contract verification

- Load 7,000 simulations from `data/database.csv`
- Inspect 10 building features, 12 ground-motion features, 100 responses
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

### Step 2: 2-Fold Cross-Validation & Training
- Train neural network surrogate with **optimal hyperparameters**:
  - Epochs: 200
  - Batch size: 32
  - Learning rate: 2e-5
  - L2 regularization: 1e-4
  - Dropout: [0.3, 0.3, 0.2]
- Collect per-epoch loss history (training curves)
- Compute fold-level metrics: MSE, MAE, R²
- Generate visualizations:
  - `training_curves.png` – Per-fold train/val loss over 200 epochs
  - `learning_curves.png` – MSE, MAE, R² across folds

### Step 3: Final Model Training
- Retrain model on **full dataset** (all 7,000 simulations)
- Use same hyperparameters as cross-validation
- Display training progress at key epochs (1, 50, 100, 200)
- Model ready for deployment

### Step 4: Surrogate Validation
- Generate predictions on 500 holdout samples
- Compute RMSE and MAE to verify surrogate accuracy
- Ensure model generalizes before optimization

### Step 5: Multi-Objective Optimization (NSGA-II)
- Use trained surrogate as **fast objective function**
- Run genetic algorithm with:
  - Population size: 40 designs
  - Generations: 50
  - Objectives: 8 responses (acceleration, drift, stress, mass, etc.)
- Extract Pareto-optimal designs (best trade-offs)

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

## What Makes This Example "Complete"?

Unlike typical "quick-start" examples, this pipeline is:

1. **Production-ready**: Full preprocessing, validation, and error handling
2. **Educational**: Clear print statements explain each step
3. **Reproducible**: Fixed random seeds and deterministic splits
4. **Well-documented**: Every function has docstrings explaining inputs/outputs/methodology
5. **Visualization-rich**: Training curves, learning curves, and hyperparameter summary plots
6. **Export-ready**: Results saved as CSV for further analysis

## Hyperparameter Justification

All hyperparameters (epochs, batch size, learning rate, etc.) were **optimized via trial-and-error** in the original Jupyter notebook. For detailed rationale, see:

**[docs/HYPERPARAMETERS.md](../docs/HYPERPARAMETERS.md)**

Key insights:
- **Epochs = 200**: Validation loss plateaus; further training risks overfitting
- **Batch size = 32**: Tuned for 7,000-sample dataset (tested 16, 32, 64, 128)
- **Learning rate = 2e-5**: Very low for fine-tuned convergence (tested 1e-5, 5e-5, 1e-4)
- **Weight decay = 1e-4**: L2 regularization prevents overfitting without underfitting
- **Dropout = [0.3, 0.3, 0.2]**: Graduated rates (aggressive early, moderate late)

## Customization

### Modify hyperparameters

Edit `step_2_cross_validation_and_training()`:

```python
config = NeuralNetworkConfig(
    num_epochs=250,           # Increase to 250 epochs
    batch_size=64,            # Change to larger batch
    learning_rate=1e-5,       # Lower learning rate
    l2_weight_decay=5e-4,     # Stronger regularization
    k_fold_splits=3,          # Change to 3-fold CV
    device='cuda',            # Use GPU if available
)
```

### Modify optimization objectives

Edit `step_5_multi_objective_optimization()`:

```python
optimizer = NsGAIIOptimizer(
    surrogate_trainer=trainer,
    pop_size=100,             # Larger population
    n_gen=100,                # More generations
    seed=123,                 # Different random seed
)
```

### Use a different surrogate

Replace `ModelTrainer` (in `src.models`) with any regressor implementing:
- `.fit(x_building, x_gm, y)` – train
- `.predict(x_building, x_gm)` → (n_samples, 100) array

## Related Documentation

- [GETTING_STARTED.md](../GETTING_STARTED.md) – Full walkthrough
- [docs/HYPERPARAMETERS.md](../docs/HYPERPARAMETERS.md) – Why these values?
- [docs/METHODOLOGY.md](../docs/METHODOLOGY.md) – Technical background
- [docs/DATASET.md](../docs/DATASET.md) – Feature/response definitions

## Notes

- Examples are **self-contained** and use only the public package API
- The repository includes **source code**, not trained weights (see GETTING_STARTED)
- Both examples are designed to **run without modification** on the included dataset