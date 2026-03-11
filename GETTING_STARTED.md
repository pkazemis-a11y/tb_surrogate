# Getting Started with Surrogate-Based Design Optimization

This repository provides a **complete, end-to-end pipeline** for surrogate-model training and multi-objective optimization of tall buildings with outer diagrids.

## Overview

The workflow consists of **6 main steps**:

1. **Data Loading & Preprocessing** – Load 7,000 simulations and prepare inputs/outputs
2. **Cross-Validation & Training** – Train a neural network surrogate with 2-fold CV
3. **Final Model Training** – Retrain on full dataset for deployment
4. **Surrogate Validation** – Verify predictor accuracy on holdout data
5. **Multi-Objective Optimization** – Use NSGA-II to find Pareto-optimal designs
6. **Visualization & Export** – Plot training metrics and save optimal designs

All hyperparameters are pre-tuned (based on extensive notebook trial-and-error) for best performance.

## Prerequisites

- **Python**: 3.8 to 3.11
- **Poetry**: For dependency management

## Quick Start (10 minutes)

### 1. Install dependencies

```bash
poetry install
```

### 2. Verify the environment is ready

```bash
poetry run python -c "from src.core import DataLoader; print('✓ Ready to go!')"
```

### 3. Run the complete pipeline

```bash
poetry run python examples/complete_pipeline.py
```

This will execute all 6 steps, print progress at each stage, and generate:
- **Visualizations**: `outputs/visualizations/` (training curves, learning curves, hyperparameter summary)
- **Optimization results**: `outputs/optimization/` (Pareto-optimal designs, response objectives)

Expected runtime: **~15-30 minutes** (depending on GPU availability)

## Step-by-Step Walkthrough

### Step 1: Data Loading & Preprocessing

The script loads the database and extracts:
- **Building features** (10): Top/bottom geometry, height, taper, twist, etc.
- **Ground-motion features** (12): Magnitude, distance, Vs30, intensity, etc.
- **Structural responses** (100): Accelerations, drifts, stresses, displacements, etc.

**What happens**:
- Raw CSV is validated for required columns
- Inputs are separated by modality (building vs. seismic)
- Responses are extracted and prepared for training

### Step 2: 2-Fold Cross-Validation & Training

Trains the neural network surrogate using **optimal hyperparameters** from notebook optimization:

| Hyperparameter | Value | Why? |
|---|---|---|
| Epochs | 200 | After 200, validation loss plateaus |
| Batch size | 32 | Tuned for this dataset size (7,000 samples) |
| Learning rate | 2e-5 | Low for fine-tuned, stable convergence |
| Weight decay | 1e-4 | L2 regularization to prevent overfitting |
| Dropout | [0.3, 0.3, 0.2] | Early layers more aggressive for regularization |
| K-folds | 2 | Balances robustness vs. computational cost |

**What happens**:
- Data is split into 2 folds
- For each fold:
  - Separate preprocessors fit on training fold
  - 200 epochs of PyTorch training with Adam optimizer
  - Per-epoch loss tracked (train and validation)
  - Fold-level metrics computed (MSE, MAE, R²)
- Visualization saved: `training_curves.png`, `learning_curves.png`

**Output**: Per-fold MSE, MAE, R² scores

### Step 3: Final Model Training

After validation, retrain the model on **all 7,000 simulations** with the same hyperparameters.

**What happens**:
- Single preprocessor fit on full dataset
- 200-epoch training run
- Per-epoch training loss tracked and displayed
- Model ready for predictions

**Output**: Trained in-memory model (no weights saved, following notebook methodology)

### Step 4: Surrogate Validation

Test the trained surrogate on a holdout subset (500 random samples).

**What happens**:
- Generate predictions on holdout data
- Compute RMSE and MAE
- Verify surrogate accuracy before using for optimization

**Output**: Holdout metrics (RMSE, MAE)

### Step 5: Multi-Objective Optimization (NSGA-II)

Use the trained surrogate as a **fast objective function** to optimize building designs.

**Algorithm**: NSGA-II (Non-dominated Sorting Genetic Algorithm II)
- Population size: 40 designs
- Generations: 50
- Objectives: 8 structural/economic responses

**What happens**:
- NSGA-II generates candidate designs
- Surrogate predicts responses for each candidate
- Pareto-optimal designs are identified
- Results stored as "pareto front"

**Output**: Pareto-optimal designs and their predicted responses

### Step 6: Summary & Exports

Generate final visualizations and export results.

**What happens**:
- Hyperparameter summary plot saved
- Pareto-optimal designs exported to CSV
- Response objectives exported to CSV

**Output**:
- `pareto_optimal_designs.csv` – Design variables for each Pareto solution
- `pareto_objectives.csv` – Predicted responses for each solution
- `hyperparameter_summary.png` – Training config and final metrics

## Understanding the Architecture

### Surrogate Model: Two-Branch MIMO Network

```
Building Features (10)      Ground-Motion Features (12)
        |                              |
    Dense(512)                    Dense(512)
    ReLU                          ReLU
    Dropout(0.3)                  Dropout(0.3)
        |                              |
        +----------->Merge<-----------+
                      |
                  Dense(512)
                  ReLU
                  Dropout(0.3)
                      |
                  Dense(256)
                  ReLU
                  Dropout(0.2)
                      |
                  Dense(128)
                  ReLU
                      |
            100 Response Outputs
            (8 groups for different
             structural categories)
```

**Why two branches?**
- Separate "encoders" for building and ground-motion modalities
- Each learns domain-specific transformations
- Merged representation combines complementary information
- Dropout regularization prevents co-adaptation

### Optimizing with the Surrogate

NSGA-II searches the design space to minimize **8 objectives simultaneously**:

1. Overall max acceleration
2. Max displacement
3. Max inter-story drift
4. Max von Mises stress
5. Max torsional response
6. Max response magnitude (high freq)
7. Max response magnitude (mid freq)
8. Total structural mass

**Trade-offs**: No single design minimizes all objectives. The Pareto front shows feasible trade-offs (e.g., strong buildings are heavier).

## Hyperparameter Tuning Rationale

For detailed justification of every hyperparameter, see **[docs/HYPERPARAMETERS.md](docs/HYPERPARAMETERS.md)**.

Key insights:
- **Learning rate 2e-5** was found optimal vs. 1e-4, 5e-5, 1e-5 (stable & fast convergence)
- **Batch size 32** beat 16, 64, 128 (better val loss, smoother curves)
- **Weight decay 1e-4** balanced regularization (reduced overfitting gap without underfitting)
- **Dropout [0.3, 0.3, 0.2]** leverages "early features are more general" principle
- **200 epochs** provides best validation loss; further training shows overfitting

These were found via trial-and-error in `surrogate-26.06.2024.ipynb` over many iterations.

## Customization & Extension

### Modify Hyperparameters

Edit `examples/complete_pipeline.py`, function `step_2_cross_validation_and_training()`:

```python
config = NeuralNetworkConfig(
    num_epochs=200,           # Change to 250 for more epochs
    batch_size=32,            # Change to 64 for larger batches
    learning_rate=2e-5,       # Change learning rate
    l2_weight_decay=1e-4,     # Adjust regularization
    k_fold_splits=2,          # Change CV folds
    device='cpu',             # Use 'cuda' for GPU
)
```

### Change Optimization Objectives

In `examples/complete_pipeline.py`, function `step_5_multi_objective_optimization()`:

```python
optimizer = NsGAIIOptimizer(
    surrogate_trainer=trainer,
    pop_size=40,              # Change population size
    n_gen=50,                 # Change generations
    seed=42,                  # Change random seed
)
```

### Use a Different Surrogate Model

The pipeline is agnostic to the surrogate implementation. Replace `ModelTrainer` in `examples/complete_pipeline.py` with any regressor that has:
- `.fit(x_building, x_gm, y)` → train
- `.predict(x_building, x_gm)` → (n_samples, 100) predictions

## Common Commands

### Run the full pipeline

```bash
poetry run python examples/complete_pipeline.py
```

### Load and inspect the data

```bash
poetry run python examples/load_data.py
```

### Run tests

```bash
poetry run pytest
```

### Check GPU availability

```bash
poetry run python -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

## Folder Structure

```
surrogate-based-generative-optimisation-of-diagrid-tall-buildings/
├── data/
│   └── database.csv                    # 7,000 simulations
├── src/
│   ├── core/                           # Data loading, preprocessing
│   ├── models/                         # Neural network, trainer
│   ├── optimization/                   # NSGA-II optimizer
│   └── visualization.py                # Training curve plots
├── examples/
│   ├── complete_pipeline.py            # 6-step workflow (START HERE)
│   └── load_data.py                    # Simple data inspection
├── docs/
│   ├── HYPERPARAMETERS.md              # Why these values?
│   └── METHODOLOGY.md                  # Technical background
└── outputs/
    ├── visualizations/                 # Training curves, learning curves
    └── optimization/                   # Pareto fronts, designs
```

## Troubleshooting

### ImportError: No module named 'torch'

Ensure poetry environment is activated:
```bash
poetry install
poetry run python examples/complete_pipeline.py
```

### CUDA out of memory

Edit `examples/complete_pipeline.py` and change `device='cuda'` to `device='cpu'`.

### Data file not found

Ensure you run the script from the repository root:
```bash
cd surrogate-based-generative-optimisation-of-diagrid-tall-buildings
poetry run python examples/complete_pipeline.py
```

## What's NOT Included

- **Trained weights**: Model is fit in-memory; weights are not persisted. This keeps the repo lightweight and focuses attention on the pipeline, not artifacts.
- **Pre-computed results**: Optimization runs fresh each time to reflect any code/data changes.

## Next Steps

1. **Understand the architecture**: Review [src/models/network.py](src/models/network.py) for the two-branch MIMO design
2. **Learn hyperparameter choices**: Read [docs/HYPERPARAMETERS.md](docs/HYPERPARAMETERS.md)
3. **Modify objectives**: Edit the 8 optimization targets in [src/optimization/nsga2.py](src/optimization/nsga2.py)
4. **Swap the surrogate**: Replace `ModelTrainer` with your own regressor
5. **Inspect the notebook**: Original exploratory work is in `Tall-buildings-with-outer-diagrids-design-exploration/_archive/surrogate-26.06.2024.ipynb`

## Questions?

Refer to:
- [docs/METHODOLOGY.md](docs/METHODOLOGY.md) – Technical method overview
- [docs/HYPERPARAMETERS.md](docs/HYPERPARAMETERS.md) – Hyperparameter justification
- [docs/DATASET.md](docs/DATASET.md) – Feature/response definitions