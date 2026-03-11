# Getting Started with Surrogate-Based Design Optimization

This repository is organized as one pipeline: **data import → preprocessing → training → validation → optimization**. Each stage depends on the previous one being done correctly.

That ordering matters. If the data has not been prepared, training is unreliable. If the predictor has not been trained and checked, optimization is just searching on top of noise.

## Overview

The workflow consists of **6 main steps**:

1. **Data Loading & Preprocessing** – Load 7,000 simulations and prepare inputs/outputs
2. **Cross-Validation & Training** – Train a neural network surrogate with 2-fold CV
3. **Final Model Training** – Retrain on full dataset for deployment
4. **Surrogate Validation** – Verify predictor accuracy on holdout data
5. **Multi-Objective Optimization** – Use NSGA-II to find Pareto-optimal designs
6. **Visualization & Export** – Plot training metrics and save optimal designs

All hyperparameters are pre-tuned based on repeated comparative experiments for best performance.

## Prerequisites

- **Python**: 3.8 to 3.13
- **Poetry**: For dependency management

## Quick Start (10 minutes)

### 1. Install dependencies

```bash
poetry install
```

### 2. Verify the environment is ready

```bash
poetry run python examples/load_data.py
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
- **Structural responses** (91): Accelerations, drifts, stresses, displacements, etc.

**What happens**:
- Raw CSV is validated for required columns
- Inputs are separated by modality (building vs. seismic)
- Input features are standardized for stable numerical behavior during training
- Responses are scaled so multi-output learning remains well-conditioned
- Responses are extracted and prepared for training

**Why this comes first**:
- Without preprocessing, the training stage mixes variables with very different ranges and semantics.
- Without a fitted preprocessing stage, optimization would send invalid inputs into the predictor.

### Step 2: 2-Fold Cross-Validation & Training

Trains the neural network surrogate using **pre-tuned hyperparameters**:

| Hyperparameter | Value | Why? |
|---|---|---|
| Epochs | 100 | After 100, validation loss plateaus |
| Batch size | 32 | Tuned for this dataset size (7,000 samples) |
| Learning rate | 2e-5 | Low for fine-tuned, stable convergence |
| Weight decay | 1e-4 | L2 regularization to prevent overfitting |
| Dropout | [0.3, 0.3, 0.2] | Early layers more aggressive for regularization |
| K-folds | 2 | Balances robustness vs. computational cost |

**What happens**:
- Data is split into 2 folds
- For each fold:
  - preprocessing is fitted on the training partition only
  - 100 epochs of training are run with Adam
  - per-epoch loss is tracked for both training and validation
  - fold-level metrics are computed (MSE, MAE, R²)
- Visualizations are saved: `training_curves.png`, `learning_curves.png`

**Output**: Per-fold MSE, MAE, R² scores

**Why this comes before optimization**:
- Optimization needs a predictor that can evaluate many candidate designs cheaply.
- That predictor only exists after training has converged and validation shows acceptable error.

### Step 3: Final Model Training

After validation, retrain the model on **all 7,000 simulations** with the same hyperparameters.

**What happens**:
- Single preprocessor fit on full dataset
- 100-epoch training run
- Per-epoch training loss tracked and displayed
- Model ready for predictions

**Output**: Trained in-memory model (no weights saved)

This step turns the validated workflow into a usable prediction stage for downstream design search.

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
- Population size: 100 designs
- Generations: 10
- Objectives: 8 structural/economic responses

**What happens**:
- NSGA-II generates candidate designs
- Surrogate predicts responses for each candidate
- Pareto-optimal designs are identified
- Results stored as "pareto front"

**Output**: Pareto-optimal designs and their predicted responses

This stage is downstream by design: if the predictor is not trained and validated first, the optimization results have no practical meaning.

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

## Core Design Choices

### Input and Output Handling

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
            91 Response Outputs
            (8 groups for different
             structural categories)
```

**Why separate the inputs?**
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
- **100 epochs** provides best validation loss; further training shows overfitting

These values were selected through repeated comparative tuning over many iterations.

## Customization & Extension

### Adjust training behavior

Typical tuning directions:

- increase epochs if validation loss is still falling steadily
- increase batch size only if runtime or memory is the constraint
- lower learning rate when optimization is unstable
- increase regularization when training error drops much faster than validation error
- change fold count if you need a stronger estimate of generalization

### Adjust optimization behavior

Typical tuning directions:

- increase population size when you want a broader Pareto front
- increase number of generations when convergence is still improving
- narrow the objective set when you want more interpretable trade-offs
- tighten design bounds when the search space is too large for the available data coverage

### Use a Different Predictor

The pipeline is agnostic to the predictor implementation as long as it can:
- learn from preprocessed building inputs, ground-motion inputs, and response targets
- return one full response vector per candidate design
- support repeated evaluation during optimization

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
tb_surrogate/
├── data/                              # training dataset
├── docs/                              # workflow notes and rationale
├── examples/                          # runnable examples
├── src/                               # implementation of the pipeline
├── tests/                             # smoke tests
└── outputs/                           # generated plots and optimization results
```

## Troubleshooting

### ImportError: No module named 'torch'

Ensure poetry environment is activated:
```bash
poetry install
poetry run python examples/complete_pipeline.py
```

### CUDA out of memory

Run on CPU instead of GPU or reduce batch size.

### Data file not found

Ensure you run the script from the repository root:
```bash
cd tb_surrogate
poetry run python examples/complete_pipeline.py
```

## What's NOT Included

- **Trained weights**: Model is fit in-memory; weights are not persisted. This keeps the repo lightweight and focuses attention on the pipeline, not artifacts.
- **Pre-computed results**: Optimization runs fresh each time to reflect any code/data changes.

## Next Steps

1. **Understand the workflow**: Read this guide once from start to finish as a pipeline, not as isolated steps
2. **Learn hyperparameter choices**: Read [docs/HYPERPARAMETERS.md](docs/HYPERPARAMETERS.md)
3. **Review design trade-offs**: Inspect the optimization outputs and Pareto front behavior
4. **Adapt the workflow**: Tune preprocessing, training, and optimization together rather than changing only one stage

## Questions?

Refer to:
- [docs/METHODOLOGY.md](docs/METHODOLOGY.md) – Technical method overview
- [docs/HYPERPARAMETERS.md](docs/HYPERPARAMETERS.md) – Hyperparameter justification
- [docs/DATASET.md](docs/DATASET.md) – Feature/response definitions