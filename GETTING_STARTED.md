# Getting Started

## Prerequisites

- Python 3.8 to 3.11
- Poetry

Install dependencies:

```bash
poetry install
```

## 5-Minute Path

### 1. Verify the environment

```bash
poetry run python -c "from src.core import DataLoader; print('ready')"
```

### 2. Inspect the dataset contract

```bash
poetry run python examples/load_data.py
```

### 3. Train the surrogate

```bash
poetry run train-model \
    --data-path data/database.csv \
    --epochs 200 \
    --batch-size 32 \
    --output-dir models/
```

Expected artifacts:

- `models/mimo_fnn_model.pth`
- `models/building_feature_preprocessor.pkl`
- `models/gm_feature_preprocessor.pkl`
- `models/response_scaler.pkl`

### 4. Optimize candidate designs

```bash
poetry run optimize-designs \
    --model models/mimo_fnn_model.pth \
    --building-prep models/building_feature_preprocessor.pkl \
    --gm-prep models/gm_feature_preprocessor.pkl \
    --response-scaler models/response_scaler.pkl \
    --data-path data/database.csv \
    --population 100 \
    --generations 10 \
    --objectives Overall_Max_Acc Max_Displacement "Total costs/TGA" \
    --gm-strategy mean \
    --output results/optimized_designs.csv
```

## What the Training Step Actually Does

1. Loads the raw CSV and checks the required columns.
2. Extracts 10 building features, 12 ground-motion features, and 100 response variables.
3. Fits preprocessors inside each validation fold rather than on the full dataset.
4. Resets the model before each fold so validation metrics are not contaminated by previous training.
5. Trains one final model on the full preprocessed dataset and saves deployable artifacts.

## What the Optimization Step Actually Does

1. Loads the trained model and all persisted preprocessing artifacts.
2. Builds a fixed ground-motion template from either the dataset mean or a selected row.
3. Searches over observed building-feature bounds using NSGA-II.
4. Predicts the selected objectives through the same preprocessing and response-scaling path used in training.
5. Ranks and exports candidate designs.

## Common Commands

Run wrappers directly if you do not want to use the Poetry scripts:

```bash
poetry run python scripts/train_model.py --help
poetry run python scripts/optimize_designs.py --help
```

Run the end-to-end example:

```bash
poetry run python examples/complete_pipeline.py
```

Run tests:

```bash
poetry run pytest
```

## Model Scope

**Training:** The MIMO neural network predicts all 100 response variables from the dataset.

**Optimization:** By default, NSGA-II minimizes 8 key structural and economic responses. You can specify different objectives:

```bash
poetry run optimize-designs \
  --objectives Overall_Max_Acc Max_Displacement "Total costs/TGA"
```

Each response in the dataset (all 100) is available as an optimization objective.

The codebase is self-contained: inputs load from CSV, outputs are saved as model files or result tables, and the CLI covers the main workflow.