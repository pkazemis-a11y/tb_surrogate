# Surrogate-based Generative Optimisation of Diagrid Tall Buildings

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8-3.11](https://img.shields.io/badge/Python-3.8--3.11-blue.svg)](https://www.python.org/)

This repository contains the codebase accompanying the manuscript "Surrogate-based generative optimisation of diagrid tall buildings". It implements a multi-input, multi-output feed-forward neural network surrogate and a multi-objective optimisation workflow for early-stage design exploration of tall buildings with outer diagrids.

The package trains on the generated simulation dataset and uses the fitted surrogate to evaluate candidate designs without repeatedly running the full structural analysis workflow.

## Related Repository

The companion dataset repository is [Tall-buildings-with-outer-diagrids-design-exploration](https://github.com/pkazemis-a11y/Tall-buildings-with-outer-diagrids-design-exploration). It contains the database, dataset-oriented documentation, and supporting reference material used to build the surrogate model in this repository.

## What This Repository Provides

1. **Train** a MIMO feed-forward neural network on generated tall-building response data.
2. **Predict** structural, economic, and environmental response variables from building and ground-motion inputs.
3. **Optimise** candidate designs with NSGA-II using a configurable objective set.
4. **Persist** preprocessing and model artifacts for reproducible inference.

## Repository Structure

```
src/
├── cli/              # train-model and optimize-designs commands
├── core/             # Configuration, data loading, preprocessing
├── models/           # Neural network architecture and training
└── optimization/     # NSGA-II problem setup and solution ranking

examples/             # Demonstration scripts
docs/                 # Methodology, FAQ, and usage notes
data/                 # Training data
tests/                # Smoke tests for contracts
```

## How It Works

**Training:** K-fold cross-validation with fold-local preprocessing prevents data leakage while evaluating model stability. Once validated, the model is trained on all data and saved to `models/`.

**Optimisation:** NSGA-II searches over building design parameters while holding the selected ground-motion input fixed. Candidate designs are evaluated with the trained surrogate and ranked before export to CSV.

**Inputs to model:**
- 10 building features (geometry, proportions, structural parameters)
- 12 ground-motion parameters (seismic characteristics)

**Outputs (100 responses):** accelerations, displacements, stresses, torsion, moments, costs, embodied carbon, etc.

**Default optimization targets (8):** peak acceleration, maximum displacement, story drift, equivalent stress, torsion, reaction moments, and cost per area.

## Installation

Requires Python 3.8–3.11. Uses Poetry for dependency management.

```bash
poetry install
```

Run commands without activating a shell:

```bash
poetry run python examples/load_data.py
```

Or activate the environment first:

```bash
poetry shell
```

More setup detail is in [SETUP.md](SETUP.md).

## Training

Use the installable CLI:

```bash
poetry run train-model \
    --data-path data/database.csv \
    --epochs 200 \
    --batch-size 32 \
    --learning-rate 2e-5 \
    --output-dir models/
```

The training pipeline:

- loads the CSV once through `DataLoader`
- fits building and ground-motion preprocessors separately
- performs fold-local preprocessing during cross-validation to avoid leakage
- resets model weights between folds
- retrains a final model on the full dataset after validation

Artifacts written to `models/`:

- `mimo_fnn_model.pth`
- `building_feature_preprocessor.pkl`
- `gm_feature_preprocessor.pkl`
- `response_scaler.pkl`

## Optimization

Use the trained artifacts to run NSGA-II over the observed building-design space:

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

`--gm-strategy` controls how ground-motion inputs are provided during optimization:

- `mean`: use the column-wise mean ground-motion profile from the dataset
- `row`: reuse one observed ground-motion row via `--gm-row-index`

## Implementation Notes

- Configuration is centralized in `src/core/config.py`.
- Training and optimization share the same persisted preprocessing artifacts.
- Cross-validation fits preprocessors inside each fold instead of on the full dataset.
- Optimization uses the fitted preprocessing path rather than a separate inference shortcut.
- Examples stay close to the package API.

## Quick Verification

```bash
poetry run pytest
poetry run python examples/load_data.py
```

## Documentation Map

- [GETTING_STARTED.md](GETTING_STARTED.md): end-to-end usage
- [SETUP.md](SETUP.md): environment setup
- [docs/METHODOLOGY.md](docs/METHODOLOGY.md): modeling decisions
- [docs/DATASET.md](docs/DATASET.md): raw data vs training contract
- [docs/FAQ.md](docs/FAQ.md): implementation questions

## Manuscript Context

The associated manuscript is currently under review. Until the paper is formally published, please cite this repository as software and reference the manuscript title in related project material when needed.

## License

MIT. See [LICENSE](LICENSE).