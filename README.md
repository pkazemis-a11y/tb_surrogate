# Surrogate-based Generative Optimisation of Diagrid Tall Buildings

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8-3.13](https://img.shields.io/badge/Python-3.8--3.13-blue.svg)](https://www.python.org/)

This repository presents a clean machine-learning pipeline for surrogate-based design exploration of tall buildings with outer diagrids. The focus is straightforward: import the data, preprocess it carefully, train a predictor, validate it properly, and only then use it for multi-objective optimization.

The repository includes working code for that pipeline, but it intentionally does not ship trained weights or frozen preprocessing artifacts. Anyone using the repo is expected to run those stages for themselves and adapt them to their own design questions.

## Related Repository

The companion dataset repository is [tb_database](https://github.com/pkazemis-a11y/tb_database). It contains the database, dataset-oriented documentation, and supporting reference material used to build the surrogate model in this repository.

## What This Repository Provides

1. **A clear learning contract** built around 22 inputs and 91 response variables.
2. **A reproducible data pipeline** for loading, checking, and preparing the dataset.
3. **A full training sequence** with preprocessing, validation, and performance tracking.
4. **A downstream optimization stage** that depends on a trained predictor rather than bypassing it.
5. **Documentation that explains the workflow as a pipeline**, not as a collection of disconnected scripts.

## Repository Structure

```
data/                 # training dataset
docs/                 # workflow explanation and technical notes
examples/             # runnable pipeline examples
src/                  # implementation of the workflow
tests/                # smoke tests
```

## How It Works

This repository is structured as a strict pipeline:

1. **Import data** and verify the expected feature and response columns.
2. **Preprocess data** by keeping building and ground-motion inputs separate, standardizing the inputs, and scaling the responses for stable optimization during training.
3. **Train and validate** the surrogate with fold-local preprocessing, monitored loss curves, and holdout checks.
4. **Re-fit on the full dataset** once the training configuration has been validated.
5. **Run optimization** only after the surrogate can produce reliable response predictions.

If preprocessing is skipped, training is not meaningful. If training is skipped, optimization is not meaningful. The workflow is intentionally sequential.

**Inputs to model:**
- 10 building features (geometry, proportions, structural parameters)
- 12 ground-motion parameters (seismic characteristics)

**Outputs (91 responses):** accelerations, displacements, stresses, torsion, geometry-derived quantities, total costs, and embodied carbon.

**Default optimization targets (8):** peak acceleration, maximum displacement, story drift, equivalent stress, torsion, reaction resultants, and total structural mass.

## Installation

Requires Python 3.8–3.13. Uses Poetry for dependency management.

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

## Public Scope

This repository includes:

- the 10 building-feature inputs
- the 12 ground-motion inputs
- the 91-response output contract used in the archived notebook workflow
- a complete preprocessing, training, validation, and optimization workflow
- loss tracking and visual diagnostics for training quality
- multi-objective optimization utilities for design exploration

This repository does not include:

- trained model weights
- serialized preprocessing artifacts
- installable training or inference CLIs

## Training Example

Run the clean end-to-end example:

```bash
poetry run python examples/complete_pipeline.py
```

The example walks through the full sequence of preprocessing, training, validation, and optimization without writing model files.

## Optimization

Optimization is treated as the final stage of the workflow, not as a standalone entry point. Candidate designs are evaluated only after the response-prediction stage is in place.

## Workflow Design Notes

- building and ground-motion inputs stay separate because they describe different physical processes
- response scaling stays explicit so training remains stable across many output targets
- cross-validation comes before final fitting so the workflow is not tuned to one lucky split
- optimization sits downstream of prediction, so it is treated as the last step rather than the starting point

## Quick Verification

```bash
poetry run pytest
poetry run python examples/load_data.py
poetry run python examples/complete_pipeline.py
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