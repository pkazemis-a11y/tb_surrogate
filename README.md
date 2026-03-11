# Surrogate-based Generative Optimisation of Diagrid Tall Buildings

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8-3.11](https://img.shields.io/badge/Python-3.8--3.11-blue.svg)](https://www.python.org/)

This repository contains the public code sample accompanying the manuscript "Surrogate-based generative optimisation of diagrid tall buildings". It packages the cleaned parts of the original exploratory notebook into a structured codebase: dataset contract, preprocessing, a reference surrogate model, and optimization utilities for early-stage design exploration of tall buildings with outer diagrids.

The repository includes the reference model code, but it intentionally does not ship trained weights or serialized preprocessing artifacts. Users can retrain, modify, or replace the model while keeping the documented 22-input / 100-response contract.

## Related Repository

The companion dataset repository is [Tall-buildings-with-outer-diagrids-design-exploration](https://github.com/pkazemis-a11y/Tall-buildings-with-outer-diagrids-design-exploration). It contains the database, dataset-oriented documentation, and supporting reference material used to build the surrogate model in this repository.

## What This Repository Provides

1. **Define** the 22-input / 100-response contract used by the surrogate workflow.
2. **Load** and validate the tall-building dataset through reusable data utilities.
3. **Train** a clean notebook-derived reference surrogate in memory.
4. **Optimise** candidate designs with generic NSGA-II utilities once a predictor is supplied.
5. **Document** the modeling assumptions and response targets used in the manuscript workflow.

## Repository Structure

```
src/
├── core/             # Configuration, data loading, preprocessing
├── models/           # Reference surrogate architecture and training utilities
└── optimization/     # NSGA-II problem setup and solution ranking

examples/             # Demonstration scripts
docs/                 # Methodology, FAQ, and usage notes
data/                 # Training data
tests/                # Smoke tests for contracts
```

## How It Works

**Training:** The repository includes a clean two-branch surrogate network and an in-memory trainer derived from the original notebook workflow.

**Optimisation:** The included NSGA-II utilities search over building design parameters and rank candidate designs once you provide a predictor callable.

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

## Public Scope

This repository includes:

- the 10 building-feature inputs
- the 12 ground-motion inputs
- the 100-response output contract
- reusable preprocessing utilities
- a reference PyTorch surrogate model and trainer
- generic NSGA-II optimization utilities

This repository does not include:

- trained model weights
- serialized preprocessing artifacts
- installable training or inference CLIs

## Training Example

Run the clean end-to-end example:

```bash
poetry run python examples/complete_pipeline.py
```

The example trains and evaluates the reference surrogate in memory and reports summary metrics without writing model files.

## Optimization

The `src.optimization` package remains usable, but it is now documented as a generic optimization layer. To use it, provide a predictor callable that maps candidate designs to the 100-response vector.

## Implementation Notes

- Configuration is centralized in `src/core/config.py`.
- Building and ground-motion preprocessing remain separate.
- The reference trainer uses fold-local preprocessing during cross-validation.
- The response contract still covers all 100 outputs used in the manuscript workflow.
- Optimization is exposed as reusable Python classes rather than a bundled inference CLI.
- Examples stay close to the package API.

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