# Surrogate-based Generative Optimisation of Diagrid Tall Buildings

[![DOI](https://img.shields.io/badge/DOI-10.XXXX%2Fjournal.XXXX-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Surrogate](https://img.shields.io/badge/Surrogate-MIMO--FNN-red)]()
[![Optimisation](https://img.shields.io/badge/Optimisation-NSGA--II-orange)]()

This repository implements a data-driven framework that integrates generative parametric modelling, high-fidelity finite element simulation and deep learning to enable rapid and accurate performance assessment of diagrid tall buildings under seismic loading. A dataset of 1,000 parametric building models — each analysed under seven ground motion records from the NGA-West2 database — was generated through automated architectural modelling in Grasshopper and structural simulation in OpenSeesPy, resulting in 7,000 labelled samples. A multi-input, multi-output feed-forward neural network (MIMO-FNN) serves as a surrogate model, predicting 91 response variables spanning structural demands, geometric properties, cost and embodied carbon. The trained surrogate is embedded in an NSGA-II multi-objective optimisation loop, generating Pareto-optimal building configurations that improve inter-story drift, material efficiency and environmental impact compared to baseline designs. The full methodology is described in the companion paper submitted to *Computer-Aided Civil and Infrastructure Engineering* (CACAIE).

## Companion Repository

The dataset generation, documentation and interactive visualisations are maintained separately:

| Repository | Content |
|---|---|
| **[tb_database](https://github.com/pkazemis-a11y/tb_database)** | 7,000-row parametric database, variable descriptions, interactive HTML plots |
| **tb_surrogate** (this repo) | Surrogate model, training pipeline, NSGA-II optimisation code |

## Workflow Overview

The proposed framework consists of four stages, as described in the paper:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  1. DATA           2. PREPROCESSING     3. SURROGATE        4. NSGA-II │
│                                                                         │
│  1000 buildings    StandardScaler       MIMO-FNN             Multi-     │
│  × 7 ground       (inputs)             2-branch             objective  │
│  motions           MaxAbsScaler        pyramidal             Pareto     │
│  = 7000 samples    (responses)          architecture          front     │
│                    LabelEncoder                                         │
│  10 building       (categorical)       512→256→128→91                   │
│  + 12 GM features                      outputs                         │
└─────────────────────────────────────────────────────────────────────────┘
```

1. **Data loading** — Import the 7,000-sample simulation database (1,000 buildings × 7 earthquakes) and validate the 22-input / 91-output contract.
2. **Preprocessing** — Standardise building and ground-motion inputs separately; scale responses with MaxAbsScaler; handle categorical and conditional features (X6–X10 masking).
3. **Surrogate training** — Train a pyramidal MIMO-FNN with k-fold cross-validation, fold-local preprocessing and monitored loss convergence.
4. **Generative optimisation** — Embed the trained surrogate in NSGA-II to explore the design space and generate Pareto-optimal building configurations beyond the original dataset.

## Model Architecture

The MIMO-FNN follows a pyramidal topology selected after comparing hourglass, bottleneck, residual and wide-layer alternatives (see paper, Section 4):

```
Building features (10) ──→ Dense(512) + ReLU ─┐
                                               ├─→ Concat(1024)
GM features (12) ─────────→ Dense(512) + ReLU ─┘
                                  │
                           Dense(512) + ReLU + Dropout(0.3)
                                  │
                           Dense(256) + ReLU + Dropout(0.3)
                                  │
                           Dense(128) + ReLU + Dropout(0.2)
                                  │
                           91 response outputs
```

**Inputs (22 features):**

| Group | Count | Examples |
|---|---|---|
| Building design parameters (X1–X10) | 10 | plan geometry, number of stories, floor height, tapering, twisting angle, curvilinearity |
| Ground motion descriptors | 12 | magnitude, mechanism, Rjb, Rrup, Vs30, Arias intensity, duration |

**Outputs (91 responses):**

| Category | Count | Examples |
|---|---|---|
| Structural demands | 69 | acceleration, displacement, inter-story drift, von Mises stress, torsion, base reactions |
| Geometry-dependent | 8 | total gross area, aspect ratio, façade area, diagrid angles, total mass |
| Cost | 2 | total cost, cost per gross floor area |
| Embodied carbon | 12 | EC for three steel types × two floor types, EC per GIA |

**Hyperparameters (Table 5 in paper):**

| Parameter | Value |
|---|---|
| Hidden layers | 512 → 256 → 128 |
| Activation | ReLU |
| Dropout | 0.3, 0.3, 0.2 |
| Batch size | 32 |
| Learning rate | 2 × 10⁻⁵ |
| Weight decay (L2) | 1 × 10⁻⁴ |
| Epochs | 100 |
| Optimiser | Adam |
| CV folds | 2 |

## Optimisation

NSGA-II searches the 10-dimensional building design space using the trained surrogate as a fast evaluator. Default objectives target eight structural and mass responses:

1. **Overall_Max_Acc** — peak floor acceleration
2. **Max_Displacement** — maximum lateral displacement
3. **Overall_Max_Drift** — maximum inter-story drift
4. **Total_Max_Von_Mises_tot** — peak equivalent stress
5. **Overall_Max_Torsion** — maximum torsional response
6. **Total_Max_Magnitude_R** — peak reaction resultant
7. **Total_Max_Magnitude_M** — peak moment resultant
8. **TotalMass** — total structural mass

Users can specify any subset of the 91 predicted responses as objectives for their own design criteria (e.g. cost, embodied carbon).

## Repository Structure

```
tb_surrogate/
├── data/
│   └── database.csv              # 7,000-row simulation database
├── src/
│   ├── core/
│   │   ├── config.py             # Feature/response definitions, hyperparameters
│   │   └── data.py               # DataLoader and preprocessing utilities
│   ├── models/
│   │   ├── network.py            # MIMO-FNN architecture (PyTorch)
│   │   └── trainer.py            # Cross-validation, training, prediction
│   ├── optimization/
│   │   └── nsga2.py              # NSGA-II problem definition and optimizer
│   └── visualization.py          # Training curve and metric plots
├── examples/
│   ├── load_data.py              # Quick data inspection
│   └── complete_pipeline.py      # Full 6-step pipeline
├── tests/
│   └── test_smoke.py             # Contract and shape verification
├── docs/
│   ├── DATASET.md                # Data pipeline and response contract
│   ├── METHODOLOGY.md            # Surrogate and optimisation methodology
│   ├── HYPERPARAMETERS.md        # Hyperparameter justification
│   └── FAQ.md                    # Common questions
├── pyproject.toml                # Poetry project and dependencies
├── CITATION.cff                  # Citation metadata
└── README.md
```

## Installation

Requires Python 3.8–3.13. Uses Poetry for dependency management.

```bash
poetry install
```

## Quick Start

```bash
# Verify the environment and data contract
poetry run python examples/load_data.py

# Run the full pipeline: data → CV → training → validation → NSGA-II
poetry run python examples/complete_pipeline.py

# Run smoke tests
poetry run pytest tests/
```

## Citation

If you use this code, please cite:

```
Kazemi P., Turrin M., Andriotis C., Entezami A., Mariani S., Ghisi A. (2025),
"Surrogate-based generative optimisation of diagrid tall buildings",
Computer-Aided Civil and Infrastructure Engineering.
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.