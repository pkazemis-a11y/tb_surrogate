# Surrogate-based Generative Optimisation of Diagrid Tall Buildings

[![DOI](https://img.shields.io/badge/DOI-10.XXXX%2Fjournal.XXXX-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Surrogate](https://img.shields.io/badge/Surrogate-MIMO--FNN-red)]()
[![Optimisation](https://img.shields.io/badge/Optimisation-NSGA--II-orange)]()

This repository provides a reproducible machine-learning pipeline for surrogate-based design exploration of diagrid tall buildings under seismic loading. It covers the complete workflow — from data loading and preprocessing, through training and cross-validation of a deep neural network surrogate, to multi-objective optimisation with NSGA-II — so that new building configurations can be evaluated in seconds rather than hours of finite element simulation. The codebase is built around a strict 22-input / 91-output data contract derived from 7,000 parametric simulations (1,000 buildings × 7 earthquakes) and is designed to be extended with custom objectives, alternative architectures or additional datasets. The full methodology is described in the companion paper submitted to *Computer-Aided Civil and Infrastructure Engineering* (CACAIE).

## Companion Repository

The surrogate model and optimisation code in this repository build on the parametric database maintained separately in
[tb_database](https://github.com/pkazemis-a11y/tb_database). That repository contains the 7,000-row simulation database, variable descriptions, interactive HTML visualisations and dataset-oriented documentation.

## 📁 Repository Structure

```
├── data/                          # Training dataset
│   └── database.csv              # 7,000-row simulation database
├── src/                           # Implementation of the workflow
│   ├── core/
│   │   ├── config.py             # Feature/response definitions, hyperparameters
│   │   └── data.py               # DataLoader and preprocessing utilities
│   ├── models/
│   │   ├── network.py            # MIMO-FNN architecture (PyTorch)
│   │   └── trainer.py            # Cross-validation, training, prediction
│   ├── optimization/
│   │   └── nsga2.py              # NSGA-II problem definition and optimizer
│   └── visualization.py          # Training curve and metric plots
├── examples/                      # Runnable pipeline examples
│   ├── load_data.py              # Quick data inspection
│   └── complete_pipeline.py      # Full 6-step pipeline
├── tests/                         # Smoke tests
│   └── test_smoke.py             # Contract and shape verification
├── docs/                          # Documentation and resources
│   ├── DATASET.md                # Data pipeline and response contract
│   ├── METHODOLOGY.md            # Surrogate and optimisation methodology
│   ├── HYPERPARAMETERS.md        # Hyperparameter justification
│   └── FAQ.md                    # Frequently asked questions
├── CITATION.cff                  # Citation information
├── CONTRIBUTING.md               # Contribution guidelines
├── CHANGELOG.md                  # Version history
├── pyproject.toml                # Poetry project and dependencies
├── LICENSE                       # Repository license
└── README.md                     # This file
```

## 🚀 Getting Started

New to this repository? Start here:

1. **[Getting Started Guide](GETTING_STARTED.md)** - Environment setup and first run
2. **[Example Scripts](examples/)** - Ready-to-run pipeline examples
3. **[Dataset Documentation](docs/DATASET.md)** - Data pipeline and response contract
4. **[Methodology](docs/METHODOLOGY.md)** - Surrogate and optimisation methodology
5. **[FAQ](docs/FAQ.md)** - Frequently asked questions

### Quick Start

```bash
# Clone the repository
git clone https://github.com/pkazemis-a11y/tb_surrogate.git
cd tb_surrogate

# Install dependencies (requires Poetry)
poetry install

# Verify the environment and data contract
poetry run python examples/load_data.py

# Run the full pipeline: data → CV → training → validation → NSGA-II
poetry run python examples/complete_pipeline.py

# Run smoke tests
poetry run pytest tests/
```

## 📄 Abstract

Predicting the performance of tall buildings at the early design stage is a challenging task due to the complex interplay among geometric variability, architectural transformations and dynamic structural behaviour. This study introduces a data-driven framework that integrates generative parametric modelling, high-fidelity simulation and deep learning to enable rapid and accurate assessment of the responses of tall buildings to moderate seismic loads across a broad spectrum of geometries. A dataset comprising 1,000 tall building models, each one subjected to multiple ground motion records, is generated through automated architectural and structural modelling. A multi-input, multi-output feed-forward neural network is constructed as a surrogate model to predict 91 performance indicators including structural demands, geometric properties and sustainability-related metrics. The surrogate is trained in a supervised fashion exploiting numerical simulations and is embedded in a multi-objective optimisation scheme, yielding solutions that significantly improve inter-story drift, material efficiency and embodied carbon compared to baseline designs.

## 🔬 Workflow Overview

The proposed framework consists of four stages, as described in the paper:

```
  ╔══════════════════╗     ╔══════════════════╗     ╔══════════════════╗     ╔══════════════════╗
  ║   1 · DATA       ║     ║ 2 · PREPROCESS   ║     ║  3 · SURROGATE   ║     ║   4 · NSGA-II    ║
  ╠══════════════════╣     ╠══════════════════╣     ╠══════════════════╣     ╠══════════════════╣
  ║                  ║     ║                  ║     ║                  ║     ║                  ║
  ║  1000 buildings  ║     ║  StandardScaler  ║     ║    MIMO-FNN      ║     ║  Multi-objective ║
  ║  × 7 earthquakes ║────▶║  (inputs)        ║────▶║    2-branch      ║────▶║  Pareto front    ║
  ║  = 7000 samples  ║     ║  MaxAbsScaler    ║     ║    pyramidal     ║     ║                  ║
  ║                  ║     ║  (responses)     ║     ║                  ║     ║  ~2000 optimal   ║
  ║  10 building     ║     ║  LabelEncoder    ║     ║  512 → 256 →     ║     ║  designs across  ║
  ║  + 12 GM inputs  ║     ║  (categorical)   ║     ║  128 → 91 out    ║     ║  competing goals ║
  ║                  ║     ║                  ║     ║                  ║     ║                  ║
  ╚══════════════════╝     ╚══════════════════╝     ╚══════════════════╝     ╚══════════════════╝
```

1. **Data loading** — Import the 7,000-sample simulation database (1,000 buildings × 7 earthquakes) and validate the 22-input / 91-output contract.
2. **Preprocessing** — Standardise building and ground-motion inputs separately; scale responses with MaxAbsScaler; handle categorical and conditional features (X6–X10 masking).
3. **Surrogate training** — Train a pyramidal MIMO-FNN with k-fold cross-validation, fold-local preprocessing and monitored loss convergence.
4. **Generative optimisation** — Embed the trained surrogate in NSGA-II to explore the design space and generate Pareto-optimal building configurations beyond the original dataset.

## 🧠 Model Architecture

The MIMO-FNN follows a pyramidal topology selected after comparing hourglass, bottleneck, residual and wide-layer alternatives (see paper, Section 4):

```
                    ┌───────────────────────┐
                    │  Building features    │
                    │  (10 variables)       │
                    └──────────┬────────────┘
                               │
                    ┌──────────▼────────────┐
                    │  Dense(512) + ReLU    │
                    └──────────┬────────────┘
                               │
                               ├──── Concatenate (1024) ────┐
                               │                            │
                    ┌──────────▼────────────┐    ┌──────────┴────────────┐
                    │  GM features          │    │  Dense(512) + ReLU    │
                    │  (12 variables)       │    └──────────┬────────────┘
                    └───────────────────────┘               │
                                                ┌──────────▼────────────┐
                                                │  Dense(512) + ReLU    │
                                                │  Dropout(0.3)         │
                                                └──────────┬────────────┘
                                                           │
                                                ┌──────────▼────────────┐
                                                │  Dense(256) + ReLU    │
                                                │  Dropout(0.3)         │
                                                └──────────┬────────────┘
                                                           │
                                                ┌──────────▼────────────┐
                                                │  Dense(128) + ReLU    │
                                                │  Dropout(0.2)         │
                                                └──────────┬────────────┘
                                                           │
                                          ┌────────────────┼────────────────┐
                                          │                │                │
                                ┌─────────▼──────┐ ┌──────▼───────┐ ┌──────▼───────┐
                                │  69 structural │ │  8 geometry  │ │  14 cost/EC  │
                                │  responses     │ │  responses   │ │  responses   │
                                └────────────────┘ └──────────────┘ └──────────────┘
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

## 🎯 Optimisation

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

## 👥 Authors

- **Pooyan Kazemi** (Corresponding author) - seyedpooyan.kazemi@polimi.it
- **Michela Turrin**
- **Charalampos Andriotis**
- **Alireza Entezami** - alireza.entezami@polimi.it
- **Stefano Mariani** - stefano.mariani@polimi.it
- **Aldo Ghisi** - aldo.ghisi@polimi.it

*Department of Civil and Environmental Engineering, Politecnico di Milano, Piazza Leonardo da Vinci 32, 20133 Milano, Italy*

## ⚖️ License

See [LICENSE](LICENSE) for details.

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{kazemi2025surrogate,
  title={Surrogate-based generative optimisation of diagrid tall buildings},
  author={Kazemi, Pooyan and Turrin, Michela and Andriotis, Charalampos and Entezami, Alireza and Mariani, Stefano and Ghisi, Aldo},
  journal={Computer-Aided Civil and Infrastructure Engineering},
  year={2025},
  doi={10.XXXX/journal.XXXX},
  publisher={Wiley},
  address={Department of Civil and Environmental Engineering, Politecnico di Milano, Milano, Italy}
}
```

**Paper DOI**: [10.XXXX/journal.XXXX]() *(Update with actual DOI upon publication)*

## 📧 Contact

For questions, issues, or collaboration inquiries, please:
- Open a GitHub issue
- Contact the corresponding author: Pooyan Kazemi (seyedpooyan.kazemi@polimi.it)

## 🙏 Acknowledgments

This research was conducted at the Department of Civil and Environmental Engineering, Politecnico di Milano, and the Architectural Engineering and Technology Department, Delft University of Technology. Finite element simulations were carried out on the Delft Blue supercomputing cluster. The ground motion data is sourced from the NGA-West2 database.

---

**Keywords**: Surrogate Modelling • Deep Learning • MIMO-FNN • NSGA-II • Multi-objective Optimisation • Diagrid Tall Buildings • Seismic Performance • Embodied Carbon • Early-stage Design