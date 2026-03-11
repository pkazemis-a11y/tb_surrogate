# Surrogate Modeling Methodology

## Overview

The workflow in this repository predicts 91 structural, geometric, economic, and carbon response variables from 22 inputs describing building design and ground motion.

The goal here is not to distribute frozen artifacts. The goal is to show a disciplined pipeline: prepare the data carefully, train against the full response contract, validate the predictor, and then use it for design-space exploration.

**Model inputs:**
- 10 building design parameters (geometry, proportions, vertical distribution)
- 12 ground-motion characteristics (magnitude, distance, duration, etc.)

**Model outputs:**
- All 91 response variables (accelerations, displacements, stresses, geometry-derived responses, total costs, and embodied carbon)

## Architecture

The manuscript workflow used separate input branches that merge into shared
hidden layers:

```
Building (10) ─┐
               ├─→ Merge ─→ Shared Layers ─→ 91 Outputs
GM (12) ──────┘
```

**Design rationale:**
- Separate branches respect different feature scales and distributions.
- Branch-specific projections keep the input structure simple.
- Shared layers learn interactions between seismic input and building response.

## Training: All 91 Responses

In the manuscript workflow, the model is trained with k-fold cross-validation to
minimize MSE across all 91 outputs simultaneously. Each fold:

1. Fits preprocessing independently on training data only (prevents leakage)
2. Resets model weights (prevents carryover learning between folds)
3. Evaluates on held-out validation set using train-fitted scaler

This produces a fully-conditioned surrogate that can predict any subset of responses.

## Optimization: Selected Objectives

The response contract contains 91 outputs in total. Optimization often focuses
on a smaller subset of responses that are useful for early-stage design tradeoff
studies:

- `Overall_Max_Acc`
- `Max_Displacement`
- `Overall_Max_Drift`
- `Total_Max_Von_Mises_tot`
- `Overall_Max_Torsion`
- `Total_Max_Magnitude_R`
- `Total_Max_Magnitude_M`
- `TotalMass`

The notebook also explored cost and carbon objectives as candidate additions,
but the archived optimization implementation used the 8-objective set above.

## Training Method

Cross-validation is the main validation path in this workflow.

For each fold:

1. split the raw data into training and validation partitions
2. fit preprocessing on the training partition only
3. scale the targets from the training partition only
4. reset the predictor weights
5. train on the fold
6. evaluate on the held-out partition

This design avoids two common ML portfolio mistakes:

- preprocessing leakage from validation data into training statistics
- carrying learned weights from one fold into the next

After cross-validation, users can train a final model on the full preprocessed dataset and save it using their own deployment conventions.

## Preprocessing

The preprocessing stage does four jobs:
- handle missing or categorical values where needed
- scale building inputs and ground-motion inputs separately
- scale the response space independently from the inputs
- keep each fold isolated so validation data never leaks into training statistics

Building and ground-motion inputs are kept separate because they represent distinct domains with different statistical structure.

## Multi-Objective Optimization

NSGA-II searches the building-design feature space using a predictor that has already been trained and checked.

**Defaults:** Optimizes 8 structural and mass responses:
- Structural demands: acceleration, displacement, drift, stress, torsion, resultants
- Mass: total structural mass

**Flexibility:** Users can specify any subset of the 91 predicted responses once
their predictor returns the full response vector.

**Implementation:**
- Design bounds inferred from observed feature ranges
- Objectives validated against all 91 available responses
- Optimization is kept separate from any project-specific model implementation