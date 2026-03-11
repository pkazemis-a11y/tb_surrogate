# Surrogate Modeling Methodology

## Overview

The underlying research workflow uses a multi-input multi-output surrogate to
predict 100 structural and economic response variables from 22 building design
and ground-motion inputs.

This public repository includes a cleaned reference implementation of that
workflow while keeping the focus on readable source code rather than distributing
trained artifacts.

**Model inputs:**
- 10 building design parameters (geometry, proportions, vertical distribution)
- 12 ground-motion characteristics (magnitude, distance, duration, etc.)

**Model outputs:**
- All 100 response variables (accelerations, displacements, stresses, costs, etc.)

## Architecture

The manuscript workflow used separate input branches that merge into shared
hidden layers:

```
Building (10) ─┐
               ├─→ Merge ─→ Shared Layers ─→ 100 Outputs
GM (12) ──────┘
```

**Design rationale:**
- Separate branches respect different feature scales and distributions.
- Branch-specific projections keep the input structure simple.
- Shared layers learn interactions between seismic input and building response.

## Training: All 100 Responses

In the manuscript workflow, the model is trained with k-fold cross-validation to
minimize MSE across all 100 outputs simultaneously. Each fold:

1. Fits preprocessing independently on training data only (prevents leakage)
2. Resets model weights (prevents carryover learning between folds)
3. Evaluates on held-out validation set using train-fitted scaler

This produces a fully-conditioned surrogate that can predict any subset of responses.

## Optimization: Selected Objectives

The response contract contains 100 outputs in total. Optimization often focuses
on a smaller subset of responses that are useful for early-stage design tradeoff
studies:

- `Overall_Max_Acc`
- `Max_Displacement`
- `Overall_Max_Drift`
- `Total_Max_Von_Mises_tot`
- `Overall_Max_Torsion`
- `Total_Max_Magnitude_R`
- `Total_Max_Magnitude_M`
- `Total costs/TGA`

## Training Method

Cross-validation is the primary validation path in the included reference
trainer.

For each fold:

1. split raw DataFrames into train and validation partitions
2. fit the building preprocessor on training building features only
3. fit the ground-motion preprocessor on training ground-motion features only
4. fit the response scaler on training targets only
5. reset the model to its initial weights
6. train and validate on the fold

This design avoids two common ML portfolio mistakes:

- preprocessing leakage from validation data into training statistics
- carrying learned weights from one fold into the next

After cross-validation, users can train a final model on the full preprocessed dataset and save it using their own deployment conventions.

## Preprocessing

`FeaturePreprocessor` handles:
- Missing value imputation
- Optional categorical encoding (e.g., mechanism type)
- Feature scaling via StandardScaler
- Fit/transform pattern with no leakage between CV folds

`ResponseScaler` handles:
- Independent response scaling (separate from input scaling)
- MaxAbsScaler for sign preservation
- Reversible for post-inference reporting

The code maintains separate preprocessors for building and GM inputs because they represent distinct domains with different statistical structures.

## Multi-Objective Optimization

NSGA-II searches the building-design feature space using a user-supplied predictor.

**Defaults:** Optimizes 8 structural and economic responses:
- Structural demands: acceleration, displacement, drift, stress, torsion, moments
- Economic: cost per floor area

**Flexibility:** Users can specify any subset of the 100 predicted responses once
their predictor returns the full response vector.

**Implementation:**
- Design bounds inferred from observed feature ranges
- Objectives validated against all 100 available responses
- Optimization is kept separate from any project-specific model implementation