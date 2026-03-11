# Surrogate Modeling Methodology

## Overview

A multi-input multi-output (MIMO) neural network learns to predict 100 structural and economic response variables from 22 building design and ground-motion inputs.

**Model inputs:**
- 10 building design parameters (geometry, proportions, vertical distribution)
- 12 ground-motion characteristics (magnitude, distance, duration, etc.)

**Model outputs:**
- All 100 response variables (accelerations, displacements, stresses, costs, etc.)

## Architecture

The network uses separate input branches that merge into shared hidden layers:

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

The model is trained with k-fold cross-validation to minimize MSE across all 100 outputs simultaneously. Each fold:

1. Fits preprocessing independently on training data only (prevents leakage)
2. Resets model weights (prevents carryover learning between folds)
3. Evaluates on held-out validation set using train-fitted scaler

This produces a fully-conditioned surrogate that can predict any subset of responses.

## Optimization: Selected Objectives

The network produces one scalar output per response variable, for 100 outputs in total. Optimization defaults to a smaller subset of responses that are useful for early-stage design tradeoff studies:

- `Overall_Max_Acc`
- `Max_Displacement`
- `Overall_Max_Drift`
- `Total_Max_Von_Mises_tot`
- `Overall_Max_Torsion`
- `Total_Max_Magnitude_R`
- `Total_Max_Magnitude_M`
- `Total costs/TGA`

## Training Method

Cross-validation is the primary validation path.

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

After cross-validation, a final model is trained on the full preprocessed dataset and saved for deployment.

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

NSGA-II searches the building-design feature space using the trained surrogate.

**Defaults:** Optimizes 8 structural and economic responses:
- Structural demands: acceleration, displacement, drift, stress, torsion, moments
- Economic: cost per floor area

**Flexibility:** Users can specify any subset of the 100 predicted responses via `--objectives`:

```bash
poetry run optimize-designs \
  --objectives Overall_Max_Acc Max_Displacement "Total costs/TGA"
```

**Implementation:**
- Design bounds inferred from observed feature ranges
- Objectives validated against all 100 available responses
- Optimization uses persisted preprocessors and scaler from training
- Ground-motion template fixed during search (mean profile or observed record)