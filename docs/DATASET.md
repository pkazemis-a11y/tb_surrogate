# Data Pipeline and Training Contract

## Raw Dataset

The dataset in `data/database.csv` contains 7,000 simulations with building design parameters, ground-motion characteristics, and structural response measurements.

**Column breakdown:**
- 2 metadata columns (Model ID, Ground Motion ID)
- 10 building design features (X1–X10 geometric/structural parameters)
- 12 ground-motion parameters (magnitude, distance, duration, etc.)
- **100 response variables** (acceleration, displacement, stress, cost, etc.)

## Model Training Contract

The MIMO neural network is trained to predict all **100 response variables** from the inputs:

```
Input:  10 building features + 12 GM parameters  → Output: 100 responses
```

This comprehensive prediction task allows flexible post-hoc selection of optimization objectives.

## Optimization: Selected Response Targets

While the model predicts all 100 responses, optimization uses a subset of 8 key responses chosen for structural and economic importance:

- `Overall_Max_Acc` - Peak acceleration
- `Max_Displacement` - Maximum lateral displacement
- `Overall_Max_Drift` - Story drift ratio
- `Total_Max_Von_Mises_tot` - Equivalent stress
- `Overall_Max_Torsion` - Torsional response
- `Total_Max_Magnitude_R` - Reaction force magnitude
- `Total_Max_Magnitude_M` - Moment magnitude
- `Total costs/TGA` - Cost per gross floor area

Users can specify different objectives via the CLI `--objectives` flag.

## Why the Contract Is Narrower Than the Raw CSV

The raw CSV also contains metadata fields and other context columns that are not part of the model input-output contract. The package makes the learning problem explicit by separating:

- 10 building design inputs
- 12 ground-motion inputs
- 100 supervised response outputs

Optimization then works from that full prediction space but defaults to a smaller set of responses for design tradeoff studies.

## Pipeline Summary

1. `DataLoader` validates the CSV path and required columns.
2. The code extracts building features, ground-motion features, and all response variables.
3. Building and ground-motion features are preprocessed separately.
4. Targets are scaled independently.
5. The same persisted artifacts are reused at optimization time.