# Data Pipeline and Training Contract

## Raw Dataset

The dataset in `data/database.csv` contains 7,000 simulations with building design parameters, ground-motion characteristics, and structural response measurements.

**Column breakdown:**
- 2 metadata columns (Model ID, Ground Motion ID)
- 10 building design features (X1–X10 geometric/structural parameters)
- 12 ground-motion parameters (magnitude, distance, duration, etc.)
- **91 response variables** (acceleration, displacement, stress, geometry-dependent quantities, total costs, embodied carbon)

## Training Contract

The learning task is to predict all **91 response variables** from the available inputs:

```
Input: 10 building features + 12 ground-motion parameters -> Output: 91 responses
```

That full prediction space is what makes later objective selection flexible.

## Optimization: Selected Response Targets

While the model predicts all 91 responses, optimization uses a subset of 8 key responses chosen in the archived notebook implementation:

- `Overall_Max_Acc` - Peak acceleration
- `Max_Displacement` - Maximum lateral displacement
- `Overall_Max_Drift` - Story drift ratio
- `Total_Max_Von_Mises_tot` - Equivalent stress
- `Overall_Max_Torsion` - Torsional response
- `Total_Max_Magnitude_R` - Reaction force magnitude
- `Total_Max_Magnitude_M` - Moment magnitude
- `TotalMass` - Total structural mass

Users can specify different objectives in their own optimization wiring.

## Why the Contract Is Narrower Than the Raw CSV

The raw CSV also contains metadata fields and other context columns that are not part of the learning contract. The workflow makes the problem explicit by separating:

- 10 building design inputs
- 12 ground-motion inputs
- 91 supervised response outputs

Optimization then works from that full prediction space but defaults to a smaller set of responses for design trade-off studies.

## Pipeline Summary

1. validate the CSV path and required columns
2. extract building features, ground-motion features, and response variables
3. preprocess the two input groups separately
4. scale the targets independently
5. train and validate the predictor
6. use that predictor in optimization