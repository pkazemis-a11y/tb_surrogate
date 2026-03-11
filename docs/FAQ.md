# FAQ

## Why are building and ground-motion features separated?

They are distinct domains with different statistical properties. Separate branches and independent preprocessors keep the feature handling explicit and avoid mixing assumptions between geometry and seismic inputs.

## Does the model predict all 100 responses or just 8?

The model predicts **all 100 response variables**. Eight of these are selected as default optimization objectives because they were chosen based on structural engineering importance and feasibility. Users can optimize different objectives by specifying `--objectives` at runtime.

## Why can optimization objectives be customized?

Because the surrogate predicts all 100 responses, different design problems might prioritize different objectives. Users can trade off acceleration vs. cost, or focus on displacement only, depending on their design criteria.

## How is data leakage prevented?

Preprocessors and the response scaler are fit inside each cross-validation fold using training partitions only. Validation rows are transformed with statistics learned from the corresponding training split.

## Why reset the model between folds?

Without a reset, fold 2 would start from weights already trained on fold 1. That contaminates the validation process and makes the reported metrics optimistic.

## Why save two preprocessors?

The building and ground-motion inputs are persisted independently as:

- `building_feature_preprocessor.pkl`
- `gm_feature_preprocessor.pkl`

That mirrors the two-input model architecture and keeps deployment explicit.

## How does optimization handle ground-motion features?

Optimization keeps ground motion fixed while searching over building variables.

- `--gm-strategy mean` uses the dataset mean profile
- `--gm-strategy row --gm-row-index N` reuses one observed ground-motion row

This keeps the optimization problem well defined while still letting you test different seismic scenarios.

## Is Poetry required?

Yes for the documented workflow in this repository. The package metadata, development tooling, and CLI scripts are configured through `pyproject.toml`.

## What should I run first to verify the repo?

```bash
poetry install
poetry run pytest
poetry run python examples/load_data.py
```