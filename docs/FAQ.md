# FAQ

## Why are building and ground-motion features separated?

They describe different things and behave differently statistically. Keeping them separate makes preprocessing clearer and helps the training stage preserve that structure.

## Does this repository ship a predictor workflow?

Yes. It includes a working training pipeline that goes from prepared data to validated prediction and then to optimization.

## Why are trained weights not included?

Because the goal of this public repo is to show the workflow and code quality,
not to distribute frozen experimental artifacts. Users can retrain the model on
`data/database.csv` or adapt the implementation to their own needs.

## Why can optimization objectives be customized?

Because the response contract contains 91 outputs, different design problems
might prioritize different objectives. Users can trade off acceleration vs. cost,
or focus on displacement only, depending on their design criteria.

## How should I use the optimization stage?

Use it only after you have a predictor that can return the full response vector for candidate designs. The optimization stage assumes the prediction stage is already in place and trustworthy.

## How is data leakage prevented?

Preprocessors and the response scaler are fit inside each cross-validation fold using training partitions only. Validation rows are transformed with statistics learned from the corresponding training split.

## Why keep building and ground-motion preprocessing separate?

That split reflects the modeling workflow. Building variables and
seismic variables have different semantics and statistical structure, so the repo
keeps them explicit rather than hiding everything inside one combined transform.

## Is Poetry required?

Yes for the documented workflow in this repository. Dependencies and development tooling are configured through `pyproject.toml`.

## What should I run first to verify the repo?

```bash
poetry install
poetry run pytest
poetry run python examples/load_data.py
poetry run python examples/complete_pipeline.py
```