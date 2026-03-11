# FAQ

## Why are building and ground-motion features separated?

They are distinct domains with different statistical properties. Separate branches and independent preprocessors keep the feature handling explicit and avoid mixing assumptions between geometry and seismic inputs.

## Does this repository ship a surrogate model?

Yes. It includes a cleaned reference implementation of the two-branch surrogate
network and the associated in-memory training workflow.

## Why are trained weights not included?

Because the goal of this public repo is to show the workflow and code quality,
not to distribute frozen experimental artifacts. Users can retrain the model on
`data/database.csv` or adapt the implementation to their own needs.

## Why can optimization objectives be customized?

Because the response contract contains 100 outputs, different design problems
might prioritize different objectives. Users can trade off acceleration vs. cost,
or focus on displacement only, depending on their design criteria.

## How should I use the optimization utilities?

Provide a predictor callable that returns the full response vector for a
population of candidate designs. You can build that callable around the included
reference trainer or around your own surrogate implementation. The
`src.optimization` classes handle objective selection, NSGA-II search, and
solution ranking.

## How is data leakage prevented?

Preprocessors and the response scaler are fit inside each cross-validation fold using training partitions only. Validation rows are transformed with statistics learned from the corresponding training split.

## Why keep building and ground-motion preprocessing separate?

That split comes directly from the notebook workflow. Building variables and
seismic variables have different semantics and statistical structure, so the repo
keeps them explicit rather than hiding everything inside one combined transform.

## Is Poetry required?

Yes for the documented workflow in this repository. The package metadata, dependencies, and development tooling are configured through `pyproject.toml`.

## What should I run first to verify the repo?

```bash
poetry install
poetry run pytest
poetry run python examples/load_data.py
poetry run python examples/complete_pipeline.py
```