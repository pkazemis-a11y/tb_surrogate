# Contributing

Contributions should improve correctness, clarity, maintainability, or reproducibility.

## Development setup

```bash
poetry install
poetry run pytest
```

## Before opening a pull request

Run:

```bash
poetry run black src scripts examples tests
poetry run isort src scripts examples tests
poetry run mypy src
poetry run pylint src
poetry run pytest
```

## Contribution priorities

- fix correctness bugs in training or optimization
- improve tests around data loading, persistence, and CLI behavior
- extend the target set only if training, persistence, docs, and optimization are all updated together
- keep public documentation aligned with the actual code contract

## Coding expectations

- use type hints on public functions
- prefer small, composable modules
- avoid parallel inference or preprocessing paths that bypass persisted artifacts
- preserve the separation between building and ground-motion preprocessing unless there is a clear reason to change it

## Documentation expectations

If you change the public contract, update the matching docs in the same pull request:

- `README.md`
- `GETTING_STARTED.md`
- `docs/METHODOLOGY.md`
- `docs/FAQ.md`

The repository is strongest when code and documentation make the same claims.