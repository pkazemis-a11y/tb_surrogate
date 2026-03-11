# Setup Guide

## Supported Setup

This project is configured for Poetry and currently targets Python 3.8 to 3.11.
The public repository includes the cleaned surrogate-model code, preprocessing
helpers, and optimization utilities. Trained weights are intentionally not
bundled here.

```bash
poetry install
```

If you want an interactive shell:

```bash
poetry shell
```

If you prefer one-off commands:

```bash
poetry run python examples/load_data.py
```

## Verify the Installation

```bash
poetry run python -c "from src.core import DataLoader; print('installation successful')"
poetry run pytest
```

## Useful Commands

```bash
poetry run black src scripts examples tests
poetry run isort src scripts examples tests
poetry run mypy src
poetry run pylint src
```

## Troubleshooting

### Poetry is not on PATH

Install Poetry from https://python-poetry.org/docs/ and restart the shell.

### Imports fail inside the editor

Make sure VS Code is using the Poetry environment created for this workspace.

### CLI commands are not found

This public snapshot focuses on Python modules and examples rather than
installable training or inference CLIs. Use the Python examples instead:

```bash
poetry run python examples/load_data.py
poetry run python examples/complete_pipeline.py
```

### Tests fail because dependencies are missing

Run `poetry install` first. The source tree depends on third-party packages such as `torch`, `pandas`, `scikit-learn`, `pymoo`, and `joblib`.