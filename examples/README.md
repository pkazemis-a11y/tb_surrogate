# Examples

Run examples from the repository root.

```bash
poetry run python examples/load_data.py
poetry run python examples/complete_pipeline.py
```

## load_data.py

Purpose:

- verify the dataset loads correctly
- inspect the 10 building features, 12 ground-motion features, and 100 response variables
- show the data contract used by the package

## complete_pipeline.py

Purpose:

- demonstrate preprocessing for both input domains
- train through the same model and trainer abstractions used by the CLI
- save the fitted model and preprocessing artifacts

## Notes

- Examples are intentionally small and code-focused.
- They are not a second implementation path; they mirror the package API.
- If an example breaks after a code change, that is a signal the public API changed and the docs should be updated.

Related docs:

- [GETTING_STARTED.md](../GETTING_STARTED.md)
- [METHODOLOGY.md](../docs/METHODOLOGY.md)