# Results

```
results/<model>/<suite>/<timestamp>/
    results.json    scores, standard errors, versions, sample counts
    meta.json       model entry, suite, device, versions, git state, fingerprints
    samples/        one line per question (ignored by Git)
```

Runs are never overwritten. `report` uses the latest run of each model and suite.

The generated report lives in `reports/<suite>.md`, next to two plots:
`<suite>_vs_tokens.png` (score against tokens seen) and `<suite>_vs_params.png`
(score against non-embedding parameters). See [Workflow](workflow.md) for when they
are rebuilt.

## Tests

`uv run pytest` runs the adapter against hand computations, inside the real `lm-eval`
harness on a tiny local task, and the registry, runner and report logic. These tests
use fake models and need no network. They run on the server too: the package refuses
to import on macOS, so there is no local test path either. `tests/test_native.py` is
skipped unless PyTorch and the training package are installed.
