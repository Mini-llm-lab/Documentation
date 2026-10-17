# Workflow

## Command reference

| Command                      | What it does                                                   |
| ---------------------------- | -------------------------------------------------------------- |
| `list`                       | Shows the registered models and suites                         |
| `fingerprint MODEL`          | Prints the SHA-256 values to copy into `models.yaml`           |
| `check MODEL [--device mps]` | Batch invariance, device agreement with cpu, untrained control |
| `run SUITE MODEL... [--all]` | Evaluates models, saves each run in `results/`, rebuilds report + PNGs |
| `report SUITE`               | Rebuilds `reports/<suite>.md` and the two plots from existing results |

Regular workflow: `run` always rebuilds `reports/<suite>.md` plus the two plots
(`<suite>_vs_tokens.png`, `<suite>_vs_params.png`) from the latest runs, so a
benchmark never ends without images. Use `--no-report` to skip it, and `report` alone
to rebuild from existing results.

Useful options for `run`:

| Option             | Effect                                                                   |
| ------------------ | ------------------------------------------------------------------------ |
| `--device`         | `cpu` \| `mps` \| `cuda`                                                 |
| `--limit N`        | Truncates the number of examples (smoke tests only)                      |
| `--allow-unpinned` | Allows models without a fingerprint (test runs only)                     |
| `--require-clean`  | Refuses to run unless the code tree is committed (`results/`/`reports/` outputs excluded) |

Typical sequence:

```
uv run mini-eval list
uv run mini-eval fingerprint mini-llm-34M-12000steps
uv run mini-eval check mini-llm-34M-12000steps
uv run mini-eval run smoke mini-llm-34M-12000steps pythia-70m-step128
uv run mini-eval run small_lm --all --require-clean
uv run mini-eval report small_lm
uv run pytest
```

## Adding a model

1. A **reference model**: add an entry with `kind: hf`, the Hub id and a `revision`,
   plus its parameter counts and the tokens it saw.
2. One of **your models**: copy the checkpoint into the training repository's
   `models/` folder (the `tok.json` next to it is used), add an entry with
   `kind: native` and `checkpoint: <file name>`, run
   `mini-eval fingerprint NAME`, and paste the two hashes into `models.yaml`.

## See also

- [Protocol](protocol.md) — the fixed settings these commands enforce.
- [Results](results.md) — what a run writes to disk.
- [Backend](backend.md) — how native checkpoints are evaluated.
