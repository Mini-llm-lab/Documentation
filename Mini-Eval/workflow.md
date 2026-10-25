# Workflow

Protocol v2 runs on the Linux GPU server only: the package refuses to import on
macOS, so every command below — `pytest` included — runs on the server, with
`--device cuda` for anything that scores a model.

## Command reference

| Command                        | What it does                                                   |
| ------------------------------ | -------------------------------------------------------------- |
| `list`                         | Shows the registered models and suites                         |
| `fingerprint MODEL`            | Prints the SHA-256 values to copy into `models.yaml`           |
| `check MODEL [--device cuda]`  | Batch invariance, device agreement with cpu, untrained control |
| `run SUITE MODEL... [--all]`   | Evaluates models, saves each run in `results/`, rebuilds report + PNGs |
| `report SUITE`                 | Rebuilds `reports/<suite>.md` (+ plots, HTML) from existing results |

Regular workflow: `run` always rebuilds `reports/<suite>.md` plus the two plots
(`<suite>_vs_tokens.png`, `<suite>_vs_params.png`) from the latest runs, so a
benchmark never ends without images. The previous report and its plots are moved to
`reports/history/<suite>/<generation date>/` — no evaluation replaces an older one —
and a self-contained HTML version (`reports/<suite>.html`) is written with the same
charts plus tooltips (hover a point to see model, score and run date). Use
`--no-report` to skip it, and `report` alone to rebuild from existing results.

Useful options for `run`:

| Option             | Effect                                                                   |
| ------------------ | ------------------------------------------------------------------------ |
| `--device`         | `cuda` on the GPU server; `cpu` only for debugging (the device is recorded and the report flags mixed runs) |
| `--limit N`        | Truncates the number of examples (smoke tests only)                      |
| `--allow-unpinned` | Allows models without a fingerprint (test runs only)                     |
| `--require-clean`  | Refuses to run unless the code tree is committed (`results/`/`reports/` outputs excluded) |

Typical sequence:

```
uv run mini-eval list
uv run mini-eval fingerprint mini-llm-40M-35096steps
uv run mini-eval check mini-llm-40M-35096steps --device cuda
uv run mini-eval run smoke mini-llm-40M-35096steps pythia-70m-step128 --device cuda
uv run mini-eval run small_lm --all --device cuda --require-clean
uv run mini-eval report small_lm
uv run pytest
```

## Adding a model

1. A **reference model**: add an entry with `kind: hf`, the Hub id and a `revision`,
   plus its parameter counts and the tokens it saw. Models run at their native
   context; a `max_length: N` cap is still supported (it is passed to lm-eval and
   recorded in the run metadata) but protocol v2 does not use one.
2. One of **your models**: copy the checkpoint into the training repository's
   `models/` folder (the `tok.json` next to it is used), add an entry with
   `kind: native` and `checkpoint: <file name>`, run
   `mini-eval fingerprint NAME`, and paste the two hashes into `models.yaml`.

## See also

- [Protocol](protocol.md) — the fixed settings these commands enforce.
- [Results](results.md) — what a run writes to disk.
- [Backend](backend.md) — how native checkpoints are evaluated.
