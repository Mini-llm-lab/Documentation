# Mini-Eval

Evaluation of [Mini-LLM](../Mini-LLM/README.md) models against reference models
(Pythia, ...) with
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). The
repository is separate from the training repository on purpose: it only asks the
models for probabilities, and `lm-eval` computes the scores.

Source repository: [github.com/GautierPicon/mini-eval](https://github.com/GautierPicon/mini-eval).

## Pages

| Page                          | Content                                                          |
| ----------------------------- | ---------------------------------------------------------------- |
| [Protocol](protocol.md)       | The rules that make every number comparable (v1)                 |
| [Workflow](workflow.md)       | `list`, `fingerprint`, `check`, `run`, `report`; adding a model  |
| [Backend](backend.md)         | How native `.pt` models are evaluated without an export step     |
| [Results](results.md)         | Where runs are stored and what a report contains                 |

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```
uv sync
cp local.example.yaml local.yaml
```

Edit `local.yaml` so that `models_dir` points to the `models/` folder of the training
repository (or set `MINI_EVAL_MODELS_DIR`). This file stays on your machine and is
ignored by Git.

The training repository is a dependency, declared in `pyproject.toml`:

```
[tool.uv.sources]
mini-llm = { path = "../mini-llm", editable = true }
```

That is convenient while developing. For reported numbers, pin an exact tag instead
(`mini-llm = { git = "<repo url>", tag = "..." }`) and run `uv lock`, so the code that
produces the probabilities is identified by a commit.

## Commands

```
uv run mini-eval list
uv run mini-eval fingerprint mini-llm-34M-12000steps
uv run mini-eval check mini-llm-34M-12000steps
uv run mini-eval run smoke mini-llm-34M-12000steps pythia-70m-step128
uv run mini-eval run small_lm --all --require-clean
uv run mini-eval report small_lm
uv run pytest
```

See [Workflow](workflow.md) for what each command does, including the regular
workflow rule: `run` always rebuilds `reports/<suite>.md` plus the two plots
(`<suite>_vs_tokens.png`, `<suite>_vs_params.png`) from the latest runs, so a
benchmark never ends without images. Use `--no-report` to skip it, and `report` alone
to rebuild from existing results.
