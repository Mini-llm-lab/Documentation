# Mini-Eval

Evaluation of [Mini-LLM](../Mini-LLM/README.md) models against reference models
(Pythia, ...) with
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). The
repository is separate from the training repository on purpose: it only asks the
models for probabilities, and `lm-eval` computes the scores.

Source repository: [github.com/Mini-llm-lab/Mini-Eval](https://github.com/Mini-llm-lab/Mini-Eval).

**Server only.** Protocol v2 numbers are produced on the Linux GPU server (CUDA),
from a committed tree. The package refuses to import on macOS: there is no local
path to results, and `uv run pytest` also runs on the server.

## Pages

| Page                          | Content                                                          |
| ----------------------------- | ---------------------------------------------------------------- |
| [Protocol](protocol.md)       | The rules that make every number comparable (v2)                 |
| [Workflow](workflow.md)       | `list`, `fingerprint`, `check`, `run`, `report`; adding a model  |
| [Backend](backend.md)         | How native `.pt` models are evaluated without an export step     |
| [Results](results.md)         | Where runs are stored and what a report contains                 |

## Setup

On the server, requires [uv](https://docs.astral.sh/uv/).

```
uv sync --extra native   # reference-only installs can use plain `uv sync`
cp local.example.yaml local.yaml
```

Edit `local.yaml` so that `models_dir` points to the `models/` folder of the training
repository (or set `MINI_EVAL_MODELS_DIR`). This file stays on your machine and is
ignored by Git.

The native backend is an **optional** dependency (`[project.optional-dependencies]
native` in `pyproject.toml`) pointed at a checkout of the training repository:

```
[tool.uv.sources]
mini-llm = { path = "../Mini-LLM", editable = true }
```

That is convenient while developing. For reported numbers, pin an exact tag instead
(`mini-llm = { git = "<repo url>", tag = "..." }`) and run `uv lock`, so the code that
produces the probabilities is identified by a commit.

## Commands

```
uv run mini-eval list
uv run mini-eval fingerprint mini-llm-40M-35096steps
uv run mini-eval check mini-llm-40M-35096steps --device cuda
uv run mini-eval run smoke mini-llm-40M-35096steps pythia-70m-step128 --device cuda
uv run mini-eval run small_lm --all --device cuda --require-clean
uv run mini-eval report small_lm
uv run pytest
```

See [Workflow](workflow.md) for what each command does, including the regular
workflow rule: `run` always rebuilds `reports/<suite>.md` plus the two plots
(`<suite>_vs_tokens.png`, `<suite>_vs_params.png`) from the latest runs, so a
benchmark never ends without images — previous reports go to
`reports/history/<suite>/<date>/` and an interactive `reports/<suite>.html` is
written alongside. Use `--no-report` to skip it, and `report` alone
to rebuild from existing results.
