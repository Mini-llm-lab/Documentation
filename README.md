# Mini documentation

Documentation for the **Mini** project: small language models (~40M and ~150M on the
32k tokenizer lineage, plus the earlier ~34M and ~70M) designed, trained and run from
scratch, with the tools built around them.

This repository is the central place for everything that spans several repositories or
that does not belong in a README. Each code repository keeps a short README with a
quickstart and points here for the details.

## The project

```
                       ┌────────────────────────────────────────────┐
                       │                 Mini-LLM                   │
                       │  training from scratch (PyTorch, CUDA)     │
                       │  tokenizer → data → pretrain → sft →       │
                       │  checkpoints (.pt) → HF export             │
                       └───────────────┬──────────────┬─────────────┘
                                       │              │
                    models (.pt, tok.json)      models + suite
                                       │              │
                       ┌───────────────▼───┐   ┌──────▼─────────────┐
                       │      Mini-CLI     │   │      Mini-Eval     │
                       │  local chat/run   │   │  scores against    │
                       │  (Go + Python)    │   │  reference models  │
                       └───────────────────┘   └────────────────────┘
```

- **[Mini-LLM](Mini-LLM/README.md)** — the training repository. Everything is implemented
  from scratch: BPE tokenizer, data preparation, the transformer decoder architecture,
  pretraining, supervised fine-tuning, chat, and export to Hugging Face format.
- **[Mini-Eval](Mini-Eval/README.md)** — the evaluation repository. Runs the models
  against reference models (Pythia, ...) with `lm-evaluation-harness` and produces
  comparable scores, reports and plots.
- **[Mini-CLI](Mini-CLI/README.md)** — the local CLI. `mini chat`, `mini run` and
  `mini models` talk to a model on your machine through a small Python engine. No
  account, no API key, no remote inference.

The repositories are siblings on disk:

```
Mini/
├── Mini-LLM/        (source repository: github.com/Mini-llm-lab/Mini-LLM)
├── Mini-Eval/       (source repository: github.com/Mini-llm-lab/Mini-Eval)
├── Mini-CLI/        (source repository: github.com/Mini-llm-lab/Mini-CLI)
└── Documentation/   (this repository)
```

## Start here

- **I want to chat with a model locally** → [Mini-CLI](Mini-CLI/README.md), then
  [usage](Mini-CLI/usage.md). Works with the checkpoints already in Mini-LLM.
- **I want to understand the models** → [Mini-LLM architecture](Mini-LLM/architecture.md).
- **I want to train / continue training** → [Mini-LLM training](Mini-LLM/training.md)
  and [cloud GPU](Mini-LLM/cloud-gpu.md).
- **I want to evaluate a model** → [Mini-Eval workflow](Mini-Eval/workflow.md) and the
  [protocol](Mini-Eval/protocol.md).

## Conventions

- Documentation is plain Markdown. It is meant to be served by
  [Starlight](https://starlight.astro.build/) later; keep the hierarchy simple.
- Links between categories are relative (`../Mini-LLM/...`), so the pages render both
  in a local checkout and in a future site build.
- The code repositories are the source of truth for their own code. When a page
  describes behaviour that lives in code, it points to the exact file. If you change
  the code and the page becomes wrong, fix the page in the same session.
- Code repositories keep only: role, prerequisites, quickstart, link here.

## Repository map

| Category                  | What it documents                                                    |
| ------------------------- | -------------------------------------------------------------------- |
| [Mini-LLM](Mini-LLM/)     | Model architecture, training pipeline, checkpoints, data, export      |
| [Mini-Eval](Mini-Eval/)   | Evaluation protocol, workflow, native backend, results                |
| [Mini-CLI](Mini-CLI/)     | CLI usage, inference engine, model storage, configuration file        |
