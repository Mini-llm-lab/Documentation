# Native backend

`src/mini_eval/backend.py` implements the `lm-eval` model interface on top of your
checkpoints: no export and no `config.json` are needed. Context and continuation are
encoded with the same rules as the Hugging Face backend of `lm-eval` (trailing spaces
move to the continuation, an empty context starts with the end-of-text token, long
inputs are truncated on the left, long texts use disjoint rolling windows), so scores
are computed the same way for references and for your models.

## What it relies on from the training repository

- `mini_llm.checkpoints.load_model(path, device)` returns `(model, cfg)`, and
  `load_tokenizer(path)` reads the `tok.json` next to the checkpoint.
- `model(x)` returns `(logits, loss)`, and `model.c` is the configuration with `ctx`.
- `tok.json` defines an `<eos>` token.

These entry points are also what the training code itself uses — see
[Mini-LLM architecture](../Mini-LLM/architecture.md#sizes). If one of them changes,
the native backend breaks: that is the coupling to watch.

`tests/test_native.py` checks these assumptions against the real training package when
it is installed.

## See also

- [Workflow](workflow.md) — which command uses this backend and when.
- [Mini-LLM checkpoints](../Mini-LLM/checkpoints.md) — the files being loaded.
