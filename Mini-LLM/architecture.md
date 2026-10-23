# Architecture

A decoder-only transformer: RMSNorm, RoPE (rotary position embeddings), causal attention,
SwiGLU feed-forward, and an embedding matrix shared with the output head. Everything
lives in `Cfg` (`PRESETS` in
[`config.py`](https://github.com/Mini-llm-lab/Mini-LLM/blob/main/src/mini_llm/config.py)).

## Sizes

| Setting                   | 34M        | 40M        | 70M        | 150M        |
| ------------------------- | ---------- | ---------- | ---------- | ----------- |
| Parameters                | 34,087,424 | 42,476,032 | 69,975,680 | 152,596,224 |
| `dim` (width)             | 512        | 512        | 640        | 768         |
| `layers` (depth)          | 8          | 8          | 12         | 18          |
| `heads` (attention heads) | 8          | 8          | 10         | 12          |
| `ffn` (feed-forward size) | 1408       | 1408       | 1728       | 2048        |
| Vocabulary (BPE)          | 16,384     | 32,768     | 16,384     | 32,768      |
| Context length            | 512 tokens | 512 tokens | 512 tokens | 512 tokens  |

The 40M preset is the 34M architecture on the 32k tokenizer lineage: the only
difference is the larger embedding, so both models have the same 25,698,816
non-embedding parameters.

A single public interface exists for both the training code and the deployed model:

- `model(x)` returns `(logits, loss)`.
- `model.c` is the configuration (`model.c.ctx` is the context length).
- `mini_llm.checkpoints.load_model(path, device)` returns `(model, cfg)`, and
  `load_tokenizer(path)` reads the `tok.json` next to the checkpoint.

Mini-Eval builds its evaluation backend on exactly these entry points; see
[Mini-Eval backend](../Mini-Eval/backend.md).

## Trained checkpoints

Trained checkpoints live in `models/`:

| Checkpoint                               | Steps | Tokens seen | Val loss |
| ---------------------------------------- | ----- | ----------- | -------- |
| `mini-llm-base-34M-12000steps.pt`        | 12000 | ~0.20B      | 3.618    |
| `mini-llm-base-70M-22000steps.pt`        | 22000 | ~1.44B      | 3.030    |
| `mini-llm-base-150M-34332steps.pt`       | 34332 | ~4.50B      | 2.941    |
| `mini-llm-assistant-34M.pt` (epoch 3)    | 5598  | SFT         | 3.194    |
| `mini-llm-assistant-70M.pt` (epoch 2/3)  | 3800  | SFT         | 2.718    |
| `mini-llm-assistant-150M.pt` (epoch 2/2) | 3880  | SFT         | 2.780    |

The 70M base was trained on the cloud (`cloud/run.sh`: `SIZE=70M`, `STEPS=22000`,
`PREP_MILLIONS=1500`, CUDA batch `32x4x512`, so `65536` tokens/step), and so was the
150M base (`SIZE=150M`, `STEPS=34332`, CUDA batch `64x4x512`, so `131072` tokens/step,
4.50B tokens over the 32k vocabulary). The 34M-12000steps model is a continued run
(`--init` from 6000 steps, `phase_start=6000`), so its `val.bin` differs: base val
losses are not directly comparable across data mixes (see
[Training](training.md#training-on-more-data)). The 32k-lineage rows also predict
among 32k tokens instead of 16k: val losses are not comparable across tokenizers
either — compare lineages with bits per byte (see
[Training a 40M or 150M model](training.md#training-a-40m-or-150m-model)).

## Conversation format

The prompt format is defined once, in
[`prompt.py`](https://github.com/Mini-llm-lab/Mini-LLM/blob/main/src/mini_llm/prompt.py),
and shared by `sft` and `chat`:

```
User: <question>\nAssistant:
```

Multi-turn history is never sent: these small models anchor on their own previous
answers and degrade when they see them, so `chat` (Mini-LLM and Mini-CLI alike)
answers each question on its own and keeps a display transcript only.

The format is exactly the training format. Changing it changes the behaviour of every
model already trained with it, so it is treated as fixed. Mini-CLI reuses the same
convention so that what you type in `mini chat` matches what the model saw in
training; see [Mini-CLI inference](../Mini-CLI/inference.md).

## See also

- [Checkpoints](checkpoints.md) — what each `.pt` file contains and how one is chosen.
- [Design notes](design-notes.md) — constraints to respect when changing the code.
