# Training

The pipeline has four steps, run in this order. Long runs belong on a cloud GPU; see
[Cloud GPU](cloud-gpu.md). For a first local check:

```
uv run prep
uv run pretrain 300
uv run sft 1
uv run chat
```

| Command                                                  | What it does                                                                  | Notes                                   |
| -------------------------------------------------------- | ----------------------------------------------------------------------------- | --------------------------------------- |
| `uv run prep [millions]`                                 | Trains the tokenizer (if missing) and tokenizes a FineWeb-Edu + Wikipedia mix | Full data via `cloud/run.sh prep`       |
| `uv run pretrain [steps] [--resume] [--init checkpoint]` | Pretraining: learn to predict the next token                                  | Full run via `cloud/run.sh train`       |
| `uv run sft [epochs] [base] [--resume]`                  | Supervised fine-tuning on question/answer pairs                               | Full run via `cloud/run.sh sft`         |
| `uv run chat [checkpoint]`                               | Chat with the assistant (Ctrl+C to quit)                                      |                                         |
| `uv run export [checkpoint] [--out dir]`                 | Exports a checkpoint to Hugging Face format                                   | [Export](export-hf.md)                  |

To check that everything works before a long run: `uv run pretrain 300`, then
`uv run sft 1`.

## Resuming a run

`pretrain` saves its full state (weights, optimizer, step) every 500 steps, and `sft`
does it after every epoch. If a run stops, continue it with the same arguments plus
`--resume`:

```
uv run pretrain 6000 --resume
uv run sft 3 --resume
```

The number of steps (or epochs) must match the original run, since the learning-rate
schedule depends on it. Starting a `pretrain` run whose output already exists is
refused, so a finished model is never overwritten by accident.

`--resume` from an old 34M `.state.pt` file is no longer compatible: the optimizer now
uses two parameter groups with different weight decay.

## Training on more data

A finished base model can be trained further on fresh data instead of starting over.
`prep` keeps the existing `tok.json` and rewrites `data/` with a new random sample of
the sources, then `--init` loads the weights of a finished model and runs more steps
on it:

```
uv run prep 250
uv run pretrain 12000 --init mini-llm-base-34M-6000steps.pt
uv run sft 3
```

`12000` is the final step count, so this adds 6000 steps. The continued phase uses its
own short warmup and a lower peak learning rate (3e-4), and the optimizer starts
fresh. The result is saved as `mini-llm-base-34M-12000steps.pt`, which `sft` and
`chat` then prefer over the 6000-step model.

Constraints:

- `--init` and `--resume` cannot be combined, and `--resume` works on a continued run
  like on any other.
- Because the data mix changes, `val.bin` changes too: validation losses from before
  and after are not directly comparable. This is why the 34M-12000steps val loss
  (3.618) is in its own lineage in the [checkpoints table](architecture.md#trained-checkpoints).
  Compare such runs with bits per byte instead — see
  [Mini-Eval](../Mini-Eval/protocol.md#suite-small_lm).

## Training a 150M model

Preset `150M` (`Cfg(dim=768, layers=18, heads=12, ffn=2048, vocab=32768)`,
152,596,224 parameters, embeddings shared): the 34M/70M presets are unchanged and
their checkpoints still load with the 16k tokenizer.

- **Tokenizer (32k).** `models/tok.json` at the repo root is the 32k BPE, trained by
  `prep` (`--vocab 32768`, 300k documents, same normalizer/pre-tokenizer/`<eos>` as
  16k). The old 16k lineage lives in `models/16k/` (`tok.json` + 34M/70M
  checkpoints); every checkpoint reads the `tok.json` next to it, so old models keep
  working.
- **Data budget.** 34,332 steps × 131,072 tokens/step ≈ 4.5B tokens (~29
  tokens/param, ≤1 pass over `train.bin`). `prep` writes `data/meta.json` (`vocab`,
  tokenizer hash, train/val token counts); keep it next to the `.bin` files.
  `pretrain` aborts if the vocab or tokenizer hash does not match.
- **Run.** Everything goes through `cloud/run.sh` — see [Cloud GPU](cloud-gpu.md).
  `sft` does 2 epochs (the 70M overfit at epoch 3). `pack` archives `models/tok.json`,
  the final 150M base + assistant and `data/meta.json` (milestones excluded,
  `.state.pt` excluded).
- **Comparing across tokenizers.** Val losses of the 16k and 32k lineages are **not
  comparable** (predicting among 32k tokens is a different task). Compare with bits
  per byte instead — see the `wikitext` column in [Mini-Eval](../Mini-Eval/protocol.md).

## See also

- [Data](data.md) — what `prep` tokenizes and what `sft` trains on.
- [Checkpoints](checkpoints.md) — where the outputs land and how `sft`/`chat` pick a
  base model.
- [Export to Hugging Face](export-hf.md) — the step after training.
