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
| `uv run pretrain [steps] [--resume] [--init checkpoint] [--teacher checkpoint]` | Pretraining: learn to predict the next token | Full run via `cloud/run.sh train` |
| `uv run sft [epochs] [base] [--resume]`                  | Supervised fine-tuning on question/answer pairs                               | Full run via `cloud/run.sh sft`         |
| `uv run chat [checkpoint]`                               | Chat with the assistant (Ctrl+C to quit)                                      |                                         |
| `uv run export [checkpoint] [--out dir]`                 | Exports a checkpoint to Hugging Face format                                   | [Export](export-hf.md)                  |

Every training detail is given on the command line, never hidden in the code:

- `pretrain` takes the architecture (`--dim`, `--layers`, `--heads`, `--ffn`,
  `--vocab`, `--ctx` — the `--size` preset is only the fallback) and the schedule
  (`--lr-max`, `--lr-cont` (peak lr of a continued run), `--lr-min`, `--warmup`); it
  prints the full schedule (`lr: peak x | cont y | min z | warmup n`) before starting.
- `sft` takes `--lr-max`, `--lr-min` and `--warmup`.
- The per-size values in `pretrain.py` (`HPARAMS` and the `PRESETS`) are only
  fallbacks; `cloud/run.sh` spells every value out explicitly in the commands it
  launches.

`prep` also accepts `--lineage digit32k` (or `--size 150M`, which implies it since 1.1):
that trains the digit-split tokenizer into `models/digit32k/tok.json` and marks
`data/meta.json` with the lineage. `pretrain` refuses data whose stored lineage differs.

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

`--resume` from state files saved by older versions of the code (the 1.0-era runs) is
no longer compatible: the optimizer now uses two parameter groups with different
weight decay.

## Training on more data

A finished base model can be trained further on fresh data instead of starting over.
`prep` keeps the lineage `tok.json` and rewrites `data/` with a new random sample of
the sources, then `--init` loads the weights of a finished model and runs more steps
on it:

```
uv run prep 250
uv run pretrain 12000 --init mini-llm-base-40M-30000steps.pt --lr-max 5e-4 --lr-cont 3e-4 --lr-min 5e-5 --warmup 1000
uv run sft 2
```

`12000` is the final step count, so this adds 6000 steps. The continued phase uses its
own short warmup and a lower peak learning rate (`--lr-cont 3e-4`, passed explicitly),
and the optimizer starts fresh. The result is saved as
`mini-llm-base-40M-12000steps.pt`, which `sft` and `chat` then prefer over the
30000-step model.

Constraints:

- `--init` and `--resume` cannot be combined, and `--resume` works on a continued run
  like on any other.
- Because the data mix changes, `val.bin` changes too: validation losses from before
  and after are not directly comparable. Change `prep` first, then `--init` from the
  checkpoint of the finished run.

## Training a 40M or 150M model

The 32k lineage has two presets (see the [sizes table](architecture.md#sizes)):

- **40M** — the 1.0-generation architecture (`dim=512, layers=8, heads=8, ffn=1408`,
  preserved from the 34M era, kept in the
  [archive section](architecture.md#generation-10)) with a 32,768-token vocabulary:
  42,476,032 parameters, 25,698,816 non-embedding. The 40M base is the direct
  successor of the 1.0 runs; new embedding parameters (32,768 × 512) are the only
  difference from the old 34M shape.
- **150M** — `dim=768, layers=18, heads=12, ffn=2048`: 152,596,224 parameters,
  embeddings shared.

The 1.0 presets (`34M`, `70M`) are unchanged and their checkpoints still load with the
16k tokenizer, but no new run should target them — new runs are 40M or 150M.

- **Tokenizer (32k).** `models/32k/tok.json` is the 32k BPE, trained by `prep`
  (`--vocab 32768`, 300k documents, same normalizer/pre-tokenizer/`<eos>` as 16k).
  Each lineage keeps a single `tok.json` (`models/16k/` for the 1.0 checkpoints,
  `models/32k/` for 40M/150M) and each size stores its checkpoints in its own subfolder
  (`models/32k/40M/`, `models/32k/150M/`), so old models keep working.
- **Data budget.** Cloud runs use 131,072 tokens/step (`64x4x512`): 40M for 35,096
  steps ≈ 4.6B tokens (~108 tokens/param), 150M for 34,332 steps ≈ 4.5B tokens
  (~29 tokens/param, ≤1 pass over `train.bin`). `prep` writes `data/meta.json`
  (`vocab`, tokenizer hash, train/val token counts); keep it next to the `.bin`
  files. `pretrain` aborts if the vocab, the tokenizer hash or the `.bin` token
  counts do not match — a 16k-tokenized `train.bin` next to a 32k `meta.json` would
  pass the older checks, so the lengths are compared too.
- **Run.** Everything goes through `cloud/run.sh` — see [Cloud GPU](cloud-gpu.md).
  `sft` does 2 epochs (the 70M overfit at epoch 3). `pack` archives
  `models/32k/tok.json`, the final base + assistant and `data/meta.json`
  (milestones excluded, `.state.pt` excluded).
- **Comparing across tokenizers.** Val losses of the 16k and 32k lineages are **not
  comparable** (predicting among 32k tokens is a different task). Compare with bits
  per byte instead — see the `wikitext` column in [Mini-Eval](../Mini-Eval/protocol.md).

## Distillation from the 150M base

Legacy lineages only. The teacher (`teacher-150M-base.pt`) was trained on the classic
32k tokenizer at ctx 512; `validate_teacher` requires the same tokenizer lineage and a
teacher ctx ≥ student ctx, so a 1.1 student (`digit32k`, ctx 2048) **cannot** distill
from it — the 1.1 150M, once trained, becomes the new teacher for a future 1.1 40M (the
student must then also move to `digit32k`).

For a legacy student: the teacher is downloaded from `GautierP/MiniLLM-150M` (subfolder `base/`), checked against a
pinned `sha256`, converted to the native format and verified against the HF logits:

```
uv run teacher
uv run pretrain 35096 --size 40M --teacher models/32k/150M/teacher-150M-base.pt
```

The teacher is frozen and the loss becomes `(1 - a) * CE + a * T² * KL`, with
`a = 0.5` (`--kd-weight`) and `T = 2.0` (`--kd-temp`). Both models share the 32k
tokenizer, so the logits align token for token; `kd_loss` chunks the 32k logits
instead of materializing them all at once.

- The student still starts from random weights: this is distillation, not `--init`.
  `--init` remains reserved for continuing the same configuration.
- The data and `val.bin` are the 150M ones, so the val loss is directly comparable
  to the 150M's 2.941 and to a 40M run without a teacher.
- A distillation step costs about twice a plain 40M step: expect ~6-7 h for the
  35,096-step run on the GPU that trained the 150M in 8 h.
- `--resume` needs the same `--teacher/--kd-weight/--kd-temp`; the settings are
  stored in the `.state.pt` and a mismatch is refused.
- Keep the converted teacher as `teacher-150M-base.pt`. The canonical
  `mini-llm-base-150M-34332steps.pt` has a pinned `sha256` in Mini-Eval, and
  re-saving it with the conversion changes its bytes. `pack` only archives
  `mini-llm-base-*` and `mini-llm-assistant-*`, so the teacher is not shipped.

## See also

- [Data](data.md) — what `prep` tokenizes and what `sft` trains on.
- [Checkpoints](checkpoints.md) — where the outputs land and how `sft`/`chat` pick a
  base model.
- [Export to Hugging Face](export-hf.md) — the step after training.
