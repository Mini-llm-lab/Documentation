# Training

The pipeline has five stages, run in this order: prepare the data, pretrain, run the
format stage, SFT, chat. Long runs belong on a cloud GPU; see
[Cloud GPU](cloud-gpu.md). For a first local check (the classic lineage, no format
stage):

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
| `uv run midtrain [steps] --init checkpoint`              | Format stage: learn the `System:/User:/Assistant:` markup before SFT          | ~600 steps, `cloud/run.sh midtrain`     |
| `uv run sft [epochs] [base] [--resume]`                  | Supervised fine-tuning on question/answer pairs                               | Full run via `cloud/run.sh sft`         |
| `uv run chat [checkpoint]`                               | Chat with the assistant (Ctrl+C to quit)                                      |                                         |
| `uv run heldout [checkpoint] [--math N] [--stories N] [--dates N] [--format N] [--base]` | Greedy generation on held-out arithmetic, word problems, dated prompts and system-prompt obedience | Assistant by default, `--base` for a base or format-stage checkpoint |
| `uv run export [checkpoint] [--out dir]`                 | Exports a checkpoint to Hugging Face format                                   | [Export](export-hf.md)                  |

`heldout` is the check that says whether a recipe actually improved: it generates
greedy answers on problems drawn with a different seed than the SFT mix (bare
arithmetic and narrative word problems), on dated system prompts, and on the
instruction-style prompts of `system_prompts.VARIATIONS` plus the uncertainty pairs.
It prints accuracy per operation (`add`, `sub`, `mul`, `div`), the exact-match rate of
word problems and dates, and a style score that compares the shape of the answer
(one sentence, three bullets, five numbered items) with the shape of the reference.
Bullet and numbered answers must match the reference count exactly; a prose answer
passes when it really is prose and is not longer than the reference. It
builds its prompts the same way `chat` does, so it also verifies that the injected date
is read correctly.

Every training detail is given on the command line, never hidden in the code:

- `pretrain` takes the architecture (`--dim`, `--layers`, `--heads`, `--ffn`,
  `--vocab`, `--ctx` — the `--size` preset is only the fallback) and the schedule
  (`--lr-max`, `--lr-cont` (peak lr of a continued run), `--lr-min`, `--warmup`); it
  prints the full schedule (`lr: peak x | cont y | min z | warmup n`) before starting.
- Before the first step, `pretrain` prints the data accounting recorded by `prep`:
  `bytes/token`, and for each source its share of the training tokens and the number of
  passes the run makes over it. It also reports `bits/byte` next to every validation
  loss (loss is in nats/token, so it cannot be compared across tokenizers).
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
- **Data budget.** Cloud runs use 131,072 tokens/step (`64x2x1024` for 40M,
  `32x2x2048` for 150M): the 40M recipe runs 35,096 steps ≈ 4.6B tokens (~108
  tokens/param) over 4.6B prepared tokens (`PREP_MILLIONS=4600`), the 150M 1.1 recipe
  53,375 steps ≈ 7.0B tokens (~46 tokens/param, about one pass over the 7.3B tokens of
  `PREP_MILLIONS=7300`). The published 150M
  (`mini-llm-base-150M-34332steps.pt`) is the older, shorter 1.0-era run: 34,332 steps
  ≈ 4.5B tokens (~29 tokens/param), classic 32k tokenizer, ctx 512, no format stage.
  `prep` writes `data/meta.json`
  (`vocab`, tokenizer hash, train/val token counts); keep it next to the `.bin`
  files. `pretrain` aborts if the vocab, the tokenizer hash or the `.bin` token
  counts do not match — a 16k-tokenized `train.bin` next to a 32k `meta.json` would
  pass the older checks, so the lengths are compared too.
- **Run.** Everything goes through `cloud/run.sh` — see [Cloud GPU](cloud-gpu.md).
  After `train`, `cloud/run.sh midtrain` runs the format stage (see below) and `sft`
  picks its output automatically; `sft` does 2 epochs (the 70M overfit at epoch 3).
  `pack` archives
  `models/32k/tok.json`, every base checkpoint of that size (pretrained and format
  stage), the assistant and `data/meta.json` (milestones excluded, `.state.pt`
  excluded).
- **Comparing across tokenizers.** Val losses of the 16k and 32k lineages are **not
  comparable** (predicting among 32k tokens is a different task). Compare with bits
  per byte instead — see the `wikitext` column in [Mini-Eval](../Mini-Eval/protocol.md).

## Format stage

Between pretraining and SFT, a short stage teaches the chat markup itself. A base
trained on raw web text has seen `System:`, `User:` and `Assistant:` only as ordinary
words, so SFT would have to teach the markup *and* the answers in the same gradients —
and the assistant tends to keep copying the prompt. The base checkpoint answers
`Explain what a linked list is used for.` with
`Explain what a linked list is used for.\nAssistant: ...`; the format stage is what
turns that into an actual answer.

```
uv run midtrain 600 --init models/32k/150M/mini-llm-base-150M-53375steps.pt --device cuda
```

- 600 steps (`MID_STEPS`), lr 1e-4 → 1e-5 cosine (`MID_LR_MAX`/`MID_LR_MIN`), `--warmup`
  50 (capped at a twentieth of the run, so 30 here — the same rule as `pretrain`), the
  same 131,072 tokens/step as pretraining.
- Every step draws 70 % format batches, 30 % replay (`--replay`, `MID_REPLAY`). Format
  batches use the SFT supervised shape (`sft.format_examples`, prompt tokens masked
  with `-100`, only the answer is scored); replay batches are plain next-token windows
  from `data/train.bin`, so the base distribution does not decay while the markup is
  learnt. `train.bin` and `val.bin` are only read, never rewritten, and the run aborts
  if `data/meta.json` does not describe them (same guard as `pretrain`).
- The output is `mini-llm-base-<size>-<base_steps+600>steps.pt` — a base line with a
  higher step count, so `sft` picks it over the pretrained checkpoint on its own and
  the pretrained file stays untouched. `val.bin` loss and `bits/byte` are still printed,
  but the stage changes the loss on format text, not on web text.
- Go/no-go is the format bench on a base checkpoint:
  `uv run heldout <base> --base --format 1 --math 0 --dates 0 --stories 0`. Before the
  stage it scores ~0/17 on style and ~0/5 on uncertainty, because the model repeats the
  prompt instead of answering; a working stage moves both up.
- `midtrain` refuses an assistant checkpoint (the stage starts from a base) and a
  classic-lineage base: the 1.0 model never saw a system turn, so there is nothing to
  reinforce.

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
