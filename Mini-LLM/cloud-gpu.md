# Cloud GPU

Real training runs happen on a cloud GPU — it is the only supported path. The scripts
in `cloud/` target a RunPod pod with a persistent volume on `/workspace`.

## Usage

Defaults for `cloud/run.sh`: the **150M 1.1 recipe** — `SIZE=150M`, `STEPS=53375`,
`PREP_MILLIONS=7300` (~7.0B tokens ≈ 46 tokens/param) on the `digit32k` lineage
(digit-split tokenizer, ctx 2048). Checkpoints land in `models/digit32k/150M/`.

Previous recipes stay reproducible with a single variable: `SIZE=40M cloud/run.sh train`
puts everything back in `models/32k/40M/` (classic tokenizer, built-in recipe). All
built-in recipes are listed in the notes below.

```
cloud/run.sh setup   # uv sync, CUDA check, then cloud/preflight.sh
cloud/run.sh prep    # 7300M tokens, digit32k tokenizer, background
cloud/run.sh teacher # download + convert the legacy 150M base teacher (legacy runs only, see note)
cloud/run.sh smoke   # 100-step check, note the tok/s
cloud/run.sh train   # 53,375 steps, background, adds --resume automatically
cloud/run.sh sft     # 2 epochs on the base run above
cloud/run.sh eval    # protocol v2 suite, on the pod GPU
cloud/run.sh test    # pytest, training repo + sibling Mini-Eval
cloud/run.sh export-hf # tests, then export base + assistant
cloud/run.sh status
cloud/run.sh pack
```

`setup` runs `cloud/preflight.sh`, a quick check that the environment and the vocab
and data guards are in place before `prep` is started.

## Notes

- `SIZE`, `STEPS` and `PREP_MILLIONS` are read from the environment; each `SIZE` has a
  built-in recipe (steps + token budget) used when you do not override them. Switch a
  run without editing the script:

  | `SIZE`  | Default `STEPS` | Default `PREP_MILLIONS` | Lineage    |
  | ------- | --------------- | ----------------------- | ---------- |
  | `150M`  | 53,375          | 7,300                   | `digit32k` |
  | `40M`   | 35,096          | 4,600                   | `32k`      |
  | `70M`   | 22,000          | 1,500                   | `16k`      |

  ```
  SIZE=40M cloud/run.sh train                      # 40M recipe verbatim
  SIZE=40M STEPS=40000 PREP_MILLIONS=5000 cloud/run.sh train   # custom recipe
  ```

  The 150M 1.1 preset relies on `pretrain`'s ctx-aware batch
  defaults (micro-batches stay ≤ 65,536 tokens: 32×2×2048 = 131,072 tokens/step).
- The learning-rate schedule is passed explicitly in the command, not hidden in the
  code: `train`/`smoke` run `pretrain` with `--lr-max --lr-cont --lr-min --warmup` and
  `sft` with `--lr-max --lr-min --warmup`. Defaults per `SIZE` (40M/150M:
  5e-4 / 3e-4 / 5e-5 / 1000; 70M: 6e-4 / 3e-4 / 6e-5 / 500; SFT: 2e-4 / 2e-5 / 50) can
  be overridden from the environment with `LR_MAX`, `LR_CONT`, `LR_MIN`, `LR_WARMUP`,
  `SFT_LR_MAX`, `SFT_LR_MIN`, `SFT_LR_WARMUP`. `pretrain` prints the full schedule
  before starting.
- The architecture is passed explicitly too: `train`/`smoke` also carry
  `--dim --layers --heads --ffn --vocab --ctx` (40M: 512 / 8 / 8 / 1408 / 32768 / 1024;
  70M: 640 / 12 / 10 / 1728 / 16384 / 512; 150M 1.1: 768 / 18 / 12 / 2048 / 32768 /
  2048). Override any field from the environment to train a custom model without
  touching the code:

  ```
  SIZE=40M DIM=640 LAYERS=12 HEADS=10 FFN=1728 cloud/run.sh train
  SIZE=40M CTX=2048 cloud/run.sh train                  # longer context, same shape
  SIZE=70M VOCAB=32768 cloud/run.sh prep                # new tokenizer + data to match
  ```

  The checkpoints keep the tag computed from the real parameter count
  (`mini-llm-base-<X>M-...`), so a custom model never collides with the recipe it
  started from. `pretrain` refuses `dim` not divisible by `heads`, a `--vocab`
  override without matching `data/` (`prep` regenerates it), and non-positive values.
- `prep` refuses to reuse a `data/` folder whose `meta.json` is not for the lineage
  the script targets; `PREP_FORCE=1` overwrites it. `pretrain` additionally refuses
  `.bin` files whose token counts do not match `meta.json`, with or without the flag.
- `teacher` downloads `base/` from `GautierP/MiniLLM-150M` **for legacy runs** — the
  legacy teacher cannot serve a `digit32k` student (`validate_teacher` blocks it).
  It checks the pinned sha256, converts it to `models/32k/150M/teacher-150M-base.pt`
  and verifies the logits against the HF model. `train` adds `--teacher` plus the KD
  arguments when `TEACHER` is set:
  `TEACHER=models/32k/150M/teacher-150M-base.pt cloud/run.sh train`.
  See [Training](training.md#distillation-from-the-150m-base).
- `train` adds `--resume` automatically, so the same command restarts a run that was
  interrupted. See [Training](training.md#resuming-a-run) for the rules `--resume`
  must follow.
- `--resume` from state files saved by older versions of the code (the 1.0-era cloud
  runs) is no longer compatible (the optimizer now uses two parameter groups with
  different weight decay).
- `setup` warns when no Hugging Face token is found (env `HF_TOKEN` / legacy
  `HUGGING_FACE_HUB_TOKEN` or an earlier `hf auth login`). Anonymous downloads are
  rate-limited and slow the paid hours down; `uv run hf auth login` once per pod is
  enough because `HF_HOME` points at the persistent `/workspace/.hf`. Get a Read token
  at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens); never
  commit one.
- `smoke` is a 100-step check whose reported `tok/s` is the number to watch when
  choosing a GPU for the full run. With `TEACHER` set it also runs the distillation
  path and logs `ce` and `kd` separately.
- `sft` passes the base checkpoint explicitly
  (`models/<lineage>/<SIZE>/mini-llm-base-<SIZE>-<STEPS>steps.pt`): automatic selection
  ranks by step count, which can pick another size (70M, 150M).
- `eval` runs `mini-eval check` then the `small_lm` suite on protocol v2 (six models
  by default, CUDA, full test sets) from the sibling `../Mini-Eval` checkout
  (`MINI_EVAL_DIR` overrides the path). It never runs automatically after training —
  it is a separate command. The evaluated models can be overridden with `EVAL_MODELS`
  (e.g. `EVAL_MODELS="mini-llm-40M-35096steps pythia-70m-step143000"`, `EVAL_CHECK`
  for the check target). It needs `uv sync --extra native` in Mini-Eval and the new
  checkpoint's `sha256` filled in its `models.yaml` (`mini-eval fingerprint`);
  `mini-eval` itself refuses to run anywhere but Linux.
- `test` runs `uv run pytest` in this repository and, when the sibling `../Mini-Eval`
  checkout exists on Linux (`uv sync --extra native` there first), its suite too.
  `export-hf` runs it first, then exports
  the base and assistant of the configured `SIZE`; `export` also checks native/HF
  logit parity on its own.
- `pack` archives the deliverable files: the lineage `tok.json` (e.g.
  `models/32k/tok.json` for 40M), the final base + assistant checkpoints and
  `data/meta.json`. Milestones and `.state.pt` files are excluded.

## See also

- [Training](training.md) — what each phase does.
- [Checkpoints](checkpoints.md) — which files are worth copying back.
