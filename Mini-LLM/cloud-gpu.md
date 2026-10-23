# Cloud GPU

Real training runs happen on a cloud GPU — it is the only supported path. The scripts
in `cloud/` target a RunPod pod with a persistent volume on `/workspace`.

## Usage

Defaults for `cloud/run.sh`: the **150M v2 recipe** — `SIZE=150M`, `STEPS=53375`,
`PREP_MILLIONS=7300` (~7.0B tokens ≈ 46 tokens/param) on the `digit32k` lineage
(digit-split tokenizer, ctx 2048). Checkpoints land in `models/digit32k/150M/`.

Previous recipes stay reproducible by setting the variables: `SIZE=40M`, `STEPS=35096`,
`PREP_MILLIONS=4600` puts everything back in `models/32k/40M/` (classic tokenizer).

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

- `SIZE`, `STEPS` and `PREP_MILLIONS` at the top of `run.sh` are the single place to
  switch a run: `150M`/53,375 (v2, `digit32k`) and `40M`/35,096 (legacy, `32k`) are the
  two 32k-lineage recipes. The 150M v2 preset relies on `pretrain`'s ctx-aware batch
  defaults (micro-batches stay ≤ 65,536 tokens: 32×2×2048 = 131,072 tokens/step).
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
- `--resume` from an old 34M `.state.pt` produced by a previous cloud run is no longer
  compatible (the optimizer now uses two parameter groups with different weight
  decay).
- `smoke` is a 100-step check whose reported `tok/s` is the number to watch when
  choosing a GPU for the full run. With `TEACHER` set it also runs the distillation
  path and logs `ce` and `kd` separately.
- `sft` passes the base checkpoint explicitly
  (`models/<lineage>/<SIZE>/mini-llm-base-<SIZE>-<STEPS>steps.pt`): automatic selection
  ranks by step count, which can pick another size (70M, 150M).
- `eval` runs `mini-eval check` then the `small_lm` suite on protocol v2 (six models,
  CUDA, full test sets) from the sibling `../Mini-Eval` checkout (`MINI_EVAL_DIR`
  overrides the path). It needs `uv sync` in Mini-Eval and the new checkpoint's
  `sha256` filled in its `models.yaml` (`mini-eval fingerprint`); `mini-eval` itself
  refuses to run anywhere but Linux.
- `test` runs `uv run pytest` in this repository and, when the sibling `../Mini-Eval`
  checkout exists on Linux, its suite too. `export-hf` runs it first, then exports
  the base and assistant of the configured `SIZE`; `export` also checks native/HF
  logit parity on its own.
- `pack` archives the deliverable files: the lineage `tok.json` (e.g.
  `models/32k/tok.json` for 40M), the final base + assistant checkpoints and
  `data/meta.json`. Milestones and `.state.pt` files are excluded.

## See also

- [Training](training.md) — what each phase does.
- [Checkpoints](checkpoints.md) — which files are worth copying back.
