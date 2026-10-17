# Cloud GPU

Real training runs happen on a cloud GPU — it is the only supported path. The scripts
in `cloud/` target a RunPod pod with a persistent volume on `/workspace`.

## Usage

Defaults for `cloud/run.sh`: the 150M preset, 34,332 steps, 4600M tokens.

```
cloud/run.sh setup
cloud/run.sh prep    # 4600M tokens, background
cloud/run.sh smoke   # 100-step check, note the tok/s
cloud/run.sh train   # 34,332 steps, background, adds --resume automatically
cloud/run.sh sft     # 2 epochs
cloud/run.sh status
cloud/run.sh pack
```

`setup` runs `cloud/preflight.sh`, a quick check that the environment and the vocab
guards are in place before `prep` is started.

## Notes

- `train` adds `--resume` automatically, so the same command restarts a run that was
  interrupted. See [Training](training.md#resuming-a-run) for the rules `--resume`
  must follow.
- `--resume` from an old 34M `.state.pt` produced by a previous cloud run is no longer
  compatible (the optimizer now uses two parameter groups with different weight
  decay).
- `smoke` is a 100-step check whose reported `tok/s` is the number to watch when
  choosing a GPU for the full run.
- `pack` archives the deliverable files: `models/tok.json`, the final base +
  assistant checkpoints and `data/meta.json`. Milestones and `.state.pt` files are
  excluded.

## See also

- [Training](training.md) — what each phase does.
- [Checkpoints](checkpoints.md) — which files are worth copying back.
