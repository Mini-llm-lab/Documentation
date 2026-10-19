# Checkpoints

Every checkpoint stores its weights, its configuration and metadata (`step`,
`total_steps`, `val_loss`). Files in `models/`:

| File                                             | Content                                                                                       |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| `models/<lineage>/tok.json`                      | The lineage tokenizer: `32k` = 32,768 tokens (40M/150M), `16k` = 16,384 (34M/70M)             |
| `models/16k/mini-llm-base-<size>-<steps>steps.pt` or `models/32k/<size>/mini-llm-base-<size>-<steps>steps.pt` | Base model, written once pretraining is complete (`16k` 34M-12000steps / 70M-22000steps, `32k` 40M-35096steps / 150M-34332steps) |
| `models/16k/mini-llm-base-<size>-<steps>steps.state.pt` or `models/32k/<size>/mini-llm-base-<size>-<steps>steps.state.pt` | Resume state (includes optimizer), updated every 500 steps and deleted when the run completes |
| `models/16k/mini-llm-assistant-<size>.pt` or `models/32k/<size>/mini-llm-assistant-<size>.pt` | Best assistant so far, by validation loss (`16k` assistant-34M/70M, `32k` assistant-40M/150M) |
| `models/16k/mini-llm-assistant-<size>.state.pt` or `models/32k/<size>/mini-llm-assistant-<size>.state.pt` | Resume state of the SFT run, deleted when the run completes |
| `data/train.bin`, `data/val.bin`, `data/meta.json` | Training data, already converted to tokens, plus vocab/tokenizer provenance                  |

## Automatic selection

How checkpoints are chosen when no path is given:

- `sft` picks the base model with the highest step count in its file name, then the
  most recent file. A quick 300-step test therefore never takes priority over a
  6000-step model. `.state.pt` files are never selected automatically.
- `chat` does the same among the assistant files.
- A path can always be passed explicitly, either absolute or relative to `models/`.

The tokenizer is read next to the checkpoint, or in the lineage folder one level up:
each lineage keeps a single `tok.json` (`models/16k/` for 34M/70M, `models/32k/` for
40M/150M), and each size stores its checkpoints in its own subfolder
(`models/32k/40M/`, `models/32k/150M/`). `prep` derives the lineage folder from the
preset vocab (16,384 → `16k`, 32,768 → `32k`); `pretrain` adds the size tag.

## Compatibility rules

- Checkpoints store the configuration as a plain dictionary. Adding a field with a
  default value to `Cfg` keeps old checkpoints loadable; renaming or removing a field
  does not.
- Checkpoints are loaded with `weights_only=True`. Only legacy checkpoints that
  pickled the `Cfg` object itself fall back to unrestricted loading. Never load a
  `.pt` file from an untrusted source.

## See also

- [Training](training.md) — how these files are produced and resumed.
- [Export to Hugging Face](export-hf.md) — `.pt` in, deployable directory out.
- [Mini-Eval backend](../Mini-Eval/backend.md) — how a checkpoint is evaluated
  directly, with no export step.
