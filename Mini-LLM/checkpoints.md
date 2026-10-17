# Checkpoints

Every checkpoint stores its weights, its configuration and metadata (`step`,
`total_steps`, `val_loss`). Files in `models/`:

| File                                             | Content                                                                                       |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| `models/tok.json`                                | The tokenizer of the current lineage (32,768 tokens for 150M)                                 |
| `models/16k/tok.json`                            | The previous 16,384-token tokenizer, next to the 34M/70M checkpoints                           |
| `models/mini-llm-base-<size>-<steps>steps.pt`    | Base model, written once pretraining is complete (`34M-12000steps`, `70M-22000steps`)         |
| `models/mini-llm-base-<size>-<steps>steps.state.pt` | Resume state (includes optimizer), updated every 500 steps and deleted when the run completes |
| `models/mini-llm-assistant-<size>.pt`            | Best assistant so far, by validation loss (`assistant-34M`, `assistant-70M`)                  |
| `models/mini-llm-assistant-<size>.state.pt`      | Resume state of the SFT run, deleted when the run completes                                   |
| `data/train.bin`, `data/val.bin`, `data/meta.json` | Training data, already converted to tokens, plus vocab/tokenizer provenance                  |

## Automatic selection

How checkpoints are chosen when no path is given:

- `sft` picks the base model with the highest step count in its file name, then the
  most recent file. A quick 300-step test therefore never takes priority over a
  6000-step model. `.state.pt` files are never selected automatically.
- `chat` does the same among the assistant files.
- A path can always be passed explicitly, either absolute or relative to `models/`.

The tokenizer is always read from the folder that contains the checkpoint, so older
models can be archived in a subfolder together with their own `tok.json`
(`models/16k/` is exactly that: the previous lineage).

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
