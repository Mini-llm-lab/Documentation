# Design notes

Things to keep in mind when changing the code:

- Token ids are stored as `uint16` in the `.bin` files, so `Cfg.vocab` must stay at or
  below 65,536. Changing `vocab` **or switching tokenizer lineage** invalidates
  `tok.json` and the data: delete them and run `prep` again (for the v2 digit-split
  tokenizer, `prep --lineage digit32k` writes `models/digit32k/tok.json`; the legacy
  `models/32k/tok.json` is left untouched so the 40M keeps working). See
  [Data](data.md#tokenized-files).
- The prompt format is defined once, in
  [`prompt.py`](https://github.com/Mini-llm-lab/Mini-LLM/blob/main/src/mini_llm/prompt.py),
  and is **versioned by tokenizer lineage** (`Cfg.lineage`): legacy lineages train and
  infer with `User:/Assistant:`, the `digit32k` lineage always sends a System line. A
  checkpoint carries its own configuration, so presets can change without breaking old
  models; do not change a lineage's format silently. See
  [Architecture](architecture.md#conversation-format).
- Checkpoints store the configuration as a plain dictionary. Adding a field with a
  default value to `Cfg` keeps old checkpoints loadable; renaming or removing a field
  does not. See [Checkpoints](checkpoints.md#compatibility-rules).
- Checkpoints are loaded with `weights_only=True`. Only legacy checkpoints that
  pickled the `Cfg` object itself fall back to unrestricted loading. Never load a
  `.pt` file from an untrusted source.
- The distillation teacher is a derived file (`teacher-150M-base.pt`). Do not
  overwrite `mini-llm-base-150M-34332steps.pt` with a converted copy: Mini-Eval pins
  its `sha256`, and re-saving changes the bytes even when the weights are identical.
- `prep.py` ends with `os._exit(0)` on purpose: the Hugging Face streaming workers can
  keep the interpreter from exiting after the data is written. The files are already
  closed at that point.
- SFT masks everything that is not an answer (system header, questions) with labels
  `-100` and pads with the `<eos>` id, because the tokenizer only defines `<eos>` as a
  special token. Multi-turn conversations supervise every assistant turn and close
  each one with `<eos>`, matching what generation would emit.
- `*.pt`, `*.safetensors`, `*.bin`, `data/` and `exports/` are ignored by Git, and
  the lineage `tok.json` files (`models/16k/tok.json`, `models/32k/tok.json`) are
  tracked. Do not `git add -f` model files.
- `model.py` (training) and `modeling_mini_llm.py` (HF deploy) implement the same
  math; `tests/test_hf_export.py` checks logit parity. Change both together. The
  native RoPE buffers are non-persistent, the HF ones persistent (transformers'
  loader needs them in `model.safetensors`). See [Export](export-hf.md).

## See also

- [Architecture](architecture.md) — the model these constraints apply to.
- [Training](training.md) — the pipeline around them.
