# Export to Hugging Face

Training checkpoints (`.pt`, pickle) stay as-is: they carry optimizer state for
`--resume`. `export` converts the finished weights to the deployable convention, with
zero logit difference (checked automatically, `max abs diff` must be `< 1e-5` before
and after a `from_pretrained` reload):

```
uv run export models/32k/40M/mini-llm-base-40M-35096steps.pt --base
uv run export models/32k/40M/mini-llm-assistant-40M.pt
uv run export models/16k/mini-llm-base-34M-12000steps.pt --base
uv run export models/16k/mini-llm-assistant-34M.pt
uv run export models/16k/mini-llm-base-70M-22000steps.pt --base
uv run export models/16k/mini-llm-assistant-70M.pt
uv run export models/32k/150M/mini-llm-base-150M-34332steps.pt --base
uv run export models/32k/150M/mini-llm-assistant-150M.pt
```

`--base` is for base models: the tokenizer ships without the chat template and
`generation_config.json` keeps the model defaults instead of the assistant's sampling.
Output directories are `exports/<checkpoint stem>-hf` at the repository root
(`--out` overrides); `exports/` and `*.safetensors` are ignored by Git.

## Output

Each output directory is self-contained and Hub-ready:

| File                        | Content                                                              |
| --------------------------- | -------------------------------------------------------------------- |
| `config.json`               | Model configuration                                                  |
| `model.safetensors`         | Weights, tied `head` deduplicated                                    |
| `tokenizer.json` + `tokenizer_config.json` | Tokenizer (`chat_template.jinja` for assistants, matching the `User:/Assistant:` training format) |
| `generation_config.json`    | Assistant: same sampling as `chat.py`; base: model defaults          |
| `configuration_mini_llm.py` / `modeling_mini_llm.py` | For `trust_remote_code`                                 |

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("exports/mini-llm-assistant-40M-hf", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("exports/mini-llm-assistant-40M-hf", trust_remote_code=True)
```

## Upload

The 40M pair is published as two repositories, `GautierP/MiniLLM-40M-base` (base
model) and `GautierP/MiniLLM-40M` (assistant):

```
hf upload GautierP/MiniLLM-40M-base exports/mini-llm-base-40M-35096steps-hf .
hf upload GautierP/MiniLLM-40M      exports/mini-llm-assistant-40M-hf .
```

For the 150M and 70M models, the base weights live in the **same repository as the
assistant**, under `base/`. The assistant stays at the repository root, so existing
`from_pretrained("GautierP/MiniLLM-150M")` calls are unchanged:

```
hf upload GautierP/MiniLLM-150M exports/mini-llm-base-150M-34332steps-hf base
hf upload GautierP/MiniLLM-70M  exports/mini-llm-base-70M-22000steps-hf base
```

Load the base model with `subfolder="base"`:

```python
tok = AutoTokenizer.from_pretrained(repo, subfolder="base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo, subfolder="base", trust_remote_code=True)
```

Copy each model card (`hf/mini-llm-assistant-*/README.md`) into its export directory
before uploading, so the Hub renders the details and the cross-links. The 150M/70M
cards document the base weights in their "Base model" section; upload those cards to
the repository root alongside the assistant files. `uv run pytest` covers the
native/HF parity (`tests/test_hf_export.py`).

## See also

- [Design notes](design-notes.md) — `model.py` and `modeling_mini_llm.py` implement
  the same math and must change together.
- [Mini-CLI inference](../Mini-CLI/inference.md) — the exported directory is one of
  the two formats the CLI can load.
