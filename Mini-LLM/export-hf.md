# Export to Hugging Face

Training checkpoints (`.pt`, pickle) stay as-is: they carry optimizer state for
`--resume`. `export` converts the finished weights to the deployable convention, with
zero logit difference (checked automatically, `max abs diff` must be `< 1e-5` before
and after a `from_pretrained` reload):

```
uv run export mini-llm-base-34M-12000steps.pt
uv run export mini-llm-assistant-34M.pt
uv run export mini-llm-base-70M-22000steps.pt
uv run export mini-llm-assistant-70M.pt
```

## Output

Each output directory is self-contained and Hub-ready:

| File                        | Content                                                              |
| --------------------------- | -------------------------------------------------------------------- |
| `config.json`               | Model configuration                                                  |
| `model.safetensors`         | Weights, tied `head` deduplicated                                    |
| `tokenizer.json` + `tokenizer_config.json` | Tokenizer (+ `chat_template.jinja` matching the `User:/Assistant:` training format) |
| `generation_config.json`    | Same sampling as `chat.py`                                           |
| `configuration_mini_llm.py` / `modeling_mini_llm.py` | For `trust_remote_code`                                 |

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("exports/mini-llm-assistant-34M-hf", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("exports/mini-llm-assistant-34M-hf", trust_remote_code=True)
```

## Upload

```
huggingface-cli upload <org>/<repo> exports/<name>-hf .
```

`exports/` and `*.safetensors` are ignored by Git. `uv run pytest` covers the
native/HF parity (`tests/test_hf_export.py`).

## See also

- [Design notes](design-notes.md) — `model.py` and `modeling_mini_llm.py` implement
  the same math and must change together.
- [Mini-CLI inference](../Mini-CLI/inference.md) — the exported directory is one of
  the two formats the CLI can load.
