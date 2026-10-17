# Usage

```bash
# mock (no model needed)
./mini run --mock "Hello"
./mini chat --mock

# real local model (needs MINI_PYTHON + MINI_ENGINE_PATH)
./mini run --model ../Mini-LLM/models/16k "Hello"
./mini run --model ../Mini-LLM/exports/mini-llm-assistant-34M-hf "Hello" --max-tokens 64 --temperature 0.5

# models
./mini models list
./mini models info 16k
./mini models info ../Mini-LLM/exports/mini-llm-assistant-34M-hf
./mini models pull mini-llm-70m --registry https://example.com/models
```

Chat keys: `Enter` send, `/reset` clear history, `/quit` or `Ctrl+C` exit. History is
in-memory only, never written to disk.

Conversation format is exactly the training format (`User: ...\nAssistant: ...\n`, see
`../Mini-LLM/src/mini_llm/prompt.py` and
[Mini-LLM architecture](../Mini-LLM/architecture.md#conversation-format)). The Python
engine truncates oldest turns when the 512-token context is full.

## Configuration file

JSON at `ConfigFilePath()` (see `internal/config` in the source repository). Example:

```json
{
  "default_model": "mini-llm-70m",
  "models_dir": "/data/mini/models",
  "engine_path": "/opt/mini/mini_engine.py",
  "temperature": 0.5,
  "max_tokens": 256,
  "registry_url": "https://example.com/models"
}
```

## Paths

Default paths per platform:

| Platform | Models                                                    | Config                                             |
| -------- | --------------------------------------------------------- | -------------------------------------------------- |
| macOS    | `~/Library/Application Support/mini/models`               | `~/Library/Application Support/mini/config.json`   |
| Linux    | `~/.local/share/mini/models`                              | `~/.config/mini/config.json`                       |
| Windows  | `%APPDATA%/mini/models`                                   |                                                    |

## Environment variables

| Variable             | Effect                                      |
| -------------------- | ------------------------------------------- |
| `MINI_MODELS_DIR`    | Overrides the models directory              |
| `MINI_DEFAULT_MODEL` | Overrides the default model                 |
| `MINI_ENGINE_PATH`   | Path to `mini_engine.py`                    |
| `MINI_CONFIG`        | Overrides the config file path              |

`MINI_PYTHON` selects the Python interpreter that runs the engine (auto-detected,
see the [category overview](README.md#install)).

## See also

- [Inference](inference.md) — what happens when you run a prompt.
- [Models storage](models.md) — what `models list/info/pull` operate on.
