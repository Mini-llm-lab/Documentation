# Mini-CLI

Go CLI (`mini`) to download, manage and chat with Mini locally. No account, no API
key, no remote backend for inference.

Source repository: [github.com/GautierPicon/Mini-CLI](https://github.com/GautierPicon/Mini-CLI).

- `mini chat` — interactive conversation (Bubble Tea). Works offline once the model is
  installed.
- `mini run "prompt"` — single prompt, stdout = answer, stderr = diagnostics
  (scriptable).
- `mini models list | info | pull` — local registry, sha256 verification, atomic
  install.

Real inference is done by `python/mini_engine.py` (PyTorch + `mini_llm`), spawned as a
child process over JSON lines on stdio. The Go binary never reimplements the network
and never converts weights blindly — see [Inference](inference.md).

## Pages

| Page                        | Content                                                     |
| --------------------------- | ----------------------------------------------------------- |
| [Usage](usage.md)           | Commands, chat keys, configuration file, env variables      |
| [Inference](inference.md)   | The Go → Python sidecar, the JSONL protocol, offline rules  |
| [Models storage](models.md) | Installed model layout, `manifest.json`, `models pull`      |

## Install

Prerequisites:

- Go >= 1.22 (tested with 1.27.1).
- Python 3.11+ with the `Mini-LLM` venv for real inference:
  ```
  cd ../Mini-LLM
  uv sync
  ```
  Needs `torch`, `tokenizers`, `transformers`, `safetensors` (already in
  `Mini-LLM/pyproject.toml`).
- Model files: either native `models/<dir>/` (`*.pt` + `tok.json`), or an HF export
  (`config.json` + `model.safetensors` + `tokenizer.json`) — both produced by
  [Mini-LLM](../Mini-LLM/README.md).

```
cd Mini-CLI
go build -o mini ./cmd/mini
./mini --help
```

Set the engine + python for real inference (auto-detected when you run from Mini-CLI
next to Mini-LLM, no exports needed for the 70M test):

```bash
# auto: finds ../Mini-LLM/.venv/bin/python and ../Mini-LLM/src + python/mini_engine.py
# manual override only if auto-detect fails:
export MINI_ENGINE_PATH="$PWD/python/mini_engine.py"
export MINI_PYTHON="$PWD/../Mini-LLM/.venv/bin/python"
```

`mini chat --mock` / `mini run --mock` work without any model (deterministic fake
replies, dev only).

Try the 70M assistant (native `.pt`, no install needed):

```bash
./mini run --model ../Mini-LLM/models/16k "What is 2+2?" --max-tokens 32
./mini chat --model ../Mini-LLM/models/16k
```
