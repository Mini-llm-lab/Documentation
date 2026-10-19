# Inference

## How it works

```
mini (Go: Cobra + Bubble Tea)
  └─ LocalGenerator (internal/inference/local.go)
       └─ python3 python/mini_engine.py  (JSONL stdin/stdout, logs on stderr)
            ├─ native .pt via mini_llm.checkpoints.load_model
            └─ HF dir via transformers AutoModelForCausalLM
```

- No TCP port, binds nowhere. The child is killed on `Close()`; no orphan process.
- Ready for streaming later: the protocol can add `{"token": ...}` messages.

The engine reuses the training code directly: native checkpoints load through
`mini_llm.checkpoints`, so the CLI never reimplements the network and never converts
weights blindly. The interface it relies on is the one described in
[Mini-LLM architecture](../Mini-LLM/architecture.md#sizes). HF directories are the
ones produced by [Mini-LLM export](../Mini-LLM/export-hf.md).

The conversation format is the training format, handled on the Python side; each
question is sent as a single turn and truncated to the last 511 tokens of the
512-token context. See
[Mini-LLM architecture](../Mini-LLM/architecture.md#conversation-format).

## Offline

Once the model directory is present, `mini chat` / `mini run` work offline (no
network). The first download obviously needs internet — see
[Models storage](models.md#models-pull).

## Limitations

- Real generation needs Python + torch (~300 MB for 70M fp32). `go build` alone does
  not bundle them.
- The mini-llm models hallucinate; 512-token context; no conversation memory (these
  models anchor on earlier turns and degrade).
- `pull` needs a trusted registry; none is hardcoded.
- CPU is fine; CUDA used automatically by the engine when available.

## Development

Go side:

```bash
go vet ./...
go test ./...
go build -o mini ./cmd/mini
```

Python side:

```bash
cd ../Mini-LLM
uv run pytest -q
echo '{"cmd":"load","model_dir":"models/16k"}...' | uv run python ../Mini-CLI/python/mini_engine.py
```

## Layout

```
cmd/mini/main.go
internal/cli/      Cobra commands (testable)
internal/chat/     Conversation + display transcript (uses Generator)
internal/inference/ Generator interface, Mock, Local (stdio sidecar)
internal/model/    Registry, manifest verify, pull
internal/config/   OS paths, JSON config, env overrides
internal/tui/      Bubble Tea chat UI + Lip Gloss styles
python/mini_engine.py  sidecar
```
