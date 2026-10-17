# Models storage

Each installed model is a directory:

```
$MINI_MODELS_DIR/mini-llm-70m/
  manifest.json
  model.safetensors / *.pt
  config.json / tok.json
  tokenizer.json
```

`manifest.json` (when present) lists `files[{path, sha256, size}]`.

## `models list` and `models info`

`models list` shows `id, format (hf/native-pt/unknown), size, verified|reason`.
`models info` works on an installed model id or on a path (for example
`../Mini-LLM/exports/mini-llm-assistant-34M-hf`).

## `models pull`

`models pull` requires a registry URL, downloads to `*.dl-*` tmp files, verifies
sha256 + sizes, then does an atomic `rename`. It refuses to overwrite an existing
model and refuses `..` / absolute paths in manifests. No official download URL is
bundled — `pull` errors clearly until `registry_url` is configured (see
[Usage](usage.md#configuration-file)).

## See also

- [Inference](inference.md) — how these files are loaded at run time.
- [Mini-LLM checkpoints](../Mini-LLM/checkpoints.md) — the native layout these
  directories mirror.
