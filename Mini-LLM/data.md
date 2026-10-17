# Data

## Pretraining

Streamed and shuffled, ~70/30 by document count:

- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
  (`sample-10BT`): educational web pages written by humans.
- [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) (`20231101.en`):
  encyclopedia articles written by humans, for factual knowledge.

## SFT

Mostly written by humans:

- [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k):
  questions and answers written by Databricks employees (examples without extra
  context only).
- [No Robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots): 10,000
  instructions written by professional annotators, including chat-style prompts.
- [OpenAssistant oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2):
  volunteer-written conversations. Only English, first turns, and the top-ranked
  answer to each prompt are kept.
- [alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned): **AI-written**
  (generated with an OpenAI model). It adds short, simple answers, but it is capped so
  that it never exceeds 25% of the mix (`AI_MAX_SHARE` in `sft.py`); human data always
  dominates.
- [`identity.py`](https://github.com/Mini-llm-lab/Mini-LLM/blob/main/src/mini_llm/identity.py):
  examples written for this
  project (who the assistant is, greetings, thanks, how it handles "test", its
  limits) repeated three times, plus 200 generated arithmetic examples (sums,
  differences and times tables with correct answers). They are only added to the
  training split.

Examples that do not fit in the context window are dropped, and 500 examples are held
out for validation.

## Conversation format

```
User: <question>
Assistant: <answer>
```

The SFT loss is computed on the answer only, not on the question. The format is
defined in
[`prompt.py`](https://github.com/Mini-llm-lab/Mini-LLM/blob/main/src/mini_llm/prompt.py)
and is treated as fixed —
see [Architecture](architecture.md#conversation-format).

## Tokenized files

`prep` writes `data/train.bin`, `data/val.bin` and `data/meta.json` (`vocab`,
tokenizer hash, train/val token counts). Token ids are stored as `uint16`, so
`Cfg.vocab` must stay at or below 65,536. Changing `vocab` invalidates `tok.json` and
the data: delete them and run `prep` again. `pretrain` aborts if the vocab or
tokenizer hash in `meta.json` does not match.

## See also

- [Training](training.md) — the commands that use these sources.
- [Design notes](design-notes.md) — data constraints to keep in mind.
