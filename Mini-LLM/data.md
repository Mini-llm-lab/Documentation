# Data

## Pretraining

Streamed and shuffled, ~70/30 by document count:

- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
  (`sample-10BT`): educational web pages written by humans.
- [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) (`20231101.en`):
  encyclopedia articles written by humans, for factual knowledge.

## SFT

Sources (all English):

- [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k):
  questions and answers written by Databricks employees.
  - Without context: general instruction following.
  - **With context** (v2 only): the provided passage is part of the user turn, so the
    model learns to answer from a text it is given (close to open-book QA). The
    legacy recipe drops these; v2 keeps them.
- [No Robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots): 10,000
  instructions written by professional annotators, including chat-style prompts.
  v2 also keeps two-turn exchanges as full conversations.
- [OpenAssistant oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2):
  volunteer-written conversations. Only English, top-ranked answers. v2 keeps
  two-turn threads (question → top answer → follow-up → top answer).
- [alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned): **AI-written**
  (generated with an OpenAI model). It adds short, simple answers, but it is capped so
  that it never exceeds 25% of the mix (`AI_MAX_SHARE` in `sft.py`); human data always
  dominates.
- [gsm8k](https://huggingface.co/datasets/openai/gsm8k) (v2 only): grade-school word
  problems, filtered to at most 2 reasoning steps and 80 words, ~4k examples. The
  real-language bridge between the synthetic arithmetic and ordinary chat.
- [`identity.py`](https://github.com/Mini-llm-lab/Mini-LLM/blob/main/src/mini_llm/identity.py):
  identity, small-talk, limits, written for this project, repeated a few times.
- [`system_prompts.py`](https://github.com/Mini-llm-lab/Mini-LLM/blob/src/mini_llm/system_prompts.py)
  (v2 only): the default system prompt plus hand-written variations
  ("Answer in one sentence.", "Answer with a numbered list.", ...) whose answers match
  the instruction — this is what teaches system-prompt obedience and markdown output —
  and "I don't know" calibration examples (uncertainty, never refusals).
- [`math_data.py`](https://github.com/Mini-llm-lab/Mini-LLM/blob/main/src/mini_llm/math_data.py)
  (v2 only): synthetic arithmetic and token-level skills, deterministic by seed:
  - multiplication tables 1–12 and their inverse divisions, covered exhaustively;
  - additions/subtractions 1–5 digits (30% of the 4–5-digit ones echo one rounding
    step: `3472 + 512 = 3472 + 500 = 3972, then 3972 + 12 = 3984.`);
  - products: 2×2 digits, plus exhaustive 3-digit × 1-digit and sampled larger ones;
  - mini-skills: word counting, sorting small lists, reversing words/digit strings.
  ~50k examples per epoch next to the ~45k source pairs — roughly a third of SFT tokens.
- `chat.py` and prompts never format numbers with separators (3,472 is off-format for
  the digit-split lineage): every example inside numbers is a plain digit run.

Examples that do not fit in the context window are dropped, and 500 examples are held
out for validation.

## Conversation format

Defined in [`prompt.py`](https://github.com/Mini-llm-lab/Mini-LLM/blob/main/src/mini_llm/prompt.py),
versioned per tokenizer lineage (see
[Architecture](architecture.md#conversation-format)):

- legacy: `User: <question>\nAssistant:` — prompt masked, loss on the answer.
- v2 (`digit32k`): `System: ...\nUser: ...\nAssistant:` — the system header and the
  questions are masked; loss covers every assistant answer, in single-turn pairs and
  in two-turn conversations alike. Multi-turn sequences end each assistant turn with
  `<eos>`, exactly what generation emits.

## Tokenized files

`prep` writes `data/train.bin`, `data/val.bin` and `data/meta.json` (`vocab`,
tokenizer hash, train/val token counts). Token ids are stored as `uint16`, so
`Cfg.vocab` must stay at or below 65,536. Changing `vocab` invalidates `tok.json` and
the data: delete them and run `prep` again. `pretrain` aborts if the vocab, the
tokenizer hash or the `.bin` token counts in `meta.json` do not match (a
16k-tokenized file next to a 32k `meta.json` passes the first two checks, so the
lengths are compared as well).

## See also

- [Training](training.md) — the commands that use these sources.
- [Design notes](design-notes.md) — data constraints to keep in mind.
