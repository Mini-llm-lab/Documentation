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
  - **With context** (1.1 only): the provided passage is part of the user turn, so the
    model learns to answer from a text it is given (close to open-book QA). The
    legacy recipe drops these; 1.1 keeps them.
- [No Robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots): 10,000
  instructions written by professional annotators, including chat-style prompts.
  1.1 also keeps two-turn exchanges as full conversations.
- [OpenAssistant oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2):
  volunteer-written conversations. Only English, top-ranked answers. 1.1 keeps
  two-turn threads (question → top answer → follow-up → top answer).
- [alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned): **AI-written**
  (generated with an OpenAI model). It adds short, simple answers, but it is capped so
  that it never exceeds 25% of the mix (`AI_MAX_SHARE` in `sft.py`); human data always
  dominates.
- [gsm8k](https://huggingface.co/datasets/openai/gsm8k) (1.1 only): grade-school word
  problems, filtered to at most 2 reasoning steps and 80 words, ~4k examples. The
  real-language bridge between the synthetic arithmetic and ordinary chat.
- [`identity.py`](https://github.com/Mini-llm-lab/Mini-LLM/blob/main/src/mini_llm/identity.py):
  identity, small-talk, limits, written for this project, repeated a few times.
- [`system_prompts.py`](https://github.com/Mini-llm-lab/Mini-LLM/blob/src/mini_llm/system_prompts.py)
  (1.1 only): the default system prompt plus hand-written variations
  ("Answer in one sentence.", "Answer with a numbered list.",
  "Answer with a markdown list of four items.", ...) whose answers match
  the instruction — this is what teaches system-prompt obedience and markdown output
  (lists only, on purpose — no headings or bold) —
  and "I don't know" calibration examples (uncertainty, never refusals).
  The default persona is also written three ways in `SYSTEM_PARAPHRASES`, and one of
  them replaces the canonical text on ~15 % of examples: the model learns what the
  system line *does* instead of memorising one exact string. The canonical wording
  still dominates, because it is the one `chat`, the export template and Mini-CLI send
  at inference.
- [`math_data.py`](https://github.com/Mini-llm-lab/Mini-LLM/blob/main/src/mini_llm/math_data.py)
  (1.1 only): synthetic arithmetic and token-level skills, deterministic by seed:
  - multiplication tables 1–12 and their inverse divisions, covered exhaustively but
    held to ~10 % of the math mix (`table_share`), so they stay a base and not a crutch;
  - additions/subtractions 1–6 digits, with ~10 % of subtractions landing on a
    negative result; 30 % of the 4–6-digit ones echo one rounding step:
    `3472 + 512 = 3472 + 500 = 3972, then 3972 + 12 = 3984.`;
  - products from 2×1 to 5×1 and 2×2 / 3×2 / 4×2 digits; half of the multi-digit ones
    split the second factor: `47 x 38 = 47 x 30 + 47 x 8 = 1410 + 376 = 1786.`;
  - divisions are always exact: divisor 2–99, quotient 2–4 digits;
  - answers are plain text, never LaTeX — one notation to learn, not two;
  - mini-skills: word counting, sorting small lists, reversing words/digit strings.
  75k math examples and 10k skills per epoch next to the ~45k source pairs: about
  45 % of SFT tokens, which buys arithmetic without drowning the chat data.
- Date examples (1.1 only): ~400 `What is today's date?` pairs whose answer is copied
  from the `Today is YYYY-MM-DD.` line of the system prompt, with dates spread over
  2024–2027, and ~90 % of the other system prompts carry such a line (`DATE_SHARE` in
  `sft.py`). The point is to read the date, not to memorise one — see
  [Architecture](architecture.md#conversation-format).
- `chat.py` and prompts never format numbers with separators (3,472 is off-format for
  the digit-split lineage): every example inside numbers is a plain digit run.

Examples that do not fit in the context window are dropped, and 500 examples are held
out for validation.

## Conversation format

Defined in [`prompt.py`](https://github.com/Mini-llm-lab/Mini-LLM/blob/main/src/mini_llm/prompt.py),
versioned per tokenizer lineage (see
[Architecture](architecture.md#conversation-format)):

- legacy: `User: <question>\nAssistant:` — prompt masked, loss on the answer.
- 1.1 (`digit32k`): `System: ...\nUser: ...\nAssistant:` — the system header and the
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
