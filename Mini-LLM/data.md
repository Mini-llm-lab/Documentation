# Data

## Pretraining

Streamed and shuffled, 70/30 by **document** count, which lands near 69/31 **by token**
with the 32k tokenizer (a FineWeb-Edu document averages ~990 tokens, a Wikipedia
article ~1 050, so the two sources weigh about the same per document):

- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
  (`sample-10BT`): educational web pages written by humans.
- [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) (`20231101.en`):
  encyclopedia articles written by humans, for factual knowledge.

Documents shorter than `MIN_DOC_TOKENS` (200 tokens) are dropped before the split:
Wikipedia has a long tail of stubs (median ~340 tokens) that spend tokens without
teaching much.

There is deliberately no third source. At 40M parameters the model can hold roughly
10 MB of facts, while the sources are gigabytes: more encyclopedic text does not fit.
What fits is the distribution of English, the conversation format, and narrow skills
(arithmetic, lists, dates) — the first comes from these two sources, the rest from the
SFT mix below.

## Mixture accounting and a stable validation set

`prep` counts what it wrote, per source, and stores it in `data/meta.json`:
`train_tokens`, `train_bytes`, `sources[].docs/tokens/bytes`, `seed`,
`min_doc_tokens` and `complete` (false when a source ran dry before the target).
`pretrain` reads it back to print the token share and the number of passes over each
source, `bytes/token`, and `bits/byte` next to every validation loss.

`val.bin` is written **once**. `prep` records the hash of every validation document in
`data/val_docs.json`; later runs reuse `val.bin` untouched and keep those documents out
of `train.bin`. Validation losses therefore stay comparable across `prep` runs,
including the `--init` continuation below, as long as the vocabulary and the lineage do
not change. Moving `data/` aside or switching lineage starts a new validation set.

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
  1.1 also keeps multi-turn exchanges (up to four turns) as full conversations.
- [OpenAssistant oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2):
  volunteer-written conversations. Only English, top-ranked answers. 1.1 follows
  whole threads (up to four turns: question → top answer → follow-up → top answer).
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
  - narrative word problems (`story_pairs`): one-step buys, gifts, boxes and equal
    shares, plus two-step "picked a on Monday, b on Tuesday, gave c away", answered in
    the gsm8k shape — `18 + 7 = 25, then 25 - 4 = 21. The answer is 21.`. This is the
    bridge from `3472 + 512` to a sentence, so `gsm8k` is no longer the only source of
    that skill;
  - answers are plain text, never LaTeX — one notation to learn, not two;
  - mini-skills: word counting, sorting small lists, reversing words/digit strings.
  The synthetic maths is capped at `MATH_MAX_SHARE` (30 %) of the supervised SFT tokens,
  next to the ~45k human source pairs. Arithmetic saturates well before the cap; the
  tokens it frees up go to chat.
- Date examples (1.1 only): ~400 `What is today's date?` pairs whose answer is copied
  from the `Today is YYYY-MM-DD.` line of the system prompt, with dates spread over
  2024–2027, and ~90 % of the other system prompts carry such a line (`DATE_SHARE` in
  `sft.py`). The point is to read the date, not to memorise one — see
  [Architecture](architecture.md#conversation-format).
- `chat.py` and prompts never format numbers with separators (3,472 is off-format for
  the digit-split lineage): every example inside numbers is a plain digit run.

Examples that do not fit in the context window are dropped, and 500 examples are held
out for validation. SFT runs 2 epochs over this mix (the 70M run overfit at epoch 3).

The format stage (`midtrain`, before SFT) uses a chat-only slice of the same recipe:
dolly, No Robots, oasst2, alpaca (same 25 % cap), identity and small-talk, arithmetic,
system-prompt variations and dates — no gsm8k, no synthetic maths, no word problems. It
is not a second SFT: 600 steps where the prompt tokens are masked and 30 % of the
batches replay raw text from `data/train.bin`, so the markup lands without forgetting
how to write English. See [Training](training.md#format-stage).

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

`prep` writes `data/train.bin`, `data/val.bin`, `data/val_docs.json` (hashes of the
validation documents, read back on later runs) and `data/meta.json` (`vocab`, lineage,
tokenizer hash, `train_tokens`, `val_tokens`, `train_bytes`, the per-source counts
above, `seed`, `min_doc_tokens`, `complete`). Token ids are stored as `uint16`, so
`Cfg.vocab` must stay at or below 65,536. Changing `vocab` invalidates `tok.json` and
the data: delete them and run `prep` again. `pretrain` aborts if the vocab, the
tokenizer hash or the `.bin` token counts in `meta.json` do not match (a
16k-tokenized file next to a 32k `meta.json` passes the first two checks, so the
lengths are compared as well). A `meta.json` with `complete: false` means the run
stopped short and a source ran dry: the files are not the size the recipe expects.

## See also

- [Training](training.md) — the commands that use these sources.
- [Design notes](design-notes.md) — data constraints to keep in mind.
