# Evaluation protocol v2

These choices are fixed so that every number in this repository is comparable with
the others. Anything that changes a score changes the protocol version
(`PROTOCOL_VERSION` in `src/mini_eval/__init__.py`), and every model, references
included, is then evaluated again.

**What changed in v2:** runs are produced on the Linux GPU server only (CUDA), every
model runs at its native context length (the v1 cap that held SmolLM2-135M at 2048
tokens is removed, so it runs at 8192), and the suite batch size is 4.

## Fixed settings

| Setting   | Value                                               | Why                                                                                              |
| --------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Harness   | `lm-eval` 0.4.13, pinned in `uv.lock`               | Tasks and prompts change between versions                                                        |
| Few-shot  | 0                                                   | Small models gain little from examples, and published zero-shot numbers exist for the references |
| Precision | float32                                             | No rounding differences between models                                                           |
| Examples  | Full test sets for reported numbers (`limit: null`) | A limit is only for smoke tests                                                                  |
| Decoding  | Greedy, for the few generative uses                 | Deterministic                                                                                    |
| Device    | `cuda`, on the Linux GPU server; recorded in every run | The code refuses to run on macOS, so no local numbers can enter the results                    |
| Context   | Model-native (up to 8192)                           | v1 capped SmolLM2-135M at 2048 because the eval machine could not fit its native 8192           |
| Batch     | 4                                                   | Memory-safe at 8192 tokens; batch size must not change scores anyway (checked by `mini-eval check`) |

## Suite `small_lm`

The primary metric of each task is declared in `suites/small_lm.yaml` before any
result is looked at:

| Task           | Primary metric | Chance                           |
| -------------- | -------------- | -------------------------------- |
| hellaswag      | acc_norm       | 0.25                             |
| arc_easy       | acc_norm       | 0.25                             |
| arc_challenge  | acc_norm       | 0.25                             |
| piqa           | acc_norm       | 0.50                             |
| winogrande     | acc            | 0.50                             |
| openbookqa     | acc_norm       | 0.25                             |
| sciq           | acc_norm       | 0.25                             |
| lambada_openai | acc            | 0                                |
| wikitext       | bits_per_byte  | not applicable (lower is better) |

`bits_per_byte` does not depend on the tokenizer, so it is the only language-modeling
number that can be compared between models with different vocabularies. Perplexity
per token must never be compared across tokenizers. This matches the 16k vs 32k
lineage warning in [Mini-LLM training](../Mini-LLM/training.md#training-a-40m-or-150m-model).

## Models

- **References** are Hugging Face models pinned to a revision. Pythia publishes
  intermediate checkpoints (`step64`, `step128`, ...) at 2,097,152 tokens per step,
  which allows comparisons at an equal number of training tokens.
- **Native models** are identified by the SHA-256 of the checkpoint and of `tok.json`,
  stored in `models.yaml`. A run is refused if the files on disk do not match.
- Every model runs at its **native context length**; the value is recorded in the run
  metadata.
- Size is described by **non-embedding parameters** as well as total parameters,
  because embeddings are a very different share of the total from one model to
  another (about 73% for Pythia-70M).

## Reading results

- Every score comes with its standard error. A difference smaller than two standard
  errors is not a result.
- Two views are reported: score against tokens seen, and score against non-embedding
  parameters. Comparing a model trained on 0.2B tokens with a reference trained on
  300B tokens says more about the data than about the model.
- A report warns when the runs in a table differ in harness version, protocol version,
  few-shot count, example limit or device, or come from a modified working tree.

## Before trusting a number

1. **Reproduce a published result** with the Hugging Face path (for example Pythia-70M
   on HellaSwag) and compare it with the values in the model card or paper. A
   mismatch points to a configuration error.
2. Run `mini-eval check MODEL --device cuda` on every native model: batch size must not
   change scores, cuda must agree with cpu within tolerance, and an untrained model
   must stay near chance.
3. Run reported numbers with `--require-clean`, from a committed working tree, on the
   server (the `results/` and `reports/` outputs written by the runs themselves do not
   count).

## Known gaps

- **Contamination** between training data and test sets has not been measured yet. The
  plan is an n-gram overlap report stored next to the results.
- The assistant (chat) evaluation is not part of this protocol yet. The adapter
  implements generation and the training prompt format, but no generative suite is
  defined.
- Benchmark datasets are downloaded from the Hugging Face Hub at run time and are not
  pinned beyond the `lm-eval` task version.

## Maintaining this page

The master copy of this protocol lives in the source repository
(`Mini-Eval/Protocol.md`) so that a checkout of the evaluation repository alone is
self-contained. This page must be updated whenever that file changes.
