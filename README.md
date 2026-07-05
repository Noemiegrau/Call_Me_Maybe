*This project has been created as part of the 42 curriculum by nograu.*

# Call Me Maybe

## Description

The goal is to translate plain language into computer code. Given a prompt like *"What is the sum of 2 and 3?"*, the program must not answer *5* — it must figure out which function to call and with what arguments, and output that as structured JSON:

```json
{"name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

This is what powers real AI assistants: the model understands the request in plain language, and the program turns it into a precise function call that your code can actually run.

The hard part is the model: Qwen3-0.6B has only 600 million parameters and often produces broken JSON when left to generate freely — valid output only about 30% of the time. The fix is **constrained decoding**: at each generation step, before the model picks its next token (text fragment), we check every candidate in the vocabulary and discard any that would make the output invalid. The model only ever gets to choose among tokens that keep the JSON correct. The result is 100% valid, parseable output — without the model needing to "know" how to format JSON at all.

## Instructions

### 1. Setup (once)

```bash
make install
```

This runs `uv sync` and installs all dependencies (pydantic, numpy, the llm_sdk wrapper, flake8, mypy). The model weights are downloaded from Hugging Face on first run.

### 2. Run

```bash
make run                              # default input files
make run FUNCS=data/input/functions_definition.json \
         INPUT=data/input/function_calling_tests.json \
         OUTPUT=data/output/function_calling_results.json
```

Or directly:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### 3. See results at a glance

```bash
make test
```

Runs the pipeline and prints a summary table — which function was selected for each prompt — so you can verify correctness without opening the JSON file.

### 4. Check code quality

```bash
make lint         # flake8 + mypy
make lint-strict  # mypy --strict (stricter)
```

### 5. Debug

```bash
make debug
```

Drops you into Python's `pdb` debugger. Useful commands: `n` (next line), `s` (step into), `p <var>` (print), `c` (continue), `b <line>` (set breakpoint).

### 6. Clean

```bash
make clean
```

## Algorithm

### Constrained decoding

Language models generate text one token at a time. At each step the model produces a **logit** (raw probability score) for every token in its vocabulary (~150 000 tokens for Qwen3). Normally you just pick the highest — but that gives no guarantees about the output format.

Constrained decoding intervenes before the pick:

1. The model produces logits for all tokens.
2. A state machine inspects the current partial output and determines which tokens are valid next.
3. Invalid tokens get their logit set to `−∞`.
4. `argmax` is called on the surviving tokens — the result is guaranteed to be structurally correct.

This repeats for every token until generation is complete. The output is always valid, parseable JSON that matches the schema exactly.

### Two-phase generation

**Phase 1 — function selection.** The prompt context (available functions + user request) is encoded and passed to the model. A prefix trie is built from all valid function names. At each generation step, only tokens whose string representation is a valid continuation of some function name are allowed. Generation stops as soon as the accumulated string exactly matches one of the names.

**Phase 2 — parameter extraction.** Once the function is known, all parameter names and types are known too. The JSON keys are pre-filled (they are fixed strings, not generated). Only the *values* are generated, and each value is constrained by its declared type:

- **number** — the JSON number grammar `-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?` is encoded as a state machine. Tokens that keep the partial number a valid prefix are allowed; separator tokens (`,`, `}`) are allowed only when the current partial is already a complete number, which ends generation.
- **string** — any token that does not contain an unescaped `"` is allowed; the closing `"` ends generation.
- **boolean** — constrained to the tokens forming `true` or `false` exactly.

The final JSON is built by combining the fixed JSON structure with the generated values.

### Vocabulary mapping

The vocabulary file (`vocab.json`, retrieved via `get_path_to_vocab_file`) maps every token string to its ID. The reverse map (ID → string) is built once at startup and used at every generation step to translate token IDs into their character sequences, which is what the state machine inspects.

## Design decisions

**State machine over post-processing.** Fixing invalid JSON after the fact (retrying, regex patching) is fragile and can still fail. Driving the generation token-by-token with a state machine is harder to implement but makes failures structurally impossible.

**Pre-filling fixed JSON keys.** Parameter names are known before generation starts. Encoding them directly into the prompt context costs nothing and removes an entire class of potential errors (wrong key names, wrong order).

**Argmax instead of sampling.** For structured output, determinism is more valuable than diversity. Sampling introduces randomness that can still produce invalid tokens even after masking. Argmax always picks the most likely valid token.

**Pydantic for all data classes.** The subject requires it, and it earns its keep: schema validation on load catches malformed input files early, and typed models make mypy useful throughout the codebase.

## Performance analysis

The bottleneck is the model forward pass, which runs once per generated token. A typical function call with two parameters requires roughly 15–30 forward passes (function name: ~10 tokens, two values: ~5–10 tokens each). On CPU this takes a few seconds per prompt; on MPS or CUDA it is significantly faster.

The full test suite (11 prompts) runs in under 5 minutes on standard hardware, well within the subject's requirement.

Caching the reverse vocabulary map and the function-name trie at startup avoids rebuilding them for every prompt.

## Challenges faced

**Token boundary alignment.** A function name like `fn_add_numbers` might be split across multiple tokens in ways that don't align with word boundaries. The prefix trie approach handles this correctly because it operates on accumulated character strings rather than whole tokens.

**Number termination.** Knowing when a number is "done" is non-trivial: `2` is a valid number, but so is `2.0` and `2e10`. The solution is to allow terminator tokens (`,`, `}`) only once the partial string matches the complete number grammar — the model then naturally emits a terminator when it has no more digits to add.

**Qwen3's thinking mode.** Qwen3 is a reasoning model that may prepend `<think>...</think>` blocks to its output. The prompt is constructed to suppress this behavior, and the generation pipeline pre-fills the JSON prefix directly to skip any preamble entirely.

## Testing strategy

`make test` runs the full pipeline on the provided input files and prints a comparison table of selected functions vs. prompts. This is the primary sanity check.

Beyond that, edge cases to verify manually:
- Missing or malformed input files (must fail gracefully with a clear message, not a traceback)
- Prompts with large numbers, special characters, or ambiguous phrasing
- Functions with multiple parameters of mixed types
- Swapped or rephrased prompts that map to the same function

## Example usage

**Input** (`data/input/function_calling_tests.json`):

```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "Greet shrek"},
  {"prompt": "Reverse the string 'hello'"}
]
```

**Run:**

```bash
make run
```

**Output** (`data/output/function_calling_results.json`):

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": {"name": "shrek"}
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {"s": "hello"}
  }
]
```

## Resources

- [Comment FORCER une IA à produire du JSON valide à 100%](https://www.youtube.com/watch?v=QhgQKRRu3d4) — video explaining constrained decoding in practice
- [Constrained decoding for structured generation](https://huggingface.co/blog/constrained-beam-search) — HuggingFace blog post on the concept
- [Outlines library](https://github.com/dottxt-ai/outlines) — reference implementation of constrained decoding (not used here, study only)
- [JSON grammar specification (ECMA-404)](https://www.json.org/json-en.html) — used to build the number and string automata
- [BPE tokenization explained](https://huggingface.co/learn/nlp-course/chapter6/5) — necessary background for understanding token boundaries
- Qwen3 model card on Hugging Face — model behavior, context length, thinking-mode conventions

### AI usage

Claude Code was used to talk through the constrained decoding architecture, discuss the trade-offs between prefix-trie and regex-NFA approaches for function name selection, and review the state machine logic for number termination. All code in the repository was written and fully understood by the author.

The two-phase generation design (function selection first, parameter extraction second) was also discussed with peers at 42 to validate that it was the right decomposition before committing to it.
