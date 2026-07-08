import argparse
import sys

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from .loader import load_functions, load_prompts, save_results
from .pipeline import process_prompt
from .tokenizer import build_reverse_vocab, build_trie, load_vocab


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments with functions_definition, input, output, and model.
    """
    parser = argparse.ArgumentParser(
        description="Translate natural language prompts into function calls."
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        metavar="FILE",
        help="Path to the functions definition JSON file.",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        metavar="FILE",
        help="Path to the input prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        metavar="FILE",
        help="Path for the output results JSON file.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        metavar="MODEL",
        help="HuggingFace model identifier (default: Qwen/Qwen3-0.6B).",
    )
    return parser.parse_args()


def main() -> int:
    """Run the function calling pipeline.

    Returns:
        Exit code (0 on success, 1 on error).
    """
    args = parse_args()

    functions = load_functions(args.functions_definition)
    prompts = load_prompts(args.input)

    model = Small_LLM_Model(model_name=args.model)
    vocab = load_vocab(model.get_path_to_vocab_file())
    reverse_vocab = build_reverse_vocab(vocab)
    trie = build_trie([fn.name for fn in functions])

    results = []
    for prompt in prompts:
        result = process_prompt(model, prompt, functions, trie, reverse_vocab)
        results.append(result)
        print(f"  {result.name} <- {prompt.prompt}")

    save_results(results, args.output)
    print(f"\nSaved {len(results)} result(s) to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
