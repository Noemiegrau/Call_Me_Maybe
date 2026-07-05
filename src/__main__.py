import argparse
import sys


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments with functions_definition, input, and output paths.
    """
    parser = argparse.ArgumentParser(
        description="Translate natural language prompts into structured function calls."
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
    return parser.parse_args()


def main() -> int:
    """Run the function calling pipeline.

    Returns:
        Exit code (0 on success, 1 on error).
    """
    args = parse_args()
    print(f"functions_definition : {args.functions_definition}")
    print(f"input                : {args.input}")
    print(f"output               : {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
