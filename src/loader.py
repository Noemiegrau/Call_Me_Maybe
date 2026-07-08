import json
import os
import sys
from pydantic import ValidationError

from .models import FunctionCall, FunctionDefinition, Prompt


def load_functions(path: str) -> list[FunctionDefinition]:
    """Load and validate function definitions from a JSON file.

    Args:
        path: Path to the functions definition JSON file.

    Returns:
        List of validated FunctionDefinition objects.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in {path}: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        return [FunctionDefinition(**item) for item in data]
    except ValidationError as e:
        print(
            f"Error: malformed function definition in {path}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


def load_prompts(path: str) -> list[Prompt]:
    """Load and validate prompts from a JSON file.

    Args:
        path: Path to the input prompts JSON file.

    Returns:
        List of validated Prompt objects.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in {path}: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        return [Prompt(**item) for item in data]
    except ValidationError as e:
        print(f"Error: malformed prompt in {path}: {e}", file=sys.stderr)
        sys.exit(1)


def save_results(results: list[FunctionCall], path: str) -> None:
    """Serialize results to a JSON file, creating output directory if needed.

    Args:
        results: List of FunctionCall objects to serialize.
        path: Destination path for the output JSON file.
    """
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)
