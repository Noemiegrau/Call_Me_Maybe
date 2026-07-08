from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from .decoder import (
    generate_boolean,
    generate_function_name,
    generate_integer,
    generate_number,
    generate_string,
)
from .models import FunctionCall, FunctionDefinition, Prompt
from .tokenizer import Trie


def _build_prompt(
    functions: list[FunctionDefinition], user_prompt: str
) -> str:
    """Build the text context fed to the model, ending with the JSON prefix.

    Args:
        functions: Available function definitions shown to the model.
        user_prompt: The natural language request from the user.

    Returns:
        Prompt string ending with the pre-filled JSON opening.
    """
    fn_lines = []
    for fn in functions:
        params = ", ".join(
            f"{name}: {param.type}"
            for name, param in fn.parameters.items()
        )
        fn_lines.append(f"- {fn.name}({params}): {fn.description}")
    return (
        "You are a function calling assistant.\n\n"
        "Available functions:\n"
        + "\n".join(fn_lines)
        + f"\n\nUser request: {user_prompt}\n\n"
        + '{"name": "'
    )


def _encode(model: Small_LLM_Model, text: str) -> list[int]:
    """Encode text to a flat list of token IDs.

    Args:
        model: The language model whose tokenizer is used.
        text: Text to tokenize.

    Returns:
        List of integer token IDs.
    """
    return model.encode(text)[0].tolist()  # type: ignore[no-any-return]


def process_prompt(
    model: Small_LLM_Model,
    prompt: Prompt,
    functions: list[FunctionDefinition],
    trie: Trie,
    reverse_vocab: dict[int, str],
) -> FunctionCall:
    """Run the two-phase constrained generation pipeline for one prompt.

    Phase 1 selects the function name via trie-constrained decoding.
    Phase 2 generates each parameter value constrained by its declared type.

    Args:
        model: The language model used for generation.
        prompt: The user prompt to process.
        functions: All available function definitions.
        trie: Prefix trie built from function names.
        reverse_vocab: Mapping from token IDs to token strings.

    Returns:
        FunctionCall with the selected function name and extracted parameters.
    """
    input_ids = _encode(model, _build_prompt(functions, prompt.prompt))

    name, input_ids = generate_function_name(
        model, input_ids, trie, reverse_vocab
    )
    fn_def = next(f for f in functions if f.name == name)

    input_ids += _encode(model, '", "parameters": {')

    parameters: dict[str, float | int | str | bool] = {}
    param_items = list(fn_def.parameters.items())

    for idx, (param_name, param_def) in enumerate(param_items):
        is_last = idx == len(param_items) - 1
        input_ids += _encode(model, f'"{param_name}": ')

        if param_def.type == "number":
            num_val, input_ids = generate_number(
                model, input_ids, reverse_vocab
            )
            parameters[param_name] = num_val
        elif param_def.type == "integer":
            int_val, input_ids = generate_integer(
                model, input_ids, reverse_vocab
            )
            parameters[param_name] = int_val
        elif param_def.type == "string":
            input_ids += _encode(model, '"')
            str_val, input_ids = generate_string(
                model, input_ids, reverse_vocab
            )
            input_ids += _encode(model, '"')
            parameters[param_name] = str_val
        elif param_def.type == "boolean":
            bool_val, input_ids = generate_boolean(
                model, input_ids, reverse_vocab
            )
            parameters[param_name] = bool_val
        else:
            raise ValueError(f"Unsupported parameter type: {param_def.type!r}")

        input_ids += _encode(model, "}" if is_last else ", ")

    if not param_items:
        input_ids += _encode(model, "}")

    input_ids += _encode(model, "}")

    return FunctionCall(prompt=prompt.prompt, name=name, parameters=parameters)
