from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from .tokenizer import Trie, build_trie, get_valid_token_ids, is_complete_name


_TERMINATORS = {",", "}"}
_BOOL_TRIE: Trie = build_trie(["true", "false"])


def _number_next_state(state: str, char: str) -> str | None:
    """Advance the JSON number state machine by one character.

    Args:
        state: Current state of the automaton.
        char: Next character to process.

    Returns:
        New state, or None if the character is invalid in this state.
    """
    if state == "start":
        if char == "-":
            return "minus"
        if char == "0":
            return "zero"
        if char in "123456789":
            return "int"
    elif state == "minus":
        if char == "0":
            return "zero"
        if char in "123456789":
            return "int"
    elif state == "zero":
        if char == ".":
            return "dot"
        if char in "eE":
            return "exp_e"
    elif state == "int":
        if char in "0123456789":
            return "int"
        if char == ".":
            return "dot"
        if char in "eE":
            return "exp_e"
    elif state == "dot":
        if char in "0123456789":
            return "frac"
    elif state == "frac":
        if char in "0123456789":
            return "frac"
        if char in "eE":
            return "exp_e"
    elif state == "exp_e":
        if char in "+-":
            return "exp_sign"
        if char in "0123456789":
            return "exp_digit"
    elif state == "exp_sign":
        if char in "0123456789":
            return "exp_digit"
    elif state == "exp_digit":
        if char in "0123456789":
            return "exp_digit"
    return None


_NUMBER_COMPLETE = {"zero", "int", "frac", "exp_digit"}


def generate_function_name(
    model: Small_LLM_Model,
    input_ids: list[int],
    trie: Trie,
    reverse_vocab: dict[int, str],
) -> tuple[str, list[int]]:
    """Generate a function name token by token using constrained decoding.

    At each step, only tokens that extend a valid function name are allowed.
    Generation stops as soon as the accumulated string is a complete name.

    Args:
        model: The language model used to get logits.
        input_ids: Current token ID sequence (prompt context).
        trie: Prefix trie built from valid function names.
        reverse_vocab: Mapping from token IDs to token strings.

    Returns:
        Tuple of (function name, updated input_ids).
    """
    accumulated = ""
    while True:
        logits = model.get_logits_from_input_ids(input_ids)
        valid_ids = set(get_valid_token_ids(trie, accumulated, reverse_vocab))
        for i in range(len(logits)):
            if i not in valid_ids:
                logits[i] = float("-inf")
        best_id = int(max(range(len(logits)), key=lambda i: logits[i]))
        token_str = reverse_vocab[best_id]
        accumulated += token_str
        input_ids = input_ids + [best_id]
        if is_complete_name(trie, accumulated):
            break
    return accumulated, input_ids


def generate_number(
    model: Small_LLM_Model,
    input_ids: list[int],
    reverse_vocab: dict[int, str],
) -> tuple[float, list[int]]:
    """Generate a JSON number value constrained to valid float syntax.

    Uses a state machine to allow only tokens that keep the partial
    number a valid prefix. Terminators are allowed only when the
    partial string is already a complete number.

    Args:
        model: The language model used to get logits.
        input_ids: Current token ID sequence.
        reverse_vocab: Mapping from token IDs to token strings.

    Returns:
        Tuple of (float value, updated input_ids).
    """
    partial = ""
    state = "start"
    while True:
        logits = model.get_logits_from_input_ids(input_ids)
        for i in range(len(logits)):
            token_str = reverse_vocab.get(i, "")
            if not token_str:
                logits[i] = float("-inf")
                continue
            if token_str in _TERMINATORS:
                if state not in _NUMBER_COMPLETE:
                    logits[i] = float("-inf")
            else:
                tmp = state
                ok = True
                for ch in token_str:
                    nxt = _number_next_state(tmp, ch)
                    if nxt is None:
                        ok = False
                        break
                    tmp = nxt
                if not ok:
                    logits[i] = float("-inf")
        best_id = int(max(range(len(logits)), key=lambda i: logits[i]))
        token_str = reverse_vocab[best_id]
        if token_str in _TERMINATORS:
            break
        for ch in token_str:
            nxt = _number_next_state(state, ch)
            assert nxt is not None
            state = nxt
        partial += token_str
        input_ids = input_ids + [best_id]
    return float(partial), input_ids


def generate_integer(
    model: Small_LLM_Model,
    input_ids: list[int],
    reverse_vocab: dict[int, str],
) -> tuple[int, list[int]]:
    """Generate a JSON integer value constrained to digits with optional sign.

    Args:
        model: The language model used to get logits.
        input_ids: Current token ID sequence.
        reverse_vocab: Mapping from token IDs to token strings.

    Returns:
        Tuple of (int value, updated input_ids).
    """
    partial = ""
    while True:
        logits = model.get_logits_from_input_ids(input_ids)
        for i in range(len(logits)):
            token_str = reverse_vocab.get(i, "")
            if not token_str:
                logits[i] = float("-inf")
                continue
            if token_str in _TERMINATORS:
                if not partial.lstrip("-"):
                    logits[i] = float("-inf")
            else:
                candidate = partial + token_str
                valid = candidate == "-" or candidate.lstrip("-").isdigit()
                if not valid:
                    logits[i] = float("-inf")
        best_id = int(max(range(len(logits)), key=lambda i: logits[i]))
        token_str = reverse_vocab[best_id]
        if token_str in _TERMINATORS:
            break
        partial += token_str
        input_ids = input_ids + [best_id]
    return int(partial), input_ids


def generate_string(
    model: Small_LLM_Model,
    input_ids: list[int],
    reverse_vocab: dict[int, str],
) -> tuple[str, list[int]]:
    """Generate a JSON string value, stopping on the closing quote token.

    Tokens containing a quote character are blocked unless they are
    exactly the single closing quote, which ends generation.

    Args:
        model: The language model used to get logits.
        input_ids: Current token ID sequence (context includes opening quote).
        reverse_vocab: Mapping from token IDs to token strings.

    Returns:
        Tuple of (string content without quotes, updated input_ids).
    """
    partial = ""
    while True:
        logits = model.get_logits_from_input_ids(input_ids)
        for i in range(len(logits)):
            token_str = reverse_vocab.get(i, "")
            if not token_str:
                logits[i] = float("-inf")
                continue
            if '"' in token_str and token_str != '"':
                logits[i] = float("-inf")
        best_id = int(max(range(len(logits)), key=lambda i: logits[i]))
        token_str = reverse_vocab[best_id]
        if token_str == '"':
            break
        partial += token_str
        input_ids = input_ids + [best_id]
    return partial, input_ids


def generate_boolean(
    model: Small_LLM_Model,
    input_ids: list[int],
    reverse_vocab: dict[int, str],
) -> tuple[bool, list[int]]:
    """Generate a JSON boolean value constrained to 'true' or 'false'.

    Args:
        model: The language model used to get logits.
        input_ids: Current token ID sequence.
        reverse_vocab: Mapping from token IDs to token strings.

    Returns:
        Tuple of (bool value, updated input_ids).
    """
    accumulated = ""
    while True:
        logits = model.get_logits_from_input_ids(input_ids)
        valid_ids = set(
            get_valid_token_ids(_BOOL_TRIE, accumulated, reverse_vocab)
        )
        for i in range(len(logits)):
            if i not in valid_ids:
                logits[i] = float("-inf")
        best_id = int(max(range(len(logits)), key=lambda i: logits[i]))
        token_str = reverse_vocab[best_id]
        accumulated += token_str
        input_ids = input_ids + [best_id]
        if is_complete_name(_BOOL_TRIE, accumulated):
            break
    return accumulated == "true", input_ids
