import json
from typing import Any, cast


Trie = dict[str, Any]


def load_vocab(path: str) -> dict[str, int]:
    """Load the vocabulary file mapping token strings to token IDs.

    Args:
        path: Path to the vocab.json file from the model.

    Returns:
        Dictionary mapping each token string to its integer ID.
    """
    with open(path) as f:
        return cast(dict[str, int], json.load(f))


def build_reverse_vocab(vocab: dict[str, int]) -> dict[int, str]:
    """Build the reverse mapping from token IDs to token strings.

    Args:
        vocab: Dictionary mapping token strings to token IDs.

    Returns:
        Dictionary mapping each token ID to its string representation.
    """
    return {token_id: token_str for token_str, token_id in vocab.items()}


def build_trie(names: list[str]) -> Trie:
    """Build a prefix trie from a list of strings.

    Each string is inserted character by character. An empty string key
    marks the end of a complete entry.

    Args:
        names: List of strings to insert into the trie.

    Returns:
        Nested dictionary representing the trie.
    """
    root: Trie = {}
    for name in names:
        node = root
        for char in name:
            if char not in node:
                node[char] = {}
            node = node[char]
        node[""] = True
    return root


def get_valid_token_ids(
    trie: Trie,
    prefix: str,
    reverse_vocab: dict[int, str],
) -> list[int]:
    """Return all token IDs that are valid continuations of the given prefix.

    Navigates the trie to the node corresponding to prefix, then returns
    every token whose string can extend the current position without
    leaving the trie.

    Args:
        trie: Prefix trie built from valid names.
        prefix: Accumulated string generated so far.
        reverse_vocab: Mapping from token IDs to token strings.

    Returns:
        List of token IDs that are valid next tokens.
    """
    node = trie
    for char in prefix:
        if char not in node:
            return []
        node = node[char]
    valid = []
    for token_id, token_str in reverse_vocab.items():
        current = node
        ok = True
        for char in token_str:
            if char not in current:
                ok = False
                break
            current = current[char]
        if ok:
            valid.append(token_id)
    return valid


def is_complete_name(trie: Trie, name: str) -> bool:
    """Check whether the given string matches a complete entry in the trie.

    Args:
        trie: Prefix trie built from valid names.
        name: Accumulated string to check.

    Returns:
        True if name is a complete entry, False if it is only a prefix.
    """
    node = trie
    for char in name:
        if char not in node:
            return False
        node = node[char]
    return "" in node
