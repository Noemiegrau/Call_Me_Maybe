import json
from typing import cast


def load_vocab(path: str) -> dict[str, int]:
    """Load BPE vocabulary mapping token strings to token IDs.

    Args:
        path: Path to the vocab.json file.

    Returns:
        Dictionary mapping each token string to its integer ID.
    """
    with open(path) as f:
        return cast(dict[str, int], json.load(f))


def load_merges(path: str) -> list[tuple[str, str]]:
    """Load BPE merge rules from a merges.txt file.

    Each line contains two tokens separated by a space. Lines starting
    with '#' are skipped.

    Args:
        path: Path to the merges.txt file.

    Returns:
        List of merge pairs in priority order (first = highest priority).
    """
    merges: list[tuple[str, str]] = []
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split(' ', 1)
            if len(parts) == 2:
                merges.append((parts[0], parts[1]))
    return merges


def _apply_merges(
    chars: list[str],
    merge_ranks: dict[tuple[str, str], int],
) -> list[str]:
    """Apply BPE merges to a list of character tokens.

    At each step, the pair with the lowest rank (highest priority) is
    merged. Repeats until no more merges can be applied.

    Args:
        chars: Initial list of single-character tokens.
        merge_ranks: Mapping from token pairs to their merge priority.

    Returns:
        List of tokens after all possible merges are applied.
    """
    tokens = chars[:]
    while len(tokens) > 1:
        best_rank: float = float('inf')
        best_idx = -1
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            rank = merge_ranks.get(pair, float('inf'))
            if rank < best_rank:
                best_rank = rank
                best_idx = i
        if best_idx == -1:
            break
        merged = tokens[best_idx] + tokens[best_idx + 1]
        tokens = (
            tokens[:best_idx] + [merged] + tokens[best_idx + 2:]
        )
    return tokens


def encode(
    text: str,
    vocab: dict[str, int],
    merges: list[tuple[str, str]],
) -> list[int]:
    """Encode text to a list of token IDs using the BPE algorithm.

    Splits text into individual characters, then repeatedly merges the
    highest-priority pair until no more merges apply. Returns the ID
    of each resulting token found in the vocabulary.

    Args:
        text: Input text to tokenize.
        vocab: Mapping from token strings to token IDs.
        merges: BPE merge rules in priority order.

    Returns:
        List of integer token IDs.
    """
    merge_ranks: dict[tuple[str, str], int] = {
        pair: rank for rank, pair in enumerate(merges)
    }
    tokens = _apply_merges(list(text), merge_ranks)
    return [vocab[t] for t in tokens if t in vocab]


def decode(token_ids: list[int], reverse_vocab: dict[int, str]) -> str:
    """Decode a list of token IDs back to text.

    Args:
        token_ids: List of token IDs to decode.
        reverse_vocab: Mapping from token IDs to token strings.

    Returns:
        Reconstructed text string.
    """
    return ''.join(reverse_vocab.get(tid, '') for tid in token_ids)
