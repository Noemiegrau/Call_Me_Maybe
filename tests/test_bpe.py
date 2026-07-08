import json
from pathlib import Path

from src.bpe_tokenizer import (
    _apply_merges,
    decode,
    encode,
    load_merges,
    load_vocab,
)


def test_apply_merges_no_merges() -> None:
    tokens = _apply_merges(list("ab"), {})
    assert tokens == ["a", "b"]


def test_apply_merges_single_pair() -> None:
    ranks = {("a", "b"): 0}
    tokens = _apply_merges(list("ab"), ranks)
    assert tokens == ["ab"]


def test_apply_merges_priority_order() -> None:
    ranks = {("a", "b"): 0, ("b", "c"): 1}
    tokens = _apply_merges(list("abc"), ranks)
    assert tokens == ["ab", "c"]


def test_apply_merges_reverse_priority() -> None:
    ranks = {("a", "b"): 1, ("b", "c"): 0}
    tokens = _apply_merges(list("abc"), ranks)
    assert tokens == ["a", "bc"]


def test_apply_merges_full_word() -> None:
    ranks = {("h", "e"): 0, ("l", "l"): 1, ("he", "ll"): 2, ("hell", "o"): 3}
    tokens = _apply_merges(list("hello"), ranks)
    assert tokens == ["hello"]


def test_decode_simple() -> None:
    reverse_vocab = {1: "hello", 2: " ", 3: "world"}
    assert decode([1, 2, 3], reverse_vocab) == "hello world"


def test_decode_missing_id() -> None:
    reverse_vocab = {1: "hi"}
    assert decode([1, 99], reverse_vocab) == "hi"


def test_decode_empty() -> None:
    assert decode([], {}) == ""


def test_encode_with_toy_vocab_and_merges(tmp_path: Path) -> None:
    vocab = {
        "h": 0, "e": 1, "l": 2, "o": 3,
        "he": 4, "ll": 5, "hell": 6, "hello": 7,
    }
    merges_content = "#version: 0.2\nh e\nl l\nhe ll\nhell o\n"
    vocab_file = tmp_path / "vocab.json"
    merges_file = tmp_path / "merges.txt"
    vocab_file.write_text(json.dumps(vocab))
    merges_file.write_text(merges_content)
    v = load_vocab(str(vocab_file))
    m = load_merges(str(merges_file))
    assert encode("hello", v, m) == [7]


def test_encode_partial_merge(tmp_path: Path) -> None:
    vocab = {"h": 0, "e": 1, "l": 2, "o": 3, "he": 4}
    merges_content = "#version: 0.2\nh e\n"
    vocab_file = tmp_path / "vocab.json"
    merges_file = tmp_path / "merges.txt"
    vocab_file.write_text(json.dumps(vocab))
    merges_file.write_text(merges_content)
    v = load_vocab(str(vocab_file))
    m = load_merges(str(merges_file))
    result = encode("helo", v, m)
    assert result == [4, 2, 3]


def test_load_merges_skips_comments(tmp_path: Path) -> None:
    content = "#version: 0.2\n\na b\n# skip this\nc d\n"
    f = tmp_path / "merges.txt"
    f.write_text(content)
    merges = load_merges(str(f))
    assert merges == [("a", "b"), ("c", "d")]


def test_load_vocab(tmp_path: Path) -> None:
    vocab = {"fn": 0, "add": 1}
    f = tmp_path / "vocab.json"
    f.write_text(json.dumps(vocab))
    result = load_vocab(str(f))
    assert result == vocab
