from src.tokenizer import (
    build_reverse_vocab,
    build_trie,
    get_valid_token_ids,
    is_complete_name,
)


def test_build_trie_creates_nested_structure() -> None:
    trie = build_trie(["fn_add"])
    assert "f" in trie
    assert "" in trie["f"]["n"]["_"]["a"]["d"]["d"]


def test_build_trie_multiple_names() -> None:
    trie = build_trie(["fn_add", "fn_greet"])
    assert "a" in trie["f"]["n"]["_"]
    assert "g" in trie["f"]["n"]["_"]


def test_is_complete_name_exact_match() -> None:
    trie = build_trie(["fn_add", "fn_greet"])
    assert is_complete_name(trie, "fn_add") is True
    assert is_complete_name(trie, "fn_greet") is True


def test_is_complete_name_prefix_returns_false() -> None:
    trie = build_trie(["fn_add"])
    assert is_complete_name(trie, "fn_") is False
    assert is_complete_name(trie, "fn_a") is False
    assert is_complete_name(trie, "") is False


def test_is_complete_name_invalid_path() -> None:
    trie = build_trie(["fn_add"])
    assert is_complete_name(trie, "fn_xyz") is False
    assert is_complete_name(trie, "hello") is False


def test_get_valid_token_ids_empty_prefix() -> None:
    trie = build_trie(["fn_add", "fn_greet"])
    vocab = {"fn": 1, "fn_add": 2, "hello": 3, "fn_": 4}
    reverse_vocab = build_reverse_vocab(vocab)
    valid = get_valid_token_ids(trie, "", reverse_vocab)
    assert 1 in valid
    assert 2 in valid
    assert 4 in valid
    assert 3 not in valid


def test_get_valid_token_ids_with_prefix() -> None:
    trie = build_trie(["fn_add", "fn_greet"])
    vocab = {"add": 10, "greet": 11, "xyz": 12}
    reverse_vocab = build_reverse_vocab(vocab)
    valid = get_valid_token_ids(trie, "fn_", reverse_vocab)
    assert 10 in valid
    assert 11 in valid
    assert 12 not in valid


def test_get_valid_token_ids_invalid_prefix() -> None:
    trie = build_trie(["fn_add"])
    vocab = {"add": 1}
    reverse_vocab = build_reverse_vocab(vocab)
    valid = get_valid_token_ids(trie, "xyz", reverse_vocab)
    assert valid == []


def test_build_reverse_vocab() -> None:
    vocab = {"hello": 1, "world": 2, "fn": 99}
    reverse = build_reverse_vocab(vocab)
    assert reverse[1] == "hello"
    assert reverse[2] == "world"
    assert reverse[99] == "fn"
    assert len(reverse) == 3
