import json
import os
from pathlib import Path

import pytest

from src.loader import load_functions, load_prompts, save_results
from src.models import FunctionCall


def test_load_functions_valid(tmp_path: Path) -> None:
    data = [
        {
            "name": "fn_add",
            "description": "Add two numbers",
            "parameters": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "returns": {"type": "number"},
        }
    ]
    f = tmp_path / "funcs.json"
    f.write_text(json.dumps(data))
    functions = load_functions(str(f))
    assert len(functions) == 1
    assert functions[0].name == "fn_add"
    assert functions[0].parameters["a"].type == "number"


def test_load_functions_missing_file() -> None:
    with pytest.raises(SystemExit):
        load_functions("/nonexistent/path/funcs.json")


def test_load_functions_invalid_json(tmp_path: Path) -> None:
    f = tmp_path / "funcs.json"
    f.write_text("not { valid json [[[")
    with pytest.raises(SystemExit):
        load_functions(str(f))


def test_load_functions_malformed_schema(tmp_path: Path) -> None:
    data = [{"wrong_key": "no name or description"}]
    f = tmp_path / "funcs.json"
    f.write_text(json.dumps(data))
    with pytest.raises(SystemExit):
        load_functions(str(f))


def test_load_prompts_valid(tmp_path: Path) -> None:
    data = [{"prompt": "What is 2 + 2?"}, {"prompt": "Greet Alice"}]
    f = tmp_path / "prompts.json"
    f.write_text(json.dumps(data))
    prompts = load_prompts(str(f))
    assert len(prompts) == 2
    assert prompts[0].prompt == "What is 2 + 2?"
    assert prompts[1].prompt == "Greet Alice"


def test_load_prompts_missing_file() -> None:
    with pytest.raises(SystemExit):
        load_prompts("/nonexistent/prompts.json")


def test_load_prompts_invalid_json(tmp_path: Path) -> None:
    f = tmp_path / "prompts.json"
    f.write_text("{invalid}")
    with pytest.raises(SystemExit):
        load_prompts(str(f))


def test_save_results_creates_output_directory(tmp_path: Path) -> None:
    results = [
        FunctionCall(
            prompt="Add 1 and 2",
            name="fn_add",
            parameters={"a": 1.0, "b": 2.0},
        )
    ]
    output = str(tmp_path / "nested" / "dir" / "results.json")
    save_results(results, output)
    assert os.path.exists(output)


def test_save_results_correct_json(tmp_path: Path) -> None:
    results = [
        FunctionCall(
            prompt="Greet Bob",
            name="fn_greet",
            parameters={"name": "Bob"},
        )
    ]
    output = str(tmp_path / "out.json")
    save_results(results, output)
    with open(output) as f:
        data = json.load(f)
    assert data[0]["name"] == "fn_greet"
    assert data[0]["prompt"] == "Greet Bob"
    assert data[0]["parameters"]["name"] == "Bob"


def test_save_results_multiple(tmp_path: Path) -> None:
    results = [
        FunctionCall(prompt="p1", name="fn_a", parameters={"x": 1}),
        FunctionCall(prompt="p2", name="fn_b", parameters={"y": True}),
    ]
    output = str(tmp_path / "out.json")
    save_results(results, output)
    with open(output) as f:
        data = json.load(f)
    assert len(data) == 2
