PYTHON  = uv run python
SRC     = src
FUNCS   ?= data/input/functions_definition.json
INPUT   ?= data/input/function_calling_tests.json
OUTPUT  ?= data/output/function_calling_results.json

.PHONY: help install run test debug lint lint-strict clean

help:
	@echo "Usage:"
	@echo "  make install                  Install all dependencies"
	@echo "  make run                      Run with default input files"
	@echo "  make run FUNCS=<path>         Override functions definition file"
	@echo "  make run INPUT=<path>         Override input prompts file"
	@echo "  make run OUTPUT=<path>        Override output file"
	@echo "  make test                     Run and print a summary of results"
	@echo "  make debug                    Run under pdb debugger"
	@echo "  make lint                     Check style and types (flake8 + mypy)"
	@echo "  make lint-strict              Stricter mypy (--strict)"
	@echo "  make clean                    Remove __pycache__ and build artifacts"

install:
	uv sync --all-groups

run:
	$(PYTHON) -m $(SRC) \
		--functions_definition $(FUNCS) \
		--input $(INPUT) \
		--output $(OUTPUT)

test:
	@$(PYTHON) -m $(SRC) \
		--functions_definition $(FUNCS) \
		--input $(INPUT) \
		--output $(OUTPUT)
	@echo ""
	@$(PYTHON) -c "\
import json; \
d = json.load(open('$(OUTPUT)')); \
print(f'  {len(d)} result(s) — all valid JSON\n'); \
[print(f\"  {r['name']:<35s} <- {r['prompt']}\") for r in d]; \
"

debug:
	$(PYTHON) -m pdb -m $(SRC) \
		--functions_definition $(FUNCS) \
		--input $(INPUT) \
		--output $(OUTPUT)

lint:
	flake8 .
	mypy . --warn-return-any --warn-unused-ignores \
	       --ignore-missing-imports --disallow-untyped-defs \
	       --check-untyped-defs

lint-strict:
	flake8 .
	mypy . --strict

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache"  -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
