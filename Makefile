.PHONY: setup lint format test

setup:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

format:
	ruff check --fix src tests scripts
	ruff format src tests scripts

test:
	python -m pytest -q