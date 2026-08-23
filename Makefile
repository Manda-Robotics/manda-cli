PYTHON ?= python3
CONFIG ?= examples/experiment.toml

.PHONY: help validate test lint

help:
	@PYTHONPATH=src $(PYTHON) -m manda_cli --help

validate:
	@PYTHONPATH=src $(PYTHON) -m manda_cli validate-config --config $(CONFIG)

test:
	@PYTHONPATH=src $(PYTHON) -m pytest

lint:
	@ruff check src tests
