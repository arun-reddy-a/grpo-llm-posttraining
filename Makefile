.PHONY: help install install-dev test lint fmt smoke clean

help:
	@echo "install      Install the package plus the training extras"
	@echo "install-dev  Install the CPU-only test environment"
	@echo "test         Run the test suite (no GPU, no network)"
	@echo "lint         ruff check"
	@echo "fmt          ruff check --fix"
	@echo "smoke        Ten-step end-to-end training run (needs a model download)"
	@echo "clean        Remove caches and build artifacts"

install:
	pip install -e ".[train]"

install-dev:
	pip install -r requirements-dev.txt && pip install -e . --no-deps

test:
	pytest -q

lint:
	ruff check src tests

fmt:
	ruff check src tests --fix

smoke:
	bash scripts/smoke_test.sh

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
