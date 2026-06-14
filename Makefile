# Single source of truth for project commands.
#
# CI calls these same targets, and the pre-commit hooks call the same underlying
# uv tools with the same pyproject.toml config, so the local, CI, and hook gates
# cannot silently diverge. Everything runs through `uv` for a pinned, reproducible
# toolchain; do not call bare python/ruff/mypy.

.PHONY: help setup format lint type test check clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup: ## Create the env from the lockfile and install git hooks
	uv sync
	uv run pre-commit install

format: ## Auto-format and auto-fix (MUTATING)
	uv run ruff format .
	uv run ruff check --fix .

lint: ## Lint + format-check, non-mutating (CI-safe)
	uv run ruff check .
	uv run ruff format --check .

type: ## Strict static type check (mypy)
	uv run mypy

test: ## Run tests with coverage (pytest)
	uv run pytest

check: lint type test ## The full local gate: lint + type + test

clean: ## Remove tool caches
	rm -rf .mypy_cache .ruff_cache .pytest_cache .coverage htmlcov
