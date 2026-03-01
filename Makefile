.DEFAULT_GOAL := help

## help: Print this help message
.PHONY: help
help: ## Print this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

## install: Install all dependencies
.PHONY: install
install: ## Install all dependencies
	uv sync --all-extras

## lint: Run ruff and mypy
.PHONY: lint
lint: ## Run ruff and mypy
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/
	uv run mypy src/llmtrace/ --strict

## format: Auto-format code
.PHONY: format
format: ## Auto-format code
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

## test: Run tests with coverage
.PHONY: test
test: ## Run tests with coverage
	uv run pytest tests/ -v --cov=src/llmtrace --cov-report=term-missing

## check: Run lint + test
.PHONY: check
check: lint test ## Run lint + test

## build: Build package
.PHONY: build
build: ## Build package
	uv build

## clean: Remove build artifacts and caches
.PHONY: clean
clean: ## Remove build artifacts and caches
	rm -rf dist/ build/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/

## release-dry: Preview next version (dry run)
.PHONY: release-dry
release-dry: ## Preview next version (dry run)
	uvx python-semantic-release version --print --no-commit --no-tag --no-push --no-changelog
