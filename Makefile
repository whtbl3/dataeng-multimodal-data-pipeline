.PHONY: clean

clean:
	@rm -rf .pytest_cache .ruff_cache logs && find . -name __pycache__ -type d -not -path './.venv/*' -prune -exec rm -rf {} +
