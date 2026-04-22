.PHONY: help build test lint format docs clean

help:
	@echo "Available targets:"
	@echo "  build   - Build the package distribution (sdist + wheel)"
	@echo "  test    - Run the test suite with pytest"
	@echo "  lint    - Check code style with ruff"
	@echo "  format  - Auto-format code with ruff"
	@echo "  docs    - Build the HTML documentation with MkDocs"
	@echo "  clean   - Remove build, dist and cache artifacts"

build:
	python -m build

test:
	pytest

lint:
	ruff check .

format:
	ruff check --fix .
	ruff format .

docs:
	mkdocs build

clean:
	rm -rf build dist *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	rm -rf site
