# Contributing to torch-fem

Thank you for considering contributing to `torch-fem`. This document
summarises the workflow and coding standards expected for any change to
this repository.

## Reporting issues

- If you find a bug or have a feature request, please open an issue on the
  [GitHub issue tracker](https://github.com/ruibmpinto/torch-fem/issues).
- Provide as much context as possible: a minimal working example, the
  expected and observed behaviour, the Python / PyTorch version, and the
  operating system.

## Development workflow

1. **Fork** the repository and create a feature branch from `main`.
2. **Install** the project in editable mode with all optional
   dependencies:
   ```bash
   conda activate env_torchfem
   pip install -e ".[dev,tests,docs]"
   ```
3. **Make your changes** in small, logically isolated commits with clear
   messages.
4. **Run the full quality pipeline** before opening a pull request:
   ```bash
   make format   # auto-fix formatting
   make lint     # 0 errors expected
   make test     # all tests must pass
   make docs     # documentation builds without errors
   make build    # source and wheel distributions build cleanly
   ```
5. **Submit a pull request** against `main` and reference any related
   issue numbers.

## Coding style

This repository follows the
[Bessa Research Group (BRG) Python Development Code of Conduct](https://github.com/bessagroup/python_code_of_conduct).
The main points are:

- **Line length:** 79 characters.
- **Indentation:** 4 spaces, never tabs.
- **String quotes:** single quotes for regular strings, double quotes for
  triple-quoted strings (docstrings).
- **Naming:** `snake_case` for variables, functions and modules;
  `UpperCamelCase` for classes; a single leading underscore for
  non-public members.
- **Imports:** one per line, absolute, grouped as (1) standard library,
  (2) third-party, (3) local. No wildcard imports.
- **Docstrings:** NumPy style for every module, class, function and
  method. Document parameters, return values and raised exceptions.
- **Comments:** block comments only, no inline comments.

Compliance is enforced automatically by `ruff`; the exact configuration
lives under `[tool.ruff]` in `pyproject.toml`. A contribution is
acceptable only if `make lint` reports zero errors.
