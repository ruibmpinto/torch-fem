# Getting started

## Installation

The recommended workflow is to install `torch-fem` in editable mode
inside a dedicated conda environment, together with the development,
testing and documentation extras:

```bash
conda activate env_torchfem
pip install -e ".[dev,tests,docs]"
```

For a minimal user installation without development tooling:

```bash
pip install -e .
```

### Optional GPU support

For NVIDIA GPUs you need a matching PyTorch build and (optionally) a
matching CuPy build.

CUDA 11.8:

```bash
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118
pip install cupy-cuda11x
```

CUDA 12.6:

```bash
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126
pip install cupy-cuda12x
```

## Minimal example

The following script solves a small planar cantilever problem and prints
the displacement field. It is the shortest possible introduction to the
library.

```python
import torch
from torchfem import Planar
from torchfem.materials import IsotropicElasticityPlaneStress

torch.set_default_dtype(torch.float64)

material = IsotropicElasticityPlaneStress(E=1000.0, nu=0.3)

nodes = torch.tensor(
    [[0., 0.], [1., 0.], [2., 0.],
     [0., 1.], [1., 1.], [2., 1.]]
)
elements = torch.tensor([[0, 1, 4, 3], [1, 2, 5, 4]])

cantilever = Planar(nodes, elements, material)
cantilever.forces[5, 1] = -1.0
cantilever.constraints[[0, 3], :] = True

u, f, sigma, F, alpha = cantilever.solve()
print(u)
```

## Sensitivities via automatic differentiation

Because every operation inside the solver is a PyTorch tensor operation,
gradients of any quantity of interest with respect to any input tensor
are obtained by `torch.autograd`:

```python
cantilever.thickness.requires_grad = True
u, f, _, _, _ = cantilever.solve()

compliance = torch.inner(f.ravel(), u.ravel())
grad = torch.autograd.grad(compliance, cantilever.thickness)[0]
```

## Repository tour

- `src/torchfem/` — core library (elements, material models, solvers,
  utilities).
- `src/torchfem/custom_materials/` — extended material models (e.g. the
  Lou–Zhang–Yoon anisotropic elastoplastic model).
- `tests/` — `pytest` suite.
- `examples/basic/` — introductory Jupyter notebooks.
- `examples/optimization/` — topology and orientation optimisation
  examples.
- `examples/benchmark/` — performance benchmarks.
- `user_scripts/` — larger scripts used in the Bessa Research Group
  (parallel simulations, stiffness analysis, periodic boundary
  conditions, local surrogates, ...).

## Makefile targets

The project ships with a `Makefile` providing a tool-agnostic interface:

| Target        | Purpose                                    |
| ------------- | ------------------------------------------ |
| `make build`  | Build source and wheel distributions       |
| `make test`   | Run the `pytest` test suite                |
| `make lint`   | Check code style with `ruff`               |
| `make format` | Auto-format the code with `ruff`           |
| `make docs`   | Build the HTML documentation with `mkdocs` |
| `make clean`  | Remove build artefacts and caches          |
