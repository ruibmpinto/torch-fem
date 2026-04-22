# torch-fem

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torch-fem)
[![PyPI - Version](https://img.shields.io/pypi/v/torch-fem)](https://pypi.org/project/torch-fem/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/meyer-nils/torch-fem/HEAD)

| [**GitHub**](https://github.com/ruibmpinto/torch-fem)
| [**Documentation**](https://ruibmpinto.github.io/torch-fem/)
| [**Upstream**](https://github.com/meyer-nils/torch-fem)

---

## Summary

`torch-fem` is a simple, GPU-accelerated, **differentiable** finite-element
library for solid mechanics built on top of [PyTorch](https://pytorch.org).
Because every operation is differentiable, sensitivities of arbitrary
quantities of interest (compliance, stress, strain, homogenised stiffness,
etc.) with respect to the model inputs (nodal coordinates, material
parameters, element thicknesses, fibre orientations, density fields, ...)
are obtained directly with automatic differentiation, at essentially no
additional implementation cost.

Main features:

- **Elements**
    - 1D: `Bar1`, `Bar2`
    - 2D: `Quad1`, `Quad2`, `Tria1`, `Tria2`
    - 3D: `Hexa1`, `Hexa2`, `Tetra1`, `Tetra2`
    - Shell: flat-facet triangle (linear only)
- **Material models**
    - Isotropic linear elasticity
    - Orthotropic linear elasticity
    - Isotropic small-strain plasticity with isotropic hardening
    - Logarithmic finite-strain elasticity
    - Hyperelasticity (via automatic differentiation of the energy density)
    - Lou-Zhang-Yoon anisotropic elastoplasticity (custom material)
    - User-defined material interface
- **Utilities**
    - Homogenisation of orthotropic elasticity for composites
    - Mesh I/O through [meshio](https://github.com/nschloe/meshio)
    - GPU back-end via [PyTorch](https://pytorch.org) and (optionally)
      [CuPy](https://cupy.dev)

## Statement of need

Classical finite-element codes are powerful but not differentiable: obtaining
sensitivities for design optimisation, inverse problems, data-driven
constitutive modelling or machine-learning-augmented simulation typically
requires hand-derived adjoint solvers or finite differences, both of which
are labour-intensive and error-prone.

`torch-fem` fills this gap by providing a clean, research-oriented FEM
implementation in which **every step of the solver is a PyTorch operation**.
This enables:

- Gradient-based topology, shape, and material-orientation optimisation with
  a single call to `torch.autograd.grad`.
- Training of neural-network surrogates for constitutive behaviour with a
  true FEM solver in the loop.
- Seamless deployment on CPU or GPU, without writing a single line of
  accelerator-specific code.
- Rapid prototyping of custom element formulations, boundary conditions,
  and material models.

The package targets researchers in computational mechanics, materials
science, and machine learning for engineering who need a lightweight,
differentiable FEM framework that integrates naturally with the PyTorch
ecosystem.

## Authorship

**Authors:**

- Nils Meyer ([nils.meyer@uni-a.de](mailto:nils.meyer@uni-a.de)) — original
  author of the upstream `torch-fem` package.

**Maintainer of this fork:**

- Rui Barreira Morais Pinto
  ([rui_pinto@brown.edu](mailto:rui_pinto@brown.edu))

**Affiliations:**

- Bessa Research Group @ Brown University (School of Engineering)

This fork extends the original `torch-fem` with additional custom material
models, homogenisation utilities and user-level scripts used in the Bessa
Research Group.

## Getting started

### Installation

The recommended way is to install the package in editable mode from the
repository root together with the development, testing and documentation
extras:

```bash
conda activate env_torchfem
pip install -e ".[dev,tests,docs]"
```

For a minimal user installation without development tooling:

```bash
pip install -e .
```

**Optional GPU support** — install PyTorch and a matching CuPy build:

```bash
# CUDA 11.8
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118
pip install cupy-cuda11x

# CUDA 12.6
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126
pip install cupy-cuda12x
```

### Minimal example

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
```

### Tutorials and benchmarks

- `examples/basic/` — Jupyter notebooks demonstrating trusses, planar
  problems, shells and solids.
- `examples/optimization/` — topology and orientation optimisation
  examples with automatic differentiation.
- `examples/benchmark/` — performance benchmarks on cubic meshes.
- `user_scripts/` — higher-level scripts used in the Bessa Research Group
  (parallel simulations, stiffness analysis, surrogate models,
  homogenisation with periodic boundary conditions, ...).

### Common developer commands

The project uses a `Makefile` that provides a tool-agnostic interface:

| Target        | Purpose                                       |
| ------------- | --------------------------------------------- |
| `make build`  | Build the source and wheel distributions      |
| `make test`   | Run the test suite with `pytest`              |
| `make lint`   | Check the code style with `ruff`              |
| `make format` | Auto-format the code with `ruff`              |
| `make docs`   | Build the HTML documentation with `mkdocs`    |
| `make clean`  | Remove build artefacts and caches             |

## Benchmarks

Benchmarks on a cube under one-dimensional extension
(side length `1.0`, Young's modulus `1000.0`, Poisson's ratio `0.3`) with
`N x N x N` linear hexahedral elements.

### Apple M1 Pro (10 cores, 16 GB RAM) — Python 3.10, SciPy 1.14.1

|  N  |   DOFs  |  FWD Time |  BWD Time |   Peak RAM |
| --- | ------- | --------- | --------- | ---------- |
|  10 |    3000 |    0.14 s |    0.03 s |   592.2 MB |
|  20 |   24000 |    0.99 s |    0.15 s |   968.3 MB |
|  30 |   81000 |    3.42 s |    0.57 s |  1562.8 MB |
|  40 |  192000 |    8.48 s |    1.14 s |  2497.1 MB |
|  50 |  375000 |   16.46 s |    2.23 s |  3963.7 MB |
|  60 |  648000 |   28.63 s |    3.56 s |  5503.3 MB |
|  70 | 1029000 |   46.86 s |    5.91 s |  6309.5 MB |
|  80 | 1536000 |   74.12 s |   10.69 s |  6933.7 MB |
|  90 | 2187000 |  121.11 s |   16.63 s |  7663.5 MB |
| 100 | 3000000 |  179.44 s |   38.35 s |  9662.4 MB |

### AMD Threadripper PRO 5995WX (64 cores) + NVIDIA RTX 4090 — Python 3.12, CuPy 13.3.0, CUDA 11.8

|  N  |   DOFs  |  FWD Time |  BWD Time |   Peak RAM |
| --- | ------- | --------- | --------- | ---------- |
|  10 |    3000 |    0.66 s |    0.15 s |  1371.7 MB |
|  20 |   24000 |    1.00 s |    0.43 s |  1358.9 MB |
|  30 |   81000 |    1.14 s |    0.65 s |  1371.1 MB |
|  40 |  192000 |    1.37 s |    0.83 s |  1367.3 MB |
|  50 |  375000 |    1.51 s |    1.04 s |  1356.4 MB |
|  60 |  648000 |    1.94 s |    1.43 s |  1342.1 MB |
|  70 | 1029000 |    5.19 s |    4.31 s |  1366.8 MB |
|  80 | 1536000 |    7.48 s |   18.88 s |  5105.6 MB |

## Alternatives

There are several alternative FEM solvers in Python:

- Non-differentiable:
    - [scikit-fem](https://github.com/kinnala/scikit-fem)
    - [nutils](https://github.com/evalf/nutils)
    - [felupe](https://github.com/adtzlr/felupe)
- Differentiable:
    - [jax-fem](https://github.com/deepmodeling/jax-fem)
    - [PyTorch FEA](https://github.com/liangbright/pytorch_fea)

## Community Support

If you find any **issues, bugs, or problems** with this package, please use
the [GitHub issue tracker](https://github.com/ruibmpinto/torch-fem/issues)
to report them. Pull requests are welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md) for details on the contribution
workflow and the coding style enforced in this repository.

## License

Copyright (c) 2024 Nils Meyer. Fork and extensions copyright (c) 2025 Rui
Barreira Morais Pinto and the Bessa Research Group.

This project is licensed under the MIT License. See the [LICENSE](LICENSE)
file for the full license text.
