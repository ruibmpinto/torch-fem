# torch-fem

| [**GitHub**](https://github.com/ruibmpinto/torch-fem)
| [**Documentation**](https://ruibmpinto.github.io/torch-fem/)
| [**Upstream**](https://github.com/meyer-nils/torch-fem)

GPU-accelerated **differentiable** finite elements for solid mechanics
built on [PyTorch](https://pytorch.org).

---

## Summary

`torch-fem` is a lightweight, research-oriented finite-element framework
written entirely with PyTorch tensors. Because every step of the solver —
shape-function evaluation, Gauss-point integration, global assembly,
linear solve, stress recovery — is a differentiable PyTorch operation,
gradients of any scalar output with respect to any tensor input are
available through `torch.autograd`. This lets the user replace
hand-derived adjoint solvers and finite differences by a single
`torch.autograd.grad` call, and it makes the library a natural building
block for gradient-based design optimisation, inverse identification,
and hybrid physics/ML workflows.

## Statement of need

Traditional finite-element codes separate "forward simulation" from
"sensitivity analysis". The latter is typically implemented with a
hand-coded adjoint solver (tedious, specific to one formulation) or with
finite differences (expensive and noisy). Neither approach composes
naturally with modern automatic-differentiation frameworks.

`torch-fem` addresses this by:

- Implementing the full FEM pipeline directly with PyTorch tensors, so
  that sensitivities are obtained by autograd for free.
- Supporting linear and quadratic elements in 1D / 2D / 3D, a shell
  formulation, and a growing set of material models (linear elasticity,
  orthotropic elasticity, small-strain plasticity with hardening,
  hyperelasticity, Lou–Zhang–Yoon anisotropic elastoplasticity, custom
  user materials).
- Running on CPU or GPU without code changes.
- Providing homogenisation utilities for composite materials.

The target audience is researchers in computational mechanics, materials
science and machine-learning-for-engineering who need a differentiable
FEM backbone that integrates seamlessly with the PyTorch ecosystem.

## Authorship

**Authors:**

- Nils Meyer ([nils.meyer@uni-a.de](mailto:nils.meyer@uni-a.de)) —
  original author of the upstream `torch-fem` package.

**Maintainer of this fork:**

- Rui Barreira Morais Pinto
  ([rui_pinto@brown.edu](mailto:rui_pinto@brown.edu))

**Affiliations:**

- Bessa Research Group @ Brown University (School of Engineering)

## Getting started

See the [Getting started](getting_started.md) page for a detailed
installation guide and a minimal end-to-end example. The
[API reference](api.md) is auto-generated from the source docstrings and
covers every public module, class and function of the `torchfem`
package.

## Community Support

If you find any **issues, bugs, or problems** with this package, please
use the
[GitHub issue tracker](https://github.com/ruibmpinto/torch-fem/issues) to
report them. Pull requests are welcome; please consult the
[CONTRIBUTING](https://github.com/ruibmpinto/torch-fem/blob/main/CONTRIBUTING.md)
guidelines first.

## License

This project is distributed under the MIT License. The full license text
is available on the [License](license.md) page.
