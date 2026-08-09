import pytest
import torch

from torchfem.elements import (
    Hexa1,
    Hexa2,
    Quad1,
    Quad2,
    Tetra1,
    Tetra2,
    Tria1,
    Tria2,
)

# Test elements with quad shape and area 1
test_quad1 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
test_quad2 = torch.tensor(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [1.0, 0.5],
        [0.5, 1.0],
        [0.0, 0.5],
    ]
)

# Test elements with tria shape and area 1
test_tria1 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
test_tria2 = torch.tensor(
    [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]
)

# Test elements with tetra shape and volume 1
test_tetra1 = torch.tensor(
    [[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
)
test_tetra2 = torch.tensor(
    [
        [3.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.5, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.5, 0.0, 0.0],
        [1.5, 0.0, 0.5],
        [0.0, 1.0, 0.5],
        [0.0, 0.0, 0.5],
    ]
)

# Test elements with hexa shape and volume 1
test_hexa1 = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
)
test_hexa2 = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.5, 0.0, 0.0],
        [1.0, 0.5, 0.0],
        [0.5, 1.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.5, 0.0, 1.0],
        [1.0, 0.5, 1.0],
        [0.5, 1.0, 1.0],
        [0.0, 0.5, 1.0],
        [0.0, 0.0, 0.5],
        [1.0, 0.0, 0.5],
        [1.0, 1.0, 0.5],
        [0.0, 1.0, 0.5],
    ]
)


@pytest.mark.parametrize(
    "elem, nodes",
    [
        (Quad1(), test_quad1),
        (Quad2(), test_quad2),
        (Tria1(), test_tria1),
        (Tria2(), test_tria2),
        (Tetra1(), test_tetra1),
        (Tetra2(), test_tetra2),
        (Hexa1(), test_hexa1),
        (Hexa2(), test_hexa2),
    ],
)
def test_jacobian(elem, nodes):
    # The node fixtures are built at import time and therefore in
    # whatever the default dtype was then, while ipoints() follows
    # the default at call time; other test modules set it globally.
    # Align them explicitly so the test does not depend on import
    # order.
    nodes = nodes.to(elem.ipoints().dtype)
    volume = torch.zeros(1, dtype=elem.ipoints().dtype)
    for w, q in zip(elem.iweights(), elem.ipoints(), strict=False):
        J = elem.B(q) @ nodes
        detJ = torch.linalg.det(J)
        volume += w * detJ
    assert torch.allclose(
        volume, torch.ones(1, dtype=volume.dtype), atol=1e-5)


@pytest.mark.parametrize(
    "elem",
    [Quad1(), Quad2(), Tria1(), Tria2(), Tetra1(), Tetra2(), Hexa1(), Hexa2()]
)
def test_gradient(elem):
    for q in elem.ipoints():
        q.requires_grad = True
        for i in range(elem.nodes):
            grad = torch.autograd.grad(elem.N(q)[i], q)[0]
            assert torch.allclose(grad, elem.B(q)[:, i], atol=1e-5)
