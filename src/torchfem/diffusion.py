"""Scalar-field diffusion on a solid mesh.

Solves problems whose unknown is one scalar per node, either steady

    -div(k grad(phi)) = f

as in electrical conduction, or transient

    c dphi/dt - div(k grad(phi)) = f

as in heat conduction, on the same Tet4/Tet10/Hex8/Hex20 meshes the
``Solid`` element uses.

This is deliberately not a subclass of ``FEM``. That class carries one
displacement vector per node and uses ``n_dim`` both for the spatial
dimension and for the DOF count per node, which a scalar field does
not share. The element kernels are reached through ``self.etype``
exactly as ``Solid`` reaches them, so the two solvers integrate over
the same shape functions and quadrature rules.

The element conductivity matrix comes from
``operators.DiffusionOperator``, so the discretisation of the flux law
lives with the other physics operators rather than here.

Classes
-------
Diffusion
    Scalar-field diffusion problem on a solid mesh.
"""

import torch
from torch import Tensor

from .elements import Hexa1, Hexa2, Tetra1, Tetra2
from .operators import DiffusionOperator
from .sparse import CachedSolve, sparse_solve


class Diffusion:
    """Scalar-field diffusion problem on a solid mesh.

    Attributes:
        nodes (Tensor): Nodal coordinates, shape (n_nod, n_dim).
        elements (Tensor): Connectivity, shape (n_elem, n_nod_elem).
        etype (Element): Element kernel providing ``N`` and ``B``.
        n_nod (int): Number of nodes.
        n_elem (int): Number of elements.
        n_int (int): Integration points per element.
    """

    def __init__(self, nodes: Tensor, elements: Tensor):
        """Initialize the diffusion problem.

        Args:
            nodes (Tensor): Nodal coordinates of shape (n_nod, n_dim).
            elements (Tensor): Element connectivity of shape
                (n_elem, n_nodes_per_element).
        """
        self.nodes = nodes
        self.elements = elements
        # Problem size
        self.n_nod = nodes.shape[0]
        self.n_dim = nodes.shape[1]
        self.n_elem = len(elements)

        # Set element type depending on number of nodes per element
        if len(elements[0]) == 4:
            self.etype = Tetra1()
        elif len(elements[0]) == 8:
            self.etype = Hexa1()
        elif len(elements[0]) == 10:
            self.etype = Tetra2()
        elif len(elements[0]) == 20:
            self.etype = Hexa2()
        else:
            raise ValueError("Element type not supported.")

        # Integration points per element
        self.n_int = len(self.etype.iweights())
        # Assembly indices: each element fills a dense block of its nodes
        n_loc = elements.shape[1]
        self.rows = elements.unsqueeze(-1).expand(-1, -1, n_loc).reshape(-1)
        self.cols = elements.unsqueeze(-2).expand(-1, n_loc, -1).reshape(-1)

    def eval_shape_functions(self, xi: Tensor):
        """Shape functions, Cartesian gradients and Jacobian at xi.

        Args:
            xi (Tensor): Natural coordinates of one integration point.

        Returns:
            tuple[Tensor, Tensor, Tensor]: Shape functions, gradient
                operator of shape (n_elem, n_dim, n_nod_elem), and the
                Jacobian determinant of shape (n_elem,).
        """
        # Coordinates of the nodes of every element
        nodes = self.nodes[self.elements, :]
        # The element kernels follow the dtype of xi, and the stored
        # quadrature points are single precision literals
        xi = torch.as_tensor(xi, dtype=nodes.dtype, device=nodes.device)
        # Shape function derivatives in natural coordinates
        b = self.etype.B(xi)
        # Jacobian of the map from natural to physical coordinates
        J = torch.einsum("jk,mkl->mjl", b, nodes)
        detJ = torch.linalg.det(J)
        # A non-positive Jacobian means the element is inverted
        if torch.any(detJ <= 0.0):
            raise Exception("Negative Jacobian. Check element numbering.")
        # Pull the derivatives back to physical coordinates
        B = torch.einsum("jkl,lm->jkm", torch.linalg.inv(J), b)
        return self.etype.N(xi), B, detJ

    def integrate_shape_functions(self) -> Tensor:
        """Integral of every shape function over its element.

        Returns the nodal shares of the element volume, which is what a
        row-lumped capacitance and a consistent source vector need. The
        shares are integrated rather than assumed uniform, since on a
        distorted element they are not.

        Returns:
            Tensor: Nodal weights of shape (n_elem, n_nod_elem), summing
                per element to the element volume.
        """
        weights = None
        # Accumulate the quadrature sum over the element
        for w, xi in zip(*self.etype.integration_rule(self.nodes.dtype),
                         strict=False):
            N, _, detJ = self.eval_shape_functions(xi)
            # Volume this point contributes to each of its nodes
            contribution = torch.einsum("e,a->ea", w * detJ,
                                        N.to(detJ.dtype))
            weights = (contribution if weights is None
                       else weights + contribution)
        return weights

    def stiffness(self, conductivity: Tensor) -> Tensor:
        """Assemble the global conductivity matrix.

        Args:
            conductivity (Tensor): Per-element conductivity of shape
                (n_elem,).

        Returns:
            Tensor: Sparse COO matrix of shape (n_nod, n_nod).
        """
        if conductivity.shape != (self.n_elem,):
            raise ValueError(
                f"conductivity has shape {tuple(conductivity.shape)}, "
                f"expected ({self.n_elem},)."
            )
        # The flux law supplies the element kernel k * B^T B
        operator = DiffusionOperator(conductivity, self.n_dim)
        blocks = None
        # Accumulate the quadrature sum over the element
        for w, xi in zip(*self.etype.integration_rule(self.nodes.dtype),
                         strict=False):
            _, B, detJ = self.eval_shape_functions(xi)
            contribution = operator.assemble_element_stiffness(
                B, None, detJ, float(w))
            blocks = (contribution if blocks is None
                      else blocks + contribution)
        # Duplicate entries from shared nodes are summed by coalesce
        return torch.sparse_coo_tensor(
            torch.stack([self.rows, self.cols]), blocks.reshape(-1),
            size=(self.n_nod, self.n_nod)).coalesce()

    def _constrain(self, matrix: Tensor, rhs: Tensor, con: Tensor,
                   values: Tensor):
        """Eliminate constrained DOFs symmetrically.

        The constrained columns are moved to the right-hand side before
        the rows and columns are dropped, so the reduced operator stays
        symmetric and the solution carries the prescribed values
        directly, with no scatter step afterwards.

        Args:
            matrix (Tensor): Sparse COO operator.
            rhs (Tensor): Right-hand side of shape (n_nod,).
            con (Tensor): Constrained node indices.
            values (Tensor): Prescribed values, same shape as ``con``.

        Returns:
            tuple[Tensor, Tensor]: Constrained operator and right-hand
                side.
        """
        matrix = matrix.coalesce()
        rows, cols = matrix.indices()
        vals = matrix.values()
        # Flag the constrained nodes
        mask = torch.zeros(self.n_nod, dtype=torch.bool, device=rhs.device)
        mask[con] = True
        # Prescribed value at every node, zero where unconstrained
        prescribed = torch.zeros_like(rhs)
        prescribed[con] = values
        # Known columns become a load on the rows that survive
        lifted = torch.where(mask[cols] & ~mask[rows],
                             vals * prescribed[cols],
                             torch.zeros_like(vals))
        rhs = rhs - torch.zeros_like(rhs).index_add(0, rows, lifted)
        # Constrained equations read phi = value
        rhs = torch.where(mask, prescribed, rhs)
        # Drop every entry in a constrained row or column
        keep = ~(mask[rows] | mask[cols])
        # Restore a unit diagonal on the constrained equations
        rows = torch.cat([rows[keep], con])
        cols = torch.cat([cols[keep], con])
        vals = torch.cat([vals[keep], torch.ones_like(values)])
        matrix = torch.sparse_coo_tensor(
            torch.stack([rows, cols]), vals,
            size=(self.n_nod, self.n_nod)).coalesce()
        return matrix, rhs

    def solve(
        self,
        conductivity: Tensor,
        constraints: Tensor,
        constraint_values: Tensor,
        source: Tensor | None = None,
        capacity: Tensor | None = None,
        previous: Tensor | None = None,
        dt: float | None = None,
        method: str = "cholmod",
        cached_solve: CachedSolve | None = None,
    ) -> Tensor:
        """Solve one steady or transient diffusion problem.

        Supplying ``capacity``, ``previous`` and ``dt`` together selects a
        backward-Euler step of the transient equation; leaving all three
        out solves the steady problem. The capacitance is row-lumped to
        the diagonal, so the step stays a single solve on the same
        sparsity pattern as the steady operator.

        Args:
            conductivity (Tensor): Per-element conductivity, shape
                (n_elem,).
            constraints (Tensor): Constrained node indices.
            constraint_values (Tensor): Prescribed nodal values.
            source (Tensor, optional): Per-element volumetric source of
                shape (n_elem,). Defaults to no source.
            capacity (Tensor, optional): Per-element volumetric capacity
                of shape (n_elem,). Transient only.
            previous (Tensor, optional): Nodal field at the start of the
                step, shape (n_nod,). Transient only.
            dt (float, optional): Time step. Transient only.
            method (str): Linear solver backend. Defaults to "cholmod",
                which is exact to round-off on this SPD operator.
            cached_solve (CachedSolve, optional): Cache reusing the
                symbolic factorisation across solves that share a
                sparsity pattern, as when only conductivity changes.

        Returns:
            Tensor: Nodal field of shape (n_nod,), carrying the
                prescribed values at the constrained nodes.
        """
        # The three transient arguments are meaningless apart
        transient = [capacity is not None, previous is not None,
                     dt is not None]
        if any(transient) and not all(transient):
            raise ValueError(
                "A transient step needs capacity, previous and dt "
                "together; a steady solve needs none of them."
            )
        matrix = self.stiffness(conductivity)
        rhs = torch.zeros(self.n_nod, dtype=self.nodes.dtype,
                          device=self.nodes.device)
        # Nodal volume shares, needed by the source and the capacitance
        node_weights = None
        if source is not None or all(transient):
            node_weights = self.integrate_shape_functions()
        # Scatter target for per-element quantities
        flat = self.elements.reshape(-1)
        # Consistent nodal load from the volumetric source
        if source is not None:
            rhs = rhs.index_add(
                0, flat, (source[:, None] * node_weights).reshape(-1))
        if all(transient):
            # Row-lumped capacitance over the step
            lumped = torch.zeros_like(rhs).index_add(
                0, flat, (capacity[:, None] * node_weights).reshape(-1))
            lumped = lumped / float(dt)
            rhs = rhs + lumped * previous
            # Backward Euler adds it to the diagonal
            diagonal = torch.arange(self.n_nod, device=rhs.device)
            matrix = (matrix + torch.sparse_coo_tensor(
                torch.stack([diagonal, diagonal]), lumped,
                size=(self.n_nod, self.n_nod))).coalesce()
        # Impose the prescribed values before solving
        matrix, rhs = self._constrain(matrix, rhs,
                                      constraints.to(torch.long),
                                      constraint_values.to(rhs.dtype))
        # A transient cache still solves, it just cannot reuse anything
        if cached_solve is None:
            cached_solve = CachedSolve()
        return sparse_solve(matrix, rhs, None, 1e-10, None, method, None,
                            cached_solve, False)

    def flux(self, field: Tensor,
             conductivity: Tensor) -> tuple[Tensor, Tensor]:
        """Flux of a nodal field at the integration points.

        Fourier's and Ohm's laws share the form ``q = -k grad(phi)``, so
        this returns heat flux, current density or diffusive flux
        according to what the field and the conductivity are.

        Args:
            field (Tensor): Nodal field of shape (n_nod,).
            conductivity (Tensor): Per-element conductivity of shape
                (n_elem,).

        Returns:
            tuple[Tensor, Tensor]: Flux of shape (n_int, n_elem, n_dim)
                and the integration weights of shape (n_int, n_elem).
        """
        gradients, weights = self.gradient(field)
        # Flux runs down the gradient, hence the sign
        return -conductivity[None, :, None] * gradients, weights

    def power(self, field: Tensor, conductivity: Tensor) -> Tensor:
        """Total dissipated power of a nodal field.

        Evaluates ``phi^T K phi`` on the unconstrained operator, which
        is the dissipation the field actually carries. Taking it from
        the constrained operator instead would miss the boundary
        couplings that elimination removed.

        Args:
            field (Tensor): Nodal field of shape (n_nod,).
            conductivity (Tensor): Per-element conductivity of shape
                (n_elem,).

        Returns:
            Tensor: Scalar dissipated power.
        """
        return field @ self.reaction(field, conductivity)

    def reaction(self, field: Tensor, conductivity: Tensor) -> Tensor:
        """Nodal reaction ``K phi`` of a nodal field.

        Summing this over a constrained node set gives the total flow
        through that set, which is how a current or a heat rate is
        recovered from a solved potential.

        Args:
            field (Tensor): Nodal field of shape (n_nod,).
            conductivity (Tensor): Per-element conductivity of shape
                (n_elem,).

        Returns:
            Tensor: Nodal reaction of shape (n_nod,).
        """
        matrix = self.stiffness(conductivity).coalesce()
        rows, cols = matrix.indices()
        # Sparse matrix-vector product, kept differentiable
        return torch.zeros_like(field).index_add(
            0, rows, matrix.values() * field[cols])

    def gradient(self, field: Tensor) -> tuple[Tensor, Tensor]:
        """Cartesian gradient of a nodal field at the element level.

        Args:
            field (Tensor): Nodal field of shape (n_nod,).

        Returns:
            tuple[Tensor, Tensor]: Gradient of shape
                (n_int, n_elem, n_dim) and the integration weights of
                shape (n_int, n_elem).
        """
        # Gather the nodal values onto their elements
        field_elem = field[self.elements]
        gradients = []
        weights = []
        # The gradient lives at the integration points, not the nodes
        for w, xi in zip(*self.etype.integration_rule(self.nodes.dtype),
                         strict=False):
            _, B, detJ = self.eval_shape_functions(xi)
            gradients.append(torch.einsum("eik,ek->ei", B, field_elem))
            weights.append(w * detJ)
        return torch.stack(gradients), torch.stack(weights)
