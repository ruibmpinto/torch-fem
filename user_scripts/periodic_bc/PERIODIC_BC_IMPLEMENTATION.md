# Periodic Boundary Conditions Implementation

## Status
Implemented direct DOF elimination approach for periodic BCs (simpler than MPC master-slave elimination).

## Implementation Summary

### 1. DOF Mapping ([base.py:93-95](src/torchfem/base.py#L93-L95))
```python
self.dof_map = torch.arange(self.n_dofs, dtype=torch.long)
self.n_dofs_reduced = self.n_dofs
```

### 2. Set Periodic BC Method ([base.py:214-257](src/torchfem/base.py#L214-L257))
```python
fem.set_periodic_bc([
    (left_nodes, right_nodes, [0,1]),   # x,y periodic
    (bottom_nodes, top_nodes, [0,1])
])
```

Creates mapping: `dof_map[slave_dof] = master_dof`
Renumbers to contiguous range: `[0, n_dofs_reduced)`

### 3. Assembly with Mapping ([base.py:600](src/torchfem/base.py#L600))
```python
idx_mapped = self.dof_map[self.idx]
```

Assembles directly into reduced space (n_dofs_reduced × n_dofs_reduced).

## Remaining Tasks

1. **Update constraint handling** in assembly (line 620-629):
   - Map `con` to reduced space: `con_mapped = self.dof_map[con]`

2. **Update force assembly** ([base.py:640-657](src/torchfem/base.py#L640-L657)):
   - Use mapped indices
   - Initialize to `n_dofs_reduced` size

3. **Update solver** ([base.py:749-830](src/torchfem/base.py#L749-L830)):
   - Map forces/constraints to reduced space
   - Expand solution back to full space after solve

4. **Create test script** using direct API:
```python
fem.set_periodic_bc([
    (left_nodes, right_nodes, [0,1]),
    (bottom_nodes, top_nodes, [0,1])
])
```
