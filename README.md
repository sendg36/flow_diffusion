# flow_diffusion

Finite element solvers for the convection–diffusion equation, built from scratch
in self-contained C++17 — no external dependencies, no dense-matrix fallback:
matrices live in CSR from assembly to solve.

## Projects

### flow_diffusion_1D

One-dimensional stepping-stone project: 1D convection–diffusion on a uniform
grid, the same equation ladder in its simplest form.

### flow_diffusion_2D

Two-dimensional solver for

    -eps * div(grad(u)) + b . grad(u) + c * u = f

on a rectangular domain, P1 triangular elements, standard Galerkin. This is the
main teaching/acceptance project (tests T1–T8, assert-based).

- **Mesh** (`Mesh.h`): structured triangular mesh (each cell split into two
  counter-clockwise triangles), boundary nodes tagged with bit flags
  (`LEFT | RIGHT | BOTTOM | TOP`). No coordinate comparisons outside the mesh —
  this is what later lets a Gmsh importer plug in.
- **Element** (`Element.h`): closed-form P1 matrices in barycentric coordinates.
  The diffusion matrix is computed two independent ways (Gram matrix of
  `grad(lambda)` vs. the cotangent formula) and cross-checked to 1e-12 in T1.
- **Assembler** (`Assembler.h`): CSR assembly with a set-built sparsity pattern.
  Dirichlet (column shifting + row wipe), Neumann (right-hand side edge loads)
  and Robin (edge load + 2x2 boundary mass block) are all supported; natural
  boundary terms are loaded *before* essential rows are sealed, so corner nodes
  belonging to two boundary types are handled gracefully.
- **Solvers** (`Solver.h`):
  - `SparseLU` — sparse direct solver working on the CSR structure directly
    (right-looking row-wise elimination, partial pivoting, fill-in absorbed
    dynamically). Used as the reference ("ground truth") for the iterative
    solvers, and reports its fill-in (`nnz(L+U)/nnz(A)`).
  - `cg` / `bicgstab` — Krylov iterative solvers with optional Jacobi
    preconditioning, residual history recorded.
- **Post** (`Post.h`): L2 error by midpoint quadrature.

### flow_diffusion_2D_io

Interface-hardened variant of the 2D solver: the kernel files are kept
byte-identical and wrapped with real I/O — a locked Gmsh reader (`.msh` 2.2),
INI-style boundary condition input with expression evaluation, an abstract
`LinearSolver` slot (own CG/BiCGStab implementation, Eigen backend behind
an `#ifdef`), VTK output for ParaView, and an OpenMP matvec. Acceptance tests
T7–T11 live in `test_interfaces.cpp`.

## Build & run

Linux / macOS (2D teaching version):

```bash
g++ -O2 -std=c++17 flow_diffusion_2D/main.cpp -o fem2d
./fem2d
```

Windows (Visual Studio 2022): open `flow_diffusion.slnx`, build any of the
three projects as x64. The sources are UTF-8 with Chinese comments — either save them
with BOM or add `/utf-8` to the compiler flags, otherwise MSVC reads them as GBK
and may swallow the line after a comment whose last byte is 0x5C.

Expected tail of the 2D acceptance output:

```
[T8] PASS
========== ALL TESTS PASSED ==========
```

## What the 2D acceptance tests cover

| Test | Content |
|------|---------|
| T1 | Element-level cross-check: grad-lambda vs cotangent diffusion matrix, hand-checked convection / mass / load entries |
| T2 | Poisson on a 4x4 mesh, three interior nodal values against reference (0.948187 / 0.670469 / 0.481621) |
| T3 | Convergence table, L2 error 0.2763 -> 0.0057, rate -> 2 (CG + Jacobi) |
| T4 | CG / Jacobi-CG / BiCGStab against SparseLU; plain vs Jacobi iteration counts on a variable-coefficient problem |
| T5 | 2:1 rectangular domain — no hidden "unit square, nx == ny" assumptions |
| T6 | Convection-dominated case (eps = 0.01): CG stalls on the non-symmetric matrix, BiCGStab converges; both checked against SparseLU |
| T7 | Mixed boundary conditions (Dirichlet + Neumann + Robin), rate ~ 2 |
| T8 | Pure-Neumann singularity: CG diverges when no node is pinned (residual blows up), converges after pinning one corner |

Sample fill-in numbers from SparseLU (natural ordering, no AMD — the growth is
the point): Poisson nx=16 -> 3.5x, nx=32 variable-coefficient -> 8.2x. The direct
solver is the referee; Krylov methods do the large-scale work.

## Roadmap

- Newton solver for nonlinear problems, plugged into the same solver slot
- MPI parallelization (the OpenMP matvec covers single-node only so far)

## License

MIT, see [LICENSE](LICENSE).
