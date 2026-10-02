# Small-island fast path for the constraint solver

Status: design + phase 1 in progress. Branch `adenzler/island-fastpath`.

## Problem

The Newton solver treats each world as one problem: a dense `nv_pad x nv_pad` Hessian factored by
one CTA per world, one global step size, and iterations until the slowest part of the world
converges. Constraint islands (connected components of kinematic trees linked by constraints) are
independent subproblems, and the Hessian is block-diagonal across them.

The most common island is a single free body touching only static geometry: 6 DOFs, ~12-21
constraint rows. In settled scenes these dominate (pile256: 147 of 179 islands, 57% of DOFs;
aloha_clutter: 5-7 of 12 islands). Warm-started from the previous step, MuJoCo solves them in one
Newton iteration. Truly unconstrained trees (in no island) are rare (objects in flight).

## Design

Each world's problem is partitioned per step; the partition is exact because islands are
independent:

| class | definition | solve |
|---|---|---|
| U | trees in no island | `qacc = qacc_smooth`, `qfrc_constraint = 0`, no solve |
| S | island with `nv <= 8` and `nefc <= 64` (tunable) | fused fast-path kernel, each island solved to convergence |
| L | all other islands | existing Newton solver, restricted to L DOFs via compaction |

### Per step, after island discovery

1. **Bookkeeping.** Per-island DOF and constraint-row counts; constraint rows grouped by island
   (`map_iefc2efc`, `island_iefcadr`, `island_nefc`); per-world list of S islands.
2. **U.** Write `qacc_smooth` / zero constraint force for unconstrained DOFs.
3. **S fast path.** One warp per S island (launch `nworld x ntree` warps, early exit):
   - load the island's DOFs: M block (from sparse M), `qacc_smooth`, `qfrc_smooth`, warmstart;
   - load its rows into shared memory: J restricted to the island DOFs (<= 8 floats/row), D,
     aref, frictionloss, type, cone data;
   - Newton loop: lanes over rows compute Jaref, constraint state/force (incl. elliptic cones per
     contact); warp reductions form grad (<= 8) and H = M + J'DJ (+ cone curvature, <= 36 unique
     entries); 8x8 Cholesky + solve in registers; exact line search over the island's rows;
     convergence test scaled by the island's DOF count (as MuJoCo);
   - write `qacc`, `qfrc_constraint`, `efc.Ma` (implicit integrators), row force/state,
     `solver_niter` (max per world) and iteration-overflow flags.
4. **L.** Run the existing solver in compacted coordinates over L DOFs, generalizing the sleep
   compaction (compacted set = awake AND L). S and U rows are inert there: their Jacobian columns
   are outside the compacted space, so they only add a constant to the cost and nothing to the
   gradient, Hessian or line search. The S kernel runs after the L solve so S row forces written
   by the L solve are overwritten. Compacted DOFs are ordered by island, which later enables a
   block-structured (skyline or per-island tile pool) factorization for L.

### Scope (first version)

Newton only (CG keeps the current path); not the discrete integrator (its metric couples trees
through tendons and actuators outside islands); islands enabled; `ntree > 1`.

### Risks

- Two solver implementations: the S kernel must reuse the existing row-level functions
  (constraint state/force, line-search point evaluation, elliptic cone helpers) and differ only
  in data layout and reductions. Tested against the existing solver on the same islands.
- Results change: per-island line search and convergence move results toward MuJoCo and away from
  the current monolithic solve; `solver_niter` becomes the max over islands.
- Scenes that are entirely L pay the compaction gather/scatter; must be measured
  (three_humanoids).

## Phases

1. **Bookkeeping.** Make `compute_island_mapping` scale: the constraint-row grouping is
   O(nisland * nefc) (one thread per island scans all rows) and the size scan is a serial
   dependent loop. Replace with a deterministic chunked counting sort. Prerequisite for S; also
   speeds up the sleep path, which runs the mapping every step.
2. **U and L via generalized compaction.** Exclude U (and later S) DOFs from the global solve.
   Must match current results when S is empty, at similar cost.
3. **S fast path kernel.** Tested per island against the existing solver and MuJoCo.
4. **L factorization.** Block-structured factorization in the island-ordered compacted space
   (see `adenzler/island-skyline-analysis` for the skyline prototype and layout analysis).

## Phase 1 notes

Measured `compute_island_mapping` (graph-captured, per call):

| scene | nworld | nefc | nisland | total | `_island_map_constraints` | `_island_scan_sizes` |
|---|---|---|---|---|---|---|
| pile256 | 32 | 4068 | 178 | 405 us | 238 us | 42 us |
| pile64 | 512 | 1000 | 42 | 183 us | 112 us | 43 us |
| three_humanoids | 8192 | 96 | 3 | 99 us | | |

Library sorts are not an option: `segmented_sort_pairs` over worlds takes 120-2500 us and a global
`radix_sort_pairs` 120-250 us at these sizes, since both sort the full `njmax` capacity.

Counting sort: `make_constraint` emits rows grouped by type in MuJoCo's order (equality, friction,
limits, contacts; kernels run in sequence), so a stable sort of rows by island alone reproduces
MuJoCo's per-island layout (equality, friction, other; ascending within each). Rows are split
into fixed-size chunks; the counting pass also histograms rows per (world, chunk, island); an
exclusive scan over chunks gives each chunk's base offset per island; a row's rank within its
chunk is the number of earlier rows in the chunk with the same island. All steps are
deterministic and parallel.

Result (GPU kernel time per `compute_island_mapping` call, nsys; `njmax = nefc + 400`):

| scene | nworld | before | after |
|---|---|---|---|
| pile256 | 32 | 302 us | 38 us |
| pile64 | 512 | 180 us | 46 us |
| three_humanoids | 8192 | 99 us | 98 us |

The size scan is now a block-wide tile scan per world. What remains is dominated by the kernels
that run over every `njmax` row slot (`_init_efc_arrays`, `_island_count_constraints`,
`_compute_efc_tree`), which could be fused later.

## Phase 2 notes

Implemented as designed behind the experimental warp-only `opt.island_solve`: trees in no island
are excluded from the constraint solve (`qacc = qacc_smooth`, zero constraint force), the rest are
solved through the sleep path's compaction, and the compacted blocked Cholesky factors only
`round_up(ncdof + 1, 16)` rows per world instead of the static `nvmax_pad` (also used by the sleep
path).

Overhead with nothing to exclude (median of 3 interleaved runs, ns per world-step):

| scene | off | on |
|---|---|---|
| three_humanoids (nworld 8192) | 676 | 824 (-18%) |
| aloha_clutter awake (nworld 2048) | 5761 | 5961 (-3.5%) |
| pile64 / pile256 | within noise | |

Kernel attribution on three_humanoids (+41.6 ms over the profiled run): the compacted M*v
(`_mul_m_sparse_compact`, +23 ms vs -9 ms native), mapped variants of the DOF-space solver kernels
(~+20 ms), island discovery/mapping/compaction bookkeeping (~+12 ms), Cholesky (-5 ms).

Conclusion: running the whole solver in compacted coordinates is the wrong substrate for L. The
skyline prototype's approach, keeping the solver native and moving only H assembly and the
factorization into island-ordered coordinates (gather grad, scatter search), measured +6 ms on the
same scene. With L DOFs first in that order, the factorization can stop at `round_up(nL + 1, 16)`
and S/U DOFs get zero gradient and search, which is everything phase 3 needs from the L solve,
and the island order is what phase 4's block factorization needs.

Testing note: `test_data.fixture` warmstarts at the solution (`qacc_warmstart = qacc`), so solver
tests must reset the warmstart or they never exercise the factorization.

### Rework: island order for H only

`opt.island_solve` now keeps the solver in DOF coordinates. Only the Hessian is assembled and
factored in island order: rows are the island DOFs (`map_idof2dof`, from
`compute_island_mapping`); DOFs outside the factored block (unconstrained now, small islands in
phase 3) map to no row, start at their solution (`qacc_smooth` for now) and get a zero search
direction. The blocked Cholesky factors only `round_up(nidof + 1, 16)` rows; grad is gathered into
and search scattered out of the island order around it.

Overhead with nothing to exclude, same setup as above:

| scene | off | on |
|---|---|---|
| three_humanoids | 569 | 596 (-4.7%) |
| pile64 | 22076 | 22645 (-2.6%) |
| pile256 | 5776587 | 5768598 (0%) |
| aloha_clutter awake | 7135 | 7096 (0%) |

The rest is per-step island discovery and mapping, the map/gather/scatter launches, and the
mapped H assembly kernels.

## Phase 3a notes: small-island fast path (pyramidal cones)

`_small_island_solve`, one warp per island with <= 8 DOFs and <= 64 rows (each lane owns up to two
rows in registers; row sums are block reductions), runs before the main solve. It replicates the
Newton solver on the island: constraint states and forces (`_eval_constraint`), H = M + J'DJ over
quadratic rows, Cholesky in registers, the exact bracketing line search of
`_linesearch_iterative_kernel` (row costs through the same helpers), and the `_solve_done`
termination test scaled by the island's inertia trace (as MuJoCo's island solver does). It writes only `qacc`; the main
solve keeps those DOFs fixed (no H row, zero search) and computes their rows' forces, states and
`qfrc_constraint`. `solver_niter` is the max of the main solve and the small islands.

Elliptic cones are not on the fast path yet (their islands stay large).

Known approximation: the incremental solver's stable-state fast path rescales the whole stale
gradient along the ray, including the small islands' converged residual; the error is at that
residual's level.

Results (interleaved, median of 3, ns per world-step):

| scene | off | on |
|---|---|---|
| pile64 (nworld 512) | 25149 | 7559 (3.3x) |
| pile256 (nworld 32) | 7265992 | 561245 (12.9x) |
| three_humanoids (no small islands) | 698 | 735 (-5%) |
| aloha_clutter awake (elliptic) | 7744 | 8053 (-4%) |

Where the time goes with the fast path on: pile256 is 92% Cholesky of the large islands (~660
DOFs, factored densely; phase 4), pile64 is 47% Cholesky and 13% `_small_island_solve` (one thread
solves a whole island serially; a warp per island would cut its latency).

### Warp per island

The first version used one thread per island and recomputed J*qacc and J*search through two
indirections on every line-search pass. On aloha_clutter with pyramidal cones (8-DOF arm islands
with gripper contacts) it cost 1161 ms of kernel time over 300 steps against a 600 ms saving in the
main solve, making the step 15% slower. One warp per island with rows in registers and incremental
Jaref brings the kernel to 343 ms.

Line search at an island's solution: a resting island warm-started at its solution has a search
direction of ~1e-7 and a slope ~1e-16 at alpha = 0. MJWarp's line search requires a strict cost
decrease, which float32 noise never provides, so it ran into `ls_iterations` (MuJoCo accepts on the
slope alone). The fast path treats |slope(0)| < gtol as converged with alpha = 0.

Results (interleaved, median of 3, ns per world-step):

| scene | off | on |
|---|---|---|
| aloha_clutter pyramidal, 300 steps | 7712 | 7505 (1.03x) |
| aloha_clutter pyramidal, full replay | 8080 | 7082 (1.14x, noisy) |
| pile64 | 20396 | 7052 (2.9x) |
| pile256 | 5907526 | 432815 (13.7x) |
| three_humanoids (no small islands) | 546 | 580 (-6%) |
