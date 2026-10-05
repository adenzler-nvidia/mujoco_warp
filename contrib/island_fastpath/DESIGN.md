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
slope alone). The main solver has the same stall (google-deepmind/mujoco_warp#1749,
newton-physics/newton#4212); the fast path uses the same rule as that fix: a slope at alpha = 0
below the rounding bound `_ALPHA_NOISE_EPS * q1_abs` is converged with alpha = 0. (A first version
used |slope(0)| < gtol, which would accept the zero step under a loose tolerance, the case #1471
fixed.)

Results (interleaved, median of 3, ns per world-step):

| scene | off | on |
|---|---|---|
| aloha_clutter pyramidal, 300 steps | 7712 | 7505 (1.03x) |
| aloha_clutter pyramidal, full replay | 8080 | 7082 (1.14x, noisy) |
| pile64 | 20396 | 7052 (2.9x) |
| pile256 | 5907526 | 432815 (13.7x) |
| three_humanoids (no small islands) | 546 | 580 (-6%) |

## Phase 4a notes: skyline factorization of the large islands

With island_solve, h rows are the large-island DOFs in island order, so the Hessian is block
diagonal by island. The blocked Cholesky (factor and solve) now takes a per-tile envelope
`ctx.cholesky_top` (first nonzero row of each tile column; all zeros reproduces the dense
algorithm) and skips tiles above it; fill stays inside the envelope. For the island order it is
the first row of the island owning each tile's first row (`_island_h_top`). The loop bounds and the
NaN-poisoning kernel test come from the skyline prototype (`adenzler/island-skyline-analysis`).

Interleaved, median of 3, ns per world-step (three_humanoids noisy, GPU shared at the end):

| scene | island_solve off | on, before 4a | on, 4a |
|---|---|---|---|
| pile64 (nworld 512) | 25222 | 7977 | 6511 (3.9x vs off) |
| pile256 (nworld 32) | 8058485 | 565892 | 69488 (116x vs off) |
| aloha_clutter pyramidal | 9081 | 9028 | 8826 |
| aloha_clutter elliptic | 7692 | 8591 | 7419 |
| three_humanoids | 737 | 1060 | 844 |

pile64 kernel time (quiet GPU): Cholesky 239 -> 78 ms; the largest items are now
`_small_island_solve` (173 ms, 31%), Cholesky (78 ms) and `_update_gradient_init_h_sparse` (68 ms,
12%, still writes all nv_pad^2 entries). pile256: Cholesky 27%, init_h 17%, small islands 16% of a
much smaller total; with 32 worlds the per-world serial Cholesky chain is the latency (phase 4b).

### H initialization within the envelope

`_update_gradient_init_h_sparse` launched one thread per h entry (nworld * nv_pad^2, 82M threads for
pile64) and wrote the whole upper triangle every Newton step. For the blocked path with known
active rows (island order and the compact sleep path), `_update_gradient_init_h_envelope` now uses
one thread per row and writes only what the blocked Cholesky reads: the active rows, the row's
tile envelope (contiguous, since top is nondecreasing) and the last tile column (it carries the
right-hand side). Skipping entries in the per-entry kernel instead only cut 17%; the launch size was
the cost.

Kernel time, quiet GPU (4a -> with envelope init):

| scene | H init | total kernel time |
|---|---|---|
| pile64 | 65.2 -> 10.7 ms | 545 -> 506 ms |
| pile256 | ~13 -> 1.3 ms | 78 -> 63 ms |
| aloha_clutter pyramidal | 59.1 -> 39.5 ms | 4033 -> 3906 ms |

Wall clock was too noisy to resolve the aloha change (shared GPU); pile64 1.12x and pile256 1.19x.

### Small-island kernel: shared memory instead of per-lane matrices

Nsight Compute on pile64 showed `_small_island_solve` moving 2.35 GB of local-memory loads and
2.71 GB of stores per launch against 125 MB of global loads: every lane kept its own 8x8 inertia,
Hessian, Cholesky and J'DJ partial and indexed them with runtime indices, which spilled them (235
registers, 16.7% theoretical occupancy).

Now the island's inertia, Hessian (upper triangle) and constraint force live once in shared memory;
lanes add their rows into them with shared atomics (`tile_scatter_add`; every call ends in a block
barrier, so all lanes make the same calls with a mask), and the small DOF vectors, row data and the
8x8 Cholesky use fixed-size loops, so their indices are constants and they stay in registers.
Local-memory traffic is 0, registers 167, theoretical occupancy 25%, one launch 1.10 -> 0.77 ms.

| scene | small-island kernel | total kernel | wall clock |
|---|---|---|---|
| pile64 | 173 -> 99 ms | 493 -> 424 ms | 1.22x |
| pile256 | | | 1.14x |
| aloha_clutter pyramidal | 313 -> 284 ms | 3881 -> 3856 ms | 1.01x |

aloha's small islands (8-DOF arms with gripper contacts) spend relatively more time in line-search
reductions than in the Hessian work this targeted.
