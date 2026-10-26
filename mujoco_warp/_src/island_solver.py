# Copyright 2025 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Newton constraint solver over islands (opt.island_solve).

make_constraint writes an island's rows contiguously. Single trees of up to island.dense_width(m)
DOFs store their Jacobian as dense rows (island-local columns) and are solved by the dense world
solver; other groups keep sparse rows. On the island solver's block path, a group's DOFs are
gathered into tile-aligned rows, and its inertia and Hessian are dense blocks of _TILE x _TILE tiles
in a per-world pool. Every island is solved with its own step size
and convergence test, as MuJoCo's island solver does. Kernels run over the work that exists: rows,
DOFs, islands and the tile pairs of each island's Hessian.
"""

import dataclasses

import warp as wp

from mujoco_warp._src import island
from mujoco_warp._src import math
from mujoco_warp._src import types
from mujoco_warp._src.block_cholesky import create_pool_cholesky_newton_func
from mujoco_warp._src.block_cholesky import solve_search_sums
from mujoco_warp._src.solver import _JTDAJ_THREADS_PER_GROUP
from mujoco_warp._src.solver import _elliptic_hessian_entry_from_projections
from mujoco_warp._src.solver import _eval_constraint
from mujoco_warp._src.solver import _JTDACJ_sparse
from mujoco_warp._src.solver import _jtdaj_groups_per_world
from mujoco_warp._src.solver import _linesearch_iterative_kernel
from mujoco_warp._src.types import OverflowType
from mujoco_warp._src.warp_util import cache_kernel

wp.set_module_options({"enable_backward": False})

# the island tile pool's tile size (H, its factor, aligned DOF rows)
_TILE = 32

_ROW_EQUALITY = 0
_ROW_FRICTION = 1
_ROW_OTHER = 2


def _island_pool_size(nv: int, ntree: int, tile: int) -> tuple[int, int]:
  """Tiles and aligned rows the tile pool needs at most.

  A group of n DOFs takes ceil(n / tile) <= (n + tile - 1) / tile tiles per side; summed over at
  most ntree groups of nv DOFs in all.
  """
  ntile = -(-(nv * nv + 2 * (tile - 1) * nv + (tile - 1) ** 2 * ntree) // (tile * tile))
  nrow = -(-(nv + (tile - 1) * ntree) // tile) * tile
  return ntile, nrow


# threads per group of the per-group kernels (factorization as _update_gradient_cholesky)
_ISLAND_THREADS = 64


@dataclasses.dataclass
class IslandContext:
  """Per-step island-ordered solver state (see the module docstring)."""

  # layout: tiles per side of the groups this solver solves (0 for dense and sleeping groups), and
  # the aligned DOF rows a world uses
  tilebase: wp.array2d[int]
  ntile: wp.array2d[int]
  nrow_used: wp.array[int]
  vbase: wp.array2d[int]
  row2dof: wp.array2d[int]
  dof2row: wp.array2d[int]
  dof_poolrow: wp.array2d[int]
  dof_local: wp.array2d[int]
  # DOFs in tile-aligned island rows (padding is zero)
  qacc: wp.array2d[float]
  Ma: wp.array2d[float]
  qfrc_smooth: wp.array2d[float]
  frc: wp.array2d[float]
  grad: wp.array3d[float]
  search: wp.array3d[float]
  mv: wp.array2d[float]
  # per constraint row (make_constraint writes rows in island order)
  row_jaref: wp.array2d[float]
  row_jv: wp.array2d[float]
  # whether a row's curvature is in H (H is updated only where this flips), and the weight of
  # its J'J term in the next H update (D or -D where it flipped, else 0)
  row_quad: wp.array2d[bool]
  row_w: wp.array2d[float]
  # elliptic cones: per-contact quadratics along the search (see _cone_quad)
  row_cone: wp.array2d[wp.vec3]
  # per-island dense blocks: Hessian and its factor
  H: wp.array3d[float]
  U: wp.array3d[float]
  # per-island solver state
  scale: wp.array2d[float]
  done: wp.array2d[bool]
  niter: wp.array2d[int]
  alpha: wp.array2d[float]
  improvement: wp.array2d[float]
  search_dot: wp.array2d[float]
  newton_decrement: wp.array2d[float]
  # whether a row of an island flipped at the last evaluation (its H and factor are updated)
  changed: wp.array2d[bool]
  # whether a row of the island changed state at the last evaluation; without a change the quadratic
  # piece is the same, so the search direction is kept and grad = grad_scale * (stored grad), as in
  # the world solver's stable-state fast path
  state_changed: wp.array2d[bool]
  # the search direction is unchanged: the group is done or stable (as the world solver's)
  search_unchanged: wp.array2d[bool]
  grad_scale: wp.array2d[float]
  # compacted lists of (world * ntree + group), see _group_layout
  group_count: wp.array[int]
  group_list: wp.array2d[int]
  nsolving: wp.array[int]
  world_done: wp.array[bool] = None  # all False: the initial J'DJ of every world


def supported(m: types.Model) -> bool:
  """Whether the island solver covers this model's options (make_constraint writes island rows)."""
  return island.island_rows(m)


@wp.func
def _row_category(constraint_type: int) -> int:
  if constraint_type == types.ConstraintType.EQUALITY:
    return _ROW_EQUALITY
  if constraint_type == types.ConstraintType.FRICTION_DOF or constraint_type == types.ConstraintType.FRICTION_TENDON:
    return _ROW_FRICTION
  return _ROW_OTHER


@wp.func
def _pool_rc(tilebase: int, ntile: int, la: int, lb: int) -> wp.vec2i:
  """Pool (row, column) of entry (la, lb) of an island block, read from its upper triangle."""
  i = wp.min(la, lb)
  j = wp.max(la, lb)
  return wp.vec2i(_TILE * (tilebase + (i // _TILE) * ntile + j // _TILE) + i % _TILE, j % _TILE)


@wp.func
def _stable(niter: int, state_changed: bool) -> bool:
  """No row changed state since the last step: the search direction is kept."""
  return niter > 0 and not state_changed


@wp.func
def _ntile(ndof: int) -> int:
  return (ndof + _TILE - 1) // _TILE


@cache_kernel
def _group_layout(ntree: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # In:
    ngroup_in: wp.array[int],
    group_nv_in: wp.array2d[int],
    group_slot_in: wp.array2d[wp.vec4i],
    group_asleep_in: wp.array2d[bool],
    # Out:
    group_count_out: wp.array[int],
    group_list_out: wp.array2d[int],
    done_out: wp.array2d[bool],
    ntile_out: wp.array2d[int],
    scale_out: wp.array2d[float],
    tilebase_out: wp.array2d[int],
    vbase_out: wp.array2d[int],
    nrow_used_out: wp.array[int],
    row2dof_out: wp.array2d[int],
    iqacc_out: wp.array2d[float],
    iqfrc_smooth_out: wp.array2d[float],
    ifrc_out: wp.array2d[float],
    search_unchanged_out: wp.array2d[bool],
  ):
    """A world's groups: the compacted lists of the awake ones, their tiles and their layout.

    Lists 0-3: all, and by tiles per side (1, 2, >= 3); count 4: groups of several tiles. Per group:
    first pool tile and first aligned DOF row. The rows a world uses start as padding (zero,
    unmapped); _dofs fills the groups' DOFs.
    """
    worldid, lane = wp.tid()
    lanes = wp.block_dim()
    ngroup = ngroup_in[worldid]
    for g in range(lane, ntree, lanes):
      ntile = int(0)
      scale_out[worldid, g] = 0.0
      # _done sets the island solver's groups'
      search_unchanged_out[worldid, g] = True
      if g < ngroup:
        packed = worldid * ntree + g
        if group_asleep_in[worldid, g]:
          done_out[worldid, g] = True
        else:
          group_list_out[0, wp.atomic_add(group_count_out, 0, 1)] = packed  # kernel_analyzer: ignore[atomic]
          ntile = _ntile(group_nv_in[worldid, g])
          c = wp.min(ntile, 3)
          group_list_out[c, wp.atomic_add(group_count_out, c, 1)] = packed  # kernel_analyzer: ignore[atomic]
          if ntile > 1:
            wp.atomic_add(group_count_out, 4, 1)  # kernel_analyzer: ignore[atomic]
      ntile_out[worldid, g] = ntile
    _sync()
    tiles = wp.tile_load(ntile_out[worldid], shape=ntree)
    wp.tile_store(tilebase_out[worldid], wp.tile_scan_exclusive(wp.tile_map(wp.mul, tiles, tiles)))
    wp.tile_store(vbase_out[worldid], wp.tile_scan_exclusive(tiles) * _TILE)
    nrow = wp.tile_sum(tiles)[0] * _TILE
    if lane == 0:
      nrow_used_out[worldid] = nrow
    for row in range(lane, nrow, lanes):
      row2dof_out[worldid, row] = -1
      iqacc_out[worldid, row] = 0.0
      iqfrc_smooth_out[worldid, row] = 0.0
      ifrc_out[worldid, row] = 0.0

  return kernel


@cache_kernel
def _init_rows(width: int):
  """width: of the dense rows (rows are zero past a group).

  The loop over a row unrolls (max_unroll, 16 by default, would keep a loop over 28 columns
  dynamic), so a row's loads are in flight together.
  """

  @wp.kernel(module="unique", enable_backward=False, module_options={"max_unroll": width})
  def kernel(
    # Data in:
    nefc_in: wp.array[int],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    efc_aref_in: wp.array2d[float],
    efc_island_in: wp.array2d[int],
    # In:
    island_slot_in: wp.array2d[wp.vec4i],
    vbase_in: wp.array2d[int],
    dof2row_in: wp.array2d[int],
    iJ_in: wp.array3d[float],
    iqacc_in: wp.array2d[float],
    # Out:
    row_jaref_out: wp.array2d[float],
  ):
    """J qacc - aref of a row at the start point, a thread per row as _solve_init_jaref."""
    worldid, row = wp.tid()
    if row < wp.min(nefc_in[worldid], efc_aref_in.shape[1]):
      islandid = efc_island_in[worldid, row]
      block = island_slot_in[worldid, islandid]
      jaref = -efc_aref_in[worldid, row]
      if block[0] > 0:
        vbase = vbase_in[worldid, islandid]
        for c in range(wp.static(width)):
          jaref += iJ_in[worldid, row, c] * iqacc_in[worldid, vbase + c]
      else:
        rowadr = efc_J_rowadr_in[worldid, row]
        for k in range(efc_J_rownnz_in[worldid, row]):
          adr = rowadr + k
          jaref += efc_J_in[worldid, 0, adr] * iqacc_in[worldid, dof2row_in[worldid, efc_J_colind_in[worldid, 0, adr]]]
      row_jaref_out[worldid, row] = jaref

  return kernel


@cache_kernel
def _jv(width: int):
  """J search of a row for the linesearch, a thread per row as _linesearch_jv_fused_kernel.

  Groups whose search direction is unchanged (done or stable) keep it. The loop over a
  dense row unrolls, as in _init_rows.
  """

  @wp.kernel(module="unique", enable_backward=False, module_options={"max_unroll": width})
  def kernel(
    # Data in:
    nefc_in: wp.array[int],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    efc_island_in: wp.array2d[int],
    # In:
    island_slot_in: wp.array2d[wp.vec4i],
    vbase_in: wp.array2d[int],
    dof2row_in: wp.array2d[int],
    iJ_in: wp.array3d[float],
    search_in: wp.array2d[float],
    search_unchanged_in: wp.array2d[bool],
    # Out:
    row_jv_out: wp.array2d[float],
  ):
    worldid, row = wp.tid()
    if row >= wp.min(nefc_in[worldid], row_jv_out.shape[1]):
      return
    g = efc_island_in[worldid, row]
    if g < 0:
      return
    if search_unchanged_in[worldid, g]:
      return
    jv = float(0.0)
    if island_slot_in[worldid, g][0] > 0:
      vbase = vbase_in[worldid, g]
      for c in range(wp.static(width)):
        jv += iJ_in[worldid, row, c] * search_in[worldid, vbase + c]
    else:
      rowadr = efc_J_rowadr_in[worldid, row]
      for k in range(efc_J_rownnz_in[worldid, row]):
        adr = rowadr + k
        jv += efc_J_in[worldid, 0, adr] * search_in[worldid, dof2row_in[worldid, efc_J_colind_in[worldid, 0, adr]]]
    row_jv_out[worldid, row] = jv

  return kernel


@wp.func_native(snippet="WP_TILE_SYNC();")
def _sync():
  pass


@wp.kernel(module="unique", enable_backward=False)
def _pool_seed(
  # Model:
  ntree: int,
  dof_treeid: wp.array[int],
  tree_dofadr: wp.array[int],
  tree_dofnum: wp.array[int],
  M_rownnz: wp.array[int],
  M_rowadr: wp.array[int],
  M_colind: wp.array[int],
  M_hinit_i: wp.array[int],
  # Data in:
  M_in: wp.array2d[float],
  island_nv_in: wp.array2d[int],
  # In:
  nblock: int,
  group_count_in: wp.array[int],
  group_list_in: wp.array2d[int],
  row2dof_in: wp.array2d[int],
  tilebase_in: wp.array2d[int],
  ntile_in: wp.array2d[int],
  vbase_in: wp.array2d[int],
  skip_done: bool,
  done_in: wp.array2d[bool],
  group_slot_in: wp.array2d[wp.vec4i],
  # Out:
  H_out: wp.array3d[float],
):
  """The groups' blocks of M (identity on padding) in the H pool (H starts from M).

  A group's DOFs are whole trees in order and M is block-diagonal over trees, with each tree's CSR
  entries contiguous: the tiles are zeroed, then each tree's entries are written in place.
  """
  bid, lane = wp.tid()
  nthread = wp.block_dim()
  TILE2 = wp.static(_TILE * _TILE)
  for e in range(bid, group_count_in[0], nblock):
    worldid = group_list_in[0, e] // ntree
    g = group_list_in[0, e] % ntree
    if skip_done:
      if done_in[worldid, g]:
        continue
    # groups with dense rows: _dense_rows_h assembles H from M
    if group_slot_in[worldid, g][0] > 0:
      continue
    n = island_nv_in[worldid, g]
    ntile = ntile_in[worldid, g]
    tilebase = tilebase_in[worldid, g]
    vbase = vbase_in[worldid, g]
    for idx in range(lane, ntile * ntile * TILE2, nthread):
      t = idx // TILE2
      if t // ntile <= t % ntile:
        r = (idx % TILE2) // _TILE
        c = idx % _TILE
        la = (t // ntile) * _TILE + r
        H_out[worldid, _TILE * (tilebase + t) + r, c] = wp.where(la == (t % ntile) * _TILE + c and la >= n, 1.0, 0.0)
    _sync()
    col = int(0)
    while col < n:
      tree = dof_treeid[row2dof_in[worldid, vbase + col]]
      dofadr = tree_dofadr[tree]
      last = dofadr + tree_dofnum[tree] - 1
      beg = M_rowadr[dofadr]
      base = col - dofadr
      for ec in range(beg + lane, M_rowadr[last] + M_rownnz[last], nthread):
        rc = _pool_rc(tilebase, ntile, M_colind[ec] + base, M_hinit_i[ec] + base)
        H_out[worldid, rc[0], rc[1]] = M_in[worldid, ec]
      col += tree_dofnum[tree]
    _sync()


@wp.kernel
def _init_islands(
  # Model:
  ntree: int,
  M_rownnz: wp.array[int],
  M_rowadr: wp.array[int],
  M_mulm_rowadr: wp.array[int],
  M_mulm_col: wp.array[int],
  M_mulm_madr: wp.array[int],
  # Data in:
  M_in: wp.array2d[float],
  island_nv_in: wp.array2d[int],
  # In:
  group_count_in: wp.array[int],
  group_list_in: wp.array2d[int],
  vbase_in: wp.array2d[int],
  iqacc_in: wp.array2d[float],
  row2dof_in: wp.array2d[int],
  # Out:
  Ma_out: wp.array2d[float],
  scale_out: wp.array2d[float],
  done_out: wp.array2d[bool],
  niter_out: wp.array2d[int],
  alpha_out: wp.array2d[float],
  improvement_out: wp.array2d[float],
):
  """Start a group, a thread per (group, DOF): M qacc and the convergence scale.

  The scale is the inertia trace, summed into scale (zeroed by _group_layout).
  """
  e, lane = wp.tid()
  if e >= group_count_in[0]:
    return
  worldid = group_list_in[0, e] // ntree
  g = group_list_in[0, e] % ntree
  if lane == 0:
    done_out[worldid, g] = False
    niter_out[worldid, g] = 0
    alpha_out[worldid, g] = 0.0
    improvement_out[worldid, g] = 0.0
  vbase = vbase_in[worldid, g]
  trace = float(0.0)
  for la in range(lane, island_nv_in[worldid, g], _TILE):
    # sparse M: a group's DOFs are whole trees in order (column col of DOF dof: row + col - dof)
    row = vbase + la
    dof = row2dof_in[worldid, row]
    ma = float(0.0)
    for k in range(M_mulm_rowadr[dof], M_mulm_rowadr[dof + 1]):
      ma += M_in[worldid, M_mulm_madr[k]] * iqacc_in[worldid, row + M_mulm_col[k] - dof]
    Ma_out[worldid, row] = ma
    # the diagonal is the last entry of the row of the lower triangle
    trace += M_in[worldid, M_rowadr[dof] + M_rownnz[dof] - 1]
  # MuJoCo scales an island's convergence tests by its inertia trace instead of meaninertia * nv
  if trace != 0.0:
    wp.atomic_add(scale_out, worldid, g, trace)  # kernel_analyzer: ignore[atomic]


@wp.struct
class _Elliptic:
  """A row's elliptic contact (efcid0 < 0: none), as _update_constraint_efc gathers it."""

  efcid0: int
  jaref0: float
  D0: float
  mu: float
  ufrictionj: float
  TT: float


@wp.func
def _elliptic_row(
  # Model:
  opt_impratio_invsqrt: wp.array[float],
  # Data in:
  contact_friction_in: wp.array[types.vec5],
  contact_dim_in: wp.array[int],
  contact_efc_address_in: wp.array2d[int],
  efc_D_in: wp.array2d[float],
  # In:
  worldid: int,
  row: int,
  conid: int,
  row_jaref_in: wp.array2d[float],
) -> _Elliptic:
  cone = _Elliptic()
  cone.efcid0 = -1
  efcid0 = contact_efc_address_in[conid, 0]
  if efcid0 < 0:
    return cone
  friction = contact_friction_in[conid]
  cone.mu = friction[0] * opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]
  cone.jaref0 = row_jaref_in[worldid, efcid0]
  cone.D0 = efc_D_in[worldid, efcid0]
  for j in range(1, contact_dim_in[conid]):
    efcidj = contact_efc_address_in[conid, j]
    if efcidj < 0:
      return cone
    uj = row_jaref_in[worldid, efcidj] * friction[j - 1]
    cone.TT += uj * uj
    if row == efcidj:
      cone.ufrictionj = uj * friction[j - 1]
  cone.efcid0 = efcid0
  return cone


@wp.kernel
def _mul_m_search(
  # Model:
  ntree: int,
  M_mulm_rowadr: wp.array[int],
  M_mulm_col: wp.array[int],
  M_mulm_madr: wp.array[int],
  # Data in:
  M_in: wp.array2d[float],
  island_nv_in: wp.array2d[int],
  # In:
  group_count_in: wp.array[int],
  group_list_in: wp.array2d[int],
  vbase_in: wp.array2d[int],
  row2dof_in: wp.array2d[int],
  search_in: wp.array2d[float],
  search_unchanged_in: wp.array2d[bool],
  # Out:
  mv_out: wp.array2d[float],
):
  """M search of a group's DOF for the linesearch, a thread per (group, DOF), as for a world.

  An unchanged search direction (done or stable) keeps it.
  """
  e, lane = wp.tid()
  if e >= group_count_in[0]:
    return
  worldid = group_list_in[0, e] // ntree
  g = group_list_in[0, e] % ntree
  if search_unchanged_in[worldid, g]:
    return
  vbase = vbase_in[worldid, g]
  for la in range(lane, island_nv_in[worldid, g], _TILE):
    # sparse M: a group's DOFs are whole trees in order (column col of DOF dof: row + col - dof)
    row = vbase + la
    dof = row2dof_in[worldid, row]
    mv = float(0.0)
    for k in range(M_mulm_rowadr[dof], M_mulm_rowadr[dof + 1]):
      mv += M_in[worldid, M_mulm_madr[k]] * search_in[worldid, row + M_mulm_col[k] - dof]
    mv_out[worldid, row] = mv


@cache_kernel
def _rows(initial: bool, elliptic: bool):
  INITIAL = initial
  ELLIPTIC = elliptic

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    opt_impratio_invsqrt: wp.array[float],
    # Data in:
    nefc_in: wp.array[int],
    contact_friction_in: wp.array[types.vec5],
    contact_dim_in: wp.array[int],
    contact_efc_address_in: wp.array2d[int],
    efc_type_in: wp.array2d[int],
    efc_id_in: wp.array2d[int],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    efc_D_in: wp.array2d[float],
    efc_frictionloss_in: wp.array2d[float],
    efc_island_in: wp.array2d[int],
    # In:
    island_slot_in: wp.array2d[wp.vec4i],
    tilebase_in: wp.array2d[int],
    ntile_in: wp.array2d[int],
    vbase_in: wp.array2d[int],
    done_in: wp.array2d[bool],
    dof2row_in: wp.array2d[int],
    row_jaref_in: wp.array2d[float],
    iJ_in: wp.array3d[float],
    lanes: int,
    # Data out:
    efc_force_out: wp.array2d[float],
    efc_state_out: wp.array2d[int],
    # Out:
    row_quad_out: wp.array2d[bool],
    row_w_out: wp.array2d[float],
    frc_out: wp.array2d[float],
    H_out: wp.array3d[float],
    changed_out: wp.array2d[bool],
    state_changed_out: wp.array2d[bool],
  ):
    """Constraint force and state of a row, and the weight of its J'J term in the H update.

    H = M + sum of D J'J over quadratic rows: the initial pass weighs every quadratic row, later
    ones only rows that flip (D added or removed). Rows of sparse groups also add J'force to frc
    and, where they flip, w J'J to H (their initial H is assembled by _JTDACJ_sparse).
    """
    worldid, lane = wp.tid()
    for row in range(lane, wp.min(nefc_in[worldid], efc_island_in.shape[1]), lanes):
      islandid = efc_island_in[worldid, row]
      if done_in[worldid, islandid]:
        continue
      category = _row_category(efc_type_in[worldid, row])
      D = efc_D_in[worldid, row]
      cone = _Elliptic()
      cone.efcid0 = -1
      if wp.static(ELLIPTIC):
        if efc_type_in[worldid, row] == types.ConstraintType.CONTACT_ELLIPTIC:
          cone = _elliptic_row(
            opt_impratio_invsqrt,
            contact_friction_in,
            contact_dim_in,
            contact_efc_address_in,
            efc_D_in,
            worldid,
            row,
            efc_id_in[worldid, row],
            row_jaref_in,
          )
      res = _eval_constraint(
        category == _ROW_EQUALITY,
        category == _ROW_FRICTION,
        cone.efcid0 >= 0,
        row_jaref_in[worldid, row],
        D,
        efc_frictionloss_in[worldid, row],
        row,
        cone.efcid0,
        cone.jaref0,
        cone.D0,
        cone.mu,
        cone.ufrictionj,
        cone.TT,
      )
      state = int(res[1])
      if wp.static(not INITIAL):
        if efc_state_out[worldid, row] != state:
          state_changed_out[worldid, islandid] = True
      efc_force_out[worldid, row] = res[0]
      efc_state_out[worldid, row] = state
      quad = state == types.ConstraintState.QUADRATIC.value
      w = float(0.0)
      if wp.static(ELLIPTIC):
        # H is assembled again at every point (_JTDACJ_sparse): no incremental update and no
        # stable-state shortcut
        if wp.static(not INITIAL):
          changed_out[worldid, islandid] = True
          state_changed_out[worldid, islandid] = True
      elif wp.static(INITIAL):
        row_quad_out[worldid, row] = quad
        w = wp.where(quad, D, 0.0)
      elif quad != row_quad_out[worldid, row]:
        row_quad_out[worldid, row] = quad
        changed_out[worldid, islandid] = True
        w = wp.where(quad, D, -D)
      row_w_out[worldid, row] = w

      block = island_slot_in[worldid, islandid]
      if block[0] > 0:
        # dense rows: a flipped row's rank-1 update of the group's tile, as
        # _update_gradient_h_incremental_rows (_dense_rows_h assembles the initial H)
        if wp.static(not INITIAL):
          if w != 0.0:
            row0 = _TILE * tilebase_in[worldid, islandid]
            for i in range(block[0]):
              Ji = iJ_in[worldid, row, i]
              if Ji != 0.0:
                wJi = w * Ji
                for j in range(i, block[0]):
                  Jj = iJ_in[worldid, row, j]
                  if Jj != 0.0:
                    wp.atomic_add(H_out[worldid, row0 + i], j, wJi * Jj)  # kernel_analyzer: ignore[atomic]
        continue
      force = res[0]
      if wp.static(INITIAL):
        w = 0.0
      rowadr = efc_J_rowadr_in[worldid, row]
      rownnz = efc_J_rownnz_in[worldid, row]
      dof2row = dof2row_in[worldid]
      colind = efc_J_colind_in[worldid, 0]
      tilebase = tilebase_in[worldid, islandid]
      ntile = ntile_in[worldid, islandid]
      vbase = vbase_in[worldid, islandid]
      for k in range(rownnz):
        cola = dof2row[colind[rowadr + k]]
        Ja = efc_J_in[worldid, 0, rowadr + k]
        if force != 0.0:
          wp.atomic_add(frc_out[worldid], cola, Ja * force)  # kernel_analyzer: ignore[atomic]
        if w != 0.0:
          Jaw = Ja * w
          for kb in range(k, rownnz):
            rc = _pool_rc(tilebase, ntile, cola - vbase, dof2row[colind[rowadr + kb]] - vbase)
            wp.atomic_add(
              H_out[worldid, rc[0]], rc[1], Jaw * efc_J_in[worldid, 0, rowadr + kb]
            )  # kernel_analyzer: ignore[atomic]

  return kernel


@wp.func
def _quad_weight(D: float, state: int) -> float:
  return wp.where(state == types.ConstraintState.QUADRATIC.value, D, 0.0)


@cache_kernel
def _dense_rows_h(width: int, elliptic: bool):
  """H = M + J'DJ over the quadratic dense rows of a group, as _update_gradient_JTDAJ_dense_tiled.

  With the cone curvature of elliptic contacts. For every group initially, else (elliptic cones: H
  at every point) for the groups still iterating. Assembled at the row width W and stored into the
  group's tile, the rest of which is identity padding.
  """
  W = width
  K = 16
  ELLIPTIC = elliptic

  @wp.kernel(module="unique", enable_backward=False, module_options={"enable_mathdx_gemm": False})
  def kernel(
    # Model:
    ntree: int,
    opt_impratio_invsqrt: wp.array[float],
    dof_treeid: wp.array[int],
    tree_dofadr: wp.array[int],
    tree_dofnum: wp.array[int],
    M_rownnz: wp.array[int],
    M_rowadr: wp.array[int],
    M_colind: wp.array[int],
    M_hinit_i: wp.array[int],
    # Data in:
    M_in: wp.array2d[float],
    contact_friction_in: wp.array[types.vec5],
    contact_dim_in: wp.array[int],
    contact_efc_address_in: wp.array2d[int],
    efc_type_in: wp.array2d[int],
    efc_id_in: wp.array2d[int],
    efc_D_in: wp.array2d[float],
    efc_state_in: wp.array2d[int],
    # In:
    group_count_in: wp.array[int],
    group_list_in: wp.array2d[int],
    nblock: int,
    initial: bool,
    group_slot_in: wp.array2d[wp.vec4i],
    group_nv_in: wp.array2d[int],
    tilebase_in: wp.array2d[int],
    vbase_in: wp.array2d[int],
    row2dof_in: wp.array2d[int],
    done_in: wp.array2d[bool],
    row_jaref_in: wp.array2d[float],
    iJ_in: wp.array3d[float],
    # Out:
    H_out: wp.array3d[float],
  ):
    bid, lane = wp.tid()
    lanes = wp.block_dim()
    for e in range(bid, group_count_in[0], nblock):
      worldid = group_list_in[0, e] // ntree
      g = group_list_in[0, e] % ntree
      block = group_slot_in[worldid, g]
      if block[0] == 0:
        continue
      if not initial:
        if done_in[worldid, g]:
          continue
      rowadr = block[3]
      nrow = block[2]
      row0 = _TILE * tilebase_in[worldid, g]
      # views ending at the group's last row: the loads zero-fill past it (and past column W)
      J = wp.array(ptr=iJ_in[worldid].ptr, shape=(rowadr + nrow, W), dtype=float)
      D = wp.array(ptr=efc_D_in[worldid].ptr, shape=(rowadr + nrow,), dtype=float)
      state = wp.array(ptr=efc_state_in[worldid].ptr, shape=(rowadr + nrow,), dtype=int)
      # M's upper triangle, tree by tree (a group's DOFs are whole trees in order, each tree's CSR
      # entries contiguous); identity on padding. Uniform trip counts: the scatters are collective
      m_tile = wp.tile_zeros(shape=(wp.static(W * W),), dtype=float, storage="shared")
      ndof = group_nv_in[worldid, g]
      vbase = vbase_in[worldid, g]
      c = int(0)
      while c < ndof:
        tree = dof_treeid[row2dof_in[worldid, vbase + c]]
        dofadr = tree_dofadr[tree]
        last = dofadr + tree_dofnum[tree] - 1
        beg = M_rowadr[dofadr]
        nent = M_rowadr[last] + M_rownnz[last] - beg
        base = c - dofadr
        for it in range((nent + lanes - 1) // lanes):
          ent = it * lanes + lane
          enable = ent < nent
          ec = beg + wp.where(enable, ent, 0)
          pos = (M_colind[ec] + base) * W + M_hinit_i[ec] + base
          wp.tile_scatter_add(m_tile, pos, wp.where(enable, M_in[worldid, ec], 0.0), enable)
        c += tree_dofnum[tree]
      for it in range((W + lanes - 1) // lanes):
        i = it * lanes + lane
        enable = i >= ndof and i < W
        wp.tile_scatter_add(m_tile, wp.where(enable, i * W + i, 0), 1.0, enable)
      h = wp.tile_reshape(m_tile, (W, W))
      for k in range(0, nrow, K):
        J_k = wp.tile_load(J, shape=(K, W), offset=(rowadr + k, 0), bounds_check=True)
        D_k = wp.tile_map(
          _quad_weight,
          wp.tile_load(D, shape=K, offset=rowadr + k, bounds_check=True),
          wp.tile_load(state, shape=K, offset=rowadr + k, bounds_check=True),
        )
        JD = wp.tile_map(wp.mul, wp.tile_transpose(J_k), wp.tile_broadcast(D_k, shape=(W, K)))
        h += wp.tile_matmul(JD, J_k)
      wp.tile_store(H_out[worldid], h, offset=(row0, 0))
      # the rest of the tile: identity on the padding diagonal
      if wp.static(W < _TILE):
        for idx in range(lane, wp.static(_TILE * _TILE), lanes):
          r = idx // _TILE
          c = idx % _TILE
          if r >= W or c >= W:
            H_out[worldid, row0 + r, c] = wp.where(r == c, 1.0, 0.0)
      if wp.static(ELLIPTIC):
        _sync()
        # the cone curvature of the contacts in the cone (_update_gradient_JTCJ_dense); each lane
        # keeps its entries over the contacts
        for r in range(nrow):
          efcid0 = rowadr + r
          if efc_type_in[worldid, efcid0] != types.ConstraintType.CONTACT_ELLIPTIC:
            continue
          if efc_state_in[worldid, efcid0] != types.ConstraintState.CONE:
            continue
          conid = efc_id_in[worldid, efcid0]
          if contact_efc_address_in[conid, 0] != efcid0:
            continue
          condim = contact_dim_in[conid]
          fri = contact_friction_in[conid]
          mu = fri[0] * opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]
          mu2 = mu * mu
          dm = math.safe_div(efc_D_in[worldid, efcid0], mu2 * (1.0 + mu2))
          n = row_jaref_in[worldid, efcid0] * mu
          tt = float(0.0)
          for dim in range(1, condim):
            u = row_jaref_in[worldid, contact_efc_address_in[conid, dim]] * fri[dim - 1]
            tt += u * u
          inv_t = math.safe_div(1.0, wp.max(wp.sqrt(tt), types.MJ_MINVAL))
          mu_n_over_t = mu * n * inv_t
          for entry in range(lane, wp.static(W * W), wp.block_dim()):
            a = entry // W
            b = entry % W
            projection1 = float(0.0)
            projection2 = float(0.0)
            tangent_dot = float(0.0)
            for dim in range(1, condim):
              efcid = contact_efc_address_in[conid, dim]
              scale = fri[dim - 1]
              u = row_jaref_in[worldid, efcid] * scale
              z1 = scale * iJ_in[worldid, efcid, a]
              z2 = scale * iJ_in[worldid, efcid, b]
              projection1 += u * z1
              projection2 += u * z2
              tangent_dot += z1 * z2
            H_out[worldid, row0 + a, b] += _elliptic_hessian_entry_from_projections(
              dm,
              mu,
              mu_n_over_t,
              mu2 - mu_n_over_t,
              mu * iJ_in[worldid, efcid0, a],
              mu * iJ_in[worldid, efcid0, b],
              projection1 * inv_t,
              projection2 * inv_t,
              tangent_dot,
            )

  return kernel


@cache_kernel
def _done(warn_overflow: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    ntree: int,
    opt_tolerance: wp.array[float],
    opt_iterations: int,
    # In:
    group_count_in: wp.array[int],
    group_list_in: wp.array2d[int],
    group_nv_in: wp.array2d[int],
    vbase_in: wp.array2d[int],
    scale_in: wp.array2d[float],
    niter_in: wp.array2d[int],
    alpha_in: wp.array2d[float],
    improvement_in: wp.array2d[float],
    grad_in: wp.array2d[float],
    grad_scale_in: wp.array2d[float],
    newton_decrement_in: wp.array2d[float],
    state_changed_in: wp.array2d[bool],
    # Data out:
    solver_niter_out: wp.array[int],
    overflow_out: wp.array[int],
    # Out:
    done_out: wp.array2d[bool],
    nsolving_out: wp.array[int],
    search_unchanged_out: wp.array2d[bool],
  ):
    """Convergence test of a group still iterating, a thread per group, as _solve_done per world.

    While a group's states hold, its gradient and Newton decrement are grad_scale times the stored
    ones (the stable-state fast path). Counts the groups still iterating, and marks those whose
    search direction stays (done or stable), as the world solver's search_unchanged.
    """
    e = wp.tid()
    if e >= group_count_in[0]:
      return
    worldid = group_list_in[0, e] // ntree
    g = group_list_in[0, e] % ntree
    if done_out[worldid, g]:
      return
    niter = niter_in[worldid, g]
    converged = False
    if niter > 0:
      vbase = vbase_in[worldid, g]
      grad_scale = grad_scale_in[worldid, g]
      grad_dot = float(0.0)
      for la in range(group_nv_in[worldid, g]):
        grad_dot += grad_in[worldid, vbase + la] * grad_in[worldid, vbase + la]
      grad_dot *= grad_scale * grad_scale
      scale = scale_in[worldid, g]
      tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]
      improvement = improvement_in[worldid, g]
      converged = (
        (alpha_in[worldid, g] == 0.0)
        or (improvement > 0.0 and improvement / scale < tolerance)
        or (wp.sqrt(grad_dot) / scale < tolerance)
        or (0.5 * newton_decrement_in[worldid, g] * grad_scale * grad_scale / scale < tolerance)
      )
    search_unchanged_out[worldid, g] = True
    if converged or niter >= opt_iterations:
      done_out[worldid, g] = True
      if not converged and opt_iterations > 0:
        if wp.static(bool(warn_overflow & OverflowType.ITERATIONS)):
          wp.printf(
            "solver iterations limit reached - please increase iterations beyond %u\n"
            "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.ITERATIONS (or = 0 for all)\n",
            opt_iterations,
          )
        wp.atomic_or(overflow_out, worldid, wp.static(OverflowType.ITERATIONS))
      wp.atomic_max(solver_niter_out, worldid, niter)  # kernel_analyzer: ignore[atomic]
    else:
      wp.atomic_add(nsolving_out, 0, 1)  # kernel_analyzer: ignore[atomic]
      search_unchanged_out[worldid, g] = _stable(niter, state_changed_in[worldid, g])

  return kernel


@cache_kernel
def _grad(final: bool):
  FINAL = final

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    ntree: int,
    # Data in:
    efc_force_in: wp.array2d[float],
    # In:
    group_count_in: wp.array[int],
    group_list_in: wp.array2d[int],
    nblock: int,
    group_slot_in: wp.array2d[wp.vec4i],
    iJ_in: wp.array3d[float],
    ntile_in: wp.array2d[int],
    vbase_in: wp.array2d[int],
    done_in: wp.array2d[bool],
    niter_in: wp.array2d[int],
    state_changed_in: wp.array2d[bool],
    Ma_in: wp.array2d[float],
    iqfrc_smooth_in: wp.array2d[float],
    grad_scale_in: wp.array2d[float],
    # Out:
    frc_out: wp.array2d[float],
    grad_out: wp.array2d[float],
    nsolving_out: wp.array[int],
  ):
    """J'force (dense rows) and the gradient of a group, a warp per group and a lane per column.

    As _update_constraint_init_qfrc_constraint_dense and _update_gradient_grad for a world; sparse
    rows' J'force is accumulated by _rows. Where the last iteration skipped J'force, the final pass
    recovers it (qfrc_constraint) from the gradient, as _qfrc_constraint_from_grad: while the states
    hold, the gradient is grad_scale times the stored one.
    """
    slot, lane = wp.tid()
    if wp.static(not FINAL):
      # _done counts the groups still iterating (a memset node costs more)
      if slot == 0 and lane == 0:
        nsolving_out[0] = 0
    for e in range(slot, group_count_in[0], nblock):
      worldid = group_list_in[0, e] // ntree
      g = group_list_in[0, e] % ntree
      if wp.static(not FINAL):
        if done_in[worldid, g] or _stable(niter_in[worldid, g], state_changed_in[worldid, g]):
          continue
      block = group_slot_in[worldid, g]
      if wp.static(FINAL):
        # J'force is at the solution unless the last iteration was stable (it skips J'force)
        if block[0] == 0 or not _stable(niter_in[worldid, g], state_changed_in[worldid, g]):
          continue
      vbase = vbase_in[worldid, g]
      if wp.static(FINAL):
        grad_scale = grad_scale_in[worldid, g]
        for col in range(lane, ntile_in[worldid, g] * _TILE, _TILE):
          row = vbase + col
          frc_out[worldid, row] = Ma_in[worldid, row] - iqfrc_smooth_in[worldid, row] - grad_scale * grad_out[worldid, row]
        continue
      for col in range(lane, ntile_in[worldid, g] * _TILE, _TILE):
        row = vbase + col
        frc = float(0.0)
        if block[0] > 0:
          if col < block[0]:
            # independent loads (no test of the force first): the row loop is latency bound
            for r in range(block[3], block[3] + block[2]):
              frc += iJ_in[worldid, r, col] * efc_force_in[worldid, r]
          frc_out[worldid, row] = frc
        else:
          frc = frc_out[worldid, row]
        if wp.static(not FINAL):
          grad_out[worldid, row] = Ma_in[worldid, row] - iqfrc_smooth_in[worldid, row] - frc

  return kernel


@cache_kernel
def _factor(max_size: int):
  factorize = create_pool_cholesky_newton_func(_TILE, max_size, True)
  solve_only = create_pool_cholesky_newton_func(_TILE, max_size, False)

  @wp.kernel(module="unique", enable_backward=False, module_options={"enable_mathdx_gemm": False})
  def kernel(
    # Model:
    ntree: int,
    # Data in:
    nisland_in: wp.array[int],
    # In:
    group_count_in: wp.array[int],
    group_list_in: wp.array2d[int],
    cls: int,
    nblock: int,
    tilebase_in: wp.array2d[int],
    ntile_in: wp.array2d[int],
    vbase_in: wp.array2d[int],
    done_in: wp.array2d[bool],
    niter_in: wp.array2d[int],
    state_changed_in: wp.array2d[bool],
    grad_in: wp.array3d[float],
    H_in: wp.array3d[float],
    # Out:
    changed_out: wp.array2d[bool],
    U_out: wp.array3d[float],
    search_out: wp.array3d[float],
    search_dot_out: wp.array2d[float],
    newton_decrement_out: wp.array2d[float],
    grad_scale_out: wp.array2d[float],
  ):
    """Newton search direction of an island over two tiles per side (blocked Cholesky)."""
    bid, lane = wp.tid()
    for e in range(bid, group_count_in[cls], nblock):
      worldid = group_list_in[cls, e] // ntree
      islandid = group_list_in[cls, e] % ntree
      if islandid >= nisland_in[worldid]:
        continue
      if done_in[worldid, islandid]:
        continue
      ntile = ntile_in[worldid, islandid]
      if ntile <= 2:
        continue
      if _stable(niter_in[worldid, islandid], state_changed_in[worldid, islandid]):
        continue
      vbase = vbase_in[worldid, islandid]
      tilebase = tilebase_in[worldid, islandid]
      if changed_out[worldid, islandid] or niter_in[worldid, islandid] == 0:
        sums = wp.static(factorize)(
          H_in[worldid], grad_in[worldid], tilebase, ntile, vbase, U_out[worldid], search_out[worldid]
        )
      else:
        sums = wp.static(solve_only)(
          H_in[worldid], grad_in[worldid], tilebase, ntile, vbase, U_out[worldid], search_out[worldid]
        )
      _sync()  # every thread has read the flag
      if lane == 0:
        changed_out[worldid, islandid] = False
        search_dot_out[worldid, islandid] = sums[0]
        newton_decrement_out[worldid, islandid] = sums[1]
        grad_scale_out[worldid, islandid] = 1.0

  return kernel


def _tile_newton(n: int, ntile: int):
  """Newton direction of a group of ntile x ntile pool tiles, factored as one n x n tile.

  One tile (n <= _TILE: the group fits n) is factored every time, as in the world solver; two keep
  their factor in U and only solve with it while no row changes.
  """

  @wp.func
  def one_tile(
    # In:
    H_in: wp.array2d[float],
    grad_in: wp.array[float],
    row0: int,
    vbase: int,
    factorize: bool,
    # Out:
    U_out: wp.array2d[float],
    search_out: wp.array[float],
  ) -> wp.vec2:
    # as _update_gradient_cholesky: refactoring a tile costs less than keeping its factor
    H = wp.tile_load(H_in, shape=(n, n), offset=(row0, 0), storage="shared")
    wp.tile_cholesky_inplace(H, fill_mode="upper")
    b1 = wp.tile_load(grad_in, shape=n, offset=vbase)
    x1 = wp.tile_cholesky_solve(H, b1, fill_mode="upper")
    wp.tile_store(search_out, wp.tile_map(wp.neg, x1), offset=vbase)
    return wp.tile_reduce(wp.add, wp.tile_map(solve_search_sums, b1, x1))[0]

  if ntile == 1:
    return one_tile

  @wp.func
  def func(
    # In:
    H_in: wp.array2d[float],
    grad_in: wp.array[float],
    row0: int,
    vbase: int,
    factorize: bool,
    # Out:
    U_out: wp.array2d[float],
    search_out: wp.array[float],
  ) -> wp.vec2:
    U = wp.tile_zeros(shape=(n, n), dtype=float, storage="shared")
    # the block's upper tiles (0, 0), (0, 1), (1, 1); the factorization reads no others
    if factorize:
      wp.tile_assign(U, wp.tile_load(H_in, shape=(_TILE, _TILE), offset=(row0, 0)))
      if wp.static(ntile == 2):
        wp.tile_assign(U, wp.tile_load(H_in, shape=(_TILE, _TILE), offset=(row0 + _TILE, 0)), offset=(0, _TILE))
        wp.tile_assign(U, wp.tile_load(H_in, shape=(_TILE, _TILE), offset=(row0 + 3 * _TILE, 0)), offset=(_TILE, _TILE))
      wp.tile_cholesky_inplace(U, fill_mode="upper")
      wp.tile_store(U_out, wp.tile_view(U, offset=(0, 0), shape=(_TILE, _TILE)), offset=(row0, 0))
      if wp.static(ntile == 2):
        wp.tile_store(U_out, wp.tile_view(U, offset=(0, _TILE), shape=(_TILE, _TILE)), offset=(row0 + _TILE, 0))
        wp.tile_store(U_out, wp.tile_view(U, offset=(_TILE, _TILE), shape=(_TILE, _TILE)), offset=(row0 + 3 * _TILE, 0))
    else:
      wp.tile_assign(U, wp.tile_load(U_out, shape=(_TILE, _TILE), offset=(row0, 0)))
      if wp.static(ntile == 2):
        wp.tile_assign(U, wp.tile_load(U_out, shape=(_TILE, _TILE), offset=(row0 + _TILE, 0)), offset=(0, _TILE))
        wp.tile_assign(U, wp.tile_load(U_out, shape=(_TILE, _TILE), offset=(row0 + 3 * _TILE, 0)), offset=(_TILE, _TILE))
    b = wp.tile_load(grad_in, shape=n, offset=vbase)
    x = wp.tile_cholesky_solve(U, b, fill_mode="upper")
    wp.tile_store(search_out, wp.tile_map(wp.neg, x), offset=vbase)
    return wp.tile_reduce(wp.add, wp.tile_map(solve_search_sums, b, x))[0]

  return func


@cache_kernel
def _dense_factor(ntile: int):
  """Newton directions of the groups of ntile tiles per side, as _update_gradient_cholesky.

  One-tile groups of at most half a tile of DOFs are factored at half the size (cuSolverDx is
  slower at sizes between).
  """
  N = ntile * _TILE
  HALF = _TILE // 2
  full = _tile_newton(N, ntile)
  half = _tile_newton(HALF, 1) if ntile == 1 else full

  @wp.kernel(module="unique", enable_backward=False, module_options={"enable_mathdx_gemm": False})
  def kernel(
    # Model:
    ntree: int,
    # Data in:
    nisland_in: wp.array[int],
    island_nv_in: wp.array2d[int],
    # In:
    group_count_in: wp.array[int],
    group_list_in: wp.array2d[int],
    cls: int,
    nblock: int,
    tilebase_in: wp.array2d[int],
    ntile_in: wp.array2d[int],
    vbase_in: wp.array2d[int],
    done_in: wp.array2d[bool],
    niter_in: wp.array2d[int],
    changed_in: wp.array2d[bool],
    state_changed_in: wp.array2d[bool],
    grad_in: wp.array2d[float],
    H_in: wp.array3d[float],
    # Out:
    U_out: wp.array3d[float],
    search_out: wp.array2d[float],
    search_dot_out: wp.array2d[float],
    newton_decrement_out: wp.array2d[float],
    grad_scale_out: wp.array2d[float],
  ):
    bid, lane = wp.tid()
    for e in range(bid, group_count_in[cls], nblock):
      worldid = group_list_in[cls, e] // ntree
      islandid = group_list_in[cls, e] % ntree
      if islandid >= nisland_in[worldid]:
        continue
      if done_in[worldid, islandid]:
        continue
      if ntile_in[worldid, islandid] != wp.static(ntile):
        continue
      if _stable(niter_in[worldid, islandid], state_changed_in[worldid, islandid]):
        continue
      vbase = vbase_in[worldid, islandid]
      row0 = _TILE * tilebase_in[worldid, islandid]
      factorize = changed_in[worldid, islandid] or niter_in[worldid, islandid] == 0
      sums = wp.vec2(0.0)
      if wp.static(ntile == 1) and island_nv_in[worldid, islandid] <= HALF:
        sums = wp.static(half)(H_in[worldid], grad_in[worldid], row0, vbase, factorize, U_out[worldid], search_out[worldid])
      else:
        sums = wp.static(full)(H_in[worldid], grad_in[worldid], row0, vbase, factorize, U_out[worldid], search_out[worldid])
      if lane == 0:
        search_dot_out[worldid, islandid] = sums[0]
        newton_decrement_out[worldid, islandid] = sums[1]
        grad_scale_out[worldid, islandid] = 1.0

  return kernel


@wp.kernel
def _scatter_dofs(
  # In:
  lanes: int,
  nrow_used_in: wp.array[int],
  row2dof_in: wp.array2d[int],
  iqacc_in: wp.array2d[float],
  frc_in: wp.array2d[float],
  iMa_in: wp.array2d[float],
  # Data out:
  qacc_out: wp.array2d[float],
  qfrc_constraint_out: wp.array2d[float],
  efc_Ma_out: wp.array2d[float],
):
  worldid, lane = wp.tid()
  for row in range(lane, nrow_used_in[worldid], lanes):
    dofid = row2dof_in[worldid, row]
    if dofid >= 0:
      qacc_out[worldid, dofid] = iqacc_in[worldid, row]
      qfrc_constraint_out[worldid, dofid] = frc_in[worldid, row]
      efc_Ma_out[worldid, dofid] = iMa_in[worldid, row]


@wp.kernel
def _dofs(
  # Model:
  dof_treeid: wp.array[int],
  # Data in:
  qacc_warmstart_in: wp.array2d[float],
  tree_awake_in: wp.array2d[int],
  qfrc_smooth_in: wp.array2d[float],
  qacc_smooth_in: wp.array2d[float],
  map_dof2idof_in: wp.array2d[int],
  # In:
  warmstart: bool,
  dof_group_in: wp.array2d[int],
  group_idofadr_in: wp.array2d[int],
  tilebase_in: wp.array2d[int],
  ntile_in: wp.array2d[int],
  vbase_in: wp.array2d[int],
  # Data out:
  qacc_out: wp.array2d[float],
  qfrc_constraint_out: wp.array2d[float],
  efc_Ma_out: wp.array2d[float],
  # Out:
  row2dof_out: wp.array2d[int],
  dof2row_out: wp.array2d[int],
  dof_poolrow_out: wp.array2d[int],
  dof_local_out: wp.array2d[int],
  iqacc_out: wp.array2d[float],
  iqfrc_smooth_out: wp.array2d[float],
):
  """A DOF's aligned row, pool row (first tile column), local index, start point and smooth force.

  DOFs outside all islands have no constraints (M qacc_smooth = qfrc_smooth); sleeping ones rest.
  """
  worldid, dofid = wp.tid()
  row = int(-1)
  poolrow = int(-1)
  local = int(-1)
  g = dof_group_in[worldid, dofid]
  if g >= 0:
    ntile = ntile_in[worldid, g]
    if ntile > 0:
      local = map_dof2idof_in[worldid, dofid] - group_idofadr_in[worldid, g]
      row = vbase_in[worldid, g] + local
      poolrow = _TILE * (tilebase_in[worldid, g] + (local // _TILE) * ntile) + local % _TILE
      row2dof_out[worldid, row] = dofid
      iqacc_out[worldid, row] = wp.where(warmstart, qacc_warmstart_in[worldid, dofid], qacc_smooth_in[worldid, dofid])
      iqfrc_smooth_out[worldid, row] = qfrc_smooth_in[worldid, dofid]
  dof2row_out[worldid, dofid] = row
  dof_poolrow_out[worldid, dofid] = poolrow
  dof_local_out[worldid, dofid] = local
  if tree_awake_in:
    if tree_awake_in[worldid, dof_treeid[dofid]] == 0:
      qacc_out[worldid, dofid] = 0.0
      qfrc_constraint_out[worldid, dofid] = 0.0
      efc_Ma_out[worldid, dofid] = 0.0
      return
  if g < 0:
    qacc_out[worldid, dofid] = qacc_smooth_in[worldid, dofid]
    qfrc_constraint_out[worldid, dofid] = 0.0
    efc_Ma_out[worldid, dofid] = qfrc_smooth_in[worldid, dofid]


def _create_context(m: types.Model, d: types.Data) -> IslandContext:
  ntile_pool, nrow = _island_pool_size(m.nv, m.ntree, _TILE)
  nworld, njmax, ntree = d.nworld, d.njmax, m.ntree
  pool = (nworld, ntile_pool * _TILE, _TILE)
  return IslandContext(
    tilebase=wp.empty((nworld, ntree), dtype=int),
    ntile=wp.empty((nworld, ntree), dtype=int),
    nrow_used=wp.empty(nworld, dtype=int),
    vbase=wp.empty((nworld, ntree), dtype=int),
    row2dof=wp.empty((nworld, nrow), dtype=int),
    dof2row=wp.empty((nworld, m.nv), dtype=int),
    dof_poolrow=wp.empty((nworld, m.nv), dtype=int),
    dof_local=wp.empty((nworld, m.nv), dtype=int),
    qacc=wp.empty((nworld, nrow), dtype=float),
    Ma=wp.zeros((nworld, nrow), dtype=float),
    qfrc_smooth=wp.empty((nworld, nrow), dtype=float),
    frc=wp.zeros((nworld, nrow), dtype=float),
    grad=wp.zeros((nworld, nrow, 1), dtype=float),
    search=wp.zeros((nworld, nrow, 1), dtype=float),
    mv=wp.zeros((nworld, nrow), dtype=float),
    row_jaref=wp.empty((nworld, njmax), dtype=float),
    row_jv=wp.empty((nworld, njmax), dtype=float),
    row_quad=wp.zeros((nworld, njmax), dtype=bool),
    row_w=wp.zeros((nworld, njmax), dtype=float),
    row_cone=wp.empty((nworld, njmax) if m.opt.cone == types.ConeType.ELLIPTIC else (0, 0), dtype=wp.vec3),
    H=wp.empty(pool, dtype=float),
    U=wp.empty(pool, dtype=float),
    scale=wp.empty((nworld, ntree), dtype=float),
    done=wp.empty((nworld, ntree), dtype=bool),
    niter=wp.empty((nworld, ntree), dtype=int),
    alpha=wp.empty((nworld, ntree), dtype=float),
    improvement=wp.empty((nworld, ntree), dtype=float),
    search_dot=wp.empty((nworld, ntree), dtype=float),
    newton_decrement=wp.empty((nworld, ntree), dtype=float),
    changed=wp.zeros((nworld, ntree), dtype=bool),
    state_changed=wp.zeros((nworld, ntree), dtype=bool),
    search_unchanged=wp.empty((nworld, ntree), dtype=bool),
    grad_scale=wp.ones((nworld, ntree), dtype=float),
    group_count=wp.zeros(5, dtype=int),
    world_done=wp.zeros(nworld, dtype=bool),
    group_list=wp.empty((4, nworld * ntree), dtype=int),
    nsolving=wp.zeros(1, dtype=int),
  )


def _lanes(nworld: int, nrow: int) -> int:
  """Threads per world of the row kernels, which loop over the rows that exist."""
  return min(nrow, max(32, -(-(1 << 19) // nworld // 32) * 32))


def _nblock(per_sm: int = 1) -> int:
  """Persistent blocks for the group kernels (they stride over the groups that exist)."""
  return wp.get_device().sm_count * 32 * per_sm


def _seed_pool(m: types.Model, d: types.Data, ctx: IslandContext, g: island.SlotBuffers, skip_done: bool = False):
  """H = M (identity on padding) for the block path's groups (skip_done: those still iterating)."""
  wp.launch_tiled(
    _pool_seed,
    dim=_nblock(),
    inputs=[
      m.ntree,
      m.dof_treeid,
      m.tree_dofadr,
      m.tree_dofnum,
      m.M_rownnz,
      m.M_rowadr,
      m.M_colind,
      m.M_hinit_i,
      d.M,
      g.group_nv,
      _nblock(),
      ctx.group_count,
      ctx.group_list,
      ctx.row2dof,
      ctx.tilebase,
      ctx.ntile,
      ctx.vbase,
      skip_done,
      ctx.done,
      g.group_slot,
    ],
    outputs=[ctx.H],
    block_dim=_ISLAND_THREADS,
  )


def _evaluate(m: types.Model, d: types.Data, ctx: IslandContext, g: island.SlotBuffers, initial: bool):
  """Constraint state at the current point, J'force of sparse islands and the H updates."""
  wp.launch(
    _rows(initial, m.opt.cone == types.ConeType.ELLIPTIC),
    dim=(d.nworld, _lanes(d.nworld, d.njmax)),
    inputs=[
      m.opt.impratio_invsqrt,
      d.nefc,
      d.contact.friction,
      d.contact.dim,
      d.contact.efc_address,
      d.efc.type,
      d.efc.id,
      d.efc.J_rownnz,
      d.efc.J_rowadr,
      d.efc.J_colind,
      d.efc.J,
      d.efc.D,
      d.efc.frictionloss,
      g.efc_group,
      g.group_slot,
      ctx.tilebase,
      ctx.ntile,
      ctx.vbase,
      ctx.done,
      ctx.dof2row,
      ctx.row_jaref,
      g.J,
      _lanes(d.nworld, d.njmax),
    ],
    outputs=[d.efc.force, d.efc.state, ctx.row_quad, ctx.row_w, ctx.frc, ctx.H, ctx.changed, ctx.state_changed],
  )

  elliptic = m.opt.cone == types.ConeType.ELLIPTIC

  def sparse_h():
    # dense rows have no constraint groups; elliptic: one block per constraint group (the cone
    # terms are shared per block)
    block_dim = (
      (1 if wp.get_device().is_cpu else _JTDAJ_THREADS_PER_GROUP) if elliptic else m.block_dim.update_gradient_JTDAJ_sparse
    )
    kernel = _JTDACJ_sparse(False, m.opt.cone, max(3, int(m.nmaxcondim)) if elliptic else 3, block_dim, _TILE)
    groups_per_world = _jtdaj_groups_per_world(d.nworld, d.njmax, kernel)
    wp.launch(
      kernel,
      dim=(d.nworld, groups_per_world, _JTDAJ_THREADS_PER_GROUP),
      inputs=[
        m.opt.impratio_invsqrt,
        d.contact.friction,
        d.contact.dim,
        d.efc.id,
        d.efc.jtdaj_adr,
        d.efc.jtdaj_nrow,
        d.efc.jtdaj_nblock,
        d.efc.J_rownnz,
        d.efc.J_rowadr,
        d.efc.J_colind,
        d.efc.J,
        d.efc.D,
        d.efc.state,
        ctx.dof_poolrow,
        ctx.dof_local,
        ctx.row_jaref,
        ctx.world_done,
        groups_per_world,
        g.efc_group,
        ctx.done,
      ],
      outputs=[ctx.H],
      block_dim=block_dim,
    )

  def dense_h():
    wp.launch_tiled(
      _dense_rows_h(g.J.shape[2], elliptic),
      dim=_nblock(),
      inputs=[
        m.ntree,
        m.opt.impratio_invsqrt,
        m.dof_treeid,
        m.tree_dofadr,
        m.tree_dofnum,
        m.M_rownnz,
        m.M_rowadr,
        m.M_colind,
        m.M_hinit_i,
        d.M,
        d.contact.friction,
        d.contact.dim,
        d.contact.efc_address,
        d.efc.type,
        d.efc.id,
        d.efc.D,
        d.efc.state,
        ctx.group_count,
        ctx.group_list,
        _nblock(),
        initial,
        g.group_slot,
        g.group_nv,
        ctx.tilebase,
        ctx.vbase,
        ctx.row2dof,
        ctx.done,
        ctx.row_jaref,
        g.J,
      ],
      outputs=[ctx.H],
      block_dim=_ISLAND_THREADS,
    )

  def h():
    # elliptic cones: H at every point (the cone curvature depends on jaref)
    if not initial and elliptic:
      _seed_pool(m, d, ctx, g, skip_done=True)
    if initial or elliptic:
      sparse_h()
      dense_h()

  # H is assembled initially and, with elliptic cones, at every point; else _rows updates it
  h()


def _launch_grad(d: types.Data, ctx: IslandContext, g: island.SlotBuffers, final: bool):
  nrow = ctx.qacc.shape[1]
  # warp slots striding over the groups, several per block (full occupancy; no tile operations)
  nslot = _nblock(4)
  wp.launch(
    _grad(final),
    dim=(nslot, _TILE),
    inputs=[
      g.group_slot.shape[1],
      d.efc.force,
      ctx.group_count,
      ctx.group_list,
      nslot,
      g.group_slot,
      g.J,
      ctx.ntile,
      ctx.vbase,
      ctx.done,
      ctx.niter,
      ctx.state_changed,
      ctx.Ma,
      ctx.qfrc_smooth,
      ctx.grad_scale,
    ],
    outputs=[ctx.frc, ctx.grad.reshape((d.nworld, nrow)), ctx.nsolving],
    block_dim=4 * _TILE,
  )


def _direction(m: types.Model, d: types.Data, ctx: IslandContext, g: island.SlotBuffers):
  """Gradient and Newton direction at the current point (as the world solver's _update_gradient)."""
  nrow = ctx.qacc.shape[1]
  _launch_grad(d, ctx, g, final=False)

  def dense_factor(ntile: int):
    wp.launch_tiled(
      _dense_factor(ntile),
      dim=_nblock(),
      inputs=[
        m.ntree,
        g.ngroup,
        g.group_nv,
        ctx.group_count,
        ctx.group_list,
        ntile,
        _nblock(),
        ctx.tilebase,
        ctx.ntile,
        ctx.vbase,
        ctx.done,
        ctx.niter,
        ctx.changed,
        ctx.state_changed,
        ctx.grad.reshape((d.nworld, nrow)),
        ctx.H,
      ],
      outputs=[ctx.U, ctx.search.reshape((d.nworld, nrow)), ctx.search_dot, ctx.newton_decrement, ctx.grad_scale],
      block_dim=_ISLAND_THREADS,
    )

  def blocked_factor():
    wp.launch_tiled(
      _factor(-(-m.nv // _TILE) * _TILE),
      dim=_nblock(),
      inputs=[
        m.ntree,
        g.ngroup,
        ctx.group_count,
        ctx.group_list,
        3,
        _nblock(),
        ctx.tilebase,
        ctx.ntile,
        ctx.vbase,
        ctx.done,
        ctx.niter,
        ctx.state_changed,
        ctx.grad,
        ctx.H,
      ],
      outputs=[ctx.changed, ctx.U, ctx.search, ctx.search_dot, ctx.newton_decrement, ctx.grad_scale],
      block_dim=_ISLAND_THREADS,
    )

  def multi_tile():
    dense_factor(2)
    blocked_factor()

  # groups of several tiles only when some group has them (one-tile groups nearly always)
  dense_factor(1)
  if m.opt.graph_conditional:
    wp.capture_if(ctx.group_count[4:5], on_true=multi_tile)
  else:
    multi_tile()


def _launch_done(m: types.Model, d: types.Data, ctx: IslandContext, g: island.SlotBuffers):
  # ctx.nsolving is zeroed by _grad
  nrow = ctx.qacc.shape[1]
  wp.launch(
    _done(int(m.opt.warn_overflow)),
    dim=d.nworld * m.ntree,
    inputs=[
      m.ntree,
      m.opt.tolerance,
      m.opt.iterations,
      ctx.group_count,
      ctx.group_list,
      g.group_nv,
      ctx.vbase,
      ctx.scale,
      ctx.niter,
      ctx.alpha,
      ctx.improvement,
      ctx.grad.reshape((d.nworld, nrow)),
      ctx.grad_scale,
      ctx.newton_decrement,
      ctx.state_changed,
    ],
    outputs=[d.solver_niter, d.overflow, ctx.done, ctx.nsolving, ctx.search_unchanged],
  )


def _iteration(m: types.Model, d: types.Data, ctx: IslandContext, g: island.SlotBuffers):
  """A Newton iteration, as the world solver's: linesearch, constraints, direction, done test."""
  nrow = ctx.qacc.shape[1]
  wp.launch(
    _mul_m_search,
    dim=(d.nworld * m.ntree, _TILE),
    inputs=[
      m.ntree,
      m.M_mulm_rowadr,
      m.M_mulm_col,
      m.M_mulm_madr,
      d.M,
      g.group_nv,
      ctx.group_count,
      ctx.group_list,
      ctx.vbase,
      ctx.row2dof,
      ctx.search.reshape((d.nworld, nrow)),
      ctx.search_unchanged,
    ],
    outputs=[ctx.mv],
  )
  wp.launch(
    _jv(g.J.shape[2]),
    dim=(d.nworld, d.njmax),
    inputs=[
      d.nefc,
      d.efc.J_rownnz,
      d.efc.J_rowadr,
      d.efc.J_colind,
      d.efc.J,
      g.efc_group,
      g.group_slot,
      ctx.vbase,
      ctx.dof2row,
      g.J,
      ctx.search.reshape((d.nworld, nrow)),
      ctx.search_unchanged,
    ],
    outputs=[ctx.row_jv],
  )
  # the world solver's linesearch, a block per group (_linesearch_iterative_kernel, island mode)
  flat = d.nworld * m.ntree
  wp.launch_tiled(
    _linesearch_iterative_kernel(m.opt.ls_iterations, m.opt.cone, False, True, False, int(m.opt.warn_overflow), True),
    dim=flat,
    inputs=[
      0,
      m.ntree,
      m.opt.tolerance,
      m.opt.ls_tolerance,
      m.opt.impratio_invsqrt,
      g.group_ne.reshape(flat),
      g.group_nf.reshape(flat),
      g.group_nefc.reshape(flat),
      ctx.qfrc_smooth,
      d.contact.friction,
      d.contact.dim,
      d.contact.efc_address,
      d.efc.type,
      d.efc.id,
      d.efc.J_rownnz,
      d.efc.J_rowadr,
      d.efc.J_colind,
      d.efc.J,
      d.efc.D,
      d.efc.frictionloss,
      d.njmax,
      d.nacon,
      None,
      None,
      ctx.row_jaref,
      ctx.search.reshape((d.nworld, nrow)),
      ctx.search_dot.reshape(flat),
      ctx.mv,
      ctx.row_jv,
      ctx.row_cone,
      ctx.done.reshape(flat),
      g.group_iefcadr.reshape(flat),
      ctx.group_count[0:1],
      ctx.group_list[0],
      ctx.vbase.reshape(flat),
      g.group_nv.reshape(flat),
      ctx.scale.reshape(flat),
    ],
    outputs=[
      ctx.qacc,
      ctx.Ma,
      d.overflow,
      ctx.row_jaref,
      ctx.row_jv,
      ctx.row_cone,
      ctx.improvement.reshape(flat),
      ctx.alpha.reshape(flat),
      None,
      ctx.frc,
      ctx.niter.reshape(flat),
      ctx.grad_scale.reshape(flat),
      ctx.changed.reshape(flat),
      ctx.state_changed.reshape(flat),
    ],
    block_dim=m.block_dim.linesearch_iterative,
  )
  _evaluate(m, d, ctx, g, initial=False)
  _direction(m, d, ctx, g)
  _launch_done(m, d, ctx, g)


def _solve_groups(m: types.Model, d: types.Data, ctx: IslandContext, g: island.SlotBuffers):
  """Solve the awake groups: one-tile groups whole, wider ones on the block path."""
  nrow = ctx.qacc.shape[1]
  _seed_pool(m, d, ctx, g)
  wp.launch(
    _init_rows(g.J.shape[2]),
    dim=(d.nworld, d.njmax),
    inputs=[
      d.nefc,
      d.efc.J_rownnz,
      d.efc.J_rowadr,
      d.efc.J_colind,
      d.efc.J,
      d.efc.aref,
      g.efc_group,
      g.group_slot,
      ctx.vbase,
      ctx.dof2row,
      g.J,
      ctx.qacc,
    ],
    outputs=[ctx.row_jaref],
  )
  wp.launch(
    _init_islands,
    dim=(d.nworld * m.ntree, _TILE),
    inputs=[
      m.ntree,
      m.M_rownnz,
      m.M_rowadr,
      m.M_mulm_rowadr,
      m.M_mulm_col,
      m.M_mulm_madr,
      d.M,
      g.group_nv,
      ctx.group_count,
      ctx.group_list,
      ctx.vbase,
      ctx.qacc,
      ctx.row2dof,
    ],
    outputs=[ctx.Ma, ctx.scale, ctx.done, ctx.niter, ctx.alpha, ctx.improvement],
  )
  # initial constraint evaluation (H = M + J'DJ over the quadratic rows), direction and done test
  _evaluate(m, d, ctx, g, initial=True)
  _direction(m, d, ctx, g)
  _launch_done(m, d, ctx, g)

  # each iteration ends with ctx.nsolving = the number of islands still iterating
  if m.opt.graph_conditional:
    wp.capture_while(ctx.nsolving, while_body=_iteration, m=m, d=d, ctx=ctx, g=g)
  else:
    for _ in range(m.opt.iterations):
      _iteration(m, d, ctx, g)

  # J'force where the last iteration skipped it, then scatter back
  _launch_grad(d, ctx, g, final=True)
  wp.launch(
    _scatter_dofs,
    dim=(d.nworld, _lanes(d.nworld, nrow)),
    inputs=[_lanes(d.nworld, nrow), ctx.nrow_used, ctx.row2dof, ctx.qacc, ctx.frc, ctx.Ma],
    outputs=[d.qacc, d.qfrc_constraint, d.efc.Ma],
  )


def solve(m: types.Model, d: types.Data) -> IslandContext:
  """Solve every island with its own Newton iteration and scatter the results back."""
  # make_constraint found the islands and wrote the rows in island order
  ctx = _create_context(m, d)
  # islands are solved in groups (see island.MERGE)
  g = island.slot_buffers(m, d)
  d.solver_niter.zero_()
  nrow = ctx.qacc.shape[1]
  wp.launch_tiled(
    _group_layout(m.ntree),
    dim=d.nworld,
    inputs=[g.ngroup, g.group_nv, g.group_slot, g.group_asleep],
    outputs=[
      ctx.group_count,
      ctx.group_list,
      ctx.done,
      ctx.ntile,
      ctx.scale,
      ctx.tilebase,
      ctx.vbase,
      ctx.nrow_used,
      ctx.row2dof,
      ctx.qacc,
      ctx.qfrc_smooth,
      ctx.frc,
      ctx.search_unchanged,
    ],
    block_dim=m.block_dim.island_scan_sizes,
  )
  wp.launch(
    _dofs,
    dim=(d.nworld, m.nv),
    inputs=[
      m.dof_treeid,
      d.qacc_warmstart,
      d.tree_awake if m.opt.enableflags & types.EnableBit.SLEEP else None,
      d.qfrc_smooth,
      d.qacc_smooth,
      d.map_dof2idof,
      not (m.opt.disableflags & types.DisableBit.WARMSTART),
      g.dof_group,
      g.group_idofadr,
      ctx.tilebase,
      ctx.ntile,
      ctx.vbase,
    ],
    outputs=[
      d.qacc,
      d.qfrc_constraint,
      d.efc.Ma,
      ctx.row2dof,
      ctx.dof2row,
      ctx.dof_poolrow,
      ctx.dof_local,
      ctx.qacc,
      ctx.qfrc_smooth,
    ],
  )
  # unconditional: entering a conditional body costs more than kernels without groups
  _solve_groups(m, d, ctx, g)

  return ctx
