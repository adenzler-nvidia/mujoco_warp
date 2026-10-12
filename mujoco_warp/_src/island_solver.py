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

make_constraint writes an island's rows contiguously. Islands of up to island.DENSE_NV DOFs store
their Jacobian as one dense block of whole 16-DOF tiles (island-local columns), wider ones as sparse
rows. Each step, an island's DOFs are gathered into tile-aligned rows, and its inertia and Hessian
are dense blocks of 16x16 tiles in a per-world pool. Every island is solved with its own step size
and convergence test, as MuJoCo's island solver does. Kernels run over the work that exists: rows,
DOFs, islands and the tile pairs of each island's Hessian.
"""

import dataclasses

import warp as wp

from mujoco_warp._src import island
from mujoco_warp._src import support
from mujoco_warp._src import types
from mujoco_warp._src.block_cholesky import create_pool_cholesky_newton_func
from mujoco_warp._src.block_cholesky import solve_search_sums
from mujoco_warp._src.solver import _ALPHA_NOISE_EPS
from mujoco_warp._src.solver import _JTDAJ_THREADS_PER_GROUP
from mujoco_warp._src.solver import _ROW_EQUALITY
from mujoco_warp._src.solver import _ROW_FRICTION
from mujoco_warp._src.solver import _ROW_OTHER
from mujoco_warp._src.solver import _active_check
from mujoco_warp._src.solver import _eval_constraint
from mujoco_warp._src.solver import _eval_pt
from mujoco_warp._src.solver import _eval_pt_3alphas
from mujoco_warp._src.solver import _in_bracket
from mujoco_warp._src.solver import _island_pool_maps
from mujoco_warp._src.solver import _island_pool_size
from mujoco_warp._src.solver import _JTDACJ_sparse
from mujoco_warp._src.solver import _jtdaj_groups_per_world
from mujoco_warp._src.solver import _row_eval_pt
from mujoco_warp._src.solver import _row_eval_pt_alpha_zero
from mujoco_warp._src.types import OverflowType
from mujoco_warp._src.warp_util import cache_kernel

wp.set_module_options({"enable_backward": False})

_TILE = types.TILE_SIZE_JTDAJ_DENSE
# threads per island: factorization as _update_gradient_cholesky, linesearch one warp as
# _linesearch_iterative_kernel; one block per Hessian tile pair
_ISLAND_THREADS = 64
_LINESEARCH_THREADS = 32
_PAIR_THREADS = 64


@dataclasses.dataclass
class IslandContext:
  """Per-step island-ordered solver state (see the module docstring)."""

  # layout
  tilebase: wp.array2d[int]
  ntile: wp.array2d[int]
  vbase: wp.array2d[int]
  # first tile pair (bi <= bj) of each island's Hessian, in a per-world list of pairs
  pairbase: wp.array2d[int]
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
  # per-island dense blocks: inertia, Hessian and its factor
  M: wp.array3d[float]
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
  grad_scale: wp.array2d[float]
  # compacted lists of (world * ntree + group), see _group_lists
  group_count: wp.array[int]
  group_list: wp.array2d[int]
  group_problem: wp.array2d[int]
  nsolving: wp.array[int]
  # the dense world solver's model, data and context for two-tile groups (see _create_dense)
  dense: object = None


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


@wp.func
def _select(keep: float, x: float) -> float:
  return wp.where(keep != 0.0, x, 0.0)


@wp.func
def _npair(ntile: int) -> int:
  return ntile * (ntile + 1) // 2


@cache_kernel
def _layout(ntree: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Data in:
    island_nv_in: wp.array2d[int],
    # Out:
    tilebase_out: wp.array2d[int],
    ntile_out: wp.array2d[int],
    vbase_out: wp.array2d[int],
    pairbase_out: wp.array2d[int],
  ):
    """Per island: tiles per side, first pool tile, first aligned DOF row and first tile pair."""
    worldid = wp.tid()
    ntile = wp.tile_map(_ntile, wp.tile_load(island_nv_in[worldid], shape=ntree))
    wp.tile_store(ntile_out[worldid], ntile)
    wp.tile_store(tilebase_out[worldid], wp.tile_scan_exclusive(wp.tile_map(wp.mul, ntile, ntile)))
    wp.tile_store(vbase_out[worldid], wp.tile_scan_exclusive(ntile) * _TILE)
    wp.tile_store(pairbase_out[worldid], wp.tile_scan_exclusive(wp.tile_map(_npair, ntile)))

  return kernel


@wp.kernel
def _group_lists(
  # Model:
  ntree: int,
  # In:
  ngroup_in: wp.array[int],
  ntile_in: wp.array2d[int],
  island_slot_in: wp.array2d[wp.vec4i],
  # Out:
  group_count_out: wp.array[int],
  group_list_out: wp.array2d[int],
  group_problem_out: wp.array2d[int],
  done_out: wp.array2d[bool],
):
  """Compacted lists of the groups that exist, by solve path.

  Lists 0-3: the block path, sparse rows (all, and by tiles per side: 1, 2, >= 3; count 5 is the
  same as count 0). List 4: dense one-tile groups, solved whole by _solve_small. List 6: dense
  two-tile groups, solved by the dense world solver (see _solve_dense), problem index in
  group_problem. Groups of lists 4 and 6 are done for the block path.
  """
  worldid, g = wp.tid()
  if g >= ngroup_in[worldid]:
    return
  packed = worldid * ntree + g
  width = island_slot_in[worldid, g][0]
  group_problem_out[worldid, g] = -1
  if width == _TILE:
    group_list_out[4, wp.atomic_add(group_count_out, 4, 1)] = packed
    done_out[worldid, g] = True
    return
  if width > _TILE:
    p = wp.atomic_add(group_count_out, 6, 1)
    group_list_out[6, p] = packed
    group_problem_out[worldid, g] = p
    done_out[worldid, g] = True
    return
  group_list_out[0, wp.atomic_add(group_count_out, 0, 1)] = packed
  wp.atomic_add(group_count_out, 5, 1)
  c = wp.min(ntile_in[worldid, g], 3)
  group_list_out[c, wp.atomic_add(group_count_out, c, 1)] = packed


@wp.kernel
def _gather_dofs(
  # Data in:
  qacc_warmstart_in: wp.array2d[float],
  qfrc_smooth_in: wp.array2d[float],
  qacc_smooth_in: wp.array2d[float],
  # In:
  warmstart: bool,
  row2dof_in: wp.array2d[int],
  # Out:
  iqacc_out: wp.array2d[float],
  iqfrc_smooth_out: wp.array2d[float],
  ifrc_out: wp.array2d[float],
):
  """DOF start point and smooth force in aligned island rows; padding is zero."""
  worldid, row = wp.tid()
  dofid = row2dof_in[worldid, row]
  qacc = float(0.0)
  qfrc = float(0.0)
  if dofid >= 0:
    qacc = wp.where(warmstart, qacc_warmstart_in[worldid, dofid], qacc_smooth_in[worldid, dofid])
    qfrc = qfrc_smooth_in[worldid, dofid]
  iqacc_out[worldid, row] = qacc
  iqfrc_smooth_out[worldid, row] = qfrc
  ifrc_out[worldid, row] = 0.0


@wp.kernel
def _pool_init(
  # Model:
  M_elemid: wp.array2d[int],
  # Data in:
  nisland_in: wp.array[int],
  M_in: wp.array2d[float],
  island_nv_in: wp.array2d[int],
  # In:
  h_row2dof_in: wp.array2d[int],
  island_tilebase_in: wp.array2d[int],
  island_ntile_in: wp.array2d[int],
  island_vbase_in: wp.array2d[int],
  island_slot_in: wp.array2d[wp.vec4i],
  # Out:
  ctx_h_out: wp.array3d[float],
):
  """Upper triangle of one row of a group's block in the tile pool: M, identity on padding.

  Two-tile dense groups (solved by the dense world solver) have no block.
  """
  worldid, row = wp.tid()
  nisland = nisland_in[worldid]
  if nisland == 0:
    return

  # the owning island is the last one whose aligned rows start at or before this row
  lo = int(0)
  hi = nisland
  while hi - lo > 1:
    mid = (lo + hi) // 2
    if island_vbase_in[worldid, mid] <= row:
      lo = mid
    else:
      hi = mid
  if island_slot_in[worldid, lo][0] > _TILE:
    return
  TILE = wp.static(types.TILE_SIZE_JTDAJ_DENSE)
  ntile = island_ntile_in[worldid, lo]
  vbase = island_vbase_in[worldid, lo]
  la = row - vbase
  if la >= TILE * ntile:
    return

  n = island_nv_in[worldid, lo]
  tilebase = island_tilebase_in[worldid, lo]
  dofa = h_row2dof_in[worldid, row]
  for lb in range(la, TILE * ntile):
    value = float(0.0)
    if la < n and lb < n:
      dofb = h_row2dof_in[worldid, vbase + lb]
      elemid = M_elemid[wp.max(dofa, dofb), wp.min(dofa, dofb)]
      if elemid >= 0:
        value = M_in[worldid, elemid]
    elif la == lb:
      value = 1.0
    ctx_h_out[worldid, TILE * (tilebase + (la // TILE) * ntile + lb // TILE) + la % TILE, lb % TILE] = value


@wp.kernel
def _init_rows(
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
  """J qacc - aref of a row at the start point."""
  worldid, row = wp.tid()
  if row >= nefc_in[worldid]:
    return
  islandid = efc_island_in[worldid, row]
  block = island_slot_in[worldid, islandid]
  # dense groups wider than a tile: the dense world solver computes its own
  if block[0] > _TILE:
    return
  jaref = -efc_aref_in[worldid, row]
  if block[0] > 0:
    vbase = vbase_in[worldid, islandid]
    r = row - block[3]
    for c in range(block[0]):
      jaref += iJ_in[worldid, row, c] * iqacc_in[worldid, vbase + c]
  else:
    rowadr = efc_J_rowadr_in[worldid, row]
    for k in range(efc_J_rownnz_in[worldid, row]):
      adr = rowadr + k
      jaref += efc_J_in[worldid, 0, adr] * iqacc_in[worldid, dof2row_in[worldid, efc_J_colind_in[worldid, 0, adr]]]
  row_jaref_out[worldid, row] = jaref


@wp.func
def _block_matvec(
  # In:
  pool: wp.array2d[float],
  tilebase: int,
  ntile: int,
  vbase: int,
  ndof: int,
  x: wp.array[float],
  la: int,
) -> float:
  """Row la of a symmetric island block times the island's part of x."""
  acc = float(0.0)
  for lb in range(ndof):
    rc = _pool_rc(tilebase, ntile, la, lb)
    acc += pool[rc[0], rc[1]] * x[vbase + lb]
  return acc


@wp.func
def _copy_upper_tiles(
  # In:
  lane: int,
  nthread: int,
  tilebase: int,
  ntile: int,
  src: wp.array2d[float],
  # Out:
  dst_out: wp.array2d[float],
):
  """Copy the upper tiles of an island block (the factorization reads no others)."""
  TILE2 = wp.static(_TILE * _TILE)
  for e in range(lane, ntile * ntile * TILE2, nthread):
    t = e // TILE2
    if t // ntile <= t % ntile:
      row = _TILE * (tilebase + t) + (e % TILE2) // _TILE
      dst_out[row, e % _TILE] = src[row, e % _TILE]


@wp.func_native(snippet="WP_TILE_SYNC();")
def _sync():
  pass


@wp.kernel
def _init_islands(
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
  iqacc_in: wp.array2d[float],
  iM_in: wp.array3d[float],
  # Out:
  Ma_out: wp.array2d[float],
  H_out: wp.array3d[float],
  scale_out: wp.array2d[float],
  done_out: wp.array2d[bool],
  niter_out: wp.array2d[int],
  alpha_out: wp.array2d[float],
  improvement_out: wp.array2d[float],
  nsolving_out: wp.array[int],
):
  """Start every island: M qacc, H = M, and its convergence scale (the inertia trace)."""
  bid, lane = wp.tid()
  for e in range(bid, group_count_in[cls], nblock):
    worldid = group_list_in[cls, e] // ntree
    islandid = group_list_in[cls, e] % ntree
    if islandid >= nisland_in[worldid]:
      if lane == 0:
        done_out[worldid, islandid] = True
      continue
    nthread = wp.block_dim()
    ndof = island_nv_in[worldid, islandid]
    tilebase = tilebase_in[worldid, islandid]
    ntile = ntile_in[worldid, islandid]
    vbase = vbase_in[worldid, islandid]
    trace = float(0.0)
    for la in range(lane, ndof, nthread):
      Ma_out[worldid, vbase + la] = _block_matvec(iM_in[worldid], tilebase, ntile, vbase, ndof, iqacc_in[worldid], la)
      rc = _pool_rc(tilebase, ntile, la, la)
      trace += iM_in[worldid, rc[0], rc[1]]
    _copy_upper_tiles(lane, nthread, tilebase, ntile, iM_in[worldid], H_out[worldid])
    # MuJoCo scales an island's convergence tests by its inertia trace instead of meaninertia * nv
    trace = wp.tile_reduce(wp.add, wp.tile(trace))[0]
    if lane == 0:
      scale_out[worldid, islandid] = trace
      done_out[worldid, islandid] = False
      niter_out[worldid, islandid] = 0
      alpha_out[worldid, islandid] = 0.0
      improvement_out[worldid, islandid] = 0.0
      wp.atomic_add(nsolving_out, 0, 1)


@cache_kernel
def _rows(initial: bool):
  INITIAL = initial

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Data in:
    nefc_in: wp.array[int],
    efc_type_in: wp.array2d[int],
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
    worldid, row = wp.tid()
    if row >= nefc_in[worldid]:
      return
    islandid = efc_island_in[worldid, row]
    if done_in[worldid, islandid]:
      return
    category = _row_category(efc_type_in[worldid, row])
    D = efc_D_in[worldid, row]
    res = _eval_constraint(
      category == _ROW_EQUALITY,
      category == _ROW_FRICTION,
      False,
      row_jaref_in[worldid, row],
      D,
      efc_frictionloss_in[worldid, row],
      0,
      -1,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
    )
    state = int(res[1])
    if wp.static(not INITIAL):
      if efc_state_out[worldid, row] != state:
        state_changed_out[worldid, islandid] = True
    efc_force_out[worldid, row] = res[0]
    efc_state_out[worldid, row] = state
    quad = state == types.ConstraintState.QUADRATIC.value
    w = float(0.0)
    if wp.static(INITIAL):
      row_quad_out[worldid, row] = quad
      w = wp.where(quad, D, 0.0)
    elif quad != row_quad_out[worldid, row]:
      row_quad_out[worldid, row] = quad
      changed_out[worldid, islandid] = True
      w = wp.where(quad, D, -D)
    row_w_out[worldid, row] = w

    if island_slot_in[worldid, islandid][0] > 0:
      return
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
        wp.atomic_add(frc_out[worldid], cola, Ja * force)
      if w != 0.0:
        Jaw = Ja * w
        for kb in range(k, rownnz):
          rc = _pool_rc(tilebase, ntile, cola - vbase, dof2row[colind[rowadr + kb]] - vbase)
          wp.atomic_add(H_out[worldid, rc[0]], rc[1], Jaw * efc_J_in[worldid, 0, rowadr + kb])

  return kernel


@cache_kernel
def _grad(final: bool):
  FINAL = final

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Data in:
    efc_J_in: wp.array3d[float],
    efc_force_in: wp.array2d[float],
    dof_island_in: wp.array2d[int],
    # In:
    island_slot_in: wp.array2d[wp.vec4i],
    iJ_in: wp.array3d[float],
    row2dof_in: wp.array2d[int],
    vbase_in: wp.array2d[int],
    done_in: wp.array2d[bool],
    niter_in: wp.array2d[int],
    state_changed_in: wp.array2d[bool],
    Ma_in: wp.array2d[float],
    iqfrc_smooth_in: wp.array2d[float],
    # Out:
    frc_out: wp.array2d[float],
    grad_out: wp.array2d[float],
  ):
    """J'force (dense islands) and the gradient of an aligned island DOF row (zero on padding).

    Sparse islands' J'force is accumulated by _sparse_rows. The final pass only computes J'force
    (qfrc_constraint) of dense islands.
    """
    worldid, row = wp.tid()
    dofid = row2dof_in[worldid, row]
    if dofid < 0:
      if wp.static(not FINAL):
        grad_out[worldid, row] = 0.0
      return
    islandid = dof_island_in[worldid, dofid]
    if wp.static(not FINAL):
      if done_in[worldid, islandid] or _stable(niter_in[worldid, islandid], state_changed_in[worldid, islandid]):
        return
    block = island_slot_in[worldid, islandid]
    frc = float(0.0)
    if block[0] == _TILE:
      col = row - vbase_in[worldid, islandid]
      for r in range(block[2]):
        force = efc_force_in[worldid, block[3] + r]
        if force != 0.0:
          frc += iJ_in[worldid, block[3] + r, col] * force
      frc_out[worldid, row] = frc
    elif wp.static(FINAL) or block[0] > 0:
      return
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


@cache_kernel
def _dense_factor(ntile: int):
  N = ntile * _TILE

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
    grad_in: wp.array2d[float],
    H_in: wp.array3d[float],
    # Out:
    search_out: wp.array2d[float],
    search_dot_out: wp.array2d[float],
    newton_decrement_out: wp.array2d[float],
    grad_scale_out: wp.array2d[float],
  ):
    """Newton search direction of an island of ntile tiles per side, factored as one tile.

    As _update_gradient_cholesky for a world: the island's H (the pool's upper tiles) is factored
    in shared memory every iteration.
    """
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
      # the block's upper tiles (0, 0), (0, 1), (1, 1); the factorization reads no others
      U = wp.tile_zeros(shape=(N, N), dtype=float, storage="shared")
      wp.tile_assign(U, wp.tile_load(H_in[worldid], shape=(_TILE, _TILE), offset=(row0, 0)), offset=(0, 0))
      if wp.static(ntile == 2):
        wp.tile_assign(U, wp.tile_load(H_in[worldid], shape=(_TILE, _TILE), offset=(row0 + _TILE, 0)), offset=(0, _TILE))
        wp.tile_assign(
          U, wp.tile_load(H_in[worldid], shape=(_TILE, _TILE), offset=(row0 + 3 * _TILE, 0)), offset=(_TILE, _TILE)
        )
      wp.tile_cholesky_inplace(U, fill_mode="upper")
      b = wp.tile_load(grad_in[worldid], shape=N, offset=vbase)
      x = wp.tile_cholesky_solve(U, b, fill_mode="upper")
      wp.tile_store(search_out[worldid], wp.tile_map(wp.neg, x), offset=vbase)
      sums = wp.tile_reduce(wp.add, wp.tile_map(solve_search_sums, b, x))[0]
      if lane == 0:
        search_dot_out[worldid, islandid] = sums[0]
        newton_decrement_out[worldid, islandid] = sums[1]
        grad_scale_out[worldid, islandid] = 1.0

  return kernel


@cache_kernel
def _newton_step(ls_iterations: int, warn_overflow: int):
  LS_ITERATIONS = ls_iterations

  @wp.func
  def step(
    # Model:
    opt_tolerance: wp.array[float],
    opt_ls_tolerance: wp.array[float],
    opt_iterations: int,
    # Data in:
    nisland_in: wp.array[int],
    efc_type_in: wp.array2d[int],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    efc_D_in: wp.array2d[float],
    efc_frictionloss_in: wp.array2d[float],
    island_nv_in: wp.array2d[int],
    island_nefc_in: wp.array2d[int],
    island_iefcadr_in: wp.array2d[int],
    # In:
    worldid: int,
    islandid: int,
    lane: int,
    tilebase_in: wp.array2d[int],
    ntile_in: wp.array2d[int],
    vbase_in: wp.array2d[int],
    scale_in: wp.array2d[float],
    iqfrc_smooth_in: wp.array2d[float],
    iM_in: wp.array3d[float],
    search_in: wp.array2d[float],
    island_slot_in: wp.array2d[wp.vec4i],
    dof2row_in: wp.array2d[int],
    iJ_in: wp.array3d[float],
    grad_in: wp.array2d[float],
    search_dot_in: wp.array2d[float],
    newton_decrement_in: wp.array2d[float],
    state_changed_in: wp.array2d[bool],
    # Data out:
    solver_niter_out: wp.array[int],
    overflow_out: wp.array[int],
    # Out:
    done_out: wp.array2d[bool],
    niter_out: wp.array2d[int],
    alpha_out: wp.array2d[float],
    improvement_out: wp.array2d[float],
    iqacc_out: wp.array2d[float],
    Ma_out: wp.array2d[float],
    frc_out: wp.array2d[float],
    mv_out: wp.array2d[float],
    row_jaref_out: wp.array2d[float],
    row_jv_out: wp.array2d[float],
    nsolving_out: wp.array[int],
    grad_scale_out: wp.array2d[float],
  ):
    """Convergence test, linesearch and update of one group (see _linesearch_iterative_kernel)."""
    nthread = wp.block_dim()
    ndof = island_nv_in[worldid, islandid]
    nrow = island_nefc_in[worldid, islandid]
    rowadr0 = island_iefcadr_in[worldid, islandid]
    tilebase = tilebase_in[worldid, islandid]
    ntile = ntile_in[worldid, islandid]
    vbase = vbase_in[worldid, islandid]
    scale = scale_in[worldid, islandid]
    block = island_slot_in[worldid, islandid]
    tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]
    ls_tolerance = opt_ls_tolerance[worldid % opt_ls_tolerance.shape[0]]
    niter = niter_out[worldid, islandid]

    # --- termination (see _solve_done); the gradient is grad_scale * the stored one ---
    stable = _stable(niter, state_changed_in[worldid, islandid])
    grad_scale = grad_scale_out[worldid, islandid]
    grad_dot = float(0.0)
    for la in range(lane, ndof, nthread):
      grad_dot += grad_in[worldid, vbase + la] * grad_in[worldid, vbase + la]
    grad_dot = wp.tile_reduce(wp.add, wp.tile(grad_dot))[0] * grad_scale * grad_scale
    converged = False
    if niter > 0:
      improvement = improvement_out[worldid, islandid]
      converged = (
        (alpha_out[worldid, islandid] == 0.0)
        or (improvement > 0.0 and improvement / scale < tolerance)
        or (wp.sqrt(grad_dot) / scale < tolerance)
        or (0.5 * newton_decrement_in[worldid, islandid] * grad_scale * grad_scale / scale < tolerance)
      )
    if converged or niter >= opt_iterations:
      if lane == 0:
        done_out[worldid, islandid] = True
        if not converged and opt_iterations > 0:
          if wp.static(bool(warn_overflow & OverflowType.ITERATIONS)):
            wp.printf(
              "solver iterations limit reached - please increase iterations beyond %u\n"
              "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.ITERATIONS (or = 0 for all)\n",
              opt_iterations,
            )
          wp.atomic_or(overflow_out, worldid, wp.static(OverflowType.ITERATIONS))
        wp.atomic_max(solver_niter_out, worldid, niter)
      return

    # --- linesearch along search ---
    search = search_in[worldid]
    local_dof = wp.vec2(0.0)
    for la in range(lane, ndof, nthread):
      row = vbase + la
      # an unchanged search direction keeps its M search and J search
      mv = float(0.0)
      if stable:
        mv = mv_out[worldid, row]
      else:
        mv = _block_matvec(iM_in[worldid], tilebase, ntile, vbase, ndof, search, la)
        mv_out[worldid, row] = mv
      s = search[row]
      local_dof += wp.vec2(s * (Ma_out[worldid, row] - iqfrc_smooth_in[worldid, row]), s * mv)
    local_gauss = wp.vec2(0.0)
    local_p0 = wp.vec3(0.0)
    for r in range(lane, nrow, nthread):
      row = rowadr0 + r
      jv = float(0.0)
      if stable:
        jv = row_jv_out[worldid, row]
      else:
        if block[0] > 0:
          for c in range(block[0]):
            jv += iJ_in[worldid, row, c] * search[vbase + c]
        else:
          rowadr = efc_J_rowadr_in[worldid, row]
          for k in range(efc_J_rownnz_in[worldid, row]):
            adr = rowadr + k
            jv += efc_J_in[worldid, 0, adr] * search[dof2row_in[worldid, efc_J_colind_in[worldid, 0, adr]]]
        row_jv_out[worldid, row] = jv
      category = _row_category(efc_type_in[worldid, row])
      D = efc_D_in[worldid, row]
      jaref = row_jaref_out[worldid, row]
      if category == _ROW_EQUALITY:
        jvD = jv * D
        local_gauss += wp.vec2(jvD * jaref, 0.5 * jv * jvD)
      else:
        local_p0 += _row_eval_pt_alpha_zero(category, D, efc_frictionloss_in[worldid, row], jaref, jv)
    dof_sums = wp.tile_reduce(wp.add, wp.tile(local_dof, preserve_type=True))[0]
    gauss_rows = wp.tile_reduce(wp.add, wp.tile(local_gauss, preserve_type=True))[0]
    p0_rows = wp.tile_reduce(wp.add, wp.tile(local_p0, preserve_type=True))[0]
    _sync()  # jv of all rows is written

    gauss1 = gauss_rows[0] + dof_sums[0]
    gauss2 = gauss_rows[1] + 0.5 * dof_sums[1]
    quad_gauss = wp.vec3(0.0, gauss1, gauss2)

    gtol = wp.max(tolerance * ls_tolerance * wp.sqrt(search_dot_in[worldid, islandid]) * scale, 1e-6)
    p0 = wp.vec3(0.0, gauss1, 2.0 * gauss2) + p0_rows
    p0_delta = wp.vec3(0.0, p0[1], p0[2])
    q1_abs = wp.sqrt(2.0 * wp.max(p0_rows[0], 0.0) * wp.max(p0_rows[2], 0.0)) + wp.abs(gauss1)
    gtol_accept = wp.max(gtol, _ALPHA_NOISE_EPS * q1_abs)

    lo_alpha_in = -wp.where(p0[2] != 0.0, p0[1] / p0[2], 0.0)
    local_lo_in = wp.vec3(0.0)
    for r in range(lane, nrow, nthread):
      row = rowadr0 + r
      category = _row_category(efc_type_in[worldid, row])
      if category != _ROW_EQUALITY:
        local_lo_in += _row_eval_pt(
          category,
          lo_alpha_in,
          efc_D_in[worldid, row],
          efc_frictionloss_in[worldid, row],
          row_jaref_out[worldid, row],
          row_jv_out[worldid, row],
        )
    lo_in = _eval_pt(quad_gauss, lo_alpha_in) + wp.tile_reduce(wp.add, wp.tile(local_lo_in, preserve_type=True))[0]

    alpha = float(0.0)
    improvement = float(0.0)
    # a slope at alpha = 0 below rounding noise means the start is already the line minimum
    stationary = wp.abs(p0[1]) < _ALPHA_NOISE_EPS * q1_abs
    ls_converged = stationary or (wp.abs(lo_in[1]) < gtol_accept and lo_in[0] < 0.0)
    if stationary:
      pass
    elif ls_converged:
      alpha = lo_alpha_in
      improvement = -lo_in[0]
    else:
      lo_less = lo_in[1] < p0[1]
      lo = wp.where(lo_less, lo_in, p0_delta)
      lo_alpha = wp.where(lo_less, lo_alpha_in, 0.0)
      hi = wp.where(lo_less, p0_delta, lo_in)
      hi_alpha = wp.where(lo_less, 0.0, lo_alpha_in)

      for _ in range(LS_ITERATIONS):
        lo_next_alpha = lo_alpha - wp.where(lo[2] != 0.0, lo[1] / lo[2], 0.0)
        hi_next_alpha = hi_alpha - wp.where(hi[2] != 0.0, hi[1] / hi[2], 0.0)
        mid_alpha = 0.5 * (lo_alpha + hi_alpha)

        local_combined = wp.mat33(0.0)
        for r in range(lane, nrow, nthread):
          row = rowadr0 + r
          category = _row_category(efc_type_in[worldid, row])
          if category != _ROW_EQUALITY:
            D = efc_D_in[worldid, row]
            f = efc_frictionloss_in[worldid, row]
            jaref = row_jaref_out[worldid, row]
            jv = row_jv_out[worldid, row]
            plo = _row_eval_pt(category, lo_next_alpha, D, f, jaref, jv)
            phi = _row_eval_pt(category, hi_next_alpha, D, f, jaref, jv)
            pmid = _row_eval_pt(category, mid_alpha, D, f, jaref, jv)
            # columns: lo, hi, mid
            local_combined += wp.mat33(plo[0], phi[0], pmid[0], plo[1], phi[1], pmid[1], plo[2], phi[2], pmid[2])
        result = wp.tile_reduce(wp.add, wp.tile(local_combined, preserve_type=True))[0]
        gauss_lo, gauss_hi, gauss_mid = _eval_pt_3alphas(quad_gauss, lo_next_alpha, hi_next_alpha, mid_alpha)
        lo_next = gauss_lo + wp.vec3(result[0, 0], result[1, 0], result[2, 0])
        hi_next = gauss_hi + wp.vec3(result[0, 1], result[1, 1], result[2, 1])
        mid = gauss_mid + wp.vec3(result[0, 2], result[1, 2], result[2, 2])

        conv_lo = wp.abs(lo_next[1]) < gtol_accept and lo_next[0] < 0.0
        conv_hi = wp.abs(hi_next[1]) < gtol_accept and hi_next[0] < 0.0
        conv_mid = wp.abs(mid[1]) < gtol_accept and mid[0] < 0.0
        any_converged = conv_lo or conv_hi or conv_mid
        swap_lo = False
        swap_hi = False
        if any_converged:
          conv_pt = wp.vec3(types.MJ_MAXVAL, 0.0, 0.0)
          conv_alpha = float(0.0)
          if conv_lo and lo_next[0] < conv_pt[0]:
            conv_pt = lo_next
            conv_alpha = lo_next_alpha
          if conv_hi and hi_next[0] < conv_pt[0]:
            conv_pt = hi_next
            conv_alpha = hi_next_alpha
          if conv_mid and mid[0] < conv_pt[0]:
            conv_pt = mid
            conv_alpha = mid_alpha
          lo = conv_pt
          lo_alpha = conv_alpha
          hi = conv_pt
          hi_alpha = conv_alpha
        else:
          swap_lo_lo_next = _in_bracket(lo, lo_next)
          lo = wp.where(swap_lo_lo_next, lo_next, lo)
          lo_alpha = wp.where(swap_lo_lo_next, lo_next_alpha, lo_alpha)
          swap_lo_mid = _in_bracket(lo, mid)
          lo = wp.where(swap_lo_mid, mid, lo)
          lo_alpha = wp.where(swap_lo_mid, mid_alpha, lo_alpha)
          swap_lo_hi_next = _in_bracket(lo, hi_next)
          lo = wp.where(swap_lo_hi_next, hi_next, lo)
          lo_alpha = wp.where(swap_lo_hi_next, hi_next_alpha, lo_alpha)
          swap_lo = swap_lo_lo_next or swap_lo_mid or swap_lo_hi_next

          swap_hi_hi_next = _in_bracket(hi, hi_next) or (hi[1] < 0.0 and hi_next[1] > 0.0)
          hi = wp.where(swap_hi_hi_next, hi_next, hi)
          hi_alpha = wp.where(swap_hi_hi_next, hi_next_alpha, hi_alpha)
          swap_hi_mid = _in_bracket(hi, mid)
          hi = wp.where(swap_hi_mid, mid, hi)
          hi_alpha = wp.where(swap_hi_mid, mid_alpha, hi_alpha)
          swap_hi_lo_next = _in_bracket(hi, lo_next)
          hi = wp.where(swap_hi_lo_next, lo_next, hi)
          hi_alpha = wp.where(swap_hi_lo_next, lo_next_alpha, hi_alpha)
          swap_hi = swap_hi_hi_next or swap_hi_mid or swap_hi_lo_next

        ls_done = (
          any_converged
          or (not swap_lo and not swap_hi)
          or (lo[0] < 0.0 and lo[1] < 0.0 and lo[1] > -gtol)
          or (hi[0] < 0.0 and hi[1] > 0.0 and hi[1] < gtol)
        )
        improved = lo[0] < 0.0 or hi[0] < 0.0
        lo_better = lo[0] < hi[0]
        alpha = wp.where(improved, wp.where(lo_better, lo_alpha, hi_alpha), alpha)
        improvement = wp.where(improved, -wp.where(lo_better, lo[0], hi[0]), improvement)
        if ls_done:
          ls_converged = True
          break

    if not ls_converged and lane == 0:
      if wp.static(bool(warn_overflow & OverflowType.LS_ITERATIONS)):
        wp.printf(
          "linesearch iterations limit reached - please increase ls_iterations beyond %u\n"
          "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.LS_ITERATIONS (or = 0 for all)\n",
          wp.static(LS_ITERATIONS),
        )
      wp.atomic_or(overflow_out, worldid, wp.static(OverflowType.LS_ITERATIONS))

    # --- update; sparse islands' J'force is accumulated again at the new point ---
    for la in range(lane, ndof, nthread):
      row = vbase + la
      iqacc_out[worldid, row] += alpha * search[row]
      Ma_out[worldid, row] += alpha * mv_out[worldid, row]
      if block[0] == 0:
        frc_out[worldid, row] = 0.0
    for r in range(lane, nrow, nthread):
      row = rowadr0 + r
      row_jaref_out[worldid, row] += alpha * row_jv_out[worldid, row]
    if lane == 0:
      niter_out[worldid, islandid] = niter + 1
      alpha_out[worldid, islandid] = alpha
      grad_scale_out[worldid, islandid] = grad_scale - alpha
      improvement_out[worldid, islandid] = improvement
      wp.atomic_add(nsolving_out, 0, 1)

  return step


@cache_kernel
def _linesearch(ls_iterations: int, warn_overflow: int):
  step = _newton_step(ls_iterations, warn_overflow)

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    ntree: int,
    opt_tolerance: wp.array[float],
    opt_ls_tolerance: wp.array[float],
    opt_iterations: int,
    # Data in:
    nisland_in: wp.array[int],
    efc_type_in: wp.array2d[int],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    efc_D_in: wp.array2d[float],
    efc_frictionloss_in: wp.array2d[float],
    island_nv_in: wp.array2d[int],
    island_nefc_in: wp.array2d[int],
    island_iefcadr_in: wp.array2d[int],
    # In:
    group_count_in: wp.array[int],
    group_list_in: wp.array2d[int],
    cls: int,
    nblock: int,
    tilebase_in: wp.array2d[int],
    ntile_in: wp.array2d[int],
    vbase_in: wp.array2d[int],
    scale_in: wp.array2d[float],
    iqfrc_smooth_in: wp.array2d[float],
    iM_in: wp.array3d[float],
    search_in: wp.array2d[float],
    island_slot_in: wp.array2d[wp.vec4i],
    dof2row_in: wp.array2d[int],
    iJ_in: wp.array3d[float],
    grad_in: wp.array2d[float],
    search_dot_in: wp.array2d[float],
    newton_decrement_in: wp.array2d[float],
    state_changed_in: wp.array2d[bool],
    # Data out:
    solver_niter_out: wp.array[int],
    overflow_out: wp.array[int],
    # Out:
    done_out: wp.array2d[bool],
    niter_out: wp.array2d[int],
    alpha_out: wp.array2d[float],
    improvement_out: wp.array2d[float],
    iqacc_out: wp.array2d[float],
    Ma_out: wp.array2d[float],
    frc_out: wp.array2d[float],
    mv_out: wp.array2d[float],
    row_jaref_out: wp.array2d[float],
    row_jv_out: wp.array2d[float],
    nsolving_out: wp.array[int],
    grad_scale_out: wp.array2d[float],
  ):
    """Convergence test, linesearch and update of one island (see _linesearch_iterative_kernel)."""
    bid, lane = wp.tid()
    for e in range(bid, group_count_in[cls], nblock):
      worldid = group_list_in[cls, e] // ntree
      islandid = group_list_in[cls, e] % ntree
      if islandid >= nisland_in[worldid]:
        continue
      if done_out[worldid, islandid]:
        continue
      step(
        opt_tolerance,
        opt_ls_tolerance,
        opt_iterations,
        nisland_in,
        efc_type_in,
        efc_J_rownnz_in,
        efc_J_rowadr_in,
        efc_J_colind_in,
        efc_J_in,
        efc_D_in,
        efc_frictionloss_in,
        island_nv_in,
        island_nefc_in,
        island_iefcadr_in,
        worldid,
        islandid,
        lane,
        tilebase_in,
        ntile_in,
        vbase_in,
        scale_in,
        iqfrc_smooth_in,
        iM_in,
        search_in,
        island_slot_in,
        dof2row_in,
        iJ_in,
        grad_in,
        search_dot_in,
        newton_decrement_in,
        state_changed_in,
        solver_niter_out,
        overflow_out,
        done_out,
        niter_out,
        alpha_out,
        improvement_out,
        iqacc_out,
        Ma_out,
        frc_out,
        mv_out,
        row_jaref_out,
        row_jv_out,
        nsolving_out,
        grad_scale_out,
      )

  return kernel


@wp.func
def _quad_weight(state: int, D: float, active: float) -> float:
  return wp.where(state == types.ConstraintState.QUADRATIC.value and active != 0.0, D, 0.0)


@cache_kernel
def _solve_small(ls_iterations: int, warn_overflow: int):
  step = _newton_step(ls_iterations, warn_overflow)

  @wp.kernel(module="unique", enable_backward=False, module_options={"enable_mathdx_gemm": False})
  def kernel(
    # Model:
    ntree: int,
    opt_tolerance: wp.array[float],
    opt_ls_tolerance: wp.array[float],
    opt_iterations: int,
    # Data in:
    nisland_in: wp.array[int],
    efc_type_in: wp.array2d[int],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    efc_D_in: wp.array2d[float],
    efc_frictionloss_in: wp.array2d[float],
    island_nv_in: wp.array2d[int],
    island_nefc_in: wp.array2d[int],
    island_iefcadr_in: wp.array2d[int],
    # In:
    group_count_in: wp.array[int],
    group_list_in: wp.array2d[int],
    nblock: int,
    tilebase_in: wp.array2d[int],
    ntile_in: wp.array2d[int],
    vbase_in: wp.array2d[int],
    iqfrc_smooth_in: wp.array2d[float],
    iM_in: wp.array3d[float],
    island_slot_in: wp.array2d[wp.vec4i],
    dof2row_in: wp.array2d[int],
    iJ_in: wp.array3d[float],
    # Data out:
    solver_niter_out: wp.array[int],
    efc_force_out: wp.array2d[float],
    efc_state_out: wp.array2d[int],
    overflow_out: wp.array[int],
    # Out:
    scale_out: wp.array2d[float],
    grad_out: wp.array2d[float],
    search_out: wp.array2d[float],
    search_dot_out: wp.array2d[float],
    newton_decrement_out: wp.array2d[float],
    state_changed_out: wp.array2d[bool],
    done_out: wp.array2d[bool],
    niter_out: wp.array2d[int],
    alpha_out: wp.array2d[float],
    improvement_out: wp.array2d[float],
    iqacc_out: wp.array2d[float],
    Ma_out: wp.array2d[float],
    frc_out: wp.array2d[float],
    mv_out: wp.array2d[float],
    row_jaref_out: wp.array2d[float],
    row_jv_out: wp.array2d[float],
    nsolving_out: wp.array[int],
    grad_scale_out: wp.array2d[float],
    row_quad_out: wp.array2d[bool],
    H_out: wp.array3d[float],
  ):
    """The whole Newton solve of one-tile groups (e.g. free bodies), one block per group.

    Each iteration evaluates the rows, forms J'force, the gradient and H = M + J'DJ (one tile, from
    scratch), factors it and runs the shared step (convergence test, linesearch, update).
    """
    bid, lane = wp.tid()
    nthread = wp.block_dim()
    for e in range(bid, group_count_in[4], nblock):
      worldid = group_list_in[4, e] // ntree
      islandid = group_list_in[4, e] % ntree
      ndof = island_nv_in[worldid, islandid]
      nrow = island_nefc_in[worldid, islandid]
      rowadr0 = island_iefcadr_in[worldid, islandid]
      tilebase = tilebase_in[worldid, islandid]
      vbase = vbase_in[worldid, islandid]
      block = island_slot_in[worldid, islandid]
      # start: M qacc and the convergence scale (the inertia trace)
      trace = float(0.0)
      for la in range(lane, ndof, nthread):
        Ma_out[worldid, vbase + la] = _block_matvec(iM_in[worldid], tilebase, 1, vbase, ndof, iqacc_out[worldid], la)
        trace += iM_in[worldid, _TILE * tilebase + la, la]
      trace = wp.tile_reduce(wp.add, wp.tile(trace))[0]
      if lane == 0:
        scale_out[worldid, islandid] = trace
        done_out[worldid, islandid] = False
        niter_out[worldid, islandid] = 0
        alpha_out[worldid, islandid] = 0.0
        improvement_out[worldid, islandid] = 0.0
      _sync()
      for _ in range(opt_iterations + 1):
        # constraint state at the current point
        for r in range(lane, nrow, nthread):
          row = rowadr0 + r
          category = _row_category(efc_type_in[worldid, row])
          res = _eval_constraint(
            category == _ROW_EQUALITY,
            category == _ROW_FRICTION,
            False,
            row_jaref_out[worldid, row],
            efc_D_in[worldid, row],
            efc_frictionloss_in[worldid, row],
            0,
            -1,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
          )
          efc_force_out[worldid, row] = res[0]
          efc_state_out[worldid, row] = int(res[1])
          row_quad_out[worldid, row] = int(res[1]) == types.ConstraintState.QUADRATIC.value
        _sync()
        # J'force and the gradient
        for la in range(lane, ndof, nthread):
          frc = float(0.0)
          for r in range(nrow):
            force = efc_force_out[worldid, rowadr0 + r]
            if force != 0.0:
              frc += iJ_in[worldid, rowadr0 + r, la] * force
          frc_out[worldid, vbase + la] = frc
          grad_out[worldid, vbase + la] = Ma_out[worldid, vbase + la] - iqfrc_smooth_in[worldid, vbase + la] - frc
        # H = M + J' diag(D of quadratic rows) J, then the search direction
        acc = wp.tile_zeros(shape=(_TILE, _TILE), dtype=float)
        for r in range(0, nrow, _TILE):
          active = wp.tile_map(
            _active_check, wp.tile_arange(_TILE, dtype=int), wp.tile_ones(shape=_TILE, dtype=int) * (nrow - r)
          )
          w = wp.tile_map(
            _quad_weight,
            wp.tile_load(efc_state_out[worldid], shape=_TILE, offset=rowadr0 + r, bounds_check=True),
            wp.tile_load(efc_D_in[worldid], shape=_TILE, offset=rowadr0 + r, bounds_check=True),
            active,
          )
          J = wp.tile_load(iJ_in[worldid], shape=(_TILE, _TILE), offset=(rowadr0 + r, 0), bounds_check=True)
          # rows past the block are other data, possibly uninitialized: select, not multiply
          J = wp.tile_map(_select, wp.tile_broadcast(wp.tile_reshape(active, shape=(_TILE, 1)), shape=(_TILE, _TILE)), J)
          acc += wp.tile_matmul(wp.tile_map(wp.mul, wp.tile_transpose(J), wp.tile_broadcast(w, shape=(_TILE, _TILE))), J)
        # the pool holds M's upper triangle (identity on padding); the factorization reads no other
        H = wp.tile_load(iM_in[worldid], shape=(_TILE, _TILE), offset=(_TILE * tilebase, 0)) + acc
        # H is kept in the pool as for the other groups
        wp.tile_store(H_out[worldid], H, offset=(_TILE * tilebase, 0))
        U = wp.tile_zeros(shape=(_TILE, _TILE), dtype=float, storage="shared")
        wp.tile_assign(U, H, offset=(0, 0))
        wp.tile_cholesky_inplace(U, fill_mode="upper")
        b = wp.tile_load(grad_out[worldid], shape=_TILE, offset=vbase)
        x = wp.tile_cholesky_solve(U, b, fill_mode="upper")
        wp.tile_store(search_out[worldid], wp.tile_map(wp.neg, x), offset=vbase)
        sums = wp.tile_reduce(wp.add, wp.tile_map(solve_search_sums, b, x))[0]
        if lane == 0:
          search_dot_out[worldid, islandid] = sums[0]
          newton_decrement_out[worldid, islandid] = sums[1]
          grad_scale_out[worldid, islandid] = 1.0
          # a new direction every iteration (no stable fast path)
          state_changed_out[worldid, islandid] = True
        _sync()
        step(
          opt_tolerance,
          opt_ls_tolerance,
          opt_iterations,
          nisland_in,
          efc_type_in,
          efc_J_rownnz_in,
          efc_J_rowadr_in,
          efc_J_colind_in,
          efc_J_in,
          efc_D_in,
          efc_frictionloss_in,
          island_nv_in,
          island_nefc_in,
          island_iefcadr_in,
          worldid,
          islandid,
          lane,
          tilebase_in,
          ntile_in,
          vbase_in,
          scale_out,
          iqfrc_smooth_in,
          iM_in,
          search_out,
          island_slot_in,
          dof2row_in,
          iJ_in,
          grad_out,
          search_dot_out,
          newton_decrement_out,
          state_changed_out,
          solver_niter_out,
          overflow_out,
          done_out,
          niter_out,
          alpha_out,
          improvement_out,
          iqacc_out,
          Ma_out,
          frc_out,
          mv_out,
          row_jaref_out,
          row_jv_out,
          nsolving_out,
          grad_scale_out,
        )
        _sync()
        if done_out[worldid, islandid]:
          break

  return kernel


# Groups of 17 to 32 DOFs (e.g. a humanoid) are solved by the dense world solver, each group one
# problem of m.island_dense_nv DOFs: its rows are the group's rows in place (SolverContext.efc_world
# and efc_rowoff), its Jacobian the dense rows (island.SlotBuffers.J), its inertia the world's
# sparse M.


@wp.kernel
def _dense_scalars(
  # Model:
  ntree: int,
  opt_tolerance: wp.array[float],
  opt_ls_tolerance: wp.array[float],
  island_dense_nv: int,
  M_elemid: wp.array2d[int],
  # Data in:
  M_in: wp.array2d[float],
  map_idof2dof_in: wp.array2d[int],
  # In:
  group_count_in: wp.array[int],
  group_list_in: wp.array2d[int],
  group_idofadr_in: wp.array2d[int],
  group_nv_in: wp.array2d[int],
  group_nefc_in: wp.array2d[int],
  group_ne_in: wp.array2d[int],
  group_nf_in: wp.array2d[int],
  group_iefcadr_in: wp.array2d[int],
  # Out:
  dense_nefc_out: wp.array[int],
  dense_ne_out: wp.array[int],
  dense_nf_out: wp.array[int],
  dense_meaninertia_out: wp.array[float],
  dense_tolerance_out: wp.array[float],
  dense_ls_tolerance_out: wp.array[float],
  dense_efc_world_out: wp.array[int],
  dense_efc_rowoff_out: wp.array[int],
):
  """Sizes, rows and tolerances of each dense problem."""
  p = wp.tid()
  if p >= group_count_in[6]:
    return
  worldid = group_list_in[6, p] // ntree
  g = group_list_in[6, p] % ntree
  dense_nefc_out[p] = group_nefc_in[worldid, g]
  dense_ne_out[p] = group_ne_in[worldid, g]
  dense_nf_out[p] = group_nf_in[worldid, g]
  dense_efc_world_out[p] = worldid
  dense_efc_rowoff_out[p] = group_iefcadr_in[worldid, g]
  # MuJoCo scales an island's convergence tests by its inertia trace: meaninertia * nv = trace
  idofadr = group_idofadr_in[worldid, g]
  trace = float(0.0)
  for k in range(group_nv_in[worldid, g]):
    dofk = map_idof2dof_in[worldid, idofadr + k]
    trace += M_in[worldid, M_elemid[dofk, dofk]]
  dense_meaninertia_out[p] = trace / float(island_dense_nv)
  dense_tolerance_out[p] = opt_tolerance[worldid % opt_tolerance.shape[0]]
  dense_ls_tolerance_out[p] = opt_ls_tolerance[worldid % opt_ls_tolerance.shape[0]]


@wp.kernel
def _dense_dofs(
  # Model:
  island_dense_nv: int,
  # Data in:
  qacc_warmstart_in: wp.array2d[float],
  qfrc_smooth_in: wp.array2d[float],
  qacc_smooth_in: wp.array2d[float],
  # In:
  dof_group_in: wp.array2d[int],
  dof_slotcol_in: wp.array2d[int],
  group_nv_in: wp.array2d[int],
  group_problem_in: wp.array2d[int],
  # Out:
  dense_qacc_warmstart_out: wp.array2d[float],
  dense_qfrc_smooth_out: wp.array2d[float],
  dense_qacc_smooth_out: wp.array2d[float],
  dense_problem_dof_out: wp.array2d[int],
):
  """DOF vectors and maps of the dense problems, one thread per world DOF.

  The thread of local DOF i also writes padding entries n + i, n + i + n, ...
  """
  worldid, dofid = wp.tid()
  g = dof_group_in[worldid, dofid]
  p = int(-1)
  if g >= 0:
    p = group_problem_in[worldid, g]
  if p < 0:
    return
  i = dof_slotcol_in[worldid, dofid]
  dense_qacc_warmstart_out[p, i] = qacc_warmstart_in[worldid, dofid]
  dense_qfrc_smooth_out[p, i] = qfrc_smooth_in[worldid, dofid]
  dense_qacc_smooth_out[p, i] = qacc_smooth_in[worldid, dofid]
  dense_problem_dof_out[p, i] = dofid
  n = group_nv_in[worldid, g]
  for r in range(n + i, island_dense_nv, n):
    dense_qacc_warmstart_out[p, r] = 0.0
    dense_qfrc_smooth_out[p, r] = 0.0
    dense_qacc_smooth_out[p, r] = 0.0
    dense_problem_dof_out[p, r] = -1


@wp.kernel
def _dense_rowmap(
  # Data in:
  nefc_in: wp.array[int],
  # In:
  efc_group_in: wp.array2d[int],
  group_problem_in: wp.array2d[int],
  # Out:
  efc_problem_out: wp.array2d[int],
):
  """Dense problem of each row (-1 for other rows)."""
  worldid, efcid = wp.tid()
  p = int(-1)
  if efcid < nefc_in[worldid]:
    g = efc_group_in[worldid, efcid]
    if g >= 0:
      p = group_problem_in[worldid, g]
  efc_problem_out[worldid, efcid] = p


@wp.kernel
def _dense_scatter(
  # In:
  dof_group_in: wp.array2d[int],
  dof_slotcol_in: wp.array2d[int],
  group_problem_in: wp.array2d[int],
  dense_qacc_in: wp.array2d[float],
  dense_qfrc_constraint_in: wp.array2d[float],
  dense_niter_in: wp.array[int],
  dense_overflow_in: wp.array[int],
  # Data out:
  solver_niter_out: wp.array[int],
  qacc_out: wp.array2d[float],
  qfrc_constraint_out: wp.array2d[float],
  overflow_out: wp.array[int],
):
  """Copy the dense problems' solution to their DOFs (rows' force and state are in place)."""
  worldid, dofid = wp.tid()
  g = dof_group_in[worldid, dofid]
  if g < 0:
    return
  p = group_problem_in[worldid, g]
  if p < 0:
    return
  i = dof_slotcol_in[worldid, dofid]
  qacc_out[worldid, dofid] = dense_qacc_in[p, i]
  qfrc_constraint_out[worldid, dofid] = dense_qfrc_constraint_in[p, i]
  if i == 0:
    wp.atomic_max(solver_niter_out, worldid, dense_niter_in[p])
    wp.atomic_or(overflow_out, worldid, dense_overflow_in[p])


@dataclasses.dataclass
class _Dense:
  m: types.Model
  d: types.Data
  ctx: object
  nsolving: wp.array[int]


def _create_dense(m: types.Model, d: types.Data, ctx: IslandContext, g: island.SlotBuffers) -> _Dense:
  """Model, data and solver context of the dense problems (all arrays allocated here).

  A problem's rows are its group's rows in d.efc; only DOF arrays are per problem.
  """
  from mujoco_warp._src import solver  # noqa: PLC0415 (solver dispatches to this module)

  N = m.island_dense_nv
  # a world has at most nv // 17 groups of 17 or more DOFs
  ns = d.nworld * max(1, min(m.ntree, m.nv // (_TILE + 1)))
  gc = m.opt.graph_conditional and wp.get_device().is_cuda
  opt = dataclasses.replace(
    m.opt, graph_conditional=gc, tolerance=wp.empty(ns, dtype=float), ls_tolerance=wp.empty(ns, dtype=float)
  )
  stat = dataclasses.replace(m.stat, meaninertia=wp.empty(ns, dtype=float))
  m_dense = dataclasses.replace(m, opt=opt, stat=stat, nv=N, nv_pad=N, is_sparse=False)
  efc = dataclasses.replace(d.efc, J=g.J, Ma=wp.zeros((ns, N), dtype=float))
  d_dense = dataclasses.replace(
    d,
    nworld=ns,
    nefc=wp.empty(ns, dtype=int),
    ne=wp.empty(ns, dtype=int),
    nf=wp.empty(ns, dtype=int),
    qacc_warmstart=wp.empty((ns, N), dtype=float),
    qfrc_smooth=wp.empty((ns, N), dtype=float),
    qacc_smooth=wp.empty((ns, N), dtype=float),
    qacc=wp.zeros((ns, N), dtype=float),
    qfrc_constraint=wp.zeros((ns, N), dtype=float),
    solver_niter=wp.zeros(ns, dtype=int),
    overflow=wp.zeros(ns, dtype=int),
    efc=efc,
  )
  # row workspace in the worlds' row layout, not per problem
  sctx = solver._create_solver_context(m_dense, dataclasses.replace(d_dense, njmax=0))
  sctx = dataclasses.replace(
    sctx,
    Jaref=wp.empty((d.nworld, d.njmax), dtype=float),
    jv=wp.empty((d.nworld, d.njmax), dtype=float),
    quad=wp.empty((d.nworld, d.njmax), dtype=wp.vec3),
    quad_changed_ids=wp.empty((d.nworld, d.njmax), dtype=int),
    efc_world=wp.empty(ns, dtype=int),
    efc_rowoff=wp.empty(ns, dtype=int),
    efc_problem=wp.empty((d.nworld, d.njmax), dtype=int),
    nproblem=ctx.group_count[6:7],
    problem_dof=wp.empty((ns, N), dtype=int),
  )
  return _Dense(m=m_dense, d=d_dense, ctx=sctx, nsolving=wp.empty(1, dtype=int))


def _solve_dense(m: types.Model, d: types.Data, ctx: IslandContext, g: island.SlotBuffers, dn: _Dense):
  """Fill the dense problems, run the dense world solver on them and scatter the results back."""
  from mujoco_warp._src import solver  # noqa: PLC0415

  ms, ds, sctx = dn.m, dn.d, dn.ctx
  wp.launch(
    _dense_scalars,
    dim=ds.nworld,
    inputs=[
      m.ntree,
      m.opt.tolerance,
      m.opt.ls_tolerance,
      m.island_dense_nv,
      m.M_elemid,
      d.M,
      d.map_idof2dof,
      ctx.group_count,
      ctx.group_list,
      g.group_idofadr,
      g.group_nv,
      g.group_nefc,
      g.group_ne,
      g.group_nf,
      g.group_iefcadr,
    ],
    outputs=[
      ds.nefc,
      ds.ne,
      ds.nf,
      ms.stat.meaninertia,
      ms.opt.tolerance,
      ms.opt.ls_tolerance,
      sctx.efc_world,
      sctx.efc_rowoff,
    ],
  )
  wp.launch(
    _dense_dofs,
    dim=(d.nworld, m.nv),
    inputs=[
      m.island_dense_nv,
      d.qacc_warmstart,
      d.qfrc_smooth,
      d.qacc_smooth,
      g.dof_group,
      g.dof_slotcol,
      g.group_nv,
      ctx.group_problem,
    ],
    outputs=[ds.qacc_warmstart, ds.qfrc_smooth, ds.qacc_smooth, sctx.problem_dof],
  )
  wp.launch(
    _dense_rowmap,
    dim=(d.nworld, d.njmax),
    inputs=[d.nefc, g.efc_group, ctx.group_problem],
    outputs=[sctx.efc_problem],
  )
  solver._solve(ms, ds, sctx, compact=True, nsolving=dn.nsolving)
  wp.launch(
    _dense_scatter,
    dim=(d.nworld, m.nv),
    inputs=[g.dof_group, g.dof_slotcol, ctx.group_problem, ds.qacc, ds.qfrc_constraint, ds.solver_niter, ds.overflow],
    outputs=[d.solver_niter, d.qacc, d.qfrc_constraint, d.overflow],
  )


@wp.kernel
def _scatter_dofs(
  # In:
  row2dof_in: wp.array2d[int],
  iqacc_in: wp.array2d[float],
  frc_in: wp.array2d[float],
  # Data out:
  qacc_out: wp.array2d[float],
  qfrc_constraint_out: wp.array2d[float],
):
  worldid, row = wp.tid()
  dofid = row2dof_in[worldid, row]
  if dofid >= 0:
    qacc_out[worldid, dofid] = iqacc_in[worldid, row]
    qfrc_constraint_out[worldid, dofid] = frc_in[worldid, row]


@wp.kernel
def _free_dofs(
  # Data in:
  qacc_smooth_in: wp.array2d[float],
  dof_island_in: wp.array2d[int],
  # Data out:
  qacc_out: wp.array2d[float],
  qfrc_constraint_out: wp.array2d[float],
):
  """DOFs outside all islands have no constraints."""
  worldid, dofid = wp.tid()
  if dof_island_in[worldid, dofid] < 0:
    qacc_out[worldid, dofid] = qacc_smooth_in[worldid, dofid]
    qfrc_constraint_out[worldid, dofid] = 0.0


def _create_context(m: types.Model, d: types.Data) -> IslandContext:
  ntile_pool, nrow = _island_pool_size(m.nv, m.ntree)
  nworld, njmax, ntree = d.nworld, d.njmax, m.ntree
  pool = (nworld, ntile_pool * _TILE, _TILE)
  return IslandContext(
    tilebase=wp.empty((nworld, ntree), dtype=int),
    ntile=wp.empty((nworld, ntree), dtype=int),
    vbase=wp.empty((nworld, ntree), dtype=int),
    pairbase=wp.empty((nworld, ntree), dtype=int),
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
    M=wp.empty(pool, dtype=float),
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
    grad_scale=wp.ones((nworld, ntree), dtype=float),
    group_count=wp.zeros(7, dtype=int),
    group_list=wp.empty((7, nworld * ntree), dtype=int),
    group_problem=wp.empty((nworld, ntree), dtype=int),
    nsolving=wp.zeros(1, dtype=int),
  )


def _nblock(per_sm: int = 1) -> int:
  """Persistent blocks for the group kernels (they stride over the groups that exist)."""
  return wp.get_device().sm_count * 32 * per_sm


def _evaluate(m: types.Model, d: types.Data, ctx: IslandContext, g: island.SlotBuffers, initial: bool):
  """Constraint state at the current point, J'force of sparse islands and the H updates."""
  if not initial:
    ctx.state_changed.zero_()
    ctx.changed.zero_()
  wp.launch(
    _rows(initial),
    dim=(d.nworld, d.njmax),
    inputs=[
      d.nefc,
      d.efc.type,
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
    ],
    outputs=[d.efc.force, d.efc.state, ctx.row_quad, ctx.row_w, ctx.frc, ctx.H, ctx.changed, ctx.state_changed],
  )
  # allocated outside the conditional body below (later evaluations run in the iteration loop)
  world_done = wp.zeros(d.nworld, dtype=bool) if initial else None

  def sparse_h():
    # dense rows have no constraint groups
    wp.launch(
      _JTDACJ_sparse(False, m.opt.cone, 3, True),
      dim=(d.nworld, _jtdaj_groups_per_world(d.nworld, d.njmax), _JTDAJ_THREADS_PER_GROUP),
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
        world_done,
        _jtdaj_groups_per_world(d.nworld, d.njmax),
      ],
      outputs=[ctx.H],
      block_dim=m.block_dim.update_gradient_JTDAJ_sparse,
    )

  # the initial H of sparse groups, only if there is one
  if initial:
    if m.opt.graph_conditional:
      wp.capture_if(ctx.group_count[5:6], on_true=sparse_h)
    else:
      sparse_h()


def _launch_grad(d: types.Data, ctx: IslandContext, g: island.SlotBuffers, final: bool):
  nrow = ctx.qacc.shape[1]
  wp.launch(
    _grad(final),
    dim=(d.nworld, nrow),
    inputs=[
      d.efc.J,
      d.efc.force,
      g.dof_group,
      g.group_slot,
      g.J,
      ctx.row2dof,
      ctx.vbase,
      ctx.done,
      ctx.niter,
      ctx.state_changed,
      ctx.Ma,
      ctx.qfrc_smooth,
    ],
    outputs=[ctx.frc, ctx.grad.reshape((d.nworld, nrow))],
  )


def _iteration(m: types.Model, d: types.Data, ctx: IslandContext, g: island.SlotBuffers):
  nrow = ctx.qacc.shape[1]
  ctx.nsolving.zero_()
  _launch_grad(d, ctx, g, final=False)

  def dense_factor(ntile: int):
    wp.launch_tiled(
      _dense_factor(ntile),
      dim=_nblock(),
      inputs=[
        m.ntree,
        g.ngroup,
        ctx.group_count,
        ctx.group_list,
        ntile,
        _nblock(),
        ctx.tilebase,
        ctx.ntile,
        ctx.vbase,
        ctx.done,
        ctx.niter,
        ctx.state_changed,
        ctx.grad.reshape((d.nworld, nrow)),
        ctx.H,
      ],
      outputs=[ctx.search.reshape((d.nworld, nrow)), ctx.search_dot, ctx.newton_decrement, ctx.grad_scale],
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

  # each size class only when some group has it
  for cls, launch in ((1, lambda: dense_factor(1)), (2, lambda: dense_factor(2)), (3, blocked_factor)):
    if m.opt.graph_conditional:
      wp.capture_if(ctx.group_count[cls : cls + 1], on_true=launch)
    else:
      launch()

  wp.launch_tiled(
    _linesearch(m.opt.ls_iterations, int(m.opt.warn_overflow)),
    dim=_nblock(2),
    inputs=[
      m.ntree,
      m.opt.tolerance,
      m.opt.ls_tolerance,
      m.opt.iterations,
      g.ngroup,
      d.efc.type,
      d.efc.J_rownnz,
      d.efc.J_rowadr,
      d.efc.J_colind,
      d.efc.J,
      d.efc.D,
      d.efc.frictionloss,
      g.group_nv,
      g.group_nefc,
      g.group_iefcadr,
      ctx.group_count,
      ctx.group_list,
      0,
      _nblock(2),
      ctx.tilebase,
      ctx.ntile,
      ctx.vbase,
      ctx.scale,
      ctx.qfrc_smooth,
      ctx.M,
      ctx.search.reshape((d.nworld, nrow)),
      g.group_slot,
      ctx.dof2row,
      g.J,
      ctx.grad.reshape((d.nworld, nrow)),
      ctx.search_dot,
      ctx.newton_decrement,
      ctx.state_changed,
    ],
    outputs=[
      d.solver_niter,
      d.overflow,
      ctx.done,
      ctx.niter,
      ctx.alpha,
      ctx.improvement,
      ctx.qacc,
      ctx.Ma,
      ctx.frc,
      ctx.mv,
      ctx.row_jaref,
      ctx.row_jv,
      ctx.nsolving,
      ctx.grad_scale,
    ],
    block_dim=_LINESEARCH_THREADS,
  )
  # constraint evaluation at the updated point, for the next iteration
  _evaluate(m, d, ctx, g, initial=False)


def solve(m: types.Model, d: types.Data) -> IslandContext:
  """Solve every island with its own Newton iteration and scatter the results back."""
  # make_constraint found the islands and wrote the rows in island order
  ctx = _create_context(m, d)
  nrow = ctx.qacc.shape[1]
  # islands are solved in groups (see island.MERGE)
  g = island.slot_buffers(m, d)
  dense = _create_dense(m, d, ctx, g)
  ctx.dense = dense
  d.solver_niter.zero_()
  wp.launch(_free_dofs, dim=(d.nworld, m.nv), inputs=[d.qacc_smooth, g.dof_group], outputs=[d.qacc, d.qfrc_constraint])

  # layout and gathers into island order
  wp.launch_tiled(
    _layout(m.ntree),
    dim=d.nworld,
    inputs=[g.group_nv],
    outputs=[ctx.tilebase, ctx.ntile, ctx.vbase, ctx.pairbase],
    block_dim=m.block_dim.island_scan_sizes,
  )
  wp.launch(
    _group_lists,
    dim=(d.nworld, m.ntree),
    inputs=[m.ntree, g.ngroup, ctx.ntile, g.group_slot],
    outputs=[ctx.group_count, ctx.group_list, ctx.group_problem, ctx.done],
  )
  ctx.row2dof.fill_(-1)
  wp.launch(
    _island_pool_maps,
    dim=(d.nworld, m.nv),
    inputs=[g.dof_group, g.group_idofadr, d.map_dof2idof, ctx.tilebase, ctx.ntile, ctx.vbase],
    outputs=[ctx.row2dof, ctx.dof2row, ctx.dof_poolrow, ctx.dof_local],
  )
  warmstart = not (m.opt.disableflags & types.DisableBit.WARMSTART)
  wp.launch(
    _gather_dofs,
    dim=(d.nworld, nrow),
    inputs=[d.qacc_warmstart, d.qfrc_smooth, d.qacc_smooth, warmstart, ctx.row2dof],
    outputs=[ctx.qacc, ctx.qfrc_smooth, ctx.frc],
  )
  wp.launch(
    _pool_init,
    dim=(d.nworld, nrow),
    inputs=[
      m.M_elemid,
      g.ngroup,
      d.M,
      g.group_nv,
      ctx.row2dof,
      ctx.tilebase,
      ctx.ntile,
      ctx.vbase,
      g.group_slot,
    ],
    outputs=[ctx.M],
  )
  wp.launch(
    _init_rows,
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
  wp.launch_tiled(
    _init_islands,
    dim=_nblock(),
    inputs=[
      m.ntree,
      g.ngroup,
      g.group_nv,
      ctx.group_count,
      ctx.group_list,
      0,
      _nblock(),
      ctx.tilebase,
      ctx.ntile,
      ctx.vbase,
      ctx.qacc,
      ctx.M,
    ],
    outputs=[ctx.Ma, ctx.H, ctx.scale, ctx.done, ctx.niter, ctx.alpha, ctx.improvement, ctx.nsolving],
    block_dim=_ISLAND_THREADS,
  )
  # initial constraint evaluation; H = M + J'DJ over the quadratic rows
  _evaluate(m, d, ctx, g, initial=True)

  # one-tile groups (e.g. free bodies): the whole solve in one kernel
  nrow_dof = ctx.qacc.shape[1]
  small_nsolving = wp.zeros(1, dtype=int)

  def solve_small():
    wp.launch_tiled(
      _solve_small(m.opt.ls_iterations, int(m.opt.warn_overflow)),
      dim=_nblock(2),
      inputs=[
        m.ntree,
        m.opt.tolerance,
        m.opt.ls_tolerance,
        m.opt.iterations,
        g.ngroup,
        d.efc.type,
        d.efc.J_rownnz,
        d.efc.J_rowadr,
        d.efc.J_colind,
        d.efc.J,
        d.efc.D,
        d.efc.frictionloss,
        g.group_nv,
        g.group_nefc,
        g.group_iefcadr,
        ctx.group_count,
        ctx.group_list,
        _nblock(2),
        ctx.tilebase,
        ctx.ntile,
        ctx.vbase,
        ctx.qfrc_smooth,
        ctx.M,
        g.group_slot,
        ctx.dof2row,
        g.J,
      ],
      outputs=[
        d.solver_niter,
        d.efc.force,
        d.efc.state,
        d.overflow,
        ctx.scale,
        ctx.grad.reshape((d.nworld, nrow_dof)),
        ctx.search.reshape((d.nworld, nrow_dof)),
        ctx.search_dot,
        ctx.newton_decrement,
        ctx.state_changed,
        ctx.done,
        ctx.niter,
        ctx.alpha,
        ctx.improvement,
        ctx.qacc,
        ctx.Ma,
        ctx.frc,
        ctx.mv,
        ctx.row_jaref,
        ctx.row_jv,
        small_nsolving,
        ctx.grad_scale,
        ctx.row_quad,
        ctx.H,
      ],
      block_dim=_LINESEARCH_THREADS,
    )

  if m.opt.graph_conditional:
    wp.capture_if(ctx.group_count[4:5], on_true=solve_small)
  else:
    solve_small()

  # each body ends with ctx.nsolving = the number of islands still iterating
  if m.opt.graph_conditional:
    wp.capture_while(ctx.nsolving, while_body=_iteration, m=m, d=d, ctx=ctx, g=g)
  else:
    # the last pass only marks islands at the iteration limit as done
    for _ in range(m.opt.iterations + 1):
      _iteration(m, d, ctx, g)

  # J'force at the solution (stable iterations skip it), then scatter back
  _launch_grad(d, ctx, g, final=True)
  wp.launch(_scatter_dofs, dim=(d.nworld, nrow), inputs=[ctx.row2dof, ctx.qacc, ctx.frc], outputs=[d.qacc, d.qfrc_constraint])

  # two-tile groups: the dense world solver; it scatters last, over the block path's values
  if m.opt.graph_conditional:
    wp.capture_if(ctx.group_count[6:7], on_true=lambda: _solve_dense(m, d, ctx, g, dense))
  else:
    _solve_dense(m, d, ctx, g, dense)

  support.mul_m(m, d, d.efc.Ma, d.qacc)
  return ctx
