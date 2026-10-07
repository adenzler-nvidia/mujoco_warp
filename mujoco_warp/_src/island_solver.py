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

make_constraint writes constraint rows in island order (an island's rows are contiguous). Each
step, an island's DOFs are gathered into tile-aligned rows that double as the right-hand side of
its Cholesky factorization. Each island's inertia and Hessian are dense blocks of 16x16 tiles in a
per-world pool. Every island is solved with its own step size and convergence test, as MuJoCo's
island solver does, and the results are scattered back.
"""

import dataclasses

import warp as wp

from mujoco_warp._src import island
from mujoco_warp._src import support
from mujoco_warp._src import types
from mujoco_warp._src.block_cholesky import create_pool_cholesky_newton_func
from mujoco_warp._src.solver import _ALPHA_NOISE_EPS
from mujoco_warp._src.solver import _JTDAJ_THREADS_PER_GROUP
from mujoco_warp._src.solver import _ROW_EQUALITY
from mujoco_warp._src.solver import _ROW_FRICTION
from mujoco_warp._src.solver import _ROW_OTHER
from mujoco_warp._src.solver import _eval_constraint
from mujoco_warp._src.solver import _eval_pt
from mujoco_warp._src.solver import _eval_pt_3alphas
from mujoco_warp._src.solver import _in_bracket
from mujoco_warp._src.solver import _island_pool_init
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
# threads per island for the factorization and linesearch kernels
_ISLAND_THREADS = 64


@dataclasses.dataclass
class IslandContext:
  """Per-step island-ordered solver state (see the module docstring)."""

  # layout
  tilebase: wp.array2d[int]
  ntile: wp.array2d[int]
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
  # whether a row's curvature is in H (H is updated only where this flips)
  row_quad: wp.array2d[bool]
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
  grad_dot: wp.array2d[float]
  search_dot: wp.array2d[float]
  newton_decrement: wp.array2d[float]
  # whether an island's H changed since its last factorization
  changed: wp.array2d[bool]
  nsolving: wp.array[int]


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
def _ntile(ndof: int) -> int:
  return (ndof + _TILE - 1) // _TILE


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
  ):
    """Per island: tiles per side, first pool tile and first aligned DOF row."""
    worldid = wp.tid()
    ntile = wp.tile_map(_ntile, wp.tile_load(island_nv_in[worldid], shape=ntree))
    wp.tile_store(ntile_out[worldid], ntile)
    wp.tile_store(tilebase_out[worldid], wp.tile_scan_exclusive(wp.tile_map(wp.mul, ntile, ntile)))
    wp.tile_store(vbase_out[worldid], wp.tile_scan_exclusive(ntile) * _TILE)

  return kernel


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
def _init_rows(
  # Data in:
  nefc_in: wp.array[int],
  efc_J_rownnz_in: wp.array2d[int],
  efc_J_rowadr_in: wp.array2d[int],
  efc_J_colind_in: wp.array3d[int],
  efc_J_in: wp.array3d[float],
  efc_aref_in: wp.array2d[float],
  # In:
  dof2row_in: wp.array2d[int],
  iqacc_in: wp.array2d[float],
  # Out:
  row_jaref_out: wp.array2d[float],
):
  """J qacc - aref of a row at the start point."""
  worldid, row = wp.tid()
  if row >= nefc_in[worldid]:
    return
  rowadr = efc_J_rowadr_in[worldid, row]
  jaref = -efc_aref_in[worldid, row]
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
  # Data in:
  nisland_in: wp.array[int],
  island_nv_in: wp.array2d[int],
  # In:
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
  worldid, islandid, lane = wp.tid()
  if islandid >= nisland_in[worldid]:
    if lane == 0:
      done_out[worldid, islandid] = True
    return
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
    tilebase_in: wp.array2d[int],
    ntile_in: wp.array2d[int],
    vbase_in: wp.array2d[int],
    done_in: wp.array2d[bool],
    row_jaref_in: wp.array2d[float],
    dof2row_in: wp.array2d[int],
    # Data out:
    efc_force_out: wp.array2d[float],
    efc_state_out: wp.array2d[int],
    # Out:
    row_quad_out: wp.array2d[bool],
    frc_out: wp.array2d[float],
    H_out: wp.array3d[float],
    changed_out: wp.array2d[bool],
  ):
    """Constraint force and state of a row; adds J'force to frc and updates H where it flips.

    The initial pass only records which rows are quadratic (_JTDACJ_sparse assembles H).
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
    force = res[0]
    state = int(res[1])
    efc_force_out[worldid, row] = force
    efc_state_out[worldid, row] = state
    quad = state == types.ConstraintState.QUADRATIC.value
    # H = M + sum of D J'J over quadratic rows: add or remove this row's term when it flips
    dH = float(0.0)
    if wp.static(INITIAL):
      row_quad_out[worldid, row] = quad
    elif quad != row_quad_out[worldid, row]:
      row_quad_out[worldid, row] = quad
      changed_out[worldid, islandid] = True
      dH = wp.where(quad, D, -D)

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
      if dH != 0.0:
        JaD = Ja * dH
        for kb in range(k, rownnz):
          rc = _pool_rc(tilebase, ntile, cola - vbase, dof2row[colind[rowadr + kb]] - vbase)
          wp.atomic_add(H_out[worldid, rc[0]], rc[1], JaD * efc_J_in[worldid, 0, rowadr + kb])

  return kernel


@cache_kernel
def _factor(max_size: int):
  factorize = create_pool_cholesky_newton_func(_TILE, max_size, True)
  solve_only = create_pool_cholesky_newton_func(_TILE, max_size, False)

  @wp.kernel(module="unique", enable_backward=False, module_options={"enable_mathdx_gemm": False})
  def kernel(
    # Data in:
    nisland_in: wp.array[int],
    island_nv_in: wp.array2d[int],
    # In:
    tilebase_in: wp.array2d[int],
    ntile_in: wp.array2d[int],
    vbase_in: wp.array2d[int],
    done_in: wp.array2d[bool],
    niter_in: wp.array2d[int],
    Ma_in: wp.array2d[float],
    iqfrc_smooth_in: wp.array2d[float],
    frc_in: wp.array2d[float],
    H_in: wp.array3d[float],
    # Out:
    changed_out: wp.array2d[bool],
    grad_out: wp.array3d[float],
    U_out: wp.array3d[float],
    search_out: wp.array3d[float],
    grad_dot_out: wp.array2d[float],
    search_dot_out: wp.array2d[float],
    newton_decrement_out: wp.array2d[float],
  ):
    """Gradient and Newton search direction of one island."""
    worldid, islandid, lane = wp.tid()
    if islandid >= nisland_in[worldid]:
      return
    if done_in[worldid, islandid]:
      return
    nthread = wp.block_dim()
    ndof = island_nv_in[worldid, islandid]
    ntile = ntile_in[worldid, islandid]
    vbase = vbase_in[worldid, islandid]
    grad_dot = float(0.0)
    for la in range(lane, ntile * _TILE, nthread):
      grad = float(0.0)
      if la < ndof:
        row = vbase + la
        grad = Ma_in[worldid, row] - iqfrc_smooth_in[worldid, row] - frc_in[worldid, row]
      grad_out[worldid, vbase + la, 0] = grad
      grad_dot += grad * grad
    grad_dot = wp.tile_reduce(wp.add, wp.tile(grad_dot))[0]
    _sync()
    tilebase = tilebase_in[worldid, islandid]
    if changed_out[worldid, islandid] or niter_in[worldid, islandid] == 0:
      sums = wp.static(factorize)(H_in[worldid], grad_out[worldid], tilebase, ntile, vbase, U_out[worldid], search_out[worldid])
    else:
      sums = wp.static(solve_only)(
        H_in[worldid], grad_out[worldid], tilebase, ntile, vbase, U_out[worldid], search_out[worldid]
      )
    _sync()  # every thread has read the flag
    if lane == 0:
      changed_out[worldid, islandid] = False
      grad_dot_out[worldid, islandid] = grad_dot
      search_dot_out[worldid, islandid] = sums[0]
      newton_decrement_out[worldid, islandid] = sums[1]

  return kernel


@cache_kernel
def _linesearch(ls_iterations: int, warn_overflow: int):
  LS_ITERATIONS = ls_iterations

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
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
    tilebase_in: wp.array2d[int],
    ntile_in: wp.array2d[int],
    vbase_in: wp.array2d[int],
    scale_in: wp.array2d[float],
    iqfrc_smooth_in: wp.array2d[float],
    iM_in: wp.array3d[float],
    search_in: wp.array2d[float],
    dof2row_in: wp.array2d[int],
    grad_dot_in: wp.array2d[float],
    search_dot_in: wp.array2d[float],
    newton_decrement_in: wp.array2d[float],
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
  ):
    """Convergence test, linesearch and update of one island (see _linesearch_iterative_kernel)."""
    worldid, islandid, lane = wp.tid()
    if islandid >= nisland_in[worldid]:
      return
    if done_out[worldid, islandid]:
      return
    nthread = wp.block_dim()
    ndof = island_nv_in[worldid, islandid]
    nrow = island_nefc_in[worldid, islandid]
    rowadr0 = island_iefcadr_in[worldid, islandid]
    tilebase = tilebase_in[worldid, islandid]
    ntile = ntile_in[worldid, islandid]
    vbase = vbase_in[worldid, islandid]
    scale = scale_in[worldid, islandid]
    tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]
    ls_tolerance = opt_ls_tolerance[worldid % opt_ls_tolerance.shape[0]]
    niter = niter_out[worldid, islandid]

    # --- termination (see _solve_done) ---
    converged = False
    if niter > 0:
      improvement = improvement_out[worldid, islandid]
      converged = (
        (alpha_out[worldid, islandid] == 0.0)
        or (improvement > 0.0 and improvement / scale < tolerance)
        or (wp.sqrt(grad_dot_in[worldid, islandid]) / scale < tolerance)
        or (0.5 * newton_decrement_in[worldid, islandid] / scale < tolerance)
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
      mv = _block_matvec(iM_in[worldid], tilebase, ntile, vbase, ndof, search, la)
      mv_out[worldid, row] = mv
      s = search[row]
      local_dof += wp.vec2(s * (Ma_out[worldid, row] - iqfrc_smooth_in[worldid, row]), s * mv)
    local_gauss = wp.vec2(0.0)
    local_p0 = wp.vec3(0.0)
    for r in range(lane, nrow, nthread):
      row = rowadr0 + r
      rowadr = efc_J_rowadr_in[worldid, row]
      jv = float(0.0)
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

    # --- update, and reset J'force for the next Newton step ---
    for la in range(lane, ntile * _TILE, nthread):
      row = vbase + la
      if la < ndof:
        iqacc_out[worldid, row] += alpha * search[row]
        Ma_out[worldid, row] += alpha * mv_out[worldid, row]
      frc_out[worldid, row] = 0.0
    for r in range(lane, nrow, nthread):
      row = rowadr0 + r
      row_jaref_out[worldid, row] += alpha * row_jv_out[worldid, row]
    if lane == 0:
      niter_out[worldid, islandid] = niter + 1
      alpha_out[worldid, islandid] = alpha
      improvement_out[worldid, islandid] = improvement
      wp.atomic_add(nsolving_out, 0, 1)

  return kernel


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
    row2dof=wp.empty((nworld, nrow), dtype=int),
    dof2row=wp.empty((nworld, m.nv), dtype=int),
    dof_poolrow=wp.empty((nworld, m.nv), dtype=int),
    dof_local=wp.empty((nworld, m.nv), dtype=int),
    qacc=wp.empty((nworld, nrow), dtype=float),
    Ma=wp.zeros((nworld, nrow), dtype=float),
    qfrc_smooth=wp.empty((nworld, nrow), dtype=float),
    frc=wp.empty((nworld, nrow), dtype=float),
    grad=wp.zeros((nworld, nrow, 1), dtype=float),
    search=wp.zeros((nworld, nrow, 1), dtype=float),
    mv=wp.zeros((nworld, nrow), dtype=float),
    row_jaref=wp.empty((nworld, njmax), dtype=float),
    row_jv=wp.empty((nworld, njmax), dtype=float),
    row_quad=wp.zeros((nworld, njmax), dtype=bool),
    M=wp.empty(pool, dtype=float),
    H=wp.empty(pool, dtype=float),
    U=wp.empty(pool, dtype=float),
    scale=wp.empty((nworld, ntree), dtype=float),
    done=wp.empty((nworld, ntree), dtype=bool),
    niter=wp.empty((nworld, ntree), dtype=int),
    alpha=wp.empty((nworld, ntree), dtype=float),
    improvement=wp.empty((nworld, ntree), dtype=float),
    grad_dot=wp.empty((nworld, ntree), dtype=float),
    search_dot=wp.empty((nworld, ntree), dtype=float),
    newton_decrement=wp.empty((nworld, ntree), dtype=float),
    changed=wp.zeros((nworld, ntree), dtype=bool),
    nsolving=wp.zeros(1, dtype=int),
  )


def _launch_rows(m: types.Model, d: types.Data, ctx: IslandContext, initial: bool):
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
      d.efc.island,
      ctx.tilebase,
      ctx.ntile,
      ctx.vbase,
      ctx.done,
      ctx.row_jaref,
      ctx.dof2row,
    ],
    outputs=[d.efc.force, d.efc.state, ctx.row_quad, ctx.frc, ctx.H, ctx.changed],
  )


def _iteration(m: types.Model, d: types.Data, ctx: IslandContext):
  nrow = ctx.qacc.shape[1]
  ctx.nsolving.zero_()
  wp.launch_tiled(
    _factor(-(-m.nv // _TILE) * _TILE),
    dim=(d.nworld, m.ntree),
    inputs=[
      d.nisland,
      d.island_nv,
      ctx.tilebase,
      ctx.ntile,
      ctx.vbase,
      ctx.done,
      ctx.niter,
      ctx.Ma,
      ctx.qfrc_smooth,
      ctx.frc,
      ctx.H,
    ],
    outputs=[ctx.changed, ctx.grad, ctx.U, ctx.search, ctx.grad_dot, ctx.search_dot, ctx.newton_decrement],
    block_dim=_ISLAND_THREADS,
  )
  wp.launch_tiled(
    _linesearch(m.opt.ls_iterations, int(m.opt.warn_overflow)),
    dim=(d.nworld, m.ntree),
    inputs=[
      m.opt.tolerance,
      m.opt.ls_tolerance,
      m.opt.iterations,
      d.nisland,
      d.efc.type,
      d.efc.J_rownnz,
      d.efc.J_rowadr,
      d.efc.J_colind,
      d.efc.J,
      d.efc.D,
      d.efc.frictionloss,
      d.island_nv,
      d.island_nefc,
      d.island_iefcadr,
      ctx.tilebase,
      ctx.ntile,
      ctx.vbase,
      ctx.scale,
      ctx.qfrc_smooth,
      ctx.M,
      ctx.search.reshape((d.nworld, nrow)),
      ctx.dof2row,
      ctx.grad_dot,
      ctx.search_dot,
      ctx.newton_decrement,
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
    ],
    block_dim=_ISLAND_THREADS,
  )
  # constraint evaluation at the updated point, for the next iteration
  _launch_rows(m, d, ctx, initial=False)


def solve(m: types.Model, d: types.Data) -> IslandContext:
  """Solve every island with its own Newton iteration and scatter the results back."""
  # make_constraint found the islands and wrote the rows in island order
  island.compute_island_dofs(m, d)
  ctx = _create_context(m, d)
  nrow = ctx.qacc.shape[1]

  # layout and gathers into island order
  wp.launch_tiled(
    _layout(m.ntree),
    dim=d.nworld,
    inputs=[d.island_nv],
    outputs=[ctx.tilebase, ctx.ntile, ctx.vbase],
    block_dim=m.block_dim.island_scan_sizes,
  )
  ctx.row2dof.fill_(-1)
  wp.launch(
    _island_pool_maps,
    dim=(d.nworld, m.nv),
    inputs=[d.dof_island, d.island_idofadr, d.map_dof2idof, ctx.tilebase, ctx.ntile, ctx.vbase],
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
    _island_pool_init,
    dim=(d.nworld, nrow),
    inputs=[
      m.M_elemid,
      d.nisland,
      d.M,
      d.island_nv,
      ctx.row2dof,
      ctx.tilebase,
      ctx.ntile,
      ctx.vbase,
      wp.zeros(d.nworld, dtype=bool),
    ],
    outputs=[ctx.M],
  )
  wp.launch(
    _init_rows,
    dim=(d.nworld, d.njmax),
    inputs=[d.nefc, d.efc.J_rownnz, d.efc.J_rowadr, d.efc.J_colind, d.efc.J, d.efc.aref, ctx.dof2row, ctx.qacc],
    outputs=[ctx.row_jaref],
  )
  d.solver_niter.zero_()
  wp.launch_tiled(
    _init_islands,
    dim=(d.nworld, m.ntree),
    inputs=[d.nisland, d.island_nv, ctx.tilebase, ctx.ntile, ctx.vbase, ctx.qacc, ctx.M],
    outputs=[ctx.Ma, ctx.H, ctx.scale, ctx.done, ctx.niter, ctx.alpha, ctx.improvement, ctx.nsolving],
    block_dim=_ISLAND_THREADS,
  )

  # initial constraint evaluation; H = M + J'DJ assembled by constraint groups
  _launch_rows(m, d, ctx, initial=True)
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
      wp.zeros(d.nworld, dtype=bool),
      _jtdaj_groups_per_world(d.nworld, d.njmax),
    ],
    outputs=[ctx.H],
    block_dim=m.block_dim.update_gradient_JTDAJ_sparse,
  )

  # each body ends with ctx.nsolving = the number of islands still iterating
  if m.opt.graph_conditional:
    wp.capture_while(ctx.nsolving, while_body=_iteration, m=m, d=d, ctx=ctx)
  else:
    # the last pass only marks islands at the iteration limit as done
    for _ in range(m.opt.iterations + 1):
      _iteration(m, d, ctx)

  # scatter back; DOFs outside islands take the unconstrained solution
  wp.launch(_free_dofs, dim=(d.nworld, m.nv), inputs=[d.qacc_smooth, d.dof_island], outputs=[d.qacc, d.qfrc_constraint])
  wp.launch(_scatter_dofs, dim=(d.nworld, nrow), inputs=[ctx.row2dof, ctx.qacc, ctx.frc], outputs=[d.qacc, d.qfrc_constraint])
  support.mul_m(m, d, d.efc.Ma, d.qacc)
  return ctx
