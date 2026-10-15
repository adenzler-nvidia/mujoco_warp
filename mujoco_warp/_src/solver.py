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

import dataclasses
from math import ceil
from typing import Any

import warp as wp

from mujoco_warp._src import derivative
from mujoco_warp._src import island
from mujoco_warp._src import math
from mujoco_warp._src import smooth
from mujoco_warp._src import support
from mujoco_warp._src import types
from mujoco_warp._src.block_cholesky import create_blocked_cholesky_augmented_factorize_solve_newton_func
from mujoco_warp._src.block_cholesky import create_blocked_cholesky_factorize_solve_func
from mujoco_warp._src.block_cholesky import create_blocked_cholesky_solve_newton_func
from mujoco_warp._src.block_cholesky import create_pool_cholesky_newton_func
from mujoco_warp._src.block_cholesky import solve_search_sums
from mujoco_warp._src.types import InverseContext
from mujoco_warp._src.types import OverflowType
from mujoco_warp._src.types import SolverContext
from mujoco_warp._src.warp_util import cache_kernel
from mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

_BLOCK_CHOLESKY_DIM = 32


def create_inverse_context(m: types.Model, d: types.Data) -> InverseContext:
  """Create an InverseContext with allocated workspace arrays.

  Args:
    m: Model.
    d: Data.

  Returns:
    InverseContext with allocated arrays.
  """
  nworld = d.nworld
  njmax = d.njmax

  return InverseContext(
    Jaref=wp.empty((nworld, njmax), dtype=float),
    search_dot=wp.empty((nworld,), dtype=float),
    done=wp.empty((nworld,), dtype=bool),
    quad_changed_ids=wp.empty((nworld, 0), dtype=int),
    quad_changed_count=wp.empty((0,), dtype=int),
    state_changed_count=wp.empty((0,), dtype=int),
    ls_exhausted=wp.empty((0,), dtype=bool),
  )


def _island_pool_size(nv: int, ntree: int, tile: int = types.TILE_SIZE_JTDAJ_DENSE) -> tuple[int, int]:
  """Tiles and aligned rows the island tile pool needs at most.

  An island of n DOFs takes ceil(n / 16) <= (n + 15) / 16 tiles per side, so the islands take at
  most (nv^2 + 30 nv + 225 ntree) / 256 tiles and nv + 15 ntree aligned rows (for tile = 16).
  """
  ntile = -(-(nv * nv + 2 * (tile - 1) * nv + (tile - 1) ** 2 * ntree) // (tile * tile))
  nrow = -(-(nv + (tile - 1) * ntree) // tile) * tile
  return ntile, nrow


def _create_solver_context(
  m: types.Model,
  d: types.Data,
  island: bool = False,
) -> SolverContext:
  """Create a SolverContext with allocated workspace arrays.

  Args:
    m: Model.
    d: Data.
    island: allocate h and hfactor as island tile pools (opt.island_solve).

  Returns:
    SolverContext with allocated arrays.
  """
  nworld = d.nworld
  nv = m.nv
  nv_pad = m.nv_pad
  njmax = d.njmax

  alloc_h = m.opt.solver == types.SolverType.NEWTON
  alloc_hfactor = alloc_h and nv > _BLOCK_CHOLESKY_DIM
  alloc_mgrad = m.opt.solver == types.SolverType.CG
  alloc_incremental = _use_incremental(m)

  h_shape = (nworld, nv_pad, nv_pad)
  if island:
    alloc_hfactor = alloc_h
    h_shape = (nworld, _island_pool_size(nv, m.ntree)[0] * types.TILE_SIZE_JTDAJ_DENSE, types.TILE_SIZE_JTDAJ_DENSE)

  return SolverContext(
    Jaref=wp.empty((nworld, njmax), dtype=float),
    search_dot=wp.empty((nworld,), dtype=float),
    done=wp.empty((nworld,), dtype=bool),
    grad=wp.zeros((nworld, nv_pad), dtype=float),
    grad_dot=wp.empty((nworld,), dtype=float),
    newton_decrement=wp.empty((nworld,), dtype=float),
    Mgrad=wp.empty((nworld, nv_pad), dtype=float) if alloc_mgrad else wp.empty((nworld, 0), dtype=float),
    search=wp.empty((nworld, nv), dtype=float),
    mv=wp.empty((nworld, nv), dtype=float),
    jv=wp.empty((nworld, njmax), dtype=float),
    quad=wp.empty((nworld, njmax), dtype=wp.vec3),
    alpha=wp.empty((nworld,), dtype=float),
    grad_scale=wp.empty((nworld,), dtype=float),
    improvement=wp.empty((nworld,), dtype=float),
    ls_exhausted=wp.zeros((nworld,), dtype=bool),
    search_unchanged=wp.empty((nworld,), dtype=bool),
    prev_grad=wp.empty((nworld, nv), dtype=float) if alloc_mgrad else wp.empty((nworld, 0), dtype=float),
    prev_Mgrad=wp.empty((nworld, nv), dtype=float) if alloc_mgrad else wp.empty((nworld, 0), dtype=float),
    beta=wp.empty((nworld,), dtype=float) if alloc_mgrad else wp.empty((0,), dtype=float),
    h=wp.empty(h_shape, dtype=float) if alloc_h else wp.empty((nworld, 0, 0), dtype=float),
    hfactor=wp.empty(h_shape, dtype=float) if alloc_hfactor else wp.empty((nworld, 0, 0), dtype=float),
    quad_changed_ids=wp.empty((nworld, njmax), dtype=int) if alloc_incremental else wp.empty((nworld, 0), dtype=int),
    quad_changed_count=wp.empty((nworld,), dtype=int) if alloc_incremental else wp.empty((0,), dtype=int),
    state_changed_count=wp.empty((nworld,), dtype=int) if alloc_incremental else wp.empty((0,), dtype=int),
    cholesky_top=wp.zeros((1, nv_pad // types.TILE_SIZE_JTDAJ_DENSE if alloc_hfactor else 0), dtype=int),
  )


@wp.kernel
def _add_qfrc_smooth_eff(
  # Model:
  opt_enableflags: int,
  dof_treeid: wp.array[int],
  # Data in:
  tree_awake_in: wp.array2d[int],
  qfrc_smooth_in: wp.array2d[float],
  efm_c_in: wp.array2d[float],
  efm_ca_in: wp.array2d[float],
  # Out:
  qfrc_smooth_eff_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  if (opt_enableflags & types.EnableBit.SLEEP) and tree_awake_in[worldid, dof_treeid[dofid]] == 0:
    qfrc_smooth_eff_out[worldid, dofid] = 0.0
    return
  qfrc_smooth_eff_out[worldid, dofid] = qfrc_smooth_in[worldid, dofid] + efm_c_in[worldid, dofid] + efm_ca_in[worldid, dofid]


@wp.func
def _rescale(nv: int, meaninertia: float, value: float) -> float:
  return value / (meaninertia * float(nv))


@wp.func
def _in_bracket(x: wp.vec3, y: wp.vec3) -> bool:
  return (x[1] < y[1] and y[1] < 0.0) or (x[1] > y[1] and y[1] > 0.0)


@wp.func
def _eval_pt_direct_alpha_zero(jaref: float, jv: float, d: float) -> wp.vec3:
  """Eval quadratic constraint at alpha=0."""
  jvD = jv * d
  return wp.vec3(0.5 * d * jaref * jaref, jvD * jaref, jv * jvD)


@wp.func
def _eval_pt_direct(jaref: float, jv: float, d: float, alpha: float) -> wp.vec3:
  """Eval quadratic constraint."""
  x = jaref + alpha * jv
  jvD = jv * d
  return wp.vec3(0.5 * d * x * x, jvD * x, jv * jvD)


@wp.func
def _eval_pt_direct_cost_alpha_zero(jaref: float, d: float) -> float:
  return 0.5 * d * jaref * jaref


@wp.func
def _eval_pt_direct_shifted(jaref: float, jv: float, d: float, alpha: float, offset: float) -> wp.vec3:
  """Eval quadratic constraint shifted by alpha=0, plus a constant cost offset."""
  jvD = jv * d
  hessian = jv * jvD
  alpha_h = alpha * hessian
  return wp.vec3(alpha * (jvD * jaref + 0.5 * alpha_h) + offset, jvD * jaref + alpha_h, hessian)


@wp.func
def _eval_pt_direct_3alphas(
  jaref: float, jv: float, d: float, lo_alpha: float, hi_alpha: float, mid_alpha: float
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
  """Eval quadratic constraint for 3 alphas."""
  x_lo = jaref + lo_alpha * jv
  x_hi = jaref + hi_alpha * jv
  x_mid = jaref + mid_alpha * jv
  jvD = jv * d
  hessian = jv * jvD
  half_d = 0.5 * d
  return (
    wp.vec3(half_d * x_lo * x_lo, jvD * x_lo, hessian),
    wp.vec3(half_d * x_hi * x_hi, jvD * x_hi, hessian),
    wp.vec3(half_d * x_mid * x_mid, jvD * x_mid, hessian),
  )


@wp.func
def _eval_pt_direct_shifted_3alphas(
  jaref: float, jv: float, d: float, lo_alpha: float, hi_alpha: float, mid_alpha: float, offset: float
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
  """Eval shifted quadratic constraint for 3 alphas, plus a constant cost offset."""
  jvD = jv * d
  grad0 = jvD * jaref
  hessian = jv * jvD
  lo_ah = lo_alpha * hessian
  hi_ah = hi_alpha * hessian
  mid_ah = mid_alpha * hessian
  return (
    wp.vec3(lo_alpha * (grad0 + 0.5 * lo_ah) + offset, grad0 + lo_ah, hessian),
    wp.vec3(hi_alpha * (grad0 + 0.5 * hi_ah) + offset, grad0 + hi_ah, hessian),
    wp.vec3(mid_alpha * (grad0 + 0.5 * mid_ah) + offset, grad0 + mid_ah, hessian),
  )


@wp.func
def _eval_cost(quad: wp.vec3, alpha: float) -> float:
  return alpha * alpha * quad[2] + alpha * quad[1] + quad[0]


@wp.func
def _eval_pt(quad: wp.vec3, alpha: float) -> wp.vec3:
  """Eval quad polynomial at alpha, return (cost, grad, hessian)."""
  aq2 = alpha * quad[2]
  return wp.vec3(
    alpha * aq2 + alpha * quad[1] + quad[0],
    2.0 * aq2 + quad[1],
    2.0 * quad[2],
  )


@wp.func
def _eval_pt_3alphas(quad: wp.vec3, lo_alpha: float, hi_alpha: float, mid_alpha: float) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
  """Eval quad polynomial for 3 alphas."""
  q0, q1, q2 = quad[0], quad[1], quad[2]
  hessian = 2.0 * q2
  lo_aq2 = lo_alpha * q2
  hi_aq2 = hi_alpha * q2
  mid_aq2 = mid_alpha * q2
  return (
    wp.vec3(lo_alpha * lo_aq2 + lo_alpha * q1 + q0, 2.0 * lo_aq2 + q1, hessian),
    wp.vec3(hi_alpha * hi_aq2 + hi_alpha * q1 + q0, 2.0 * hi_aq2 + q1, hessian),
    wp.vec3(mid_alpha * mid_aq2 + mid_alpha * q1 + q0, 2.0 * mid_aq2 + q1, hessian),
  )


@wp.func
def _shift_cost(pt: wp.vec3, cost0: float) -> wp.vec3:
  return wp.vec3(pt[0] - cost0, pt[1], pt[2])


@wp.func
def _eval_frictionloss_pt(x: float, f: float, rf: float, jv: float, d: float) -> wp.vec3:
  """Eval frictionloss and return (cost, grad, hessian). x = Jaref + alpha * jv."""
  if (-rf < x) and (x < rf):
    jvD = jv * d
    return wp.vec3(0.5 * d * x * x, jvD * x, jv * jvD)
  elif x <= -rf:
    return wp.vec3(f * (-0.5 * rf - x), -f * jv, 0.0)
  else:
    return wp.vec3(f * (-0.5 * rf + x), f * jv, 0.0)


@wp.func
def _eval_frictionloss_cost(x: float, f: float, rf: float, d: float) -> float:
  if (-rf < x) and (x < rf):
    return 0.5 * d * x * x
  elif x <= -rf:
    return f * (-0.5 * rf - x)
  return f * (-0.5 * rf + x)


@wp.func
def _eval_frictionloss_pt_one(x: float, f: float, rf: float, half_d: float, jvD: float, hessian: float, f_jv: float) -> wp.vec3:
  """Eval frictionloss with precomputed shared values."""
  if (-rf < x) and (x < rf):
    return wp.vec3(half_d * x * x, jvD * x, hessian)
  elif x <= -rf:
    return wp.vec3(f * (-0.5 * rf - x), -f_jv, 0.0)
  else:
    return wp.vec3(f * (-0.5 * rf + x), f_jv, 0.0)


@wp.func
def _eval_frictionloss_pt_3alphas(
  x_lo: float, x_hi: float, x_mid: float, f: float, rf: float, jv: float, d: float
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
  """Eval frictionloss for 3 x values with shared precomputation."""
  jvD = jv * d
  half_d = 0.5 * d
  hessian = jv * jvD
  f_jv = f * jv
  return (
    _eval_frictionloss_pt_one(x_lo, f, rf, half_d, jvD, hessian, f_jv),
    _eval_frictionloss_pt_one(x_hi, f, rf, half_d, jvD, hessian, f_jv),
    _eval_frictionloss_pt_one(x_mid, f, rf, half_d, jvD, hessian, f_jv),
  )


@wp.func
def _eval_elliptic_reference(
  # In:
  mu: float,
  quad: wp.vec3,
  quad1: wp.vec3,
  quad2: wp.vec3,
) -> tuple[float, float, float, int]:
  u0 = quad1[0]
  uu = quad1[2]
  dm = quad2[2]

  if uu <= 0.0:
    if u0 < 0.0:
      return quad[0], 0.0, 0.0, int(types.ConstraintState.QUADRATIC)
    return 0.0, 0.0, 0.0, int(types.ConstraintState.SATISFIED)

  T0 = wp.sqrt(uu)
  if u0 >= mu * T0:
    return 0.0, T0, 0.0, int(types.ConstraintState.SATISFIED)
  if mu * u0 + T0 <= 0.0:
    return quad[0], T0, 0.0, int(types.ConstraintState.QUADRATIC)

  r0 = u0 - mu * T0
  return 0.5 * dm * r0 * r0, T0, r0, int(types.ConstraintState.CONE)


@wp.func
def _eval_elliptic_alpha_zero(mu: float, quad: wp.vec3, quad1: wp.vec3, quad2: wp.vec3) -> wp.vec3:
  cost0, T0, r0, state0 = _eval_elliptic_reference(mu, quad, quad1, quad2)
  if state0 == int(types.ConstraintState.QUADRATIC):
    return _eval_pt(quad, 0.0)
  if state0 == int(types.ConstraintState.CONE):
    T0_inv = 1.0 / T0
    T1 = quad2[0] * T0_inv
    T2 = (quad2[1] - T1 * T1) * T0_inv
    r1 = quad1[1] - mu * T1
    dm = quad2[2]
    return wp.vec3(cost0, dm * r0 * r1, dm * (r1 * r1 - mu * r0 * T2))
  return wp.vec3(0.0)


@wp.func
def _eval_elliptic_quadratic_cone_gap(mu: float, N: float, T: float, dm: float) -> float:
  """Return quadratic cost minus cone cost at the same point."""
  boundary = mu * N + T
  return 0.5 * dm * boundary * boundary


@wp.func
def _eval_elliptic_quadratic_shifted(
  # In:
  mu: float,
  quad: wp.vec3,
  alpha: float,
  N: float,
  Tsqr: float,
  u0: float,
  T0: float,
  dm: float,
  state0: int,
) -> wp.vec3:
  aq2 = alpha * quad[2]
  cost = alpha * (aq2 + quad[1])
  if state0 == int(types.ConstraintState.CONE):
    cost += _eval_elliptic_quadratic_cone_gap(mu, u0, T0, dm)
  elif state0 == int(types.ConstraintState.SATISFIED):
    cost = 0.5 * dm * (1.0 + mu * mu) * (N * N + wp.max(Tsqr, 0.0))
  return wp.vec3(cost, 2.0 * aq2 + quad[1], 2.0 * quad[2])


@wp.func
def _eval_elliptic_shifted(
  # In:
  mu: float,
  quad: wp.vec3,
  quad1: wp.vec3,
  quad2: wp.vec3,
  alpha: float,
  cost0: float,
  T0: float,
  r0: float,
  state0: int,
) -> wp.vec3:
  u0 = quad1[0]
  v0 = quad1[1]
  uu = quad1[2]
  uv = quad2[0]
  vv = quad2[1]
  dm = quad2[2]

  N = u0 + alpha * v0
  Tsqr_delta = alpha * (2.0 * uv + alpha * vv)
  Tsqr = uu + Tsqr_delta

  if Tsqr <= 0.0:
    if N < 0.0:
      return _eval_elliptic_quadratic_shifted(mu, quad, alpha, N, Tsqr, u0, T0, dm, state0)
  else:
    T = wp.sqrt(Tsqr)
    if N >= mu * T:
      pass
    elif mu * N + T <= 0.0:
      return _eval_elliptic_quadratic_shifted(mu, quad, alpha, N, Tsqr, u0, T0, dm, state0)
    else:
      T_inv = 1.0 / T
      T1 = (uv + alpha * vv) * T_inv
      T2 = (vv - T1 * T1) * T_inv
      r = N - mu * T
      r1 = v0 - mu * T1

      if state0 == int(types.ConstraintState.CONE):
        # Rationalize T - T0 before forming the small cone residual change.
        T_delta = Tsqr_delta / (T + T0)
        r_delta = alpha * v0 - mu * T_delta
        cost = 0.5 * dm * r_delta * (2.0 * r0 + r_delta)
      elif state0 == int(types.ConstraintState.QUADRATIC):
        aq2 = alpha * quad[2]
        cost = alpha * (aq2 + quad[1]) - _eval_elliptic_quadratic_cone_gap(mu, N, T, dm)
      else:
        cost = 0.5 * dm * r * r

      return wp.vec3(
        cost,
        dm * r * r1,
        dm * (r1 * r1 + r * (-mu * T2)),
      )

  return wp.vec3(-cost0, 0.0, 0.0)


@wp.func
def _eval_elliptic_middle(
  # In:
  N: float,
  T: float,
  D0: float,
  mu: float,
  ufrictionj: float,
  is_normal: bool,
) -> wp.vec2:
  """Computes the elliptic-cone middle-zone (force, cost) for one row (cost 0 on tangent rows)."""
  dm = math.safe_div(D0, mu * mu * (1.0 + mu * mu))
  nmt = N - mu * T
  force_normal = -dm * nmt * mu
  if is_normal:
    return wp.vec2(force_normal, 0.5 * dm * nmt * nmt)
  return wp.vec2(-math.safe_div(force_normal, T) * ufrictionj, 0.0)


@wp.func
def _eval_constraint(
  # In:
  is_equality: bool,
  is_friction: bool,
  is_elliptic: bool,
  jaref: float,
  D: float,
  frictionloss: float,
  efcid: int,
  efcid0: int,
  jaref0: float,
  D0: float,
  mu: float,
  ufrictionj: float,
  TT: float,
) -> wp.vec3:
  if is_equality:
    force = -D * jaref
    cost = 0.5 * D * jaref * jaref
    return wp.vec3(force, float(types.ConstraintState.QUADRATIC.value), cost)

  if is_friction:
    rf = math.safe_div(frictionloss, D)
    if jaref <= -rf:
      return wp.vec3(frictionloss, float(types.ConstraintState.LINEARNEG.value), -frictionloss * (0.5 * rf + jaref))
    elif jaref >= rf:
      return wp.vec3(-frictionloss, float(types.ConstraintState.LINEARPOS.value), -frictionloss * (0.5 * rf - jaref))
    else:
      return wp.vec3(-D * jaref, float(types.ConstraintState.QUADRATIC.value), 0.5 * D * jaref * jaref)

  if is_elliptic:
    N = jaref0 * mu
    if TT <= 0.0:
      T = 0.0
    else:
      T = wp.sqrt(TT)

    # Top zone
    if (N >= mu * T) or ((T <= 0.0) and (N >= 0.0)):
      return wp.vec3(0.0, float(types.ConstraintState.SATISFIED.value), 0.0)
    # Bottom zone
    elif (mu * N + T <= 0.0) or ((T <= 0.0) and (N < 0.0)):
      return wp.vec3(-D * jaref, float(types.ConstraintState.QUADRATIC.value), 0.5 * D * jaref * jaref)
    # Middle zone
    else:
      is_normal = efcid == efcid0
      fc = _eval_elliptic_middle(N, T, D0, mu, ufrictionj, is_normal)
      return wp.vec3(fc[0], float(types.ConstraintState.CONE.value), fc[1])

  if jaref >= 0.0:
    return wp.vec3(0.0, float(types.ConstraintState.SATISFIED.value), 0.0)
  else:
    return wp.vec3(-D * jaref, float(types.ConstraintState.QUADRATIC.value), 0.5 * D * jaref * jaref)


# kernel_analyzer: off
@wp.func
def _compute_efc_eval_pt_pyramidal(
  efcid: int,
  alpha: float,
  ne: int,
  nf: int,
  # Per-row data:
  efc_D: float,
  efc_frictionloss: wp.array[float],
  ctx_Jaref: float,
  ctx_jv: float,
) -> wp.vec3:
  """Compute shifted cost, gradient, and hessian for pyramidal cones.

  Returns (cost(alpha) - cost(0), grad(alpha), hessian(alpha)) summed across the row.
  """
  # Limit/other constraint
  if efcid >= ne + nf:
    x = ctx_Jaref + alpha * ctx_jv
    quad0 = _eval_pt_direct_cost_alpha_zero(ctx_Jaref, efc_D)
    cost0 = wp.where(ctx_Jaref < 0.0, quad0, 0.0)
    # _eval_pt_direct_shifted returns quad(alpha) - quad(0); add back quad(0) when the
    # constraint was inactive at alpha=0 (i.e. cost(0) = 0) so we get quad(alpha) - 0.
    offset = quad0 - cost0
    if x < 0.0:
      return _eval_pt_direct_shifted(ctx_Jaref, ctx_jv, efc_D, alpha, offset)
    return wp.vec3(-cost0, 0.0, 0.0)

  # Friction constraint - needs quad for frictionloss computation
  if efcid >= ne:
    f = efc_frictionloss[efcid]
    x = ctx_Jaref + alpha * ctx_jv
    rf = math.safe_div(f, efc_D)
    return _shift_cost(_eval_frictionloss_pt(x, f, rf, ctx_jv, efc_D), _eval_frictionloss_cost(ctx_Jaref, f, rf, efc_D))

  # Equality constraint
  return _eval_pt_direct_shifted(ctx_Jaref, ctx_jv, efc_D, alpha, 0.0)


@wp.func
def _compute_efc_eval_pt_elliptic(
  efcid: int,
  alpha: float,
  ne: int,
  nf: int,
  impratio_invsqrt: float,
  # Per-row data (arrays for deferred load):
  efc_type: int,
  efc_D_in: wp.array[float],
  efc_frictionloss: wp.array[float],
  ctx_Jaref: float,
  ctx_jv: float,
  ctx_quad: wp.vec3,
  # Contact data (for elliptic):
  contact_friction: types.vec5,
  efc_address0: int,
  quad1: wp.vec3,
  quad2: wp.vec3,
) -> wp.vec3:
  """Compute shifted cost, gradient, and hessian for elliptic cones.

  Returns (cost(alpha) - cost(0), grad(alpha), hessian(alpha)) summed across the row.
  """
  # Contact/limit/other constraints
  if efcid >= ne + nf:
    if efc_type == types.ConstraintType.CONTACT_ELLIPTIC:
      if efcid != efc_address0:  # Not primary row
        return wp.vec3(0.0)
      mu = contact_friction[0] * impratio_invsqrt
      cost0, T0, r0, state0 = _eval_elliptic_reference(mu, ctx_quad, quad1, quad2)
      return _eval_elliptic_shifted(mu, ctx_quad, quad1, quad2, alpha, cost0, T0, r0, state0)

    # Limit/other constraint — direct eval (no quad read)
    x = ctx_Jaref + alpha * ctx_jv
    efc_D = efc_D_in[efcid]
    quad0 = _eval_pt_direct_cost_alpha_zero(ctx_Jaref, efc_D)
    cost0 = wp.where(ctx_Jaref < 0.0, quad0, 0.0)
    # See _compute_efc_eval_pt_pyramidal for the offset rationale.
    offset = quad0 - cost0
    if x < 0.0:
      return _eval_pt_direct_shifted(ctx_Jaref, ctx_jv, efc_D, alpha, offset)
    return wp.vec3(-cost0, 0.0, 0.0)

  # Friction constraint - load D and frictionloss only here
  if efcid >= ne:
    efc_D = efc_D_in[efcid]
    f = efc_frictionloss[efcid]
    x = ctx_Jaref + alpha * ctx_jv
    rf = math.safe_div(f, efc_D)
    return _shift_cost(_eval_frictionloss_pt(x, f, rf, ctx_jv, efc_D), _eval_frictionloss_cost(ctx_Jaref, f, rf, efc_D))

  # Equality constraint — direct eval (no quad read)
  efc_D = efc_D_in[efcid]
  return _eval_pt_direct_shifted(ctx_Jaref, ctx_jv, efc_D, alpha, 0.0)


@wp.func
def _compute_efc_eval_pt_alpha_zero_pyramidal(
  efcid: int,
  ne: int,
  nf: int,
  # Per-row data:
  efc_D: float,
  efc_frictionloss: wp.array[float],
  ctx_Jaref: float,
  ctx_jv: float,
) -> wp.vec3:
  """Optimized version for alpha=0.0, pyramidal cones."""
  # Limit/other constraint
  if efcid >= ne + nf:
    if ctx_Jaref < 0.0:
      return _eval_pt_direct_alpha_zero(ctx_Jaref, ctx_jv, efc_D)
    return wp.vec3(0.0)

  # Friction constraint - needs quad for frictionloss computation
  if efcid >= ne:
    f = efc_frictionloss[efcid]
    rf = math.safe_div(f, efc_D)
    return _eval_frictionloss_pt(ctx_Jaref, f, rf, ctx_jv, efc_D)

  # Equality constraint
  return _eval_pt_direct_alpha_zero(ctx_Jaref, ctx_jv, efc_D)


@wp.func
def _compute_efc_eval_pt_alpha_zero_elliptic(
  efcid: int,
  ne: int,
  nf: int,
  impratio_invsqrt: float,
  # Per-row data (arrays for deferred load):
  efc_type: int,
  efc_D_in: wp.array[float],
  efc_frictionloss: wp.array[float],
  ctx_Jaref: float,
  ctx_jv: float,
  ctx_quad: wp.vec3,
  # Contact data (for elliptic):
  contact_friction: types.vec5,
  efc_address0: int,
  quad1: wp.vec3,
  quad2: wp.vec3,
) -> wp.vec3:
  """Optimized version for alpha=0.0, elliptic cones."""
  # Contact/limit/other constraints
  if efcid >= ne + nf:
    if efc_type == types.ConstraintType.CONTACT_ELLIPTIC:
      if efcid != efc_address0:  # Not primary row
        return wp.vec3(0.0)
      mu = contact_friction[0] * impratio_invsqrt
      return _eval_elliptic_alpha_zero(mu, ctx_quad, quad1, quad2)

    # Limit/other constraint — direct eval (no quad read)
    if ctx_Jaref < 0.0:
      return _eval_pt_direct_alpha_zero(ctx_Jaref, ctx_jv, efc_D_in[efcid])
    return wp.vec3(0.0)

  # Friction constraint - load D and frictionloss only here
  if efcid >= ne:
    efc_D = efc_D_in[efcid]
    f = efc_frictionloss[efcid]
    rf = math.safe_div(f, efc_D)
    return _eval_frictionloss_pt(ctx_Jaref, f, rf, ctx_jv, efc_D)

  # Equality constraint — direct eval (no quad read)
  return _eval_pt_direct_alpha_zero(ctx_Jaref, ctx_jv, efc_D_in[efcid])


@wp.func
def _compute_efc_eval_pt_3alphas_pyramidal(
  efcid: int,
  lo_alpha: float,
  hi_alpha: float,
  mid_alpha: float,
  ne: int,
  nf: int,
  # Per-row data:
  efc_D: float,
  efc_frictionloss: wp.array[float],
  ctx_Jaref: float,
  ctx_jv: float,
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
  """Compute shifted cost, gradient, and hessian for 3 alphas, pyramidal cones.

  Returns a tuple of 3 vec3s for (lo_alpha, hi_alpha, mid_alpha).
  Constraint types checked in order: limit/other -> friction -> equality.
  """
  # Limit/other constraints: active only when x < 0
  if efcid >= ne + nf:
    x_lo = ctx_Jaref + lo_alpha * ctx_jv
    x_hi = ctx_Jaref + hi_alpha * ctx_jv
    x_mid = ctx_Jaref + mid_alpha * ctx_jv
    quad0 = _eval_pt_direct_cost_alpha_zero(ctx_Jaref, efc_D)
    cost0 = wp.where(ctx_Jaref < 0.0, quad0, 0.0)
    # See _compute_efc_eval_pt_pyramidal for the offset rationale.
    offset = quad0 - cost0
    pt_lo, pt_hi, pt_mid = _eval_pt_direct_shifted_3alphas(ctx_Jaref, ctx_jv, efc_D, lo_alpha, hi_alpha, mid_alpha, offset)
    inactive = wp.vec3(-cost0, 0.0, 0.0)
    return (
      wp.where(x_lo < 0.0, pt_lo, inactive),
      wp.where(x_hi < 0.0, pt_hi, inactive),
      wp.where(x_mid < 0.0, pt_mid, inactive),
    )

  # Friction constraint - needs quad for frictionloss computation
  if efcid >= ne:
    x_lo = ctx_Jaref + lo_alpha * ctx_jv
    x_hi = ctx_Jaref + hi_alpha * ctx_jv
    x_mid = ctx_Jaref + mid_alpha * ctx_jv
    f = efc_frictionloss[efcid]
    rf = math.safe_div(f, efc_D)
    cost0 = _eval_frictionloss_cost(ctx_Jaref, f, rf, efc_D)
    lo, hi, mid = _eval_frictionloss_pt_3alphas(x_lo, x_hi, x_mid, f, rf, ctx_jv, efc_D)
    return (_shift_cost(lo, cost0), _shift_cost(hi, cost0), _shift_cost(mid, cost0))

  # Equality constraint: always active
  return _eval_pt_direct_shifted_3alphas(ctx_Jaref, ctx_jv, efc_D, lo_alpha, hi_alpha, mid_alpha, 0.0)


@wp.func
def _compute_efc_eval_pt_3alphas_elliptic(
  efcid: int,
  lo_alpha: float,
  hi_alpha: float,
  mid_alpha: float,
  ne: int,
  nf: int,
  impratio_invsqrt: float,
  # Per-row data (arrays for deferred load):
  efc_type: int,
  efc_D_in: wp.array[float],
  efc_frictionloss: wp.array[float],
  ctx_Jaref: float,
  ctx_jv: float,
  ctx_quad: wp.vec3,
  # Contact data (for elliptic):
  contact_friction: types.vec5,
  efc_address0: int,
  quad1: wp.vec3,
  quad2: wp.vec3,
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
  """Compute shifted cost, gradient, and hessian for 3 alphas, elliptic cones.

  Returns a tuple of 3 vec3s for (lo_alpha, hi_alpha, mid_alpha).
  Constraint types checked in order: contact elliptic/limit/other -> friction -> equality.
  """
  # Contact/limit/other constraints
  if efcid >= ne + nf:
    # Contact elliptic: uses special elliptic cone evaluation
    if efc_type == types.ConstraintType.CONTACT_ELLIPTIC:
      if efcid != efc_address0:  # secondary rows contribute nothing
        return (wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0))
      mu = contact_friction[0] * impratio_invsqrt
      cost0, T0, r0, state0 = _eval_elliptic_reference(mu, ctx_quad, quad1, quad2)
      return (
        _eval_elliptic_shifted(mu, ctx_quad, quad1, quad2, lo_alpha, cost0, T0, r0, state0),
        _eval_elliptic_shifted(mu, ctx_quad, quad1, quad2, hi_alpha, cost0, T0, r0, state0),
        _eval_elliptic_shifted(mu, ctx_quad, quad1, quad2, mid_alpha, cost0, T0, r0, state0),
      )

    # Limit/other constraints — direct eval (no quad read)
    x_lo = ctx_Jaref + lo_alpha * ctx_jv
    x_hi = ctx_Jaref + hi_alpha * ctx_jv
    x_mid = ctx_Jaref + mid_alpha * ctx_jv
    efc_D = efc_D_in[efcid]
    quad0 = _eval_pt_direct_cost_alpha_zero(ctx_Jaref, efc_D)
    cost0 = wp.where(ctx_Jaref < 0.0, quad0, 0.0)
    # See _compute_efc_eval_pt_pyramidal for the offset rationale.
    offset = quad0 - cost0
    pt_lo, pt_hi, pt_mid = _eval_pt_direct_shifted_3alphas(ctx_Jaref, ctx_jv, efc_D, lo_alpha, hi_alpha, mid_alpha, offset)
    inactive = wp.vec3(-cost0, 0.0, 0.0)
    return (
      wp.where(x_lo < 0.0, pt_lo, inactive),
      wp.where(x_hi < 0.0, pt_hi, inactive),
      wp.where(x_mid < 0.0, pt_mid, inactive),
    )

  # Friction constraint - load D and frictionloss only here
  if efcid >= ne:
    x_lo = ctx_Jaref + lo_alpha * ctx_jv
    x_hi = ctx_Jaref + hi_alpha * ctx_jv
    x_mid = ctx_Jaref + mid_alpha * ctx_jv
    efc_D = efc_D_in[efcid]
    f = efc_frictionloss[efcid]
    rf = math.safe_div(f, efc_D)
    cost0 = _eval_frictionloss_cost(ctx_Jaref, f, rf, efc_D)
    lo, hi, mid = _eval_frictionloss_pt_3alphas(x_lo, x_hi, x_mid, f, rf, ctx_jv, efc_D)
    return (_shift_cost(lo, cost0), _shift_cost(hi, cost0), _shift_cost(mid, cost0))

  # Equality constraint — direct eval (no quad read)
  efc_D = efc_D_in[efcid]
  return _eval_pt_direct_shifted_3alphas(ctx_Jaref, ctx_jv, efc_D, lo_alpha, hi_alpha, mid_alpha, 0.0)


# kernel_analyzer: on

# =============================================================================
# Iterative Linesearch
# =============================================================================
#
# Iterative linesearch implementation using Warp's tiled execution model with
# parallel reductions over constraint (EFC) rows.
#
# Key optimizations:
#
# 1. KERNEL FUSION - Reduces kernel launch overhead by combining:
#    - linesearch_jv_fused: jv = J @ search (for small nv <= 50)
#    - linesearch_prepare_quad: quad coefficients (pyramidal: computed directly,
#      elliptic: computed in a prepare phase with __syncthreads barrier)
#    - linesearch_prepare_gauss: quad_gauss via tile reduction over DOFs
#    - linesearch_qacc_ma: qacc and Ma updates at kernel end
#    - linesearch_jaref: Jaref update at kernel end
#
# 2. PARALLEL REDUCTIONS - Uses wp.tile_reduce for summing cost/gradient/hessian
#    contributions across EFC rows within each world. The main iteration loop
#    packs 3 vec3 reductions into a single mat33 reduction for efficiency.
#
# 3. COMPILE-TIME SPECIALIZATION via factory parameters:
#    - cone_type: Eliminates elliptic cone branches for pyramidal-only models
#    - ls_iterations: Enables loop unrolling for the main bracket search
#    - fuse_jv: Conditionally includes jv computation based on nv size
#
# 4. DIRECT EVALUATION (pyramidal only) - For equality and limit constraints,
#    computes cost/gradient/hessian directly from (Jaref, jv, efc_D, alpha)
#    without intermediate quad coefficients, using _eval_pt_direct functions.
#
# 5. BATCHED 3-ALPHA EVALUATION - The main iteration loop evaluates 3 alpha
#    values per iteration (lo_next, hi_next, mid). Instead of calling
#    _compute_efc_eval_pt 3 times per constraint row (which would repeat
#    constraint type checks and data loads), we use _compute_efc_eval_pt_3alphas
#    which:
#    - Performs constraint type branching once per row
#    - Loads efc_D, efc_frictionloss, contact data once
#    - Computes x = Jaref + alpha * jv for all 3 alphas
#    - For pyramidal direct evaluation: shares jvD = jv * efc_D, hessian = jv * jvD
#    - For quad-based evaluation: uses _eval_pt_3alphas which computes the
#      constant hessian (2.0 * quad[2]) once and reuses for all 3 alphas
#
# 6. DEFERRED DATA LOADING - efc_D and efc_frictionloss are only loaded
#    inside the constraint branches where they're needed, reducing register
#    pressure for other constraint types.
#
# Trade-offs:
# - Requires block synchronization (__syncthreads) for elliptic quad preparation
# - Separate kernel compilation for each (block_dim, ls_iterations, cone_type,
#   fuse_jv) combination (cached by Warp)
#
# Optimizations attempted but not beneficial:
# - Caching EFC data (Jaref, jv, quad, etc.) in shared memory tiles for reuse
#   across the p0, lo_in, and main iteration loops.
#
# =============================================================================


@cache_kernel
def _linesearch_iterative_kernel(
  ls_iterations: int,
  cone_type: types.ConeType,
  fuse_jv: bool,
  is_sparse: bool,
  incremental: bool,
  warn_overflow: int,
  ragged: bool = False,
):
  """Factory for iterative linesearch kernel.

  Args:
    ls_iterations: Max linesearch iterations (compile-time constant for loop optimization).
    cone_type: Friction cone type (PYRAMIDAL or ELLIPTIC) for compile-time optimization.
    fuse_jv: Whether to compute jv = J @ search in-kernel (efficient for small nv).
    is_sparse: Use sparse matrix representation for constraint Jacobian.
    incremental: Use incremental linesearch updates.
    warn_overflow: Overflow warning bitmask.
    ragged: Rows of problem p start at a per-problem offset in a world's rows (SolverContext).
  """
  LS_ITERATIONS = ls_iterations
  IS_ELLIPTIC = cone_type == types.ConeType.ELLIPTIC
  FUSE_JV = fuse_jv
  INCREMENTAL = incremental
  IS_SPARSE = is_sparse
  # problem p's rows are rows [efc_rowoff[p], + nefc) of world efc_world[p] (see SolverContext)
  RAGGED = ragged

  # Native snippet for CUDA __syncthreads()
  @wp.func_native(snippet="WP_TILE_SYNC();")
  def _syncthreads():
    pass

  # Select specialized helper functions based on cone type
  if IS_ELLIPTIC:
    _compute_efc_eval_pt = _compute_efc_eval_pt_elliptic
    _compute_efc_eval_pt_alpha_zero = _compute_efc_eval_pt_alpha_zero_elliptic
    _compute_efc_eval_pt_3alphas = _compute_efc_eval_pt_3alphas_elliptic
  else:
    _compute_efc_eval_pt = _compute_efc_eval_pt_pyramidal
    _compute_efc_eval_pt_alpha_zero = _compute_efc_eval_pt_alpha_zero_pyramidal
    _compute_efc_eval_pt_3alphas = _compute_efc_eval_pt_3alphas_pyramidal

  @wp.kernel(module="unique", enable_backward=False, grid_stride=True)
  def kernel(
    # Model:
    nv: int,
    opt_tolerance: wp.array[float],
    opt_ls_tolerance: wp.array[float],
    opt_impratio_invsqrt: wp.array[float],
    stat_meaninertia: wp.array[float],
    # Data in:
    ne_in: wp.array[int],
    nf_in: wp.array[int],
    nefc_in: wp.array[int],
    qfrc_smooth_in: wp.array2d[float],
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
    njmax_in: int,
    nacon_in: wp.array[int],
    # In:
    ctx_search_unchanged_in: wp.array[bool],
    ctx_Jaref_in: wp.array2d[float],
    ctx_search_in: wp.array2d[float],
    ctx_search_dot_in: wp.array[float],
    ctx_mv_in: wp.array2d[float],
    ctx_jv_in: wp.array2d[float],
    ctx_quad_in: wp.array2d[wp.vec3],
    ctx_done_in: wp.array[bool],
    ctx_efc_world_in: wp.array[int],
    ctx_efc_rowoff_in: wp.array[int],
    # Data out:
    qacc_out: wp.array2d[float],
    efc_Ma_out: wp.array2d[float],
    overflow_out: wp.array[int],
    # Out:
    ctx_Jaref_out: wp.array2d[float],
    ctx_jv_out: wp.array2d[float],
    ctx_quad_out: wp.array2d[wp.vec3],
    ctx_improvement_out: wp.array[float],
    ctx_alpha_out: wp.array[float],
    ctx_ls_exhausted_out: wp.array[bool],
  ):
    worldid, tid = wp.tid()

    if ctx_done_in[worldid]:
      return

    ne = ne_in[worldid]
    nf = nf_in[worldid]
    nefc = wp.min(njmax_in, nefc_in[worldid])
    rw = worldid
    ro = int(0)
    if wp.static(RAGGED):
      rw = ctx_efc_world_in[worldid]
      ro = ctx_efc_rowoff_in[worldid]
    if wp.static(RAGGED):
      ne += ro
      nefc += ro

    # jv = J @ search (fused for small nv); on unchanged search the buffer
    # already holds jv (see _linesearch).
    if wp.static(FUSE_JV):
      recompute_jv = True
      if wp.static(INCREMENTAL):
        recompute_jv = not ctx_search_unchanged_in[worldid]
      if recompute_jv:
        for efcid in range(ro + tid, nefc, wp.block_dim()):
          jv = float(0.0)
          if wp.static(IS_SPARSE):
            rownnz = efc_J_rownnz_in[rw, efcid]
            rowadr = efc_J_rowadr_in[rw, efcid]
            for k in range(rownnz):
              sparseid = rowadr + k
              colind = efc_J_colind_in[rw, 0, sparseid]
              jv += efc_J_in[rw, 0, sparseid] * ctx_search_in[worldid, colind]
          else:
            for i in range(nv):
              jv += efc_J_in[rw, efcid, i] * ctx_search_in[worldid, i]
          ctx_jv_out[rw, efcid] = jv

      _syncthreads()  # ensure all jv values are written before reading

    # quad coefficients (elliptic contacts only, requires barrier sync)
    # Non-elliptic constraints (equality, friction, limit) now use direct
    # evaluation from (Jaref, jv, efc_D), avoiding quad reads entirely.
    if wp.static(IS_ELLIPTIC):
      # elliptic-only config values
      impratio_invsqrt = opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]
      nacon = nacon_in[0]

      for efcid in range(ro + tid, nefc, wp.block_dim()):
        # Only compute and store quad for CONTACT_ELLIPTIC (needs inter-row data)
        if efc_type_in[rw, efcid] == types.ConstraintType.CONTACT_ELLIPTIC:
          conid = efc_id_in[rw, efcid]
          if conid < nacon:
            efcid0 = contact_efc_address_in[conid, 0]
            if efcid == efcid0:
              Jaref = ctx_Jaref_in[rw, efcid]
              jv = ctx_jv_in[rw, efcid]
              efc_D = efc_D_in[rw, efcid]

              jvD = jv * efc_D
              quad = wp.vec3(0.5 * Jaref * Jaref * efc_D, jvD * Jaref, 0.5 * jv * jvD)

              # primary row: accumulate secondary rows and write quad, quad1, quad2
              dim = contact_dim_in[conid]
              friction = contact_friction_in[conid]
              mu = friction[0] * impratio_invsqrt

              u0 = Jaref * mu
              v0 = jv * mu

              uu = float(0.0)
              uv = float(0.0)
              vv = float(0.0)
              for j in range(1, dim):
                efcidj = contact_efc_address_in[conid, j]
                if efcidj >= 0:
                  jvj = ctx_jv_in[rw, efcidj]
                  jarefj = ctx_Jaref_in[rw, efcidj]
                  dj = efc_D_in[rw, efcidj]
                  DJj = dj * jarefj

                  quad += wp.vec3(0.5 * jarefj * DJj, jvj * DJj, 0.5 * jvj * dj * jvj)

                  # rescale to make primal cone circular
                  frictionj = friction[j - 1]
                  uj = jarefj * frictionj
                  vj = jvj * frictionj

                  uu += uj * uj
                  uv += uj * vj
                  vv += vj * vj

              ctx_quad_out[rw, efcid] = quad

              efcid1 = contact_efc_address_in[conid, 1]
              ctx_quad_out[rw, efcid1] = wp.vec3(u0, v0, uu)

              mu2 = mu * mu
              efcid2 = contact_efc_address_in[conid, 2]
              ctx_quad_out[rw, efcid2] = wp.vec3(uv, vv, efc_D / (mu2 * (1.0 + mu2)))

      _syncthreads()  # ensure all quads are written before reading

    # gtol (tolerance values loaded here, deferred from kernel start)
    tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]
    ls_tolerance = opt_ls_tolerance[worldid % opt_ls_tolerance.shape[0]]
    snorm = wp.sqrt(ctx_search_dot_in[worldid])
    meaninertia = stat_meaninertia[worldid % stat_meaninertia.shape[0]]
    scale = meaninertia * wp.float(nv)
    gtol = wp.max(tolerance * ls_tolerance * snorm * scale, 1e-6)

    # quad_gauss = [0, search.T @ Ma - search.T @ qfrc_smooth, 0.5 * search.T @ mv] + equality
    local_gauss = wp.vec2(0.0)
    for dofid in range(tid, nv, wp.block_dim()):
      search = ctx_search_in[worldid, dofid]
      local_gauss += wp.vec2(
        search * (efc_Ma_out[worldid, dofid] - qfrc_smooth_in[worldid, dofid]),
        0.5 * search * ctx_mv_in[worldid, dofid],
      )

    for efcid in range(ro + tid, ne, wp.block_dim()):
      jv = ctx_jv_in[rw, efcid]
      jvD = jv * efc_D_in[rw, efcid]
      local_gauss += wp.vec2(
        jvD * ctx_Jaref_in[rw, efcid],
        0.5 * jv * jvD,
      )

    gauss_tile = wp.tile(local_gauss, preserve_type=True)
    gauss_sum = wp.tile_reduce(wp.add, gauss_tile)
    gauss_reduced = gauss_sum[0]
    ctx_quad_gauss = wp.vec3(0.0, gauss_reduced[0], gauss_reduced[1])

    # p0 via parallel reduction over non-equality constraints
    local_p0 = wp.vec3(0.0)
    for efcid in range(ne + tid, nefc, wp.block_dim()):
      if wp.static(IS_ELLIPTIC):
        efc_type = efc_type_in[rw, efcid]
        efc_id = 0
        contact_friction = types.vec5(0.0)
        efc_addr0 = int(0)
        ctx_quad = wp.vec3(0.0)
        quad1 = wp.vec3(0.0)
        quad2 = wp.vec3(0.0)

        if efc_type == types.ConstraintType.CONTACT_ELLIPTIC:
          efc_id = efc_id_in[rw, efcid]
          contact_friction = contact_friction_in[efc_id]
          efc_addr0 = contact_efc_address_in[efc_id, 0]
          efc_addr1 = contact_efc_address_in[efc_id, 1]
          efc_addr2 = contact_efc_address_in[efc_id, 2]
          ctx_quad = ctx_quad_in[rw, efcid]
          quad1 = ctx_quad_in[rw, efc_addr1]
          quad2 = ctx_quad_in[rw, efc_addr2]

        local_p0 += _compute_efc_eval_pt_alpha_zero(
          efcid,
          ne,
          nf,
          impratio_invsqrt,
          efc_type,
          efc_D_in[rw],
          efc_frictionloss_in[rw],
          ctx_Jaref_in[rw, efcid],
          ctx_jv_in[rw, efcid],
          ctx_quad,
          contact_friction,
          efc_addr0,
          quad1,
          quad2,
        )
      else:
        # direct evaluation for pyramidal cones (no intermediate quad)
        local_p0 += _compute_efc_eval_pt_alpha_zero(
          efcid,
          ne,
          nf,
          efc_D_in[rw, efcid],
          efc_frictionloss_in[rw],
          ctx_Jaref_in[rw, efcid],
          ctx_jv_in[rw, efcid],
        )

    # at this point, every thread has computed some contributions to p0 in local_p0
    # we now create a tile of all local_p0 contributions and reduce them to a single value
    # this is done in parallel using a tile reduction
    p0_tile = wp.tile(local_p0, preserve_type=True)
    p0_sum = wp.tile_reduce(wp.add, p0_tile)

    # add quad_gauss contribution to p0
    p0 = wp.vec3(ctx_quad_gauss[0], ctx_quad_gauss[1], 2.0 * ctx_quad_gauss[2]) + p0_sum[0]
    p0_delta = wp.vec3(0.0, p0[1], p0[2])

    # The bracketing search reads derivative sums whose rounding noise is ~eps
    # times their gross magnitude, so roots below eps * |q1| / p0[2] (and never
    # below 8 ulps of the unit ray anchor) are noise, not descent: an accepted
    # step under this floor means the stale ray is exhausted and the fast path
    # must rebuild the world. Per quadratic row |q1| = sqrt(2 * cost * hessian),
    # so Cauchy-Schwarz bounds the row total by sums already reduced for p0; the
    # smooth term is added exactly. Friction linear rows fall outside the bound,
    # which only lowers the floor toward the fixed 8-ulp base.
    rows = p0_sum[0]
    q1_abs = wp.sqrt(2.0 * wp.max(rows[0], 0.0) * wp.max(rows[2], 0.0)) + wp.abs(ctx_quad_gauss[1])

    # set acceptance tolerance to avoid exceeding gtol in f32
    gtol_accept = wp.max(gtol, _ALPHA_NOISE_EPS * q1_abs)

    noise_floor = float(0.0)
    if wp.static(INCREMENTAL):
      noise_floor = _ALPHA_NOISE_EPS * wp.max(1.0, math.safe_div(q1_abs, p0[2]))

    # lo_in at lo_alpha_in = -p0[1] / p0[2]
    lo_alpha_in = -math.safe_div(p0[1], p0[2])

    local_lo_in = wp.vec3(0.0)
    for efcid in range(ne + tid, nefc, wp.block_dim()):
      if wp.static(IS_ELLIPTIC):
        efc_type = efc_type_in[rw, efcid]
        efc_id = 0
        contact_friction = types.vec5(0.0)
        efc_addr0 = int(0)
        ctx_quad = wp.vec3(0.0)
        quad1 = wp.vec3(0.0)
        quad2 = wp.vec3(0.0)

        if efc_type == types.ConstraintType.CONTACT_ELLIPTIC:
          efc_id = efc_id_in[rw, efcid]
          contact_friction = contact_friction_in[efc_id]
          efc_addr0 = contact_efc_address_in[efc_id, 0]
          efc_addr1 = contact_efc_address_in[efc_id, 1]
          efc_addr2 = contact_efc_address_in[efc_id, 2]
          ctx_quad = ctx_quad_in[rw, efcid]
          quad1 = ctx_quad_in[rw, efc_addr1]
          quad2 = ctx_quad_in[rw, efc_addr2]

        local_lo_in += _compute_efc_eval_pt(
          efcid,
          lo_alpha_in,
          ne,
          nf,
          impratio_invsqrt,
          efc_type,
          efc_D_in[rw],
          efc_frictionloss_in[rw],
          ctx_Jaref_in[rw, efcid],
          ctx_jv_in[rw, efcid],
          ctx_quad,
          contact_friction,
          efc_addr0,
          quad1,
          quad2,
        )
      else:
        # direct evaluation for pyramidal cones (no intermediate quad)
        local_lo_in += _compute_efc_eval_pt(
          efcid,
          lo_alpha_in,
          ne,
          nf,
          efc_D_in[rw, efcid],
          efc_frictionloss_in[rw],
          ctx_Jaref_in[rw, efcid],
          ctx_jv_in[rw, efcid],
        )

    lo_in_tile = wp.tile(local_lo_in, preserve_type=True)
    lo_in_sum = wp.tile_reduce(wp.add, lo_in_tile)
    lo_in = _eval_pt(ctx_quad_gauss, lo_alpha_in) + lo_in_sum[0]

    # a slope at alpha = 0 below rounding noise means the start is already the line minimum
    stationary = wp.abs(p0[1]) < _ALPHA_NOISE_EPS * q1_abs

    # accept Newton step if derivative is small and cost improved
    initial_converged = stationary or (wp.abs(lo_in[1]) < gtol_accept and lo_in[0] < 0.0)
    ls_converged = initial_converged

    # main iterative loop - skip if already converged
    if not initial_converged:
      alpha = float(0.0)
      improvement = float(0.0)

      # initialize bounds
      lo_less = lo_in[1] < p0[1]
      lo = wp.where(lo_less, lo_in, p0_delta)
      lo_alpha = wp.where(lo_less, lo_alpha_in, 0.0)
      hi = wp.where(lo_less, p0_delta, lo_in)
      hi_alpha = wp.where(lo_less, 0.0, lo_alpha_in)

      for _ in range(LS_ITERATIONS):
        lo_next_alpha = lo_alpha - math.safe_div(lo[1], lo[2])
        hi_next_alpha = hi_alpha - math.safe_div(hi[1], hi[2])
        mid_alpha = 0.5 * (lo_alpha + hi_alpha)

        local_lo = wp.vec3(0.0)
        local_hi = wp.vec3(0.0)
        local_mid = wp.vec3(0.0)

        for efcid in range(ne + tid, nefc, wp.block_dim()):
          if wp.static(IS_ELLIPTIC):
            efc_type = efc_type_in[rw, efcid]
            efc_id = 0
            contact_friction = types.vec5(0.0)
            efc_addr0 = int(0)
            ctx_quad = wp.vec3(0.0)
            quad1 = wp.vec3(0.0)
            quad2 = wp.vec3(0.0)

            if efc_type == types.ConstraintType.CONTACT_ELLIPTIC:
              efc_id = efc_id_in[rw, efcid]
              contact_friction = contact_friction_in[efc_id]
              efc_addr0 = contact_efc_address_in[efc_id, 0]
              efc_addr1 = contact_efc_address_in[efc_id, 1]
              efc_addr2 = contact_efc_address_in[efc_id, 2]
              ctx_quad = ctx_quad_in[rw, efcid]
              quad1 = ctx_quad_in[rw, efc_addr1]
              quad2 = ctx_quad_in[rw, efc_addr2]

            r_lo, r_hi, r_mid = _compute_efc_eval_pt_3alphas(
              efcid,
              lo_next_alpha,
              hi_next_alpha,
              mid_alpha,
              ne,
              nf,
              impratio_invsqrt,
              efc_type,
              efc_D_in[rw],
              efc_frictionloss_in[rw],
              ctx_Jaref_in[rw, efcid],
              ctx_jv_in[rw, efcid],
              ctx_quad,
              contact_friction,
              efc_addr0,
              quad1,
              quad2,
            )
          else:
            # direct evaluation for pyramidal cones (no intermediate quad)
            r_lo, r_hi, r_mid = _compute_efc_eval_pt_3alphas(
              efcid,
              lo_next_alpha,
              hi_next_alpha,
              mid_alpha,
              ne,
              nf,
              efc_D_in[rw, efcid],
              efc_frictionloss_in[rw],
              ctx_Jaref_in[rw, efcid],
              ctx_jv_in[rw, efcid],
            )
          local_lo += r_lo
          local_hi += r_hi
          local_mid += r_mid

        # reduce with packed mat33 (3 vec3s into columns: col0=lo, col1=hi, col2=mid)
        local_combined = wp.mat33(
          local_lo[0],
          local_hi[0],
          local_mid[0],
          local_lo[1],
          local_hi[1],
          local_mid[1],
          local_lo[2],
          local_hi[2],
          local_mid[2],
        )

        # reduce with packed mat33 (3 vec3s into columns: col0=lo, col1=hi, col2=mid)
        # this is faster than 3 vec3 reductions because it avoids synchronization barriers
        combined_tile = wp.tile(local_combined, preserve_type=True)
        combined_sum = wp.tile_reduce(wp.add, combined_tile)
        result = combined_sum[0]

        # extract columns back to vec3s and add quad_gauss contributions
        gauss_lo, gauss_hi, gauss_mid = _eval_pt_3alphas(ctx_quad_gauss, lo_next_alpha, hi_next_alpha, mid_alpha)
        lo_next = gauss_lo + wp.vec3(result[0, 0], result[1, 0], result[2, 0])
        hi_next = gauss_hi + wp.vec3(result[0, 1], result[1, 1], result[2, 1])
        mid = gauss_mid + wp.vec3(result[0, 2], result[1, 2], result[2, 2])

        # accept the lowest-cost converged candidate regardless of the sign of its derivative
        conv_lo = wp.abs(lo_next[1]) < gtol_accept and lo_next[0] < 0.0
        conv_hi = wp.abs(hi_next[1]) < gtol_accept and hi_next[0] < 0.0
        conv_mid = wp.abs(mid[1]) < gtol_accept and mid[0] < 0.0
        converged = conv_lo or conv_hi or conv_mid
        swap_lo = False
        swap_hi = False
        if converged:
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
          # bracket swapping
          # swap lo:
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

          # swap hi:
          # also accept a Newton step that crosses the minimizer from an undershooting hi,
          # turning the one-sided search into an opposite-sign bracket
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

        # check for convergence
        ls_done = (
          converged
          or (not swap_lo and not swap_hi)
          or (lo[0] < 0.0 and lo[1] < 0.0 and lo[1] > -gtol)
          or (hi[0] < 0.0 and hi[1] > 0.0 and hi[1] < gtol)
        )

        # update alpha if improved
        improved = lo[0] < 0.0 or hi[0] < 0.0
        lo_better = lo[0] < hi[0]
        best_alpha = wp.where(lo_better, lo_alpha, hi_alpha)
        best_delta = wp.where(lo_better, lo[0], hi[0])
        alpha = wp.where(improved, best_alpha, alpha)
        improvement = wp.where(improved, -best_delta, improvement)

        if ls_done:
          ls_converged = True
          break
    elif stationary:
      alpha = 0.0
      improvement = 0.0
    else:
      alpha = lo_alpha_in
      improvement = -lo_in[0]

    # qacc and Ma update
    for dofid in range(tid, nv, wp.block_dim()):
      qacc_out[worldid, dofid] += alpha * ctx_search_in[worldid, dofid]
      efc_Ma_out[worldid, dofid] += alpha * ctx_mv_in[worldid, dofid]

    # Jaref update
    for efcid in range(ro + tid, nefc, wp.block_dim()):
      ctx_Jaref_out[rw, efcid] += alpha * ctx_jv_in[rw, efcid]

    if tid == 0:
      ctx_improvement_out[worldid] = improvement
      ctx_alpha_out[worldid] = alpha
      if wp.static(INCREMENTAL):
        ctx_ls_exhausted_out[worldid] = wp.abs(alpha) < noise_floor
      if not ls_converged:
        if wp.static(bool(warn_overflow & OverflowType.LS_ITERATIONS)):
          wp.printf(
            "linesearch iterations limit reached - please increase ls_iterations beyond %u\n"
            "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.LS_ITERATIONS (or = 0 for all)\n",
            wp.static(LS_ITERATIONS),
          )
        wp.atomic_or(overflow_out, worldid, wp.static(OverflowType.LS_ITERATIONS))

  return kernel


def _linesearch_iterative(m: types.Model, d: types.Data, ctx: SolverContext, fuse_jv: bool):
  """Iterative linesearch with parallel reductions over efc rows and dofs.

  Args:
    m: Model.
    d: Data.
    ctx: SolverContext.
    fuse_jv: Whether jv is computed in-kernel (True) or pre-computed (False).
  """
  wp.launch_tiled(
    _linesearch_iterative_kernel(
      m.opt.ls_iterations,
      m.opt.cone,
      fuse_jv,
      m.is_sparse,
      _use_incremental(m),
      int(m.opt.warn_overflow),
      ctx.efc_world is not None,
    ),
    dim=d.nworld,
    inputs=[
      m.nv,
      m.opt.tolerance,
      m.opt.ls_tolerance,
      m.opt.impratio_invsqrt,
      m.stat.meaninertia,
      d.ne,
      d.nf,
      d.nefc,
      d.qfrc_smooth,
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
      ctx.search_unchanged,
      ctx.Jaref,
      ctx.search,
      ctx.search_dot,
      ctx.mv,
      ctx.jv,
      ctx.quad,
      ctx.done,
      ctx.efc_world,
      ctx.efc_rowoff,
    ],
    outputs=[
      d.qacc,
      d.efc.Ma,
      d.overflow,
      ctx.Jaref,
      ctx.jv,
      ctx.quad,
      ctx.improvement,
      ctx.alpha,
      ctx.ls_exhausted,
    ],
    block_dim=m.block_dim.linesearch_iterative,
  )


@cache_kernel
def _linesearch_zero_jv(ragged: bool = False):
  RAGGED = ragged

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Data in:
    nefc_in: wp.array[int],
    # In:
    skip_in: wp.array[bool],
    ctx_efc_problem_in: wp.array2d[int],
    # Out:
    ctx_jv_out: wp.array2d[float],
  ):
    worldid, efcid = wp.tid()
    rw = worldid
    if wp.static(RAGGED):
      # threads are the worlds' rows (jv is in the worlds' row layout)
      problem = ctx_efc_problem_in[worldid, efcid]
      if problem < 0:
        return
      worldid = problem
    elif efcid >= nefc_in[worldid]:
      return

    if skip_in[worldid]:
      return

    ctx_jv_out[rw, efcid] = 0.0

  return kernel


@cache_kernel
def _linesearch_jv_fused_kernel(is_sparse: bool, nv: int, dofs_per_thread: int, compact: bool, ragged: bool = False):
  COMPACT = compact
  RAGGED = ragged

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Data in:
    nefc_in: wp.array[int],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    dof_cdof_in: wp.array2d[int],
    # In:
    ctx_search_in: wp.array2d[float],
    skip_in: wp.array[bool],
    ctx_efc_problem_in: wp.array2d[int],
    # Out:
    ctx_jv_out: wp.array2d[float],
  ):
    worldid, efcid, dofstart = wp.tid()

    rw = worldid
    if wp.static(RAGGED):
      # threads are the worlds' rows (J and jv are in the worlds' row layout)
      problem = ctx_efc_problem_in[worldid, efcid]
      if problem < 0:
        return
      worldid = problem
    elif efcid >= nefc_in[worldid]:
      return

    if skip_in[worldid]:
      return

    jv_out = float(0.0)

    if wp.static(dofs_per_thread >= nv):
      if wp.static(is_sparse):
        # Sparse: iterate over non-zero entries in the row
        rownnz = efc_J_rownnz_in[rw, efcid]
        rowadr = efc_J_rowadr_in[rw, efcid]
        for k in range(rownnz):
          sparseid = rowadr + k
          colind = efc_J_colind_in[rw, 0, sparseid]
          if wp.static(COMPACT):
            colind = dof_cdof_in[worldid, colind]
            if colind < 0:
              continue
          jv_out += efc_J_in[rw, 0, sparseid] * ctx_search_in[worldid, colind]
      else:
        for i in range(wp.static(min(dofs_per_thread, nv))):
          jv_out += efc_J_in[rw, efcid, i] * ctx_search_in[worldid, i]
      ctx_jv_out[rw, efcid] = jv_out

    else:
      if wp.static(is_sparse):
        # Sparse: thread 0 handles entire row (sparse entries << nv typically)
        if dofstart == 0:
          rownnz = efc_J_rownnz_in[rw, efcid]
          rowadr = efc_J_rowadr_in[rw, efcid]
          for k in range(rownnz):
            sparseid = rowadr + k
            colind = efc_J_colind_in[rw, 0, sparseid]
            if wp.static(COMPACT):
              colind = dof_cdof_in[worldid, colind]
              if colind < 0:
                continue
            jv_out += efc_J_in[rw, 0, sparseid] * ctx_search_in[worldid, colind]
          ctx_jv_out[rw, efcid] = jv_out
      else:
        for i in range(wp.static(dofs_per_thread)):
          ii = dofstart * wp.static(dofs_per_thread) + i
          if ii < nv:
            jv_out += efc_J_in[rw, efcid, ii] * ctx_search_in[worldid, ii]
        wp.atomic_add(ctx_jv_out, rw, efcid, jv_out)

  return kernel


@event_scope
def _linesearch(m: types.Model, d: types.Data, ctx: SolverContext):
  """Linesearch for constraint solver.

  When state changes are tracked, worlds with ctx.search_unchanged reuse last
  iteration's mv/jv, so a fresh search requires clearing the flag (see the
  invalidation in _solve and the writer in _update_gradient_zero_grad_dot).

  Args:
    m: Model
    d: Data
    ctx: SolverContext
  """
  # mv and jv are pure functions of the search direction, and M and J are
  # constant within a solve, so worlds whose search was kept reuse last
  # iteration's values.
  skip = ctx.search_unchanged if _use_incremental(m) else ctx.done

  # mv = M @ search (common to both parallel and iterative)
  _mul_m_compact_aware(m, d, ctx, ctx.mv, ctx.search, skip)

  # Fuse jv computation in-kernel for small nv (iterative only, dense only)
  # Sparse mode requires pre-computed jv since in-kernel uses dense indexing
  # the sparse-compact J path reads the full model's sparse J structures;
  # dense full models keep the gathered dense cJ path
  sc = _sparse_compact(ctx)
  fuse_jv = m.nv <= 50 and not m.is_sparse and not sc

  # jv = J @ search (when not fused into iterative kernel)
  if not fuse_jv:
    dj = ctx.compact_d_full if sc else d
    if sc or m.is_sparse:
      # Sparse J has few nonzeros per row, one thread handles them all.
      dofs_per_thread = m.nv
      threads_per_efc = 1
    else:
      dofs_per_thread = 20 if m.nv > 50 else 50
      threads_per_efc = ceil(m.nv / dofs_per_thread)

    if threads_per_efc > 1:
      wp.launch(
        _linesearch_zero_jv(ctx.efc_problem is not None),
        dim=(d.nworld, d.njmax) if ctx.efc_problem is None else ctx.efc_problem.shape,
        inputs=[d.nefc, skip, ctx.efc_problem],
        outputs=[ctx.jv],
      )

    wp.launch(
      _linesearch_jv_fused_kernel(sc or m.is_sparse, m.nv, dofs_per_thread, sc, ctx.efc_problem is not None),
      dim=(d.nworld, d.njmax, threads_per_efc) if ctx.efc_problem is None else (*ctx.efc_problem.shape, threads_per_efc),
      inputs=[
        d.nefc,
        dj.efc.J_rownnz,
        dj.efc.J_rowadr,
        dj.efc.J_colind,
        dj.efc.J,
        dj.dof_cdof,
        ctx.search,
        skip,
        ctx.efc_problem,
      ],
      outputs=[ctx.jv],
    )

  _linesearch_iterative(m, d, ctx, fuse_jv)


@cache_kernel
def _solve_init_dof(warmstart: bool, sparse: bool):
  WARMSTART = warmstart
  SPARSE = sparse

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Data in:
    nefc_in: wp.array[int],
    qacc_warmstart_in: wp.array2d[float],
    qacc_smooth_in: wp.array2d[float],
    dof_island_in: wp.array2d[int],
    # In:
    h_dof2row_in: wp.array2d[int],
    # Data out:
    qacc_out: wp.array2d[float],
    qfrc_constraint_out: wp.array2d[float],
  ):
    worldid, dofid = wp.tid()

    # DOFs outside the factored H get no search direction, so they start at their solution:
    # small islands were solved by the fast path, unconstrained DOFs take qacc_smooth
    excluded = False
    if h_dof2row_in:
      excluded = h_dof2row_in[worldid, dofid] < 0
      if excluded:
        if dof_island_in[worldid, dofid] >= 0:
          if wp.static(SPARSE):
            if nefc_in[worldid] == 0:
              qfrc_constraint_out[worldid, dofid] = 0.0
          return
        # the world solve may be skipped, so DOFs outside all islands get their (zero) force here
        qfrc_constraint_out[worldid, dofid] = 0.0

    if wp.static(WARMSTART):
      if nefc_in[worldid] > 0 and not excluded:
        qacc_out[worldid, dofid] = qacc_warmstart_in[worldid, dofid]
      else:
        qacc_out[worldid, dofid] = qacc_smooth_in[worldid, dofid]
    else:
      qacc_out[worldid, dofid] = qacc_smooth_in[worldid, dofid]

    if wp.static(SPARSE):
      if nefc_in[worldid] == 0:
        qfrc_constraint_out[worldid, dofid] = 0.0

  return kernel


@wp.kernel(grid_stride=True)
def _solve_init_efc(
  # In:
  ctx_nproblem_in: wp.array[int],
  # Data out:
  solver_niter_out: wp.array[int],
  # Out:
  ctx_search_dot_out: wp.array[float],
  ctx_done_out: wp.array[bool],
):
  worldid = wp.tid()
  solver_niter_out[worldid] = 0
  # problems past nproblem are empty and done from the start
  done = False
  if ctx_nproblem_in:
    done = worldid >= ctx_nproblem_in[0]
  ctx_done_out[worldid] = done
  ctx_search_dot_out[worldid] = 0.0


@cache_kernel
def _solve_init_jaref_kernel(is_sparse: bool, nv: int, dofs_per_thread: int, compact: bool, ragged: bool = False):
  COMPACT = compact
  RAGGED = ragged

  @wp.kernel(module="unique", enable_backward=False, grid_stride=True)
  def kernel(
    # Data in:
    nefc_in: wp.array[int],
    qacc_in: wp.array2d[float],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    efc_aref_in: wp.array2d[float],
    dof_cdof_in: wp.array2d[int],
    # In:
    ctx_efc_world_in: wp.array[int],
    ctx_efc_rowoff_in: wp.array[int],
    ctx_efc_problem_in: wp.array2d[int],
    # Out:
    ctx_Jaref_out: wp.array2d[float],
  ):
    worldid, efcid, dofstart = wp.tid()

    if wp.static(RAGGED):
      # threads are the worlds' rows; continue as the row's problem
      problem = ctx_efc_problem_in[worldid, efcid]
      if problem < 0:
        return
      efcid -= ctx_efc_rowoff_in[problem]
      worldid = problem

    if efcid >= nefc_in[worldid]:
      return

    rw = worldid
    ro = int(0)
    if wp.static(RAGGED):
      rw = ctx_efc_world_in[worldid]
      ro = ctx_efc_rowoff_in[worldid]
    efcid += ro

    jaref = float(0.0)
    if wp.static(is_sparse):
      rownnz = efc_J_rownnz_in[rw, efcid]
      rowadr = efc_J_rowadr_in[rw, efcid]
      for i in range(rownnz):
        sparseid = rowadr + i
        colind = efc_J_colind_in[rw, 0, sparseid]
        if wp.static(COMPACT):
          colind = dof_cdof_in[worldid, colind]
          if colind < 0:
            continue
        jaref += efc_J_in[rw, 0, sparseid] * qacc_in[worldid, colind]
      ctx_Jaref_out[rw, efcid] = jaref - efc_aref_in[rw, efcid]
    else:
      if wp.static(dofs_per_thread >= nv):
        for i in range(wp.static(min(dofs_per_thread, nv))):
          jaref += efc_J_in[rw, efcid, i] * qacc_in[worldid, i]
        ctx_Jaref_out[rw, efcid] = jaref - efc_aref_in[rw, efcid]

      else:
        for i in range(wp.static(dofs_per_thread)):
          ii = dofstart * wp.static(dofs_per_thread) + i
          if ii < nv:
            jaref += efc_J_in[rw, efcid, ii] * qacc_in[worldid, ii]

        if dofstart == 0:
          wp.atomic_add(ctx_Jaref_out, rw, efcid, jaref - efc_aref_in[rw, efcid])
        else:
          wp.atomic_add(ctx_Jaref_out, rw, efcid, jaref)

  return kernel


@wp.kernel
def _solve_init_search_cg_tiled(
  # Model:
  nv: int,
  opt_tolerance: wp.array[float],
  stat_meaninertia: wp.array[float],
  # In:
  ctx_grad_in: wp.array2d[float],
  ctx_Mgrad_in: wp.array2d[float],
  # Out:
  ctx_search_out: wp.array2d[float],
  ctx_search_dot_out: wp.array[float],
  ctx_prev_grad_out: wp.array2d[float],
  ctx_prev_Mgrad_out: wp.array2d[float],
  ctx_done_out: wp.array[bool],
  nsolving_out: wp.array[int],
):
  worldid, tid = wp.tid()

  local_search_dot = float(0.0)
  local_grad_mgrad = float(0.0)
  BLOCK_DIM = wp.block_dim()

  for dofid in range(tid, nv, BLOCK_DIM):
    grad = ctx_grad_in[worldid, dofid]
    mgrad = ctx_Mgrad_in[worldid, dofid]
    search = -1.0 * mgrad
    ctx_search_out[worldid, dofid] = search
    local_search_dot += search * search
    local_grad_mgrad += grad * mgrad

    ctx_prev_grad_out[worldid, dofid] = grad
    ctx_prev_Mgrad_out[worldid, dofid] = mgrad

  search_dot_tile = wp.tile(local_search_dot, preserve_type=True)
  search_dot_sum = wp.tile_reduce(wp.add, search_dot_tile)
  grad_mgrad_tile = wp.tile(local_grad_mgrad, preserve_type=True)
  grad_mgrad_sum = wp.tile_reduce(wp.add, grad_mgrad_tile)

  if tid == 0:
    ctx_search_dot_out[worldid] = search_dot_sum[0]
    if not ctx_done_out[worldid]:
      meaninertia = stat_meaninertia[worldid % stat_meaninertia.shape[0]]
      tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]
      scale = 1.0 / (meaninertia * float(wp.max(1, nv)))
      if wp.max(0.0, 0.5 * scale * grad_mgrad_sum[0]) < tolerance:
        ctx_done_out[worldid] = True
        wp.atomic_sub(nsolving_out, 0, 1)


@cache_kernel
def _update_constraint_efc(track_changes: bool, ragged: bool = False):
  TRACK_CHANGES = track_changes
  RAGGED = ragged

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Model:
    opt_impratio_invsqrt: wp.array[float],
    # Data in:
    ne_in: wp.array[int],
    nf_in: wp.array[int],
    nefc_in: wp.array[int],
    contact_friction_in: wp.array[types.vec5],
    contact_dim_in: wp.array[int],
    contact_efc_address_in: wp.array2d[int],
    efc_type_in: wp.array2d[int],
    efc_id_in: wp.array2d[int],
    efc_D_in: wp.array2d[float],
    efc_frictionloss_in: wp.array2d[float],
    nacon_in: wp.array[int],
    # In:
    ctx_Jaref_in: wp.array2d[float],
    ctx_ls_exhausted_in: wp.array[bool],
    ctx_done_in: wp.array[bool],
    ctx_efc_world_in: wp.array[int],
    ctx_efc_rowoff_in: wp.array[int],
    ctx_efc_problem_in: wp.array2d[int],
    # Data out:
    efc_force_out: wp.array2d[float],
    efc_state_out: wp.array2d[int],
    # Out:
    quad_changed_ids_out: wp.array2d[int],
    quad_changed_count_out: wp.array[int],
    state_changed_count_out: wp.array[int],
  ):
    worldid, efcid = wp.tid()

    if wp.static(RAGGED):
      # threads are the worlds' rows; continue as the row's problem
      problem = ctx_efc_problem_in[worldid, efcid]
      if problem < 0:
        return
      efcid -= ctx_efc_rowoff_in[problem]
      worldid = problem

    if ctx_done_in[worldid]:
      return

    # The linesearch flags worlds whose accepted step was rounding noise (stale
    # ray exhausted); count it as a state change so the fast path rebuilds them.
    if wp.static(TRACK_CHANGES):
      if efcid == 0 and ctx_ls_exhausted_in[worldid]:
        wp.atomic_add(state_changed_count_out, worldid, 1)

    if efcid >= nefc_in[worldid]:
      return

    rw = worldid
    ro = int(0)
    if wp.static(RAGGED):
      rw = ctx_efc_world_in[worldid]
      ro = ctx_efc_rowoff_in[worldid]
    efcid += ro

    # Read old state before overwriting
    if wp.static(TRACK_CHANGES):
      old_state = efc_state_out[rw, efcid]

    efc_D = efc_D_in[rw, efcid]
    Jaref = ctx_Jaref_in[rw, efcid]

    ne = ro + ne_in[worldid]
    nf = nf_in[worldid]

    is_equality = efcid < ne
    is_friction = (not is_equality) and (efcid < ne + nf)
    is_elliptic = efc_type_in[rw, efcid] == types.ConstraintType.CONTACT_ELLIPTIC

    frictionloss = efc_frictionloss_in[rw, efcid] if is_friction else 0.0

    efcid0 = -1
    jaref0 = float(0.0)
    D0 = float(0.0)
    mu = float(0.0)
    ufrictionj = float(0.0)
    TT = float(0.0)

    if is_elliptic:
      conid = efc_id_in[rw, efcid]
      if conid >= nacon_in[0]:
        return
      efcid0 = contact_efc_address_in[conid, 0]
      if efcid0 < 0:
        return

      dim = contact_dim_in[conid]
      friction = contact_friction_in[conid]
      mu = friction[0] * opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]
      jaref0 = ctx_Jaref_in[rw, efcid0]
      D0 = efc_D_in[rw, efcid0]

      for j in range(1, dim):
        efcidj = contact_efc_address_in[conid, j]
        if efcidj < 0:
          return
        frictionj = friction[j - 1]
        uj = ctx_Jaref_in[rw, efcidj] * frictionj
        TT += uj * uj
        if efcid == efcidj:
          ufrictionj = uj * frictionj

    res = _eval_constraint(
      is_equality,
      is_friction,
      is_elliptic,
      Jaref,
      efc_D,
      frictionloss,
      efcid,
      efcid0,
      jaref0,
      D0,
      mu,
      ufrictionj,
      TT,
    )

    new_state = int(res[1])
    efc_force_out[rw, efcid] = res[0]
    efc_state_out[rw, efcid] = new_state

    if wp.static(TRACK_CHANGES):
      old_quad = old_state == types.ConstraintState.QUADRATIC.value
      new_quad = new_state == types.ConstraintState.QUADRATIC.value
      if old_quad != new_quad:
        idx = wp.atomic_add(quad_changed_count_out, worldid, 1)
        quad_changed_ids_out[rw, ro + idx] = efcid
      # LINEARNEG <-> LINEARPOS friction transitions change the force without
      # changing the quadratic flag (or H); the fast path must still see them.
      if old_state != new_state:
        wp.atomic_add(state_changed_count_out, worldid, 1)

  return kernel


@wp.kernel
def _zero_qfrc_constraint_sparse(
  # In:
  state_changed_count_in: wp.array[int],
  ctx_done_in: wp.array[bool],
  # Data out:
  qfrc_constraint_out: wp.array2d[float],
):
  # Only zero worlds the rebuild will repopulate; done worlds keep their value.
  worldid, dofid = wp.tid()

  if ctx_done_in[worldid]:
    return

  if state_changed_count_in[worldid] == 0:
    return

  qfrc_constraint_out[worldid, dofid] = 0.0


@cache_kernel
def _update_constraint_init_qfrc_constraint_sparse(compact: bool):
  COMPACT = compact

  @wp.kernel(module="unique", enable_backward=False, grid_stride=True)
  def kernel(
    # Data in:
    nefc_in: wp.array[int],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    efc_force_in: wp.array2d[float],
    dof_cdof_in: wp.array2d[int],
    # In:
    state_changed_count_in: wp.array[int],
    ctx_done_in: wp.array[bool],
    # Data out:
    qfrc_constraint_out: wp.array2d[float],
  ):
    worldid, efcid = wp.tid()

    if ctx_done_in[worldid]:
      return

    if state_changed_count_in[worldid] == 0:
      return

    if efcid >= nefc_in[worldid]:
      return

    force = efc_force_in[worldid, efcid]
    if force == 0.0:
      return

    rownnz = efc_J_rownnz_in[worldid, efcid]
    rowadr = efc_J_rowadr_in[worldid, efcid]
    for i in range(rownnz):
      sparseid = rowadr + i
      colind = efc_J_colind_in[worldid, 0, sparseid]
      if wp.static(COMPACT):
        colind = dof_cdof_in[worldid, colind]
        if colind < 0:
          continue
      efc_J = efc_J_in[worldid, 0, sparseid]
      wp.atomic_add(qfrc_constraint_out[worldid], colind, efc_J * force)

  return kernel


@wp.kernel
def _qfrc_constraint_from_grad(
  # Data in:
  qfrc_smooth_in: wp.array2d[float],
  efc_Ma_in: wp.array2d[float],
  # In:
  ctx_grad_in: wp.array2d[float],
  ctx_grad_scale_in: wp.array[float],
  # Data out:
  qfrc_constraint_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()

  grad = ctx_grad_scale_in[worldid] * ctx_grad_in[worldid, dofid]
  qfrc_constraint_out[worldid, dofid] = efc_Ma_in[worldid, dofid] - qfrc_smooth_in[worldid, dofid] - grad


@cache_kernel
def _update_constraint_init_qfrc_constraint_dense(stable_fast: bool, ragged: bool = False):
  STABLE_FAST = stable_fast
  RAGGED = ragged

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Data in:
    nefc_in: wp.array[int],
    efc_J_in: wp.array3d[float],
    efc_force_in: wp.array2d[float],
    njmax_in: int,
    # In:
    state_changed_count_in: wp.array[int],
    ctx_done_in: wp.array[bool],
    ctx_efc_world_in: wp.array[int],
    ctx_efc_rowoff_in: wp.array[int],
    # Data out:
    qfrc_constraint_out: wp.array2d[float],
  ):
    worldid, dofid = wp.tid()

    if ctx_done_in[worldid]:
      return

    # Fast path: stale qfrc_constraint is never read; recovered after the solve.
    if wp.static(STABLE_FAST):
      if state_changed_count_in[worldid] == 0:
        return

    rw = worldid
    ro = int(0)
    if wp.static(RAGGED):
      rw = ctx_efc_world_in[worldid]
      ro = ctx_efc_rowoff_in[worldid]
    sum_qfrc = float(0.0)
    for efcid in range(min(njmax_in, nefc_in[worldid])):
      efc_J = efc_J_in[rw, ro + efcid, dofid]
      force = efc_force_in[rw, ro + efcid]
      sum_qfrc += efc_J * force

    qfrc_constraint_out[worldid, dofid] = sum_qfrc

  return kernel


@cache_kernel
def _update_gradient_h_incremental(ragged: bool = False):
  RAGGED = ragged

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Data in:
    efc_J_in: wp.array3d[float],
    efc_D_in: wp.array2d[float],
    efc_state_in: wp.array2d[int],
    # In:
    quad_changed_ids_in: wp.array2d[int],
    quad_changed_count_in: wp.array[int],
    ctx_efc_world_in: wp.array[int],
    ctx_efc_rowoff_in: wp.array[int],
    # Out:
    ctx_h_out: wp.array3d[float],
  ):
    """Incrementally update upper triangle of H for changed constraints.

    Each thread handles one unique (i, j) element and writes it to the upper triangle.
    For each changed constraint, adds or subtracts D * J[i] * J[j].
    """
    worldid, elementid = wp.tid()

    n_changes = quad_changed_count_in[worldid]
    if n_changes == 0:
      return

    # Upper-triangle enumeration: elementid -> (row, col) where row <= col.
    col = (int(wp.sqrt(float(1 + 8 * elementid))) - 1) // 2
    row = elementid - (col * (col + 1)) // 2

    rw = worldid
    ro = int(0)
    if wp.static(RAGGED):
      rw = ctx_efc_world_in[worldid]
      ro = ctx_efc_rowoff_in[worldid]
    delta = float(0.0)
    for change_idx in range(n_changes):
      efcid = quad_changed_ids_in[rw, ro + change_idx]
      Jrow = efc_J_in[rw, efcid, row]
      if Jrow == 0.0:
        continue
      Jcol = efc_J_in[rw, efcid, col]
      if Jcol == 0.0:
        continue

      D = efc_D_in[rw, efcid]
      if efc_state_in[rw, efcid] == types.ConstraintState.QUADRATIC.value:
        delta += D * Jrow * Jcol
      else:
        delta -= D * Jrow * Jcol

    if delta != 0.0:
      ctx_h_out[worldid, row, col] += delta

  return kernel


@cache_kernel
def _update_gradient_h_incremental_rows(nv: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Data in:
    efc_J_in: wp.array3d[float],
    efc_D_in: wp.array2d[float],
    efc_state_in: wp.array2d[int],
    # In:
    quad_changed_ids_in: wp.array2d[int],
    quad_changed_count_in: wp.array[int],
    ctx_efc_rowoff_in: wp.array[int],
    ctx_efc_problem_in: wp.array2d[int],
    # Out:
    ctx_h_out: wp.array3d[float],
  ):
    """Ragged incremental H update, one thread per changed row.

    A problem's changed-row list lives in its rows' range, so the thread of row g handles list entry
    g - rowoff.
    """
    worldid, g = wp.tid()
    problem = ctx_efc_problem_in[worldid, g]
    if problem < 0:
      return
    if g - ctx_efc_rowoff_in[problem] >= quad_changed_count_in[problem]:
      return
    efcid = quad_changed_ids_in[worldid, g]
    D = efc_D_in[worldid, efcid]
    if efc_state_in[worldid, efcid] != types.ConstraintState.QUADRATIC.value:
      D = -D
    for i in range(wp.static(nv)):
      Ji = efc_J_in[worldid, efcid, i]
      if Ji != 0.0:
        DJi = D * Ji
        for j in range(i, wp.static(nv)):
          Jj = efc_J_in[worldid, efcid, j]
          if Jj != 0.0:
            wp.atomic_add(ctx_h_out[problem, i], j, DJi * Jj)

  return kernel


@wp.func
def _island_pool_add(
  # In:
  worldid: int,
  rowa: int,
  la: int,
  rowb: int,
  lb: int,
  value: float,
  TILE: int,
  # Out:
  ctx_h_out: wp.array3d[float],
):
  """Add to entry (a, b) of an island block in the tile pool, mirrored into its upper triangle.

  rowa is the pool row of island-local row la within its first tile column; tiles of a block row
  are TILE rows apart.
  """
  if la <= lb:
    wp.atomic_add(ctx_h_out[worldid, rowa + TILE * (lb // TILE)], lb % TILE, value)
  else:
    wp.atomic_add(ctx_h_out[worldid, rowb + TILE * (la // TILE)], la % TILE, value)


@cache_kernel
def _update_gradient_h_incremental_sparse(compact: bool, pool: bool = False):
  COMPACT = compact
  POOL = pool

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Data in:
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    efc_D_in: wp.array2d[float],
    efc_state_in: wp.array2d[int],
    dof_cdof_in: wp.array2d[int],
    # In:
    ctx_h_dof_local_in: wp.array2d[int],
    quad_changed_ids_in: wp.array2d[int],
    quad_changed_count_in: wp.array[int],
    slots_per_world: int,
    # Out:
    ctx_h_out: wp.array3d[float],
  ):
    """Incrementally update upper triangle of H for changed constraints (sparse J).

    One warp per changed constraint row: the lanes split the row's upper-triangular
    entries (same sqrt triangular-number decode as _JTDACJ_sparse), replacing the
    serial nnz^2 loop that dominated this kernel.
    """
    worldid, slot, lane = wp.tid()

    n_changes = quad_changed_count_in[worldid]
    for change_idx in range(slot, n_changes, slots_per_world):
      efcid = quad_changed_ids_in[worldid, change_idx]
      D = efc_D_in[worldid, efcid]
      sign = float(0.0)
      if efc_state_in[worldid, efcid] == types.ConstraintState.QUADRATIC.value:
        sign = D
      else:
        sign = -D

      rownnz = efc_J_rownnz_in[worldid, efcid]
      rowadr = efc_J_rowadr_in[worldid, efcid]
      n_entries = rownnz * (rownnz + 1) // 2

      for entry in range(lane, n_entries, wp.static(_JTDAJ_THREADS_PER_GROUP)):
        ii = int((wp.sqrt(float(8 * entry + 1)) - 1.0) * 0.5)
        jj = entry - ii * (ii + 1) // 2
        Ji = efc_J_in[worldid, 0, rowadr + ii]
        Jj = efc_J_in[worldid, 0, rowadr + jj]
        h = sign * Ji * Jj
        if h != 0.0:
          colindi = efc_J_colind_in[worldid, 0, rowadr + ii]
          colindj = efc_J_colind_in[worldid, 0, rowadr + jj]
          if wp.static(POOL):
            rowi = dof_cdof_in[worldid, colindi]
            rowj = dof_cdof_in[worldid, colindj]
            if rowi < 0 or rowj < 0:
              continue
            li = ctx_h_dof_local_in[worldid, colindi]
            lj = ctx_h_dof_local_in[worldid, colindj]
            _island_pool_add(worldid, rowi, li, rowj, lj, h, wp.static(types.TILE_SIZE_JTDAJ_DENSE), ctx_h_out)
            continue
          if wp.static(COMPACT):
            colindi = dof_cdof_in[worldid, colindi]
            colindj = dof_cdof_in[worldid, colindj]
            if colindi < 0 or colindj < 0:
              continue
          wp.atomic_add(ctx_h_out[worldid, wp.min(colindi, colindj)], wp.max(colindi, colindj), h)

  return kernel


def _update_constraint(
  m: types.Model,
  d: types.Data,
  ctx: SolverContext | InverseContext,
  track_changes: bool = False,
  stable_fast: bool = False,
):
  """Update constraint arrays after each solve iteration."""
  efc_inputs = [
    m.opt.impratio_invsqrt,
    d.ne,
    d.nf,
    d.nefc,
    d.contact.friction,
    d.contact.dim,
    d.contact.efc_address,
    d.efc.type,
    d.efc.id,
    d.efc.D,
    d.efc.frictionloss,
    d.nacon,
    ctx.Jaref,
    ctx.ls_exhausted,
    ctx.done,
    ctx.efc_world,
    ctx.efc_rowoff,
    ctx.efc_problem,
  ]

  wp.launch(
    _update_constraint_efc(track_changes, ctx.efc_world is not None),
    # ragged: one thread per row of the worlds the problems' rows are in
    dim=(d.nworld, d.njmax) if ctx.efc_problem is None else ctx.efc_problem.shape,
    inputs=efc_inputs,
    outputs=[d.efc.force, d.efc.state, ctx.quad_changed_ids, ctx.quad_changed_count, ctx.state_changed_count],
  )

  # qfrc_constraint = efc_J.T @ efc_force. Fast-path worlds with no state flips
  # skip the rebuild; the public value is recovered after the solve.
  changed = ctx.state_changed_count if stable_fast else d.nefc
  sc = _sparse_compact(ctx)
  if m.is_sparse or sc:
    dj = ctx.compact_d_full if sc else d
    wp.launch(
      _zero_qfrc_constraint_sparse,
      dim=(d.nworld, m.nv),
      inputs=[changed, ctx.done],
      outputs=[d.qfrc_constraint],
    )
    wp.launch(
      _update_constraint_init_qfrc_constraint_sparse(sc),
      dim=(d.nworld, d.njmax),
      inputs=[d.nefc, dj.efc.J_rownnz, dj.efc.J_rowadr, dj.efc.J_colind, dj.efc.J, d.efc.force, dj.dof_cdof, changed, ctx.done],
      outputs=[d.qfrc_constraint],
    )
  else:
    wp.launch(
      _update_constraint_init_qfrc_constraint_dense(stable_fast, ctx.efc_world is not None),
      dim=(d.nworld, m.nv),
      inputs=[d.nefc, d.efc.J, d.efc.force, d.njmax, changed, ctx.done, ctx.efc_world, ctx.efc_rowoff],
      outputs=[d.qfrc_constraint],
    )
  if m.opt.integrator == types.IntegratorType.DISCRETE and (m.opt.enableflags & types.EnableBit.SLEEP):
    wp.launch(
      derivative._zero_sleeping_dofs,
      dim=(d.nworld, m.nv),
      inputs=[m.dof_treeid, d.tree_awake],
      outputs=[d.qfrc_constraint],
    )


@cache_kernel
def _update_gradient_zero_grad_dot(stable_fast: bool):
  STABLE_FAST = stable_fast

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # In:
    state_changed_count_in: wp.array[int],
    ctx_alpha_in: wp.array[float],
    ctx_done_in: wp.array[bool],
    # Out:
    ctx_grad_dot_out: wp.array[float],
    ctx_newton_decrement_out: wp.array[float],
    ctx_grad_scale_out: wp.array[float],
    ctx_search_unchanged_out: wp.array[bool],
  ):
    worldid = wp.tid()

    if wp.static(STABLE_FAST):
      ctx_search_unchanged_out[worldid] = ctx_done_in[worldid] or state_changed_count_in[worldid] == 0
    else:
      ctx_search_unchanged_out[worldid] = False

    if ctx_done_in[worldid]:
      return

    # Fast path: grad stays stale at its last rebuilt value g. The true
    # gradient is grad_scale * g, and a linesearch step t along the (equally
    # stale) search direction changes it to (grad_scale - t) * g.
    if wp.static(STABLE_FAST):
      if state_changed_count_in[worldid] == 0:
        sigma = ctx_grad_scale_out[worldid]
        new_sigma = sigma - ctx_alpha_in[worldid]
        ratio = float(0.0)
        if sigma != 0.0:
          ratio = new_sigma / sigma
        ratio_sq = ratio * ratio
        ctx_grad_dot_out[worldid] *= ratio_sq
        ctx_newton_decrement_out[worldid] *= ratio_sq
        ctx_grad_scale_out[worldid] = new_sigma
        return

    ctx_grad_dot_out[worldid] = 0.0
    ctx_newton_decrement_out[worldid] = 0.0
    ctx_grad_scale_out[worldid] = 1.0

  return kernel


@cache_kernel
def _update_gradient_grad(stable_fast: bool):
  STABLE_FAST = stable_fast

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Data in:
    qfrc_smooth_in: wp.array2d[float],
    qfrc_constraint_in: wp.array2d[float],
    efc_Ma_in: wp.array2d[float],
    # In:
    state_changed_count_in: wp.array[int],
    ctx_done_in: wp.array[bool],
    # Out:
    ctx_grad_out: wp.array2d[float],
    ctx_grad_dot_out: wp.array[float],
  ):
    worldid, dofid = wp.tid()

    if ctx_done_in[worldid]:
      return

    # Fast path: grad stays stale (see _update_gradient_zero_grad_dot).
    if wp.static(STABLE_FAST):
      if state_changed_count_in[worldid] == 0:
        return

    grad = efc_Ma_in[worldid, dofid] - qfrc_smooth_in[worldid, dofid] - qfrc_constraint_in[worldid, dofid]
    ctx_grad_out[worldid, dofid] = grad
    wp.atomic_add(ctx_grad_dot_out, worldid, grad * grad)

  return kernel


@wp.kernel
def _update_gradient_grad_tiled(
  # Model:
  nv: int,
  # Data in:
  qfrc_smooth_in: wp.array2d[float],
  qfrc_constraint_in: wp.array2d[float],
  efc_Ma_in: wp.array2d[float],
  # In:
  ctx_done_in: wp.array[bool],
  # Out:
  ctx_grad_out: wp.array2d[float],
  ctx_grad_dot_out: wp.array[float],
):
  worldid, tid = wp.tid()

  if ctx_done_in[worldid]:
    return

  local_grad_dot = float(0.0)
  BLOCK_DIM = wp.block_dim()

  for dofid in range(tid, nv, BLOCK_DIM):
    grad = efc_Ma_in[worldid, dofid] - qfrc_smooth_in[worldid, dofid] - qfrc_constraint_in[worldid, dofid]
    ctx_grad_out[worldid, dofid] = grad
    local_grad_dot += grad * grad

  grad_dot_tile = wp.tile(local_grad_dot, preserve_type=True)
  grad_dot_sum = wp.tile_reduce(wp.add, grad_dot_tile)

  if tid == 0:
    ctx_grad_dot_out[worldid] = grad_dot_sum[0]


@cache_kernel
def _update_gradient_init_h_sparse(compact: bool):
  COMPACT = compact

  @wp.kernel(module="unique", enable_backward=False, grid_stride=True)
  def kernel(
    # Model:
    nv: int,
    M_elemid: wp.array2d[int],
    # Data in:
    M_in: wp.array2d[float],
    cdof_dof_in: wp.array2d[int],
    # In:
    ctx_done_in: wp.array[bool],
    # Out:
    ctx_h_out: wp.array3d[float],
  ):
    worldid, i, j = wp.tid()

    if ctx_done_in[worldid]:
      return

    # only write the upper triangle; Cholesky reads the upper triangle only
    if j < i:
      return

    if wp.static(COMPACT):
      dof_i = cdof_dof_in[worldid, i]
      dof_j = cdof_dof_in[worldid, j]
      if dof_i < 0 or dof_j < 0:
        # per-world padded block: identity keeps the factorization well conditioned
        ctx_h_out[worldid, i, j] = wp.where(i == j, 1.0, 0.0)
        return
    else:
      dof_i = i
      dof_j = j
      if i >= nv or j >= nv:
        ctx_h_out[worldid, i, j] = 0.0
        return

    # sparse M is stored in the lower triangle, so look up (larger, smaller)
    elemid = M_elemid[wp.max(dof_i, dof_j), wp.min(dof_i, dof_j)]
    if elemid >= 0:
      ctx_h_out[worldid, i, j] = M_in[worldid, elemid]
    else:
      ctx_h_out[worldid, i, j] = 0.0

  return kernel


@wp.func
def _init_h_entry(
  # Model:
  M_elemid: wp.array2d[int],
  # Data in:
  M_in: wp.array2d[float],
  cdof_dof_in: wp.array2d[int],
  # In:
  worldid: int,
  i: int,
  j: int,
) -> float:
  """M entry of the mapped rows i, j; identity on padded rows."""
  dofi = cdof_dof_in[worldid, i]
  dofj = cdof_dof_in[worldid, j]
  if dofi < 0 or dofj < 0:
    return wp.where(i == j, 1.0, 0.0)
  elemid = M_elemid[wp.max(dofi, dofj), wp.min(dofi, dofj)]
  if elemid >= 0:
    return M_in[worldid, elemid]
  return 0.0


@wp.kernel
def _update_gradient_init_h_envelope(
  # Model:
  M_elemid: wp.array2d[int],
  # Data in:
  M_in: wp.array2d[float],
  cdof_dof_in: wp.array2d[int],
  # In:
  nrow_in: wp.array[int],
  ctx_cholesky_top_in: wp.array2d[int],
  ctx_done_in: wp.array[bool],
  # Out:
  ctx_h_out: wp.array3d[float],
):
  """Initialize the upper triangle of mapped h with M, only where the blocked Cholesky reads it.

  That is the active rows and, within them, the tile envelope plus the last tile column (it carries
  the right-hand side). The envelope of a row is contiguous since top is nondecreasing.
  """
  worldid, i = wp.tid()
  if ctx_done_in[worldid]:
    return

  TILE = wp.static(types.TILE_SIZE_JTDAJ_DENSE)
  size = _blocked_size(nrow_in, worldid, TILE, ctx_h_out.shape[1])
  if i >= size:
    return

  topid = worldid % ctx_cholesky_top_in.shape[0]
  jstop = (i // TILE + 1) * TILE
  while jstop < size and ctx_cholesky_top_in[topid, jstop // TILE] <= i:
    jstop += TILE
  for j in range(i, jstop):
    ctx_h_out[worldid, i, j] = _init_h_entry(M_elemid, M_in, cdof_dof_in, worldid, i, j)
  for j in range(wp.max(jstop, size - TILE), size):
    ctx_h_out[worldid, i, j] = _init_h_entry(M_elemid, M_in, cdof_dof_in, worldid, i, j)


@wp.kernel(module="unique", enable_backward=False)
def _add_tendon_metric_dense(
  # Model:
  ten_J_rownnz: wp.array[int],
  ten_J_rowadr: wp.array[int],
  ten_J_colind: wp.array[int],
  # Data in:
  ten_J_in: wp.array2d[float],
  efm_ts_in: wp.array2d[float],
  # In:
  ctx_done_in: wp.array[bool],
  # Out:
  h_out: wp.array3d[float],
):
  worldid, t = wp.tid()
  if ctx_done_in[worldid]:
    return
  ts = efm_ts_in[worldid, t]
  if ts != 0.0:
    radr = ten_J_rowadr[t]
    rnnz = ten_J_rownnz[t]
    for i in range(rnnz):
      c1 = ten_J_colind[radr + i]
      v1 = ten_J_in[worldid, radr + i]
      ts_v1 = ts * v1
      for j in range(rnnz):
        c2 = ten_J_colind[radr + j]
        if c2 > c1:
          v2 = ten_J_in[worldid, radr + j]
          wp.atomic_add(h_out, worldid, c1, c2, ts_v1 * v2)


@wp.kernel(module="unique", enable_backward=False)
def _add_actuator_metric_dense(
  # Data in:
  moment_rownnz_in: wp.array2d[int],
  moment_rowadr_in: wp.array2d[int],
  moment_colind_in: wp.array2d[int],
  actuator_moment_in: wp.array2d[float],
  efm_as_in: wp.array2d[float],
  # In:
  ctx_done_in: wp.array[bool],
  # Out:
  h_out: wp.array3d[float],
):
  worldid, a = wp.tid()
  if ctx_done_in[worldid]:
    return
  as_val = efm_as_in[worldid, a]
  if as_val != 0.0:
    radr = moment_rowadr_in[worldid, a]
    rnnz = moment_rownnz_in[worldid, a]
    for i in range(rnnz):
      c1 = moment_colind_in[worldid, radr + i]
      v1 = actuator_moment_in[worldid, radr + i]
      as_v1 = as_val * v1
      for j in range(rnnz):
        c2 = moment_colind_in[worldid, radr + j]
        if c2 > c1:
          v2 = actuator_moment_in[worldid, radr + j]
          wp.atomic_add(h_out, worldid, c1, c2, as_v1 * v2)


@wp.kernel(module="unique", enable_backward=False)
def _add_efmK_metric_dense(
  # Model:
  efm_K_rownnz: wp.array[int],
  efm_K_rowadr: wp.array[int],
  efm_K_colind: wp.array[int],
  # Data in:
  efm_K_val_in: wp.array2d[float],
  # In:
  ctx_done_in: wp.array[bool],
  # Out:
  h_out: wp.array3d[float],
):
  worldid, r = wp.tid()
  if ctx_done_in[worldid]:
    return
  radr = efm_K_rowadr[r]
  rnnz = efm_K_rownnz[r]
  for k in range(rnnz):
    c = efm_K_colind[radr + k]
    if c >= r:
      wp.atomic_add(h_out, worldid, r, c, efm_K_val_in[worldid, radr + k])


@wp.kernel(module="unique", enable_backward=False)
def _add_flexcon_metric_dense(
  # Data in:
  contact_worldid_in: wp.array[int],
  nacon_in: wp.array[int],
  # In:
  efm_con_dof_in: wp.array2d[int],
  efm_con_val_in: wp.array2d[float],
  efm_con_scale_in: wp.array[float],
  efm_con_nnz_in: wp.array[int],
  ctx_done_in: wp.array[bool],
  # Out:
  h_out: wp.array3d[float],
):
  cid = wp.tid()
  if cid >= nacon_in[0]:
    return
  worldid = contact_worldid_in[cid]
  if ctx_done_in[worldid]:
    return
  scale = efm_con_scale_in[cid]
  nnz = efm_con_nnz_in[cid]
  if scale != 0.0 and nnz > 0:
    for i in range(nnz):
      c1 = efm_con_dof_in[cid, i]
      s_v1 = scale * efm_con_val_in[cid, i]
      for j in range(nnz):
        c2 = efm_con_dof_in[cid, j]
        if c2 >= c1:
          wp.atomic_add(h_out, worldid, c1, c2, s_v1 * efm_con_val_in[cid, j])


@wp.func
def _state_check(D: float, state: int) -> float:
  if state == types.ConstraintState.QUADRATIC.value:
    return D
  else:
    return 0.0


@wp.func
def _active_check(tid: int, threshold: int) -> float:
  if tid >= threshold:
    return 0.0
  else:
    return 1.0


@cache_kernel
def _update_gradient_JTDAJ_ragged(nv_pad: int, tile_size: int, njmax: int):
  """_update_gradient_JTDAJ_dense_tiled over the problems of the island solver.

  A problem's rows are rows of a world (efc_world, efc_rowoff). Its DOFs are whole trees in order,
  and M is block-diagonal over trees with each tree's CSR entries contiguous, so M is seeded per
  tree from the world's CSR as the world kernel does (local index = tree's column + DOF offset).
  Also writes each problem's meaninertia (first read by the line search).
  """
  if njmax < tile_size:
    tile_size = njmax

  TILE_SIZE_K = tile_size

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False, module_options={"enable_mathdx_gemm": False})
  def kernel(
    # Model:
    dof_treeid: wp.array[int],
    tree_dofadr: wp.array[int],
    tree_dofnum: wp.array[int],
    M_rownnz: wp.array[int],
    M_rowadr: wp.array[int],
    M_colind: wp.array[int],
    M_hinit_i: wp.array[int],
    # Data in:
    nefc_in: wp.array[int],
    M_in: wp.array2d[float],
    efc_J_in: wp.array3d[float],
    efc_D_in: wp.array2d[float],
    efc_state_in: wp.array2d[int],
    # In:
    ctx_done_in: wp.array[bool],
    ctx_efc_world_in: wp.array[int],
    ctx_efc_rowoff_in: wp.array[int],
    ctx_problem_dof_in: wp.array2d[int],
    # Out:
    ctx_h_out: wp.array3d[float],
    problem_meaninertia_out: wp.array[float],
  ):
    worldid, rank = wp.tid()

    if ctx_done_in[worldid]:
      return

    nefc = nefc_in[worldid]
    rw = ctx_efc_world_in[worldid]
    ro = ctx_efc_rowoff_in[worldid]

    # M's upper triangle, tree by tree; identity on padding (uniform trip counts: the scatter is
    # collective)
    m_tile = wp.tile_zeros(shape=(wp.static(nv_pad * nv_pad),), dtype=float, storage="shared")
    lanes = wp.block_dim()
    diag = float(0.0)
    c = int(0)
    while c < wp.static(nv_pad):
      dof = ctx_problem_dof_in[worldid, c]
      if dof < 0:
        break
      tree = dof_treeid[dof]
      dofadr = tree_dofadr[tree]
      last = dofadr + tree_dofnum[tree] - 1
      beg = M_rowadr[dofadr]
      nent = M_rowadr[last] + M_rownnz[last] - beg
      base = c - dofadr
      for it in range((nent + lanes - 1) // lanes):
        e = it * lanes + rank
        enable = e < nent
        ec = beg + wp.where(enable, e, 0)
        pos = (M_colind[ec] + base) * wp.static(nv_pad) + M_hinit_i[ec] + base
        value = wp.where(enable, M_in[rw, ec], 0.0)
        wp.tile_scatter_add(m_tile, pos, value, enable)
        if M_colind[ec] == M_hinit_i[ec]:
          diag += value
      c += tree_dofnum[tree]
    for it in range((wp.static(nv_pad) + lanes - 1) // lanes):
      i = it * lanes + rank
      enable = i >= c and i < wp.static(nv_pad)
      wp.tile_scatter_add(m_tile, wp.where(enable, i * wp.static(nv_pad) + i, 0), 1.0, enable)
    sum_val = wp.tile_reshape(m_tile, (nv_pad, nv_pad))
    # MuJoCo scales an island's convergence tests by its inertia trace: meaninertia * nv = trace
    trace = wp.tile_reduce(wp.add, wp.tile(diag))
    if rank == 0:
      problem_meaninertia_out[worldid] = trace[0] / float(nv_pad)

    # views ending at the problem's last row: the loads zero-fill rows past it (other problems'
    # or uninitialized rows), so neither J nor D needs masking
    J = wp.array(ptr=efc_J_in[rw].ptr, shape=(ro + nefc, efc_J_in.shape[2]), dtype=float)
    D = wp.array(ptr=efc_D_in[rw].ptr, shape=(ro + nefc,), dtype=float)
    state = wp.array(ptr=efc_state_in[rw].ptr, shape=(ro + nefc,), dtype=int)
    for k in range(0, njmax, TILE_SIZE_K):
      if k >= nefc:
        break
      J_kj = wp.tile_load(J, shape=(TILE_SIZE_K, nv_pad), offset=(ro + k, 0), bounds_check=True)
      D_k = wp.tile_load(D, shape=TILE_SIZE_K, offset=ro + k, bounds_check=True)
      state_k = wp.tile_load(state, shape=TILE_SIZE_K, offset=ro + k, bounds_check=True)
      D_k = wp.tile_map(_state_check, D_k, state_k)
      J_ki = wp.tile_map(wp.mul, wp.tile_transpose(J_kj), wp.tile_broadcast(D_k, shape=(nv_pad, TILE_SIZE_K)))
      sum_val += wp.tile_matmul(J_ki, J_kj)

    wp.tile_store(ctx_h_out[worldid], sum_val, bounds_check=False)

  return kernel


@cache_kernel
def _update_gradient_JTDAJ_dense_tiled_compact(nv_pad: int, tile_size: int, njmax: int):
  """Compact-path variant of _update_gradient_JTDAJ_dense_tiled.

  Takes M_in as a dense 3D array (nworld, nv_pad, nv_pad) -- the compacted active-DOF
  inertia block cM -- instead of a 2D CSR array.  Cholesky reads fill_mode="upper";
  cM is full-symmetric so the tile_load covers both triangles and the upper triangle
  of the result is correct.
  """
  if njmax < tile_size:
    tile_size = njmax

  TILE_SIZE_K = tile_size

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False, module_options={"enable_mathdx_gemm": False})
  def kernel(
    # Data in:
    nefc_in: wp.array[int],
    M_in: wp.array3d[float],  # kernel_analyzer: ignore; compact dense inertia (nworld, nv_pad, nv_pad)
    efc_J_in: wp.array3d[float],
    efc_D_in: wp.array2d[float],
    efc_state_in: wp.array2d[int],
    # In:
    ctx_done_in: wp.array[bool],
    # Out:
    ctx_h_out: wp.array3d[float],
  ):
    worldid, rank = wp.tid()

    if ctx_done_in[worldid]:
      return

    nefc = nefc_in[worldid]

    # Load full symmetric compact inertia tile (Cholesky only reads upper triangle).
    sum_val = wp.tile_load(M_in[worldid], shape=(wp.static(nv_pad), wp.static(nv_pad)), bounds_check=True)

    for k in range(0, njmax, TILE_SIZE_K):
      if k >= nefc:
        break

      J_kj = wp.tile_load(efc_J_in[worldid], shape=(TILE_SIZE_K, nv_pad), offset=(k, 0), bounds_check=False)

      D_k = wp.tile_load(efc_D_in[worldid], shape=TILE_SIZE_K, offset=k, bounds_check=False)
      state = wp.tile_load(efc_state_in[worldid], shape=TILE_SIZE_K, offset=k, bounds_check=False)

      D_k = wp.tile_map(_state_check, D_k, state)

      tid_tile = wp.tile_arange(TILE_SIZE_K, dtype=int)
      threshold_tile = wp.tile_ones(shape=TILE_SIZE_K, dtype=int) * (nefc - k)

      active_tile = wp.tile_map(_active_check, tid_tile, threshold_tile)
      D_k = wp.tile_map(wp.mul, active_tile, D_k)

      J_ki = wp.tile_map(wp.mul, wp.tile_transpose(J_kj), wp.tile_broadcast(D_k, shape=(nv_pad, TILE_SIZE_K)))

      sum_val += wp.tile_matmul(J_ki, J_kj)

    wp.tile_store(ctx_h_out[worldid], sum_val, bounds_check=False)

  return kernel


@cache_kernel
def _update_gradient_JTDAJ_dense_tiled(nv_pad: int, tile_size: int, njmax: int, nC: int):
  if njmax < tile_size:
    tile_size = njmax

  TILE_SIZE_K = tile_size

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False, module_options={"enable_mathdx_gemm": False})
  def kernel(
    # Model:
    M_colind: wp.array[int],  # column index of each CSR entry
    M_hinit_i: wp.array[int],  # row index of each CSR entry
    # Data in:
    nefc_in: wp.array[int],
    M_in: wp.array2d[float],  # CSR M (nworld, nC)
    efc_J_in: wp.array3d[float],
    efc_D_in: wp.array2d[float],
    efc_state_in: wp.array2d[int],
    # In:
    ctx_done_in: wp.array[bool],
    # Out:
    ctx_h_out: wp.array3d[float],
  ):
    worldid, rank = wp.tid()

    if ctx_done_in[worldid]:
      return

    nefc = nefc_in[worldid]

    # Densify M's upper triangle from CSR into the shared H tile (Cholesky reads fill_mode="upper").
    # Entry (i, col<=i) of M goes to the upper position (col, i); the lower triangle stays zero.
    m_tile = wp.tile_zeros(shape=(wp.static(nv_pad * nv_pad),), dtype=float, storage="shared")
    # Uniform trip count from the RUNTIME lane count: every lane iterates iters times (so the
    # collective tile_scatter_add is always called by all lanes -- a divergent range(rank, nC, bd)
    # deadlocks). wp.block_dim() is 1 on the CPU backend, so lane 0 then covers every entry.
    lanes = wp.block_dim()
    iters = (nC + lanes - 1) // lanes
    for it in range(iters):
      e = it * lanes + rank
      enable = e < nC
      ec = wp.where(enable, e, 0)
      pos = M_colind[ec] * nv_pad + M_hinit_i[ec]
      wp.tile_scatter_add(m_tile, pos, wp.where(enable, M_in[worldid, ec], 0.0), enable)
    sum_val = wp.tile_reshape(m_tile, (nv_pad, nv_pad))

    # Each tile processes one output tile by looping over all constraints
    for k in range(0, njmax, TILE_SIZE_K):
      if k >= nefc:
        break

      # AD: leaving bounds-check disabled here because I'm not entirely sure that
      # everything always hits the fast path. The padding takes care of any
      #  potential OOB accesses.
      J_kj = wp.tile_load(efc_J_in[worldid], shape=(TILE_SIZE_K, nv_pad), offset=(k, 0), bounds_check=False)

      # state check
      D_k = wp.tile_load(efc_D_in[worldid], shape=TILE_SIZE_K, offset=k, bounds_check=False)
      state = wp.tile_load(efc_state_in[worldid], shape=TILE_SIZE_K, offset=k, bounds_check=False)

      D_k = wp.tile_map(_state_check, D_k, state)

      # force unused elements to be zero
      tid_tile = wp.tile_arange(TILE_SIZE_K, dtype=int)
      threshold_tile = wp.tile_ones(shape=TILE_SIZE_K, dtype=int) * (nefc - k)

      active_tile = wp.tile_map(_active_check, tid_tile, threshold_tile)
      D_k = wp.tile_map(wp.mul, active_tile, D_k)

      J_ki = wp.tile_map(wp.mul, wp.tile_transpose(J_kj), wp.tile_broadcast(D_k, shape=(nv_pad, TILE_SIZE_K)))

      sum_val += wp.tile_matmul(J_ki, J_kj)

    wp.tile_store(ctx_h_out[worldid], sum_val, bounds_check=False)

  return kernel


@wp.func
def _elliptic_hessian_entry_from_projections(
  # In:
  dm: float,
  mu: float,
  mu_n_over_t: float,
  tangent_diag: float,
  z01: float,
  z02: float,
  projection1: float,
  projection2: float,
  tangent_dot: float,
) -> float:
  # Contract the diagonal-plus-rank-one curvature without materializing the cone Hessian.
  return dm * (
    z01 * z02
    - mu * (z01 * projection2 + z02 * projection1)
    + mu_n_over_t * projection1 * projection2
    + tangent_diag * tangent_dot
  )


@wp.kernel
def _update_gradient_JTCJ_dense(
  # Model:
  opt_impratio_invsqrt: wp.array[float],
  dof_tri_row: wp.array[int],
  dof_tri_col: wp.array[int],
  # Data in:
  contact_dist_in: wp.array[float],
  contact_includemargin_in: wp.array[float],
  contact_friction_in: wp.array[types.vec5],
  contact_dim_in: wp.array[int],
  contact_efc_address_in: wp.array2d[int],
  contact_worldid_in: wp.array[int],
  efc_J_in: wp.array3d[float],
  efc_D_in: wp.array2d[float],
  efc_state_in: wp.array2d[int],
  naconmax_in: int,
  nacon_in: wp.array[int],
  # In:
  ctx_Jaref_in: wp.array2d[float],
  ctx_done_in: wp.array[bool],
  nblocks_perblock: int,
  dim_block: int,
  # Out:
  ctx_h_out: wp.array3d[float],
):
  conid_start, elementid = wp.tid()

  dof1id = dof_tri_row[elementid]
  dof2id = dof_tri_col[elementid]

  for i in range(nblocks_perblock):
    conid = conid_start + i * dim_block

    if conid >= min(nacon_in[0], naconmax_in):
      return

    worldid = contact_worldid_in[conid]
    if ctx_done_in[worldid]:
      continue

    condim = contact_dim_in[conid]

    if condim == 1:
      continue

    # check contact status
    if contact_dist_in[conid] - contact_includemargin_in[conid] >= 0.0:
      continue

    efcid0 = contact_efc_address_in[conid, 0]
    if efcid0 < 0:
      continue
    if efc_state_in[worldid, efcid0] != types.ConstraintState.CONE:
      continue

    fri = contact_friction_in[conid]
    mu = fri[0] * opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]

    mu2 = mu * mu
    dm = math.safe_div(efc_D_in[worldid, efcid0], mu2 * (1.0 + mu2))

    if dm == 0.0:
      continue

    n = ctx_Jaref_in[worldid, efcid0] * mu
    z01 = mu * efc_J_in[worldid, efcid0, dof1id]
    z02 = mu * efc_J_in[worldid, efcid0, dof2id]
    tt = float(0.0)
    projection1 = float(0.0)
    projection2 = float(0.0)
    tangent_dot = float(0.0)
    for dim in range(1, condim):
      efcid = contact_efc_address_in[conid, dim]
      if efcid >= 0:
        scale = fri[dim - 1]
        u = ctx_Jaref_in[worldid, efcid] * scale
        z1 = scale * efc_J_in[worldid, efcid, dof1id]
        z2 = scale * efc_J_in[worldid, efcid, dof2id]
        tt += u * u
        projection1 += u * z1
        projection2 += u * z2
        tangent_dot += z1 * z2

    t = wp.max(wp.sqrt(tt), types.MJ_MINVAL)
    inv_t = math.safe_div(1.0, t)
    mu_n_over_t = mu * n * inv_t
    h = _elliptic_hessian_entry_from_projections(
      dm,
      mu,
      mu_n_over_t,
      mu2 - mu_n_over_t,
      z01,
      z02,
      projection1 * inv_t,
      projection2 * inv_t,
      tangent_dot,
    )

    ctx_h_out[worldid, dof1id, dof2id] += h


@cache_kernel
def _update_gradient_cholesky(tile_size: int, skip_noflip: bool = False):
  SKIP_NOFLIP = skip_noflip

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # In:
    ctx_grad_in: wp.array2d[float],
    h_in: wp.array3d[float],
    state_changed_count_in: wp.array[int],
    ctx_done_in: wp.array[bool],
    # Out:
    ctx_search_out: wp.array2d[float],
    ctx_search_dot_out: wp.array[float],
    ctx_newton_decrement_out: wp.array[float],
  ):
    worldid = wp.tid()
    TILE_SIZE = wp.static(tile_size)

    if ctx_done_in[worldid]:
      return

    # Fast path: skip the solve (see the blocked skip_unchanged variant).
    if wp.static(SKIP_NOFLIP):
      if state_changed_count_in[worldid] == 0:
        return

    mat_tile = wp.tile_load(h_in[worldid], shape=(TILE_SIZE, TILE_SIZE))
    wp.tile_cholesky_inplace(mat_tile, fill_mode="upper")
    input_tile = wp.tile_load(ctx_grad_in[worldid], shape=TILE_SIZE)
    output_tile = wp.tile_cholesky_solve(mat_tile, input_tile, fill_mode="upper")
    sums = wp.tile_reduce(wp.add, wp.tile_map(solve_search_sums, input_tile, output_tile))[0]
    ctx_search_dot_out[worldid] = sums[0]
    ctx_newton_decrement_out[worldid] = sums[1]
    wp.tile_store(ctx_search_out[worldid], wp.tile_map(wp.mul, output_tile, -1.0))

  return kernel


@wp.func
def _blocked_size(nrow_in: wp.array[int], worldid: int, tile_size: int, matrix_size: int) -> int:
  """Factor only the leading active rows plus the augmented border, rounded up to the tile size."""
  if nrow_in:
    return wp.min(matrix_size, (nrow_in[worldid] + tile_size) // tile_size * tile_size)
  return matrix_size


@cache_kernel
def _update_gradient_cholesky_blocked(tile_size: int, matrix_size: int, vector_size: int):
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False, module_options={"enable_mathdx_gemm": False})
  def kernel(
    # In:
    nrow_in: wp.array[int],
    ctx_done_in: wp.array[bool],
    ctx_grad_in: wp.array3d[float],
    ctx_h_in: wp.array3d[float],
    ctx_cholesky_top_in: wp.array2d[int],
    ctx_hfactor: wp.array3d[float],
    # Out:
    ctx_search_out: wp.array3d[float],
    ctx_search_dot_out: wp.array[float],
    ctx_newton_decrement_out: wp.array[float],
  ):
    worldid = wp.tid()
    TILE_SIZE = wp.static(tile_size)

    if ctx_done_in[worldid]:
      return

    # We need matrix size both as a runtime input as well as a static input:
    # static input is needed to specify the tile sizes for the compiler
    # runtime input is needed for the loop bounds, otherwise warp will unroll
    # unconditionally leading to shared memory capacity issues.

    sums = wp.static(create_blocked_cholesky_augmented_factorize_solve_newton_func(TILE_SIZE, matrix_size, vector_size))(
      ctx_h_in[worldid],
      ctx_grad_in[worldid],
      ctx_cholesky_top_in[worldid % ctx_cholesky_top_in.shape[0]],
      _blocked_size(nrow_in, worldid, TILE_SIZE, matrix_size),
      ctx_hfactor[worldid],
      ctx_search_out[worldid],
    )
    ctx_search_dot_out[worldid] = sums[0]
    ctx_newton_decrement_out[worldid] = sums[1]

  return kernel


@cache_kernel
def _cholesky_factorize_solve_blocked(tile_size: int, matrix_size: int):
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False, module_options={"enable_mathdx_gemm": False})
  def kernel(
    # In:
    A_in: wp.array3d[float],
    b_in: wp.array3d[float],
    # Out:
    U_out: wp.array3d[float],
    x_out: wp.array3d[float],
  ):
    worldid = wp.tid()
    TILE_SIZE = wp.static(tile_size)

    wp.static(create_blocked_cholesky_factorize_solve_func(TILE_SIZE, matrix_size))(
      A_in[worldid],
      b_in[worldid],
      matrix_size,
      U_out[worldid],
      x_out[worldid],
    )

  return kernel


@cache_kernel
def _update_gradient_cholesky_blocked_skip_unchanged(
  tile_size: int, matrix_size: int, vector_size: int, skip_noflip: bool = False
):
  """Blocked Cholesky that skips factorization when no constraints changed."""
  SKIP_NOFLIP = skip_noflip

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False, module_options={"enable_mathdx_gemm": False})
  def kernel(
    # In:
    nrow_in: wp.array[int],
    ctx_done_in: wp.array[bool],
    ctx_grad_in: wp.array3d[float],
    ctx_h_in: wp.array3d[float],
    quad_changed_count_in: wp.array[int],
    state_changed_count_in: wp.array[int],
    ctx_cholesky_top_in: wp.array2d[int],
    ctx_hfactor: wp.array3d[float],
    # Out:
    ctx_search_out: wp.array3d[float],
    ctx_search_dot_out: wp.array[float],
    ctx_newton_decrement_out: wp.array[float],
  ):
    worldid = wp.tid()
    TILE_SIZE = wp.static(tile_size)

    if ctx_done_in[worldid]:
      return

    # The linesearch is invariant to direction scale, so an unchanged ray can
    # keep its previous search direction.
    if wp.static(SKIP_NOFLIP):
      if state_changed_count_in[worldid] == 0:
        return

    size = _blocked_size(nrow_in, worldid, TILE_SIZE, matrix_size)
    top = ctx_cholesky_top_in[worldid % ctx_cholesky_top_in.shape[0]]
    if quad_changed_count_in[worldid] > 0:
      sums = wp.static(create_blocked_cholesky_augmented_factorize_solve_newton_func(TILE_SIZE, matrix_size, vector_size))(
        ctx_h_in[worldid],
        ctx_grad_in[worldid],
        top,
        size,
        ctx_hfactor[worldid],
        ctx_search_out[worldid],
      )
    else:
      sums = wp.static(create_blocked_cholesky_solve_newton_func(TILE_SIZE, matrix_size, vector_size))(
        ctx_hfactor[worldid],
        ctx_grad_in[worldid],
        top,
        size,
        ctx_search_out[worldid],
      )

    ctx_search_dot_out[worldid] = sums[0]
    ctx_newton_decrement_out[worldid] = sums[1]

  return kernel


@wp.kernel(grid_stride=True)
def _padding_h(nv: int, ctx_done_in: wp.array[bool], ctx_h_out: wp.array3d[float]):
  worldid, elementid = wp.tid()

  if ctx_done_in[worldid]:
    return

  dofid = nv + elementid
  ctx_h_out[worldid, dofid, dofid] = 1.0


def _blocked_bounds(m: types.Model, d: types.Data, ctx: SolverContext) -> tuple:
  """Active rows and tile envelope the blocked Cholesky reads (None, dense: the whole matrix)."""
  if m.nv <= _BLOCK_CHOLESKY_DIM:
    return None, ctx.cholesky_top
  nrow = d.ncdof if ctx.compact_d_full is not None else None
  return nrow, ctx.cholesky_top


def _cholesky_factorize_solve(
  m: types.Model, d: types.Data, ctx: SolverContext, skip_unchanged: bool = False, skip_noflip: bool = False
):
  """Cholesky factorize ctx.h and form the Newton search direction.

  If skip_unchanged is True (blocked path only), worlds where no constraints
  changed reuse the cached factorization in hfactor instead of refactorizing.
  """
  if ctx.island_ntile is not None:
    _island_pool_factorize_solve(m, d, ctx, skip_unchanged, skip_noflip)
    return

  grad, search = ctx.grad, ctx.search
  if m.nv <= _BLOCK_CHOLESKY_DIM:
    wp.launch_tiled(
      _update_gradient_cholesky(m.nv, skip_noflip),
      dim=d.nworld,
      inputs=[grad, ctx.h, ctx.state_changed_count if skip_noflip else d.nefc, ctx.done],
      outputs=[search, ctx.search_dot, ctx.newton_decrement],
      block_dim=m.block_dim.update_gradient_cholesky,
    )
  else:
    wp.launch(
      _padding_h,
      dim=(d.nworld, m.nv_pad - m.nv),
      inputs=[m.nv, ctx.done],
      outputs=[ctx.h],
    )

    # compacted or island-ordered DOFs are packed in front, so each world factors only that block
    nrow, _ = _blocked_bounds(m, d, ctx)

    if skip_unchanged:
      wp.launch_tiled(
        _update_gradient_cholesky_blocked_skip_unchanged(types.TILE_SIZE_JTDAJ_DENSE, m.nv_pad, m.nv, skip_noflip),
        dim=d.nworld,
        inputs=[
          nrow,
          ctx.done,
          grad.reshape(shape=(d.nworld, grad.shape[1], 1)),
          ctx.h,
          ctx.quad_changed_count,
          ctx.state_changed_count if skip_noflip else ctx.quad_changed_count,
          ctx.cholesky_top,
          ctx.hfactor,
        ],
        outputs=[
          search.reshape(shape=(d.nworld, m.nv, 1)),
          ctx.search_dot,
          ctx.newton_decrement,
        ],
        block_dim=m.block_dim.update_gradient_cholesky_blocked,
      )
    else:
      wp.launch_tiled(
        _update_gradient_cholesky_blocked(types.TILE_SIZE_JTDAJ_DENSE, m.nv_pad, m.nv),
        dim=d.nworld,
        inputs=[nrow, ctx.done, grad.reshape(shape=(d.nworld, grad.shape[1], 1)), ctx.h, ctx.cholesky_top, ctx.hfactor],
        outputs=[
          search.reshape(shape=(d.nworld, m.nv, 1)),
          ctx.search_dot,
          ctx.newton_decrement,
        ],
        block_dim=m.block_dim.update_gradient_cholesky_blocked,
      )


# ---------------------------------------------------------------------------
# Constraint groups contain consecutive rows with identical sparse support. Each thread group
# accumulates their upper-triangular Hessian block, including elliptic cone curvature.
# ---------------------------------------------------------------------------
_JTDAJ_THREADS_PER_GROUP = 32
_JTDAJ_OVERSUBSCRIBE_WAVES = 6


@cache_kernel
def _JTDACJ_sparse(compact: bool, cone_type: types.ConeType, max_condim: int, pool: int = 0):
  # pool: the tile size of the island tile pool H is stored in (0: a dense H per world)
  COMPACT = compact
  POOL = pool
  ELLIPTIC = cone_type == types.ConeType.ELLIPTIC
  MAX_CONDIM = max_condim

  def make_curvature_terms(condim: int):
    @wp.func
    def func(
      # Model:
      opt_impratio_invsqrt: wp.array[float],
      # Data in:
      efc_D_in: wp.array2d[float],
      # In:
      fri: types.vec5,
      ctx_Jaref_in: wp.array2d[float],
      worldid: int,
      efcid0: int,
      block_rows: int,
    ) -> types.vec16:
      mu = fri[0] * opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]
      mu2 = mu * mu
      dm = math.safe_div(efc_D_in[worldid, efcid0], mu2 * (1.0 + mu2))
      if dm == 0.0:
        return types.vec16()

      n = ctx_Jaref_in[worldid, efcid0] * mu
      terms = types.vec16()
      terms[6] = mu
      tt = float(0.0)
      for dim in range(1, wp.static(condim)):
        if dim < block_rows:
          efcid = efcid0 + dim
          scale = fri[dim - 1]
          u = ctx_Jaref_in[worldid, efcid] * scale
          terms[dim] = u
          terms[6 + dim] = scale
          tt += u * u

      t = wp.max(wp.sqrt(tt), types.MJ_MINVAL)
      inv_t = math.safe_div(1.0, t)
      for dim in range(1, wp.static(condim)):
        terms[dim] = terms[dim] * inv_t

      mu_n_over_t = mu * n * inv_t
      tangent_diag = mu2 - mu_n_over_t

      # Layout: normalized tangent u/t[1:6], scales[6:12], dm, mu, mu*n/t, tangent diagonal.
      terms[12] = dm
      terms[13] = mu
      terms[14] = mu_n_over_t
      terms[15] = tangent_diag
      return terms

    return func

  def make_hessian_entry(condim: int):
    @wp.func
    def func(
      # Data in:
      efc_J_in: wp.array3d[float],
      # In:
      terms: Any,
      rowadr: types.vec6i,
      worldid: int,
      pos1: int,
      pos2: int,
    ) -> float:
      z01 = terms[6] * efc_J_in[worldid, 0, rowadr[0] + pos1]
      z02 = terms[6] * efc_J_in[worldid, 0, rowadr[0] + pos2]
      projection1 = float(0.0)
      projection2 = float(0.0)
      tangent_dot = float(0.0)
      for dim in range(1, wp.static(condim)):
        z1 = terms[6 + dim] * efc_J_in[worldid, 0, rowadr[dim] + pos1]
        z2 = terms[6 + dim] * efc_J_in[worldid, 0, rowadr[dim] + pos2]
        projection1 += terms[dim] * z1
        projection2 += terms[dim] * z2
        tangent_dot += z1 * z2

      return _elliptic_hessian_entry_from_projections(
        terms[12],
        terms[13],
        terms[14],
        terms[15],
        z01,
        z02,
        projection1,
        projection2,
        tangent_dot,
      )

    return func

  curvature_terms3 = make_curvature_terms(3)
  curvature_terms4 = make_curvature_terms(4)
  curvature_terms6 = make_curvature_terms(6)
  hessian_entry3 = make_hessian_entry(3)
  hessian_entry4 = make_hessian_entry(4)
  hessian_entry6 = make_hessian_entry(6)

  @wp.func
  def curvature_terms(
    # Model:
    opt_impratio_invsqrt: wp.array[float],
    # Data in:
    efc_D_in: wp.array2d[float],
    # In:
    fri: types.vec5,
    ctx_Jaref_in: wp.array2d[float],
    worldid: int,
    efcid0: int,
    condim: int,
    block_rows: int,
  ) -> types.vec16:
    if wp.static(MAX_CONDIM == 3):
      return curvature_terms3(opt_impratio_invsqrt, efc_D_in, fri, ctx_Jaref_in, worldid, efcid0, block_rows)

    if condim == 3:
      return curvature_terms3(opt_impratio_invsqrt, efc_D_in, fri, ctx_Jaref_in, worldid, efcid0, block_rows)
    if wp.static(MAX_CONDIM == 4):
      return curvature_terms4(opt_impratio_invsqrt, efc_D_in, fri, ctx_Jaref_in, worldid, efcid0, block_rows)
    if condim == 4:
      return curvature_terms4(opt_impratio_invsqrt, efc_D_in, fri, ctx_Jaref_in, worldid, efcid0, block_rows)
    return curvature_terms6(opt_impratio_invsqrt, efc_D_in, fri, ctx_Jaref_in, worldid, efcid0, block_rows)

  @wp.func
  def hessian_entry(
    # Data in:
    efc_J_in: wp.array3d[float],
    # In:
    terms: Any,
    rowadr: types.vec6i,
    worldid: int,
    pos1: int,
    pos2: int,
    condim: int,
  ) -> float:
    if wp.static(MAX_CONDIM == 3):
      return hessian_entry3(efc_J_in, terms, rowadr, worldid, pos1, pos2)

    if condim == 3:
      return hessian_entry3(efc_J_in, terms, rowadr, worldid, pos1, pos2)
    if wp.static(MAX_CONDIM == 4):
      return hessian_entry4(efc_J_in, terms, rowadr, worldid, pos1, pos2)
    if condim == 4:
      return hessian_entry4(efc_J_in, terms, rowadr, worldid, pos1, pos2)
    return hessian_entry6(efc_J_in, terms, rowadr, worldid, pos1, pos2)

  @wp.kernel(module="unique", enable_backward=False, grid_stride=True)
  def kernel(
    # Model:
    opt_impratio_invsqrt: wp.array[float],
    # Data in:
    contact_friction_in: wp.array[types.vec5],
    contact_dim_in: wp.array[int],
    efc_id_in: wp.array2d[int],
    efc_jtdaj_adr_in: wp.array2d[int],
    efc_jtdaj_nrow_in: wp.array2d[int],
    efc_jtdaj_nblock_in: wp.array[int],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    efc_D_in: wp.array2d[float],
    efc_state_in: wp.array2d[int],
    dof_cdof_in: wp.array2d[int],
    # In:
    ctx_h_dof_local_in: wp.array2d[int],
    ctx_Jaref_in: wp.array2d[float],
    ctx_done_in: wp.array[bool],
    groups_per_world: int,
    # Out:
    h_out: wp.array3d[float],
  ):
    worldid, slot, lane = wp.tid()
    if wp.static(ELLIPTIC):
      lanes = wp.block_dim()
    else:
      lanes = wp.static(_JTDAJ_THREADS_PER_GROUP)
    if ctx_done_in[worldid]:
      return
    count = efc_jtdaj_nblock_in[worldid]
    for groupid in range(slot, count, groups_per_world):
      head_row = efc_jtdaj_adr_in[worldid, groupid]
      block_rows = efc_jtdaj_nrow_in[worldid, groupid]
      head_adr = efc_J_rowadr_in[worldid, head_row]
      support = efc_J_rownnz_in[worldid, head_row]
      n_entries = support * (support + 1) // 2

      is_cone = False
      if wp.static(ELLIPTIC):
        is_cone = efc_state_in[worldid, head_row] == types.ConstraintState.CONE
        condim = int(0)
        # Clipped cone rows retain the safe head address and a zero scale.
        cone_rowadr = types.vec6i(head_adr, head_adr, head_adr, head_adr, head_adr, head_adr)
        if is_cone:
          conid = efc_id_in[worldid, head_row]
          if wp.static(MAX_CONDIM == 3):
            condim = int(3)
          else:
            condim = contact_dim_in[conid]
          for dim in range(1, wp.static(MAX_CONDIM)):
            if dim < block_rows:
              cone_rowadr[dim] = head_adr + dim * support
          local_terms = types.vec16()
          if lane == 0:
            local_terms = curvature_terms(
              opt_impratio_invsqrt,
              efc_D_in,
              contact_friction_in[conid],
              ctx_Jaref_in,
              worldid,
              head_row,
              condim,
              block_rows,
            )
          cone_terms = wp.tile_zeros(shape=(16,), dtype=float, storage="shared")
          for term_id in range(1, 16):
            wp.tile_scatter_masked(cone_terms, term_id, local_terms[term_id], lane == 0)

      for entry in range(lane, n_entries, lanes):
        block_col = int((wp.sqrt(float(8 * entry + 1)) - 1.0) * 0.5)
        block_row = entry - block_col * (block_col + 1) // 2
        dof_row = efc_J_colind_in[worldid, 0, head_adr + block_row]
        dof_col = efc_J_colind_in[worldid, 0, head_adr + block_col]
        hval = float(0.0)
        if wp.static(ELLIPTIC):
          if is_cone:
            hval = hessian_entry(
              efc_J_in,
              cone_terms,
              cone_rowadr,
              worldid,
              block_row,
              block_col,
              condim,
            )
        if not is_cone:
          for member in range(block_rows):
            member_row = head_row + member
            if efc_state_in[worldid, member_row] == types.ConstraintState.QUADRATIC.value:
              member_adr = efc_J_rowadr_in[worldid, member_row]
              j_row = efc_J_in[worldid, 0, member_adr + block_row]
              j_col = efc_J_in[worldid, 0, member_adr + block_col]
              hval += j_row * efc_D_in[worldid, member_row] * j_col
        if hval != 0.0:
          if wp.static(POOL):
            row_a = dof_cdof_in[worldid, dof_row]
            row_b = dof_cdof_in[worldid, dof_col]
            if row_a < 0 or row_b < 0:
              continue
            la = ctx_h_dof_local_in[worldid, dof_row]
            lb = ctx_h_dof_local_in[worldid, dof_col]
            _island_pool_add(worldid, row_a, la, row_b, lb, hval, wp.static(POOL), h_out)
            continue
          if wp.static(COMPACT):
            dof_row = dof_cdof_in[worldid, dof_row]
            dof_col = dof_cdof_in[worldid, dof_col]
            if dof_row < 0 or dof_col < 0:
              continue
          wp.atomic_add(h_out[worldid, wp.min(dof_row, dof_col)], wp.max(dof_row, dof_col), hval)

  return kernel


def _jtdaj_groups_per_world(nworld: int, njmax: int) -> int:
  # njmax is capacity and often mostly empty, so cap slots at a few resident waves.
  block_size, min_grid_size = wp.get_suggested_block_size(_JTDACJ_sparse(False, types.ConeType.PYRAMIDAL, 3))
  device_warps = max(1, block_size * min_grid_size // _JTDAJ_THREADS_PER_GROUP)
  return max(1, min(njmax, _JTDAJ_OVERSUBSCRIBE_WAVES * device_warps // nworld))


def _update_gradient(m: types.Model, d: types.Data, ctx: SolverContext, compact: bool = False):
  # grad = Ma - qfrc_smooth - qfrc_constraint
  if m.opt.solver == types.SolverType.CG:
    wp.launch_tiled(
      _update_gradient_grad_tiled,
      dim=d.nworld,
      inputs=[m.nv, d.qfrc_smooth, d.qfrc_constraint, d.efc.Ma, ctx.done],
      outputs=[ctx.grad, ctx.grad_dot],
      block_dim=m.block_dim.update_gradient_grad,
    )
  else:
    wp.launch(
      _update_gradient_zero_grad_dot(False),
      dim=d.nworld,
      inputs=[d.nefc, ctx.alpha, ctx.done],
      outputs=[ctx.grad_dot, ctx.newton_decrement, ctx.grad_scale, ctx.search_unchanged],
    )
    wp.launch(
      _update_gradient_grad(False),
      dim=(d.nworld, m.nv),
      inputs=[d.qfrc_smooth, d.qfrc_constraint, d.efc.Ma, d.nefc, ctx.done],
      outputs=[ctx.grad, ctx.grad_dot],
    )

  is_discrete = m.opt.integrator == types.IntegratorType.DISCRETE
  if m.opt.solver == types.SolverType.CG:
    if is_discrete:
      if m.nefmK > 0 or m.efm0_active or (m.opt.enableflags & types.EnableBit.SLEEP):
        derivative.eff_prec(m, d, ctx.Mgrad, ctx.grad)
      else:
        smooth.solve_LD(m, d, d.qHLD, d.qHDiagInv, ctx.Mgrad, ctx.grad)
    else:
      smooth.solve_m(m, d, ctx.Mgrad, ctx.grad)
  elif m.opt.solver == types.SolverType.NEWTON:
    # h = M + (efc_J.T * efc_D * active) @ efc_J
    sc = _sparse_compact(ctx)
    if m.is_sparse or sc:
      mj = ctx.compact_m_full if sc else m
      dj = ctx.compact_d_full if sc else d
      m_mat = dj.qH if is_discrete else dj.M
      # h rows are compacted DOFs or DOFs, or h is an island tile pool
      island = ctx.island_ntile is not None
      mapped, row2dof, dof2row = sc, dj.cdof_dof, dj.dof_cdof
      if island:
        mapped, dof2row = False, ctx.h_dof_poolrow
        wp.launch(
          _island_pool_init,
          dim=(d.nworld, ctx.pgrad.shape[1]),
          inputs=[
            mj.M_elemid,
            d.nisland,
            m_mat,
            d.island_nv,
            ctx.h_row2dof,
            ctx.island_tilebase,
            ctx.island_ntile,
            ctx.island_vbase,
            ctx.done,
          ],
          outputs=[ctx.h],
        )
      nrow, top = _blocked_bounds(m, d, ctx)
      if island:
        pass
      elif mapped and nrow is not None:
        # one thread per row, writing only what the blocked Cholesky reads
        wp.launch(
          _update_gradient_init_h_envelope,
          dim=(d.nworld, m.nv_pad),
          inputs=[mj.M_elemid, m_mat, row2dof, nrow, top, ctx.done],
          outputs=[ctx.h],
        )
      else:
        wp.launch(
          _update_gradient_init_h_sparse(mapped),
          dim=(d.nworld, m.nv_pad, m.nv_pad),
          inputs=[mj.nv, mj.M_elemid, m_mat, row2dof, ctx.done],
          outputs=[ctx.h],
        )

      groups_per_world = _jtdaj_groups_per_world(d.nworld, d.njmax)
      max_condim = 3
      if m.opt.cone == types.ConeType.ELLIPTIC and m.nmaxcondim > 3:
        max_condim = int(m.nmaxcondim)
      jtdaj_kernel = _JTDACJ_sparse(mapped, m.opt.cone, max_condim, types.TILE_SIZE_JTDAJ_DENSE if island else 0)
      jtdaj_inputs = [
        m.opt.impratio_invsqrt,
        d.contact.friction,
        d.contact.dim,
        d.efc.id,
        dj.efc.jtdaj_adr,
        dj.efc.jtdaj_nrow,
        dj.efc.jtdaj_nblock,
        dj.efc.J_rownnz,
        dj.efc.J_rowadr,
        dj.efc.J_colind,
        dj.efc.J,
        d.efc.D,
        d.efc.state,
        dof2row,
        ctx.h_dof_local,
        ctx.Jaref,
        ctx.done,
        groups_per_world,
      ]
      elliptic = m.opt.cone == types.ConeType.ELLIPTIC
      threads_per_group = 1 if elliptic and wp.get_device().is_cpu else _JTDAJ_THREADS_PER_GROUP
      block_dim = threads_per_group if elliptic else mj.block_dim.update_gradient_JTDAJ_sparse
      wp.launch(
        jtdaj_kernel,
        dim=(d.nworld, groups_per_world, threads_per_group),
        inputs=jtdaj_inputs,
        outputs=[ctx.h],
        block_dim=block_dim,
      )
    else:
      m_mat = d.qH if is_discrete else d.M
      if ctx.efc_world is not None:
        wp.launch_tiled(
          _update_gradient_JTDAJ_ragged(m.nv_pad, types.TILE_SIZE_JTDAJ_DENSE, d.njmax),
          dim=d.nworld,
          inputs=[
            m.dof_treeid,
            m.tree_dofadr,
            m.tree_dofnum,
            m.M_rownnz,
            m.M_rowadr,
            m.M_colind,
            m.M_hinit_i,
            d.nefc,
            d.M,
            d.efc.J,
            d.efc.D,
            d.efc.state,
            ctx.done,
            ctx.efc_world,
            ctx.efc_rowoff,
            ctx.problem_dof,
          ],
          outputs=[ctx.h, m.stat.meaninertia],
          block_dim=m.block_dim.update_gradient_JTDAJ_dense,
        )
      elif compact:
        # compact path: d.M is the dense 3D compact inertia block (nworld, nv_pad, nv_pad)
        wp.launch_tiled(
          _update_gradient_JTDAJ_dense_tiled_compact(m.nv_pad, types.TILE_SIZE_JTDAJ_DENSE, d.njmax),
          dim=d.nworld,
          inputs=[
            d.nefc,
            m_mat,
            d.efc.J,
            d.efc.D,
            d.efc.state,
            ctx.done,
          ],
          outputs=[ctx.h],
          block_dim=m.block_dim.update_gradient_JTDAJ_dense,
        )
      else:
        wp.launch_tiled(
          _update_gradient_JTDAJ_dense_tiled(m.nv_pad, types.TILE_SIZE_JTDAJ_DENSE, d.njmax, m.M_colind.shape[0]),
          dim=d.nworld,
          inputs=[
            m.M_colind,
            m.M_hinit_i,
            d.nefc,
            m_mat,
            d.efc.J,
            d.efc.D,
            d.efc.state,
            ctx.done,
          ],
          outputs=[ctx.h],
          block_dim=m.block_dim.update_gradient_JTDAJ_dense,
        )

    if is_discrete:
      wp.launch(
        _add_tendon_metric_dense,
        dim=(d.nworld, m.ntendon),
        inputs=[m.ten_J_rownnz, m.ten_J_rowadr, m.ten_J_colind, d.ten_J, d.efm_ts, ctx.done],
        outputs=[ctx.h],
      )
      wp.launch(
        _add_actuator_metric_dense,
        dim=(d.nworld, m.nactuator),
        inputs=[d.moment_rownnz, d.moment_rowadr, d.moment_colind, d.actuator_moment, d.efm_as, ctx.done],
        outputs=[ctx.h],
      )
      if m.nefmK > 0:
        wp.launch(
          _add_efmK_metric_dense,
          dim=(d.nworld, m.nv),
          inputs=[m.efm_K_rownnz, m.efm_K_rowadr, m.efm_K_colind, d.efm_K_val, ctx.done],
          outputs=[ctx.h],
        )
      if m.has_flex_passive:
        efm_con_dof, efm_con_val, efm_con_scale, _, efm_con_nnz = derivative.build_efm_contact(m, d)
        wp.launch(
          _add_flexcon_metric_dense,
          dim=d.naconmax,
          inputs=[
            d.contact.worldid,
            d.nacon,
            efm_con_dof,
            efm_con_val,
            efm_con_scale,
            efm_con_nnz,
            ctx.done,
          ],
          outputs=[ctx.h],
        )

    if m.opt.cone == types.ConeType.ELLIPTIC and not (m.is_sparse or sc):
      # Optimization: launching update_gradient_JTCJ with limited number of blocks on a GPU.
      # Profiling suggests that only a fraction of blocks out of the original
      # d.njmax blocks do the actual work. It aims to minimize #CTAs with no
      # effective work. It launches with #blocks that's proportional to the number
      # of SMs on the GPU. We can now query the SM count:
      # https://github.com/NVIDIA/warp/commit/f3814e7e5459e5fd13032cf0fddb3daddd510f30

      if wp.get_device().is_cuda:
        sm_count = wp.get_device().sm_count

        # Here we assume one block has 256 threads. We use a factor of 6, which
        # can be changed in the future to fine-tune the perf. The optimal factor will
        # depend on the kernel's occupancy, which determines how many blocks can
        # simultaneously run on the SM. TODO: This factor can be tuned further.
        dim_block = ceil((sm_count * 6 * 256) / m.dof_tri_row.size)
      else:
        # fall back for CPU
        dim_block = d.naconmax

      nblocks_perblock = int((d.naconmax + dim_block - 1) / dim_block)

      wp.launch(
        _update_gradient_JTCJ_dense,
        dim=(dim_block, m.dof_tri_row.size),
        inputs=[
          m.opt.impratio_invsqrt,
          m.dof_tri_row,
          m.dof_tri_col,
          d.contact.dist,
          d.contact.includemargin,
          d.contact.friction,
          d.contact.dim,
          d.contact.efc_address,
          d.contact.worldid,
          d.efc.J,
          d.efc.D,
          d.efc.state,
          d.naconmax,
          d.nacon,
          ctx.Jaref,
          ctx.done,
          nblocks_perblock,
          dim_block,
        ],
        outputs=[ctx.h],
      )

    _cholesky_factorize_solve(m, d, ctx)
  else:
    raise ValueError(f"Unknown solver type: {m.opt.solver}")

  if is_discrete and (m.opt.enableflags & types.EnableBit.SLEEP):
    wp.launch(
      derivative._zero_sleeping_dofs,
      dim=(d.nworld, ctx.Mgrad.shape[1]),
      inputs=[m.dof_treeid, d.tree_awake],
      outputs=[ctx.Mgrad],
    )
    wp.launch(
      derivative._zero_sleeping_dofs,
      dim=(d.nworld, m.nv),
      inputs=[m.dof_treeid, d.tree_awake],
      outputs=[ctx.search],
    )


def _update_gradient_incremental(m: types.Model, d: types.Data, ctx: SolverContext, stable_fast: bool = False):
  """Incremental gradient update: update H for changed constraints + re-factorize.

  Skips the full J^T*D*J rebuild by applying only the delta from constraints
  that changed QUADRATIC state, then re-factorizes and solves.
  """
  changed = ctx.state_changed_count if stable_fast else d.nefc
  wp.launch(
    _update_gradient_zero_grad_dot(stable_fast),
    dim=d.nworld,
    inputs=[changed, ctx.alpha, ctx.done],
    outputs=[ctx.grad_dot, ctx.newton_decrement, ctx.grad_scale, ctx.search_unchanged],
  )

  wp.launch(
    _update_gradient_grad(stable_fast),
    dim=(d.nworld, m.nv),
    inputs=[d.qfrc_smooth, d.qfrc_constraint, d.efc.Ma, changed, ctx.done],
    outputs=[ctx.grad, ctx.grad_dot],
  )

  # Update upper triangle of H with delta from changed constraints.
  sc = _sparse_compact(ctx)
  if m.is_sparse or sc:
    dj = ctx.compact_d_full if sc else d
    slots = _jtdaj_groups_per_world(d.nworld, ctx.quad_changed_ids.shape[1])
    island = ctx.island_ntile is not None
    mapped, dof2row = sc, dj.dof_cdof
    if island:
      mapped, dof2row = False, ctx.h_dof_poolrow
    wp.launch(
      _update_gradient_h_incremental_sparse(mapped, island),
      dim=(d.nworld, slots, _JTDAJ_THREADS_PER_GROUP),
      inputs=[
        dj.efc.J_rownnz,
        dj.efc.J_rowadr,
        dj.efc.J_colind,
        dj.efc.J,
        d.efc.D,
        d.efc.state,
        dof2row,
        ctx.h_dof_local,
        ctx.quad_changed_ids,
        ctx.quad_changed_count,
        slots,
      ],
      outputs=[ctx.h],
    )
  elif ctx.efc_problem is not None:
    wp.launch(
      _update_gradient_h_incremental_rows(m.nv),
      dim=ctx.efc_problem.shape,
      inputs=[d.efc.J, d.efc.D, d.efc.state, ctx.quad_changed_ids, ctx.quad_changed_count, ctx.efc_rowoff, ctx.efc_problem],
      outputs=[ctx.h],
    )
  else:
    tri_dim = m.nv * (m.nv + 1) // 2
    wp.launch(
      _update_gradient_h_incremental(ctx.efc_world is not None),
      dim=(d.nworld, tri_dim),
      inputs=[
        d.efc.J,
        d.efc.D,
        d.efc.state,
        ctx.quad_changed_ids,
        ctx.quad_changed_count,
        ctx.efc_world,
        ctx.efc_rowoff,
      ],
      outputs=[ctx.h],
    )

  _cholesky_factorize_solve(m, d, ctx, skip_unchanged=True, skip_noflip=stable_fast)
  if m.opt.integrator == types.IntegratorType.DISCRETE and (m.opt.enableflags & types.EnableBit.SLEEP):
    wp.launch(
      derivative._zero_sleeping_dofs,
      dim=(d.nworld, ctx.Mgrad.shape[1]),
      inputs=[m.dof_treeid, d.tree_awake],
      outputs=[ctx.Mgrad],
    )
    wp.launch(
      derivative._zero_sleeping_dofs,
      dim=(d.nworld, m.nv),
      inputs=[m.dof_treeid, d.tree_awake],
      outputs=[ctx.search],
    )


@wp.kernel
def _solve_search_update_cg_tiled(
  # Model:
  nv: int,
  # In:
  ctx_grad_in: wp.array2d[float],
  ctx_Mgrad_in: wp.array2d[float],
  ctx_search_in: wp.array2d[float],
  ctx_beta_in: wp.array[float],
  ctx_done_in: wp.array[bool],
  # Out:
  ctx_search_out: wp.array2d[float],
  ctx_search_dot_out: wp.array[float],
  ctx_prev_grad_out: wp.array2d[float],
  ctx_prev_Mgrad_out: wp.array2d[float],
):
  worldid, tid = wp.tid()

  if ctx_done_in[worldid]:
    return

  local_search_dot = float(0.0)
  BLOCK_DIM = wp.block_dim()
  beta = ctx_beta_in[worldid]

  for dofid in range(tid, nv, BLOCK_DIM):
    mgrad = ctx_Mgrad_in[worldid, dofid]
    search = -1.0 * mgrad + beta * ctx_search_in[worldid, dofid]

    ctx_search_out[worldid, dofid] = search
    local_search_dot += search * search

    ctx_prev_grad_out[worldid, dofid] = ctx_grad_in[worldid, dofid]
    ctx_prev_Mgrad_out[worldid, dofid] = mgrad

  search_dot_tile = wp.tile(local_search_dot, preserve_type=True)
  search_dot_sum = wp.tile_reduce(wp.add, search_dot_tile)

  if tid == 0:
    ctx_search_dot_out[worldid] = search_dot_sum[0]


@cache_kernel
def _solve_beta_finalize_tiled(warn_overflow: int):
  WARN_OVERFLOW = warn_overflow

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    nv: int,
    opt_tolerance: wp.array[float],
    opt_iterations: int,
    stat_meaninertia: wp.array[float],
    # In:
    ctx_grad_in: wp.array2d[float],
    ctx_Mgrad_in: wp.array2d[float],
    ctx_prev_grad_in: wp.array2d[float],
    ctx_prev_Mgrad_in: wp.array2d[float],
    ctx_improvement_in: wp.array[float],
    ctx_grad_dot_in: wp.array[float],
    ctx_alpha_in: wp.array[float],
    ctx_done_in: wp.array[bool],
    # Data out:
    solver_niter_out: wp.array[int],
    overflow_out: wp.array[int],
    # Out:
    ctx_beta_out: wp.array[float],
    nsolving_out: wp.array[int],
    ctx_done_out: wp.array[bool],
  ):
    worldid, tid = wp.tid()

    if ctx_done_in[worldid]:
      return

    local_num = float(0.0)
    local_den = float(0.0)
    BLOCK_DIM = wp.block_dim()

    for dofid in range(tid, nv, BLOCK_DIM):
      prev_Mgrad = ctx_prev_Mgrad_in[worldid, dofid]
      num = ctx_grad_in[worldid, dofid] * (ctx_Mgrad_in[worldid, dofid] - prev_Mgrad)
      den = ctx_prev_grad_in[worldid, dofid] * prev_Mgrad
      local_num += num
      local_den += den

    beta_tile = wp.tile(wp.vec2(local_num, local_den), preserve_type=True)
    beta_sum = wp.tile_reduce(wp.add, beta_tile)
    reduced_beta = beta_sum[0]

    if tid == 0:
      ctx_beta_out[worldid] = wp.max(0.0, reduced_beta[0] / wp.max(types.MJ_MINVAL, reduced_beta[1]))

      solver_niter_out[worldid] += 1
      tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]
      meaninertia = stat_meaninertia[worldid % stat_meaninertia.shape[0]]

      grad_dot = ctx_grad_dot_in[worldid]

      alpha = ctx_alpha_in[worldid]
      improvement = _rescale(nv, meaninertia, ctx_improvement_in[worldid])
      gradient = _rescale(nv, meaninertia, wp.sqrt(grad_dot))
      done = (alpha == 0.0) or (improvement > 0.0 and improvement < tolerance) or (gradient < tolerance)
      if done or solver_niter_out[worldid] == opt_iterations:
        if not done and solver_niter_out[worldid] == opt_iterations:
          if wp.static(bool(WARN_OVERFLOW & OverflowType.ITERATIONS)):
            wp.printf(
              "solver iterations limit reached - please increase iterations beyond %u\n"
              "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.ITERATIONS (or = 0 for all)\n",
              opt_iterations,
            )
          overflow_out[worldid] = overflow_out[worldid] | OverflowType.ITERATIONS
        ctx_done_out[worldid] = True
        wp.atomic_add(nsolving_out, 0, -1)

  return kernel


@cache_kernel
def _solve_done(warn_overflow: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    nv: int,
    opt_tolerance: wp.array[float],
    opt_iterations: int,
    stat_meaninertia: wp.array[float],
    # In:
    ctx_grad_dot_in: wp.array[float],
    ctx_newton_decrement_in: wp.array[float],
    ctx_improvement_in: wp.array[float],
    ctx_alpha_in: wp.array[float],
    ctx_done_in: wp.array[bool],
    # Data out:
    solver_niter_out: wp.array[int],
    overflow_out: wp.array[int],
    # Out:
    nsolving_out: wp.array[int],
    ctx_done_out: wp.array[bool],
  ):
    worldid = wp.tid()

    if ctx_done_in[worldid]:
      return

    solver_niter_out[worldid] += 1
    tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]
    meaninertia = stat_meaninertia[worldid % stat_meaninertia.shape[0]]

    alpha = ctx_alpha_in[worldid]
    improvement = _rescale(nv, meaninertia, ctx_improvement_in[worldid])
    gradient = _rescale(nv, meaninertia, wp.sqrt(ctx_grad_dot_in[worldid]))
    model_improvement = _rescale(nv, meaninertia, 0.5 * ctx_newton_decrement_in[worldid])
    done = (
      (alpha == 0.0)
      or (improvement > 0.0 and improvement < tolerance)
      or (gradient < tolerance)
      or (model_improvement < tolerance)
    )
    if done or solver_niter_out[worldid] == opt_iterations:
      # if the solver has converged or the maximum number of iterations has been reached then
      # mark this world as done and remove it from the number of unconverged worlds
      if not done and solver_niter_out[worldid] == opt_iterations:
        if wp.static(bool(warn_overflow & OverflowType.ITERATIONS)):
          wp.printf(
            "solver iterations limit reached - please increase iterations beyond %u\n"
            "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.ITERATIONS (or = 0 for all)\n",
            opt_iterations,
          )
        overflow_out[worldid] = overflow_out[worldid] | OverflowType.ITERATIONS
      ctx_done_out[worldid] = True
      wp.atomic_add(nsolving_out, 0, -1)

  return kernel


# The linesearch runs in ray units anchored at the last gradient rebuild, and
# its bracketing arithmetic cannot resolve steps below ~eps times the gross
# derivative-term magnitude over the curvature (see _linesearch_iterative_kernel).
# A world whose tolerance sits below that floor could never terminate on the
# stale ray; rebuilding re-anchors the arithmetic at the current (much smaller)
# gradient scale. The 8 is an allowance for the reduction depth of the sums.
_ALPHA_NOISE_EPS = 8.0 * 1.1920929e-07  # 8 * float32 eps


def _use_incremental(m: types.Model) -> bool:
  """Whether constraint state changes are tracked for incremental H updates."""
  return m.opt.solver == types.SolverType.NEWTON and m.opt.cone != types.ConeType.ELLIPTIC


@wp.kernel(grid_stride=True)
def _zero_change_counters(
  # Out:
  quad_changed_count_out: wp.array[int],
  state_changed_count_out: wp.array[int],
):
  worldid = wp.tid()
  quad_changed_count_out[worldid] = 0
  state_changed_count_out[worldid] = 0


@event_scope
def _solver_iteration(
  m: types.Model,
  d: types.Data,
  ctx: SolverContext,
  nsolving: wp.array[int],
  compact: bool = False,
):
  _linesearch(m, d, ctx)

  # Incremental H is only valid for non-elliptic cones. The elliptic cone
  # path in _update_constraint_efc has early returns that skip state change
  # tracking, and the additional JTCJ Hessian term depends on Jaref which
  # changes every iteration.
  incremental = _use_incremental(m)

  if incremental:
    # Must complete before _update_constraint_efc which atomically increments.
    wp.launch(
      _zero_change_counters,
      dim=d.nworld,
      outputs=[ctx.quad_changed_count, ctx.state_changed_count],
    )

  # The tracking also enables the stable-state fast path: worlds with no state
  # flips this iteration were exactly quadratic over the step, so grad/search
  # only changed by a scalar along the same ray. Skip their qfrc/grad/
  # solve/search updates and track the scalar in ctx.grad_scale.
  _update_constraint(m, d, ctx, track_changes=incremental, stable_fast=incremental)

  if incremental:
    _update_gradient_incremental(m, d, ctx, stable_fast=incremental)
  else:
    _update_gradient(m, d, ctx, compact=compact)

  # polak-ribiere
  if m.opt.solver == types.SolverType.CG:
    wp.launch_tiled(
      _solve_beta_finalize_tiled(int(m.opt.warn_overflow)),
      dim=d.nworld,
      inputs=[
        m.nv,
        m.opt.tolerance,
        m.opt.iterations,
        m.stat.meaninertia,
        ctx.grad,
        ctx.Mgrad,
        ctx.prev_grad,
        ctx.prev_Mgrad,
        ctx.improvement,
        ctx.grad_dot,
        ctx.alpha,
        ctx.done,
      ],
      outputs=[
        d.solver_niter,
        d.overflow,
        ctx.beta,
        nsolving,
        ctx.done,
      ],
      block_dim=m.block_dim.solve_beta_accumulate,
    )
    wp.launch_tiled(
      _solve_search_update_cg_tiled,
      dim=d.nworld,
      inputs=[m.nv, ctx.grad, ctx.Mgrad, ctx.search, ctx.beta, ctx.done],
      outputs=[ctx.search, ctx.search_dot, ctx.prev_grad, ctx.prev_Mgrad],
      block_dim=m.block_dim.solve_search_update_cg,
    )

  else:
    wp.launch(
      _solve_done(int(m.opt.warn_overflow)),
      dim=d.nworld,
      inputs=[
        m.nv,
        m.opt.tolerance,
        m.opt.iterations,
        m.stat.meaninertia,
        ctx.grad_dot,
        ctx.newton_decrement,
        ctx.improvement,
        ctx.alpha,
        ctx.done,
      ],
      outputs=[d.solver_niter, d.overflow, nsolving, ctx.done],
    )


def init_context(m: types.Model, d: types.Data, ctx: SolverContext | InverseContext, grad: bool = True, compact: bool = False):
  if m.opt.integrator == types.IntegratorType.DISCRETE:
    if m.nefmdof > 0 and m.opt.solver == types.SolverType.CG:
      derivative.eff_prec_fold(m, d, out=d.efm_L)
    if grad:
      qfrc_smooth_eff = wp.zeros((d.nworld, m.nv_pad), dtype=float, device=d.qacc.device)
      wp.launch(
        _add_qfrc_smooth_eff,
        dim=(d.nworld, m.nv),
        inputs=[m.opt.enableflags, m.dof_treeid, d.tree_awake, d.qfrc_smooth, d.efm_c, d.efm_ca],
        outputs=[qfrc_smooth_eff],
      )
      d = dataclasses.replace(d, qfrc_smooth=qfrc_smooth_eff)

  # initialize some efc arrays
  wp.launch(
    _solve_init_efc,
    dim=d.nworld,
    inputs=[getattr(ctx, "nproblem", None)],
    outputs=[d.solver_niter, ctx.search_dot, ctx.done],
  )

  # jaref = d.efc_J @ d.qacc - d.efc_aref

  # if we are only using 1 thread, it makes sense to do more dofs as we can also skip the
  # init kernel. For more than 1 thread, dofs_per_thread is lower for better load balancing.

  if m.is_sparse:
    # Sparse J has few nonzeros per row, one thread handles them all.
    dofs_per_thread = m.nv
    threads_per_efc = 1
  elif m.nv > 50:
    dofs_per_thread = 20
    threads_per_efc = ceil(m.nv / dofs_per_thread)
  else:
    dofs_per_thread = 50
    threads_per_efc = ceil(m.nv / dofs_per_thread)
  # we need to clear the jaref array if we're doing atomic adds.
  if threads_per_efc > 1:
    ctx.Jaref.zero_()

  sc = _sparse_compact(ctx)
  dj = ctx.compact_d_full if sc else d
  if sc:
    dofs_per_thread = m.nv
    threads_per_efc = 1
  wp.launch(
    _solve_init_jaref_kernel(sc or m.is_sparse, m.nv, dofs_per_thread, sc, ctx.efc_world is not None),
    dim=(d.nworld, d.njmax, threads_per_efc) if ctx.efc_problem is None else (*ctx.efc_problem.shape, threads_per_efc),
    inputs=[
      d.nefc,
      d.qacc,
      dj.efc.J_rownnz,
      dj.efc.J_rowadr,
      dj.efc.J_colind,
      dj.efc.J,
      d.efc.aref,
      dj.dof_cdof,
      ctx.efc_world,
      ctx.efc_rowoff,
      ctx.efc_problem,
    ],
    outputs=[ctx.Jaref],
  )

  # Ma = M @ qacc
  _mul_m_compact_aware(m, d, ctx, d.efc.Ma, d.qacc, ctx.done)

  _update_constraint(m, d, ctx)

  if grad:
    _update_gradient(m, d, ctx, compact=compact)


# --- Small-island fast path (opt.island_solve) ---
#
# Islands with few DOFs and rows are solved completely by one thread each, before the main
# solve: Newton with the same constraint, Hessian, line search and termination rules, but on
# the island alone (MuJoCo semantics: island-sized convergence scaling). The main solve then
# keeps their DOFs fixed and recomputes their rows' forces from the result.

_SMALL_ISLAND_NV = 8
_SMALL_ISLAND_NEFC = 64
# islands up to this size are solved as worlds of the dense solver (see _solve_medium_islands)
_MEDIUM_ISLAND_NV = 32
_MEDIUM_ISLAND_NEFC = 64

_vec_small = wp.types.vector(length=_SMALL_ISLAND_NV, dtype=float)
_mat_small = wp.types.matrix(shape=(_SMALL_ISLAND_NV, _SMALL_ISLAND_NV), dtype=float)

# row categories, in MuJoCo's per-island order
_ROW_EQUALITY = 0
_ROW_FRICTION = 1
_ROW_OTHER = 2


@wp.func
def _is_small_island(cone: int, ndof: int, nrow: int) -> bool:
  # TODO(team): elliptic cones
  return cone == types.ConeType.PYRAMIDAL and ndof <= _SMALL_ISLAND_NV and nrow <= _SMALL_ISLAND_NEFC


@wp.func
def _is_medium_island(cone: int, ndof: int, nrow: int) -> bool:
  return (
    cone == types.ConeType.PYRAMIDAL
    and not _is_small_island(cone, ndof, nrow)
    and ndof <= _MEDIUM_ISLAND_NV
    and nrow <= _MEDIUM_ISLAND_NEFC
  )


@wp.func
def _row_eval_pt(category: int, alpha: float, D: float, f: float, jaref: float, jv: float) -> wp.vec3:
  """Shifted (cost, grad, hessian) of one pyramidal row, as _compute_efc_eval_pt_pyramidal."""
  if category == _ROW_OTHER:
    x = jaref + alpha * jv
    quad0 = _eval_pt_direct_cost_alpha_zero(jaref, D)
    cost0 = wp.where(jaref < 0.0, quad0, 0.0)
    if x < 0.0:
      return _eval_pt_direct_shifted(jaref, jv, D, alpha, quad0 - cost0)
    return wp.vec3(-cost0, 0.0, 0.0)
  if category == _ROW_FRICTION:
    rf = math.safe_div(f, D)
    return _shift_cost(_eval_frictionloss_pt(jaref + alpha * jv, f, rf, jv, D), _eval_frictionloss_cost(jaref, f, rf, D))
  return _eval_pt_direct_shifted(jaref, jv, D, alpha, 0.0)


@wp.func
def _row_eval_pt_alpha_zero(category: int, D: float, f: float, jaref: float, jv: float) -> wp.vec3:
  """(cost, grad, hessian) at alpha 0, as _compute_efc_eval_pt_alpha_zero_pyramidal."""
  if category == _ROW_OTHER:
    if jaref < 0.0:
      return _eval_pt_direct_alpha_zero(jaref, jv, D)
    return wp.vec3(0.0)
  if category == _ROW_FRICTION:
    return _eval_frictionloss_pt(jaref, f, math.safe_div(f, D), jv, D)
  return _eval_pt_direct_alpha_zero(jaref, jv, D)


@wp.func
def _small_row_category(row: int, nrow_eq: int, nrow_fr: int) -> int:
  if row < nrow_eq:
    return _ROW_EQUALITY
  if row < nrow_eq + nrow_fr:
    return _ROW_FRICTION
  return _ROW_OTHER


# One warp per small island. Each lane owns up to two constraint rows in registers; the island's
# inertia, Hessian and constraint force live once in shared memory and lanes add their rows into
# them atomically. DOF-space work is repeated per lane with fixed-size loops (padding beyond the
# island's DOFs is identity), which keeps the small vectors and the factorization in registers.
_SMALL_LANES = 32
_SMALL_ROWS_PER_LANE = (_SMALL_ISLAND_NEFC + _SMALL_LANES - 1) // _SMALL_LANES
_rows_J = wp.types.matrix(shape=(_SMALL_ROWS_PER_LANE, _SMALL_ISLAND_NV), dtype=float)
_rows_float = wp.types.vector(length=_SMALL_ROWS_PER_LANE, dtype=float)
_rows_int = wp.types.vector(length=_SMALL_ROWS_PER_LANE, dtype=int)


@wp.func_native(snippet="WP_TILE_SYNC();")
def _small_sync():
  pass


@wp.func
def _small_dot(x: _vec_small, y: _vec_small) -> float:
  acc = float(0.0)
  for i in range(wp.static(_SMALL_ISLAND_NV)):
    acc += x[i] * y[i]
  return acc


@wp.func
def _small_h(
  # In:
  M_s: wp.tile[float, _SMALL_ISLAND_NV, _SMALL_ISLAND_NV],
  JDJ_s: wp.tile[float, _SMALL_ISLAND_NV, _SMALL_ISLAND_NV],
  n: int,
  i: int,
  j: int,
) -> float:
  """Hessian entry (i >= j) from the shared inertia and upper-triangle J'DJ; identity padding."""
  if i < n:
    return wp.tile_extract(M_s, i, j) + wp.tile_extract(JDJ_s, j, i)
  return wp.where(i == j, 1.0, 0.0)


@wp.func
def _small_cholesky_solve(
  # In:
  M_s: wp.tile[float, _SMALL_ISLAND_NV, _SMALL_ISLAND_NV],
  JDJ_s: wp.tile[float, _SMALL_ISLAND_NV, _SMALL_ISLAND_NV],
  n: int,
  b: _vec_small,
) -> _vec_small:
  """Solve H x = b for the island Hessian read from shared memory, with fixed-size loops."""
  N = wp.static(_SMALL_ISLAND_NV)
  L = _mat_small()
  for j in range(N):
    s = _small_h(M_s, JDJ_s, n, j, j)
    for k in range(N):
      if k < j:
        s -= L[j, k] * L[j, k]
    ljj = wp.sqrt(wp.max(s, types.MJ_MINVAL))
    L[j, j] = ljj
    for i in range(N):
      if i > j:
        t = _small_h(M_s, JDJ_s, n, i, j)
        for k in range(N):
          if k < j:
            t -= L[i, k] * L[j, k]
        L[i, j] = t / ljj
  y = _vec_small()
  for i in range(N):
    t = b[i]
    for k in range(N):
      if k < i:
        t -= L[i, k] * y[k]
    y[i] = t / L[i, i]
  x = _vec_small()
  for ii in range(N):
    i = N - 1 - ii
    t = y[i]
    for k in range(N):
      if k > i:
        t -= L[k, i] * x[k]
    x[i] = t / L[i, i]
  return x


@cache_kernel
def _small_island_solve(warmstart: bool, ls_iterations: int, warn_overflow: int):
  WARMSTART = warmstart
  LS_ITERATIONS = ls_iterations
  N = _SMALL_ISLAND_NV

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    opt_tolerance: wp.array[float],
    opt_ls_tolerance: wp.array[float],
    opt_cone: int,
    opt_iterations: int,
    M_elemid: wp.array2d[int],
    # Data in:
    nisland_in: wp.array[int],
    qacc_warmstart_in: wp.array2d[float],
    M_in: wp.array2d[float],
    qfrc_smooth_in: wp.array2d[float],
    qacc_smooth_in: wp.array2d[float],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    efc_D_in: wp.array2d[float],
    efc_aref_in: wp.array2d[float],
    efc_frictionloss_in: wp.array2d[float],
    island_idofadr_in: wp.array2d[int],
    island_nv_in: wp.array2d[int],
    island_nefc_in: wp.array2d[int],
    island_ne_in: wp.array2d[int],
    island_nf_in: wp.array2d[int],
    island_iefcadr_in: wp.array2d[int],
    map_dof2idof_in: wp.array2d[int],
    map_idof2dof_in: wp.array2d[int],
    map_iefc2efc_in: wp.array2d[int],
    # Data out:
    qacc_out: wp.array2d[float],
    qfrc_constraint_out: wp.array2d[float],
    efc_force_out: wp.array2d[float],
    efc_state_out: wp.array2d[int],
    overflow_out: wp.array[int],
    # Out:
    niter_out: wp.array[int],
  ):
    worldid, islandid, lane = wp.tid()
    if islandid >= nisland_in[worldid]:
      return
    n = island_nv_in[worldid, islandid]
    nrow = island_nefc_in[worldid, islandid]
    if not _is_small_island(opt_cone, n, nrow):
      return

    ne = island_ne_in[worldid, islandid]
    nf = island_nf_in[worldid, islandid]
    idofadr = island_idofadr_in[worldid, islandid]
    iefcadr = island_iefcadr_in[worldid, islandid]
    tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]
    ls_tolerance = opt_ls_tolerance[worldid % opt_ls_tolerance.shape[0]]

    M_s = wp.tile_zeros(shape=(N, N), dtype=float, storage="shared")
    H_s = wp.tile_zeros(shape=(N, N), dtype=float, storage="shared")
    frc_s = wp.tile_zeros(shape=N, dtype=float, storage="shared")

    # island DOFs: start point, smooth force and inertia block
    qacc = _vec_small()
    qfrc_smooth = _vec_small()
    for i in range(N):
      if i < n:
        dofi = map_idof2dof_in[worldid, idofadr + i]
        if wp.static(WARMSTART):
          qacc[i] = qacc_warmstart_in[worldid, dofi]
        else:
          qacc[i] = qacc_smooth_in[worldid, dofi]
        qfrc_smooth[i] = qfrc_smooth_in[worldid, dofi]
    for e in range(lane, N * N, _SMALL_LANES):
      i = e // N
      j = e % N
      mij = float(0.0)
      if i < n and j < n:
        dofi = map_idof2dof_in[worldid, idofadr + i]
        dofj = map_idof2dof_in[worldid, idofadr + j]
        elemid = M_elemid[wp.max(dofi, dofj), wp.min(dofi, dofj)]
        if elemid >= 0:
          mij = M_in[worldid, elemid]
      wp.tile_scatter_masked(M_s, i, j, mij, True)
    _small_sync()
    # MuJoCo scales an island's convergence tests by its inertia trace instead of meaninertia * nv
    scale = float(0.0)
    Ma = _vec_small()
    for i in range(N):
      scale += wp.tile_extract(M_s, i, i)
      acc = float(0.0)
      for j in range(N):
        acc += wp.tile_extract(M_s, i, j) * qacc[j]
      Ma[i] = acc

    # this lane's rows: island-local Jacobian and row data
    rows_category = _rows_int()
    rows_J = _rows_J()
    rows_D = _rows_float()
    rows_f = _rows_float()
    rows_jaref = _rows_float()
    rows_jv = _rows_float()
    rows_force = _rows_float()
    rows_state = _rows_int()
    for s in range(wp.static(_SMALL_ROWS_PER_LANE)):
      row = lane + s * _SMALL_LANES
      if row < nrow:
        efcid = map_iefc2efc_in[worldid, iefcadr + row]
        category = _small_row_category(row, ne, nf)
        rows_category[s] = category
        rows_D[s] = efc_D_in[worldid, efcid]
        rows_f[s] = wp.where(category == _ROW_FRICTION, efc_frictionloss_in[worldid, efcid], 0.0)
        rowadr = efc_J_rowadr_in[worldid, efcid]
        for k in range(efc_J_rownnz_in[worldid, efcid]):
          adr = rowadr + k
          col = map_dof2idof_in[worldid, efc_J_colind_in[worldid, 0, adr]] - idofadr
          val = efc_J_in[worldid, 0, adr]
          for c in range(N):
            if c == col:
              rows_J[s, c] = val
        jaref = -efc_aref_in[worldid, efcid]
        for c in range(N):
          jaref += rows_J[s, c] * qacc[c]
        rows_jaref[s] = jaref

    niter = int(0)
    done = bool(False)
    alpha = float(0.0)
    improvement = float(0.0)
    while True:
      # --- Newton step: constraint forces, gradient, Hessian (see _solve_done for termination) ---
      _small_sync()  # every lane has read the previous step's shared Hessian and force
      # shared-tile writes end in a block barrier, so every lane makes the same calls and masks
      for e in range(lane, N * N, _SMALL_LANES):
        wp.tile_scatter_masked(H_s, e // N, e % N, 0.0, True)
      wp.tile_scatter_masked(frc_s, lane % N, 0.0, lane < N)
      _small_sync()
      for s in range(wp.static(_SMALL_ROWS_PER_LANE)):
        force = float(0.0)
        quadD = float(0.0)
        if lane + s * _SMALL_LANES < nrow:
          category = rows_category[s]
          res = _eval_constraint(
            category == _ROW_EQUALITY,
            category == _ROW_FRICTION,
            False,
            rows_jaref[s],
            rows_D[s],
            rows_f[s],
            0,
            -1,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
          )
          force = res[0]
          rows_force[s] = force
          rows_state[s] = int(res[1])
          if int(res[1]) == types.ConstraintState.QUADRATIC.value:
            quadD = rows_D[s]
        # inactive rows have zero Jacobian, so they add nothing
        for i in range(N):
          Ji = rows_J[s, i]
          wp.tile_scatter_add(frc_s, i, Ji * force, Ji != 0.0, True)
          for j in range(N):
            if j >= i:
              Jj = rows_J[s, j]
              wp.tile_scatter_add(H_s, i, j, Ji * quadD * Jj, Ji != 0.0 and Jj != 0.0 and quadD != 0.0, True)
      _small_sync()

      grad = _vec_small()
      for i in range(N):
        if i < n:
          grad[i] = Ma[i] - qfrc_smooth[i] - wp.tile_extract(frc_s, i)
      solution = _small_cholesky_solve(M_s, H_s, n, grad)
      search = -solution
      search_dot = _small_dot(search, search)
      grad_dot = _small_dot(grad, grad)
      newton_decrement = _small_dot(grad, solution)

      if niter > 0:
        done = (
          (alpha == 0.0)
          or (improvement > 0.0 and improvement / scale < tolerance)
          or (wp.sqrt(grad_dot) / scale < tolerance)
          or (0.5 * newton_decrement / scale < tolerance)
        )
      if done or niter >= opt_iterations:
        break

      # --- linesearch along search (see _linesearch_iterative_kernel) ---
      mv = _vec_small()
      for i in range(N):
        acc = float(0.0)
        for j in range(N):
          acc += wp.tile_extract(M_s, i, j) * search[j]
        mv[i] = acc
      local_gauss = wp.vec2(0.0)
      local_p0 = wp.vec3(0.0)
      for s in range(wp.static(_SMALL_ROWS_PER_LANE)):
        row = lane + s * _SMALL_LANES
        if row < nrow:
          jv = float(0.0)
          for c in range(N):
            jv += rows_J[s, c] * search[c]
          rows_jv[s] = jv
          if rows_category[s] == _ROW_EQUALITY:
            jvD = jv * rows_D[s]
            local_gauss += wp.vec2(jvD * rows_jaref[s], 0.5 * jv * jvD)
          else:
            local_p0 += _row_eval_pt_alpha_zero(rows_category[s], rows_D[s], rows_f[s], rows_jaref[s], jv)
      gauss_rows = wp.tile_reduce(wp.add, wp.tile(local_gauss, preserve_type=True))[0]
      p0_rows = wp.tile_reduce(wp.add, wp.tile(local_p0, preserve_type=True))[0]

      gauss1 = gauss_rows[0] + _small_dot(search, Ma - qfrc_smooth)
      gauss2 = gauss_rows[1] + 0.5 * _small_dot(search, mv)
      quad_gauss = wp.vec3(0.0, gauss1, gauss2)

      gtol = wp.max(tolerance * ls_tolerance * wp.sqrt(search_dot) * scale, 1e-6)
      p0 = wp.vec3(0.0, gauss1, 2.0 * gauss2) + p0_rows
      p0_delta = wp.vec3(0.0, p0[1], p0[2])
      q1_abs = wp.sqrt(2.0 * wp.max(p0_rows[0], 0.0) * wp.max(p0_rows[2], 0.0)) + wp.abs(gauss1)
      gtol_accept = wp.max(gtol, _ALPHA_NOISE_EPS * q1_abs)

      lo_alpha_in = -math.safe_div(p0[1], p0[2])
      local_lo_in = wp.vec3(0.0)
      for s in range(wp.static(_SMALL_ROWS_PER_LANE)):
        row = lane + s * _SMALL_LANES
        if row < nrow and rows_category[s] != _ROW_EQUALITY:
          local_lo_in += _row_eval_pt(rows_category[s], lo_alpha_in, rows_D[s], rows_f[s], rows_jaref[s], rows_jv[s])
      lo_in = _eval_pt(quad_gauss, lo_alpha_in) + wp.tile_reduce(wp.add, wp.tile(local_lo_in, preserve_type=True))[0]

      alpha = float(0.0)
      improvement = float(0.0)
      # a slope at alpha = 0 below rounding noise means the start is already the line minimum
      # (see _linesearch_iterative_kernel)
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
          lo_next_alpha = lo_alpha - math.safe_div(lo[1], lo[2])
          hi_next_alpha = hi_alpha - math.safe_div(hi[1], hi[2])
          mid_alpha = 0.5 * (lo_alpha + hi_alpha)

          local_lo = wp.vec3(0.0)
          local_hi = wp.vec3(0.0)
          local_mid = wp.vec3(0.0)
          for s in range(wp.static(_SMALL_ROWS_PER_LANE)):
            row = lane + s * _SMALL_LANES
            if row < nrow and rows_category[s] != _ROW_EQUALITY:
              category = rows_category[s]
              local_lo += _row_eval_pt(category, lo_next_alpha, rows_D[s], rows_f[s], rows_jaref[s], rows_jv[s])
              local_hi += _row_eval_pt(category, hi_next_alpha, rows_D[s], rows_f[s], rows_jaref[s], rows_jv[s])
              local_mid += _row_eval_pt(category, mid_alpha, rows_D[s], rows_f[s], rows_jaref[s], rows_jv[s])
          # one reduction for all three candidates (columns: lo, hi, mid)
          local_combined = wp.mat33(
            local_lo[0],
            local_hi[0],
            local_mid[0],
            local_lo[1],
            local_hi[1],
            local_mid[1],
            local_lo[2],
            local_hi[2],
            local_mid[2],
          )
          result = wp.tile_reduce(wp.add, wp.tile(local_combined, preserve_type=True))[0]
          gauss_lo, gauss_hi, gauss_mid = _eval_pt_3alphas(quad_gauss, lo_next_alpha, hi_next_alpha, mid_alpha)
          lo_next = gauss_lo + wp.vec3(result[0, 0], result[1, 0], result[2, 0])
          hi_next = gauss_hi + wp.vec3(result[0, 1], result[1, 1], result[2, 1])
          mid = gauss_mid + wp.vec3(result[0, 2], result[1, 2], result[2, 2])

          conv_lo = wp.abs(lo_next[1]) < gtol_accept and lo_next[0] < 0.0
          conv_hi = wp.abs(hi_next[1]) < gtol_accept and hi_next[0] < 0.0
          conv_mid = wp.abs(mid[1]) < gtol_accept and mid[0] < 0.0
          converged = conv_lo or conv_hi or conv_mid
          swap_lo = False
          swap_hi = False
          if converged:
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
            converged
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

      qacc += alpha * search
      Ma += alpha * mv
      for s in range(wp.static(_SMALL_ROWS_PER_LANE)):
        rows_jaref[s] += alpha * rows_jv[s]
      niter += 1

    # forces and states are those of the last evaluation, at the final qacc
    for i in range(N):
      if lane == i and i < n:
        dofi = map_idof2dof_in[worldid, idofadr + i]
        qacc_out[worldid, dofi] = qacc[i]
        qfrc_constraint_out[worldid, dofi] = wp.tile_extract(frc_s, i)
    for s in range(wp.static(_SMALL_ROWS_PER_LANE)):
      row = lane + s * _SMALL_LANES
      if row < nrow:
        efcid = map_iefc2efc_in[worldid, iefcadr + row]
        efc_force_out[worldid, efcid] = rows_force[s]
        efc_state_out[worldid, efcid] = rows_state[s]
    if lane == 0:
      if not done and niter == opt_iterations and opt_iterations > 0:
        if wp.static(bool(warn_overflow & OverflowType.ITERATIONS)):
          wp.printf(
            "solver iterations limit reached - please increase iterations beyond %u\n"
            "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.ITERATIONS (or = 0 for all)\n",
            opt_iterations,
          )
        wp.atomic_or(overflow_out, worldid, wp.static(OverflowType.ITERATIONS))
      wp.atomic_max(niter_out, worldid, niter)

  return kernel


# Medium islands run through the dense Newton solver as worlds of their own: each world has
# nslot = min(ntree, nv // 9) slots (a medium island has at least 9 DOFs), and slot v belongs to
# world v // nslot. Per step, each medium island's inertia, Jacobian rows (dense, island-local
# columns) and DOF vectors are gathered into its slot; empty slots have no constraint rows.


def _medium_nslot(m: types.Model) -> int:
  if m.opt.cone != types.ConeType.PYRAMIDAL:
    return 0
  return min(m.ntree, m.nv // (_SMALL_ISLAND_NV + 1))


@wp.kernel
def _medium_island_slots(
  # Model:
  opt_cone: int,
  # Data in:
  nisland_in: wp.array[int],
  island_nv_in: wp.array2d[int],
  island_nefc_in: wp.array2d[int],
  # In:
  nslot: int,
  # Out:
  slot_island_out: wp.array[int],
  island_slot_out: wp.array2d[int],
  any_medium_out: wp.array[int],
):
  """Give each medium island the next slot of its world; unused slots and other islands get -1."""
  worldid = wp.tid()
  k = int(0)
  for i in range(nisland_in[worldid]):
    slot = int(-1)
    if _is_medium_island(opt_cone, island_nv_in[worldid, i], island_nefc_in[worldid, i]):
      slot = worldid * nslot + k
      slot_island_out[slot] = i
      k += 1
    island_slot_out[worldid, i] = slot
  for j in range(k, nslot):
    slot_island_out[worldid * nslot + j] = -1
  if k > 0:
    any_medium_out[0] = 1


@wp.kernel
def _medium_slot_dofs(
  # Model:
  opt_tolerance: wp.array[float],
  opt_ls_tolerance: wp.array[float],
  M_elemid: wp.array2d[int],
  # Data in:
  qacc_warmstart_in: wp.array2d[float],
  M_in: wp.array2d[float],
  qfrc_smooth_in: wp.array2d[float],
  qacc_smooth_in: wp.array2d[float],
  island_idofadr_in: wp.array2d[int],
  island_nv_in: wp.array2d[int],
  island_nefc_in: wp.array2d[int],
  island_ne_in: wp.array2d[int],
  island_nf_in: wp.array2d[int],
  map_idof2dof_in: wp.array2d[int],
  # In:
  nslot: int,
  slot_island_in: wp.array[int],
  # Out:
  slot_qacc_warmstart_out: wp.array2d[float],
  slot_qfrc_smooth_out: wp.array2d[float],
  slot_qacc_smooth_out: wp.array2d[float],
  slot_nefc_out: wp.array[int],
  slot_ne_out: wp.array[int],
  slot_nf_out: wp.array[int],
  slot_meaninertia_out: wp.array[float],
  slot_tolerance_out: wp.array[float],
  slot_ls_tolerance_out: wp.array[float],
):
  """DOF vectors of each slot (zero on padding), slot sizes and tolerances."""
  slot, i = wp.tid()
  N = wp.static(_MEDIUM_ISLAND_NV)
  worldid = slot // nslot
  islandid = slot_island_in[slot]
  n = int(0)
  idofadr = int(0)
  if islandid >= 0:
    n = island_nv_in[worldid, islandid]
    idofadr = island_idofadr_in[worldid, islandid]

  warmstart = float(0.0)
  smooth_frc = float(0.0)
  smooth_acc = float(0.0)
  if i < n:
    dofi = map_idof2dof_in[worldid, idofadr + i]
    warmstart = qacc_warmstart_in[worldid, dofi]
    smooth_frc = qfrc_smooth_in[worldid, dofi]
    smooth_acc = qacc_smooth_in[worldid, dofi]
  slot_qacc_warmstart_out[slot, i] = warmstart
  slot_qfrc_smooth_out[slot, i] = smooth_frc
  slot_qacc_smooth_out[slot, i] = smooth_acc

  if i == 0:
    nefc = int(0)
    ne = int(0)
    nf = int(0)
    # MuJoCo scales an island's convergence tests by its inertia trace: meaninertia * nv = trace
    trace = float(N)
    if islandid >= 0:
      nefc = island_nefc_in[worldid, islandid]
      ne = island_ne_in[worldid, islandid]
      nf = island_nf_in[worldid, islandid]
      trace = float(0.0)
      for k in range(n):
        dofk = map_idof2dof_in[worldid, idofadr + k]
        trace += M_in[worldid, M_elemid[dofk, dofk]]
    slot_nefc_out[slot] = nefc
    slot_ne_out[slot] = ne
    slot_nf_out[slot] = nf
    slot_meaninertia_out[slot] = trace / float(N)
    slot_tolerance_out[slot] = opt_tolerance[worldid % opt_tolerance.shape[0]]
    slot_ls_tolerance_out[slot] = opt_ls_tolerance[worldid % opt_ls_tolerance.shape[0]]


@wp.kernel
def _medium_slot_rows(
  # Data in:
  efc_type_in: wp.array2d[int],
  efc_id_in: wp.array2d[int],
  efc_D_in: wp.array2d[float],
  efc_aref_in: wp.array2d[float],
  efc_frictionloss_in: wp.array2d[float],
  island_iefcadr_in: wp.array2d[int],
  map_iefc2efc_in: wp.array2d[int],
  # In:
  nslot: int,
  slot_island_in: wp.array[int],
  slot_nefc_in: wp.array[int],
  # Out:
  slot_efc_type_out: wp.array2d[int],
  slot_efc_id_out: wp.array2d[int],
  slot_efc_D_out: wp.array2d[float],
  slot_efc_aref_out: wp.array2d[float],
  slot_efc_frictionloss_out: wp.array2d[float],
):
  """Row data of each slot row; padding rows get zero D."""
  slot, row = wp.tid()
  if row < slot_nefc_in[slot]:
    worldid = slot // nslot
    efcid = map_iefc2efc_in[worldid, island_iefcadr_in[worldid, slot_island_in[slot]] + row]
    slot_efc_type_out[slot, row] = efc_type_in[worldid, efcid]
    slot_efc_id_out[slot, row] = efc_id_in[worldid, efcid]
    slot_efc_D_out[slot, row] = efc_D_in[worldid, efcid]
    slot_efc_aref_out[slot, row] = efc_aref_in[worldid, efcid]
    slot_efc_frictionloss_out[slot, row] = efc_frictionloss_in[worldid, efcid]
  else:
    slot_efc_D_out[slot, row] = 0.0


@wp.kernel
def _medium_init_M_J(
  # In:
  slot_nefc_in: wp.array[int],
  # Out:
  slot_M_out: wp.array3d[float],
  slot_efc_J_out: wp.array3d[float],
):
  """Identity slot inertia and zero Jacobian rows, up to a whole tile past the last row."""
  slot, row, col = wp.tid()
  N = wp.static(_MEDIUM_ISLAND_NV)
  TILE = wp.static(types.TILE_SIZE_JTDAJ_DENSE)
  if row < N:
    slot_M_out[slot, row, col] = wp.where(row == col, 1.0, 0.0)
  # the dense kernels read whole tiles of rows
  if row < (slot_nefc_in[slot] + TILE - 1) // TILE * TILE:
    slot_efc_J_out[slot, row, col] = 0.0


@wp.kernel
def _medium_gather_M(
  # Model:
  M_colind: wp.array[int],
  M_hinit_i: wp.array[int],
  # Data in:
  M_in: wp.array2d[float],
  dof_island_in: wp.array2d[int],
  island_idofadr_in: wp.array2d[int],
  map_dof2idof_in: wp.array2d[int],
  # In:
  island_slot_in: wp.array2d[int],
  # Out:
  slot_M_out: wp.array3d[float],
):
  """Scatter each stored inertia entry of a medium island (and its mirror) into its slot."""
  worldid, elemid = wp.tid()
  i = M_hinit_i[elemid]
  islandid = dof_island_in[worldid, i]
  if islandid < 0:
    return
  slot = island_slot_in[worldid, islandid]
  if slot < 0:
    return
  idofadr = island_idofadr_in[worldid, islandid]
  li = map_dof2idof_in[worldid, i] - idofadr
  lj = map_dof2idof_in[worldid, M_colind[elemid]] - idofadr
  mij = M_in[worldid, elemid]
  slot_M_out[slot, li, lj] = mij
  slot_M_out[slot, lj, li] = mij


@wp.kernel
def _medium_gather_J(
  # Data in:
  nefc_in: wp.array[int],
  efc_J_rownnz_in: wp.array2d[int],
  efc_J_rowadr_in: wp.array2d[int],
  efc_J_colind_in: wp.array3d[int],
  efc_J_in: wp.array3d[float],
  efc_island_in: wp.array2d[int],
  island_idofadr_in: wp.array2d[int],
  island_iefcadr_in: wp.array2d[int],
  map_dof2idof_in: wp.array2d[int],
  map_efc2iefc_in: wp.array2d[int],
  # In:
  island_slot_in: wp.array2d[int],
  # Out:
  slot_efc_J_out: wp.array3d[float],
):
  """Scatter each Jacobian nonzero of a medium island's row into its slot, island-local."""
  worldid, efcid, k = wp.tid()
  if efcid >= nefc_in[worldid] or k >= efc_J_rownnz_in[worldid, efcid]:
    return
  islandid = efc_island_in[worldid, efcid]
  if islandid < 0:
    return
  slot = island_slot_in[worldid, islandid]
  if slot < 0:
    return
  adr = efc_J_rowadr_in[worldid, efcid] + k
  row = map_efc2iefc_in[worldid, efcid] - island_iefcadr_in[worldid, islandid]
  col = map_dof2idof_in[worldid, efc_J_colind_in[worldid, 0, adr]] - island_idofadr_in[worldid, islandid]
  slot_efc_J_out[slot, row, col] = efc_J_in[worldid, 0, adr]


@wp.kernel
def _medium_scatter(
  # Data in:
  island_idofadr_in: wp.array2d[int],
  island_nv_in: wp.array2d[int],
  island_nefc_in: wp.array2d[int],
  island_iefcadr_in: wp.array2d[int],
  map_idof2dof_in: wp.array2d[int],
  map_iefc2efc_in: wp.array2d[int],
  # In:
  nslot: int,
  slot_island_in: wp.array[int],
  slot_qacc_in: wp.array2d[float],
  slot_qfrc_constraint_in: wp.array2d[float],
  slot_efc_force_in: wp.array2d[float],
  slot_efc_state_in: wp.array2d[int],
  slot_niter_in: wp.array[int],
  slot_overflow_in: wp.array[int],
  # Data out:
  qacc_out: wp.array2d[float],
  qfrc_constraint_out: wp.array2d[float],
  efc_force_out: wp.array2d[float],
  efc_state_out: wp.array2d[int],
  overflow_out: wp.array[int],
  # Out:
  niter_out: wp.array[int],
):
  """Copy each slot's solution to its island's DOFs and rows; fold iterations and overflow."""
  slot, i = wp.tid()
  islandid = slot_island_in[slot]
  if islandid < 0:
    return
  worldid = slot // nslot
  if i < island_nv_in[worldid, islandid]:
    dofi = map_idof2dof_in[worldid, island_idofadr_in[worldid, islandid] + i]
    qacc_out[worldid, dofi] = slot_qacc_in[slot, i]
    qfrc_constraint_out[worldid, dofi] = slot_qfrc_constraint_in[slot, i]
  if i < island_nefc_in[worldid, islandid]:
    efcid = map_iefc2efc_in[worldid, island_iefcadr_in[worldid, islandid] + i]
    efc_force_out[worldid, efcid] = slot_efc_force_in[slot, i]
    efc_state_out[worldid, efcid] = slot_efc_state_in[slot, i]
  if i == 0:
    wp.atomic_max(niter_out, worldid, slot_niter_in[slot])
    wp.atomic_or(overflow_out, worldid, slot_overflow_in[slot])


def _solve_medium_islands(m: types.Model, d: types.Data, niter: wp.array):
  """Solve medium islands with the dense Newton solver, one world per island slot."""
  nslot = _medium_nslot(m)
  if nslot == 0:
    return
  N = _MEDIUM_ISLAND_NV
  R = _MEDIUM_ISLAND_NEFC
  ns = d.nworld * nslot

  slot_island = wp.empty(ns, dtype=int)
  island_slot = wp.empty((d.nworld, m.ntree), dtype=int)
  any_medium = wp.zeros(1, dtype=int)
  wp.launch(
    _medium_island_slots,
    dim=d.nworld,
    inputs=[m.opt.cone, d.nisland, d.island_nv, d.island_nefc, nslot],
    outputs=[slot_island, island_slot, any_medium],
  )
  qacc_warmstart = wp.empty((ns, N), dtype=float)
  qfrc_smooth = wp.empty((ns, N), dtype=float)
  qacc_smooth = wp.empty((ns, N), dtype=float)
  nefc = wp.empty(ns, dtype=int)
  ne = wp.empty(ns, dtype=int)
  nf = wp.empty(ns, dtype=int)
  meaninertia = wp.empty(ns, dtype=float)
  tolerance = wp.empty(ns, dtype=float)
  ls_tolerance = wp.empty(ns, dtype=float)
  efc_type = wp.empty((ns, R), dtype=int)
  efc_id = wp.empty((ns, R), dtype=int)
  efc_D = wp.empty((ns, R), dtype=float)
  efc_aref = wp.empty((ns, R), dtype=float)
  efc_frictionloss = wp.empty((ns, R), dtype=float)
  M = wp.empty((ns, N, N), dtype=float)
  efc_J = wp.empty((ns, R, N), dtype=float)
  gc = m.opt.graph_conditional and wp.get_device().is_cuda
  opt = dataclasses.replace(m.opt, graph_conditional=gc, tolerance=tolerance, ls_tolerance=ls_tolerance)
  stat = dataclasses.replace(m.stat, meaninertia=meaninertia)
  m_slot = dataclasses.replace(m, opt=opt, stat=stat, nv=N, nv_pad=N, is_sparse=False)
  efc = dataclasses.replace(
    d.efc,
    type=efc_type,
    id=efc_id,
    J=efc_J,
    D=efc_D,
    aref=efc_aref,
    frictionloss=efc_frictionloss,
    state=wp.zeros((ns, R), dtype=int),
    force=wp.zeros((ns, R), dtype=float),
    Ma=wp.zeros((ns, N), dtype=float),
  )
  d_slot = dataclasses.replace(
    d,
    nworld=ns,
    njmax=R,
    M=M,
    nefc=nefc,
    ne=ne,
    nf=nf,
    qacc_warmstart=qacc_warmstart,
    qfrc_smooth=qfrc_smooth,
    qacc_smooth=qacc_smooth,
    qacc=wp.zeros((ns, N), dtype=float),
    qfrc_constraint=wp.zeros((ns, N), dtype=float),
    solver_niter=wp.zeros(ns, dtype=int),
    overflow=wp.zeros(ns, dtype=int),
    efc=efc,
  )
  ctx_slot = _create_solver_context(m_slot, d_slot)
  nsolving = wp.empty(1, dtype=int)

  def solve_slots():
    wp.launch(
      _medium_slot_dofs,
      dim=(ns, N),
      inputs=[
        m.opt.tolerance,
        m.opt.ls_tolerance,
        m.M_elemid,
        d.qacc_warmstart,
        d.M,
        d.qfrc_smooth,
        d.qacc_smooth,
        d.island_idofadr,
        d.island_nv,
        d.island_nefc,
        d.island_ne,
        d.island_nf,
        d.map_idof2dof,
        nslot,
        slot_island,
      ],
      outputs=[qacc_warmstart, qfrc_smooth, qacc_smooth, nefc, ne, nf, meaninertia, tolerance, ls_tolerance],
    )
    wp.launch(
      _medium_slot_rows,
      dim=(ns, R),
      inputs=[
        d.efc.type,
        d.efc.id,
        d.efc.D,
        d.efc.aref,
        d.efc.frictionloss,
        d.island_iefcadr,
        d.map_iefc2efc,
        nslot,
        slot_island,
        nefc,
      ],
      outputs=[efc_type, efc_id, efc_D, efc_aref, efc_frictionloss],
    )
    wp.launch(_medium_init_M_J, dim=(ns, R, N), inputs=[nefc], outputs=[M, efc_J])
    wp.launch(
      _medium_gather_M,
      dim=(d.nworld, m.M_colind.shape[0]),
      inputs=[m.M_colind, m.M_hinit_i, d.M, d.dof_island, d.island_idofadr, d.map_dof2idof, island_slot],
      outputs=[M],
    )
    # medium island rows have at most N nonzeros
    wp.launch(
      _medium_gather_J,
      dim=(d.nworld, d.njmax, N),
      inputs=[
        d.nefc,
        d.efc.J_rownnz,
        d.efc.J_rowadr,
        d.efc.J_colind,
        d.efc.J,
        d.efc.island,
        d.island_idofadr,
        d.island_iefcadr,
        d.map_dof2idof,
        d.map_efc2iefc,
        island_slot,
      ],
      outputs=[efc_J],
    )

    _solve(m_slot, d_slot, ctx_slot, compact=True, nsolving=nsolving)

    wp.launch(
      _medium_scatter,
      dim=(ns, max(N, R)),
      inputs=[
        d.island_idofadr,
        d.island_nv,
        d.island_nefc,
        d.island_iefcadr,
        d.map_idof2dof,
        d.map_iefc2efc,
        nslot,
        slot_island,
        d_slot.qacc,
        d_slot.qfrc_constraint,
        d_slot.efc.force,
        d_slot.efc.state,
        d_slot.solver_niter,
        d_slot.overflow,
      ],
      outputs=[d.qacc, d.qfrc_constraint, d.efc.force, d.efc.state, d.overflow, niter],
    )

  # the slots run only if some world has a medium island
  if m.opt.graph_conditional:
    wp.capture_if(any_medium, on_true=solve_slots)
  else:
    solve_slots()


@wp.func
def _island_ntile(cone: int, ndof: int, nrow: int) -> int:
  """Tiles per side of a large island's block in the pool; small and medium islands take none."""
  TILE = wp.static(types.TILE_SIZE_JTDAJ_DENSE)
  fast = _is_small_island(cone, ndof, nrow) or _is_medium_island(cone, ndof, nrow)
  return wp.where(fast, 0, (ndof + TILE - 1) // TILE)


@cache_kernel
def _island_pool_layout(ntree: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    opt_cone: int,
    # Data in:
    island_nv_in: wp.array2d[int],
    island_nefc_in: wp.array2d[int],
    # Out:
    island_tilebase_out: wp.array2d[int],
    island_ntile_out: wp.array2d[int],
    island_vbase_out: wp.array2d[int],
    island_large_out: wp.array[int],
  ):
    """Per large island: tiles per side, first pool tile and first aligned row."""
    worldid = wp.tid()
    nv_tile = wp.tile_load(island_nv_in[worldid], shape=ntree)
    nefc_tile = wp.tile_load(island_nefc_in[worldid], shape=ntree)
    cone_tile = wp.tile_full(shape=ntree, value=opt_cone, dtype=int)
    ntile = wp.tile_map(_island_ntile, cone_tile, nv_tile, nefc_tile)
    wp.tile_store(island_ntile_out[worldid], ntile)
    wp.tile_store(island_tilebase_out[worldid], wp.tile_scan_exclusive(wp.tile_map(wp.mul, ntile, ntile)))
    wp.tile_store(island_vbase_out[worldid], wp.tile_scan_exclusive(ntile) * wp.static(types.TILE_SIZE_JTDAJ_DENSE))
    if wp.tile_reduce(wp.max, ntile)[0] > 0:
      island_large_out[0] = 1

  return kernel


@cache_kernel
def _island_pool_maps(tile: int):
  TILE = tile

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Data in:
    dof_island_in: wp.array2d[int],
    island_idofadr_in: wp.array2d[int],
    map_dof2idof_in: wp.array2d[int],
    # In:
    island_tilebase_in: wp.array2d[int],
    island_ntile_in: wp.array2d[int],
    island_vbase_in: wp.array2d[int],
    # Out:
    h_row2dof_out: wp.array2d[int],
    h_dof2row_out: wp.array2d[int],
    h_dof_poolrow_out: wp.array2d[int],
    h_dof_local_out: wp.array2d[int],
  ):
    """Aligned vector row, pool row (first tile column) and local index of each large-island DOF."""
    worldid, dofid = wp.tid()
    row = int(-1)
    poolrow = int(-1)
    local = int(-1)
    islandid = dof_island_in[worldid, dofid]
    if islandid >= 0:
      ntile = island_ntile_in[worldid, islandid]
      if ntile > 0:
        local = map_dof2idof_in[worldid, dofid] - island_idofadr_in[worldid, islandid]
        row = island_vbase_in[worldid, islandid] + local
        poolrow = TILE * (island_tilebase_in[worldid, islandid] + (local // TILE) * ntile) + local % TILE
        h_row2dof_out[worldid, row] = dofid
    h_dof2row_out[worldid, dofid] = row
    h_dof_poolrow_out[worldid, dofid] = poolrow
    h_dof_local_out[worldid, dofid] = local

  return kernel


@wp.kernel
def _island_pool_init(
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
  ctx_done_in: wp.array[bool],
  # Out:
  ctx_h_out: wp.array3d[float],
):
  """Upper triangle of one row of an island block in the tile pool: M, identity on padding."""
  worldid, row = wp.tid()
  if ctx_done_in[worldid]:
    return
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


@cache_kernel
def _island_pool_cholesky(tile_size: int, max_size: int, skip_unchanged: bool, skip_noflip: bool):
  SKIP_UNCHANGED = skip_unchanged
  SKIP_NOFLIP = skip_noflip

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False, module_options={"enable_mathdx_gemm": False})
  def kernel(
    # Data in:
    nisland_in: wp.array[int],
    # In:
    ctx_done_in: wp.array[bool],
    quad_changed_count_in: wp.array[int],
    state_changed_count_in: wp.array[int],
    ctx_island_tilebase_in: wp.array2d[int],
    ctx_island_ntile_in: wp.array2d[int],
    ctx_island_vbase_in: wp.array2d[int],
    ctx_pgrad_in: wp.array3d[float],
    ctx_h_in: wp.array3d[float],
    # Out:
    ctx_hfactor_out: wp.array3d[float],
    ctx_psearch_out: wp.array3d[float],
    ctx_island_sums_out: wp.array2d[wp.vec2],
  ):
    """Factor (or reuse the factor of) one large island's block and solve its Newton step."""
    worldid, islandid = wp.tid()
    if ctx_done_in[worldid]:
      return
    # an unchanged ray keeps its previous search direction (see _update_gradient_cholesky_blocked)
    if wp.static(SKIP_NOFLIP):
      if state_changed_count_in[worldid] == 0:
        return
    if islandid >= nisland_in[worldid]:
      return
    ntile = ctx_island_ntile_in[worldid, islandid]
    if ntile == 0:
      return

    tilebase = ctx_island_tilebase_in[worldid, islandid]
    vbase = ctx_island_vbase_in[worldid, islandid]
    factorize = True
    if wp.static(SKIP_UNCHANGED):
      factorize = quad_changed_count_in[worldid] > 0
    if factorize:
      sums = wp.static(create_pool_cholesky_newton_func(tile_size, max_size, True))(
        ctx_h_in[worldid], ctx_pgrad_in[worldid], tilebase, ntile, vbase, ctx_hfactor_out[worldid], ctx_psearch_out[worldid]
      )
    else:
      sums = wp.static(create_pool_cholesky_newton_func(tile_size, max_size, False))(
        ctx_h_in[worldid], ctx_pgrad_in[worldid], tilebase, ntile, vbase, ctx_hfactor_out[worldid], ctx_psearch_out[worldid]
      )
    ctx_island_sums_out[worldid, islandid] = sums

  return kernel


@cache_kernel
def _island_pool_sums(skip_noflip: bool):
  SKIP_NOFLIP = skip_noflip

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Data in:
    nisland_in: wp.array[int],
    # In:
    ctx_done_in: wp.array[bool],
    state_changed_count_in: wp.array[int],
    ctx_island_ntile_in: wp.array2d[int],
    ctx_island_sums_in: wp.array2d[wp.vec2],
    # Out:
    ctx_search_dot_out: wp.array[float],
    ctx_newton_decrement_out: wp.array[float],
  ):
    """Sum the islands' Newton step sums in island order (deterministic)."""
    worldid = wp.tid()
    if ctx_done_in[worldid]:
      return
    if wp.static(SKIP_NOFLIP):
      if state_changed_count_in[worldid] == 0:
        return
    sums = wp.vec2(0.0)
    for i in range(nisland_in[worldid]):
      if ctx_island_ntile_in[worldid, i] > 0:
        sums += ctx_island_sums_in[worldid, i]
    ctx_search_dot_out[worldid] = sums[0]
    ctx_newton_decrement_out[worldid] = sums[1]

  return kernel


@wp.kernel
def _gather_h_rows(
  # In:
  ctx_done_in: wp.array[bool],
  h_row2dof_in: wp.array2d[int],
  vec_in: wp.array2d[float],
  # Out:
  vec_out: wp.array2d[float],
):
  worldid, row = wp.tid()
  if ctx_done_in[worldid]:
    return
  dofid = h_row2dof_in[worldid, row]
  vec_out[worldid, row] = wp.where(dofid >= 0, vec_in[worldid, wp.max(dofid, 0)], 0.0)


@wp.kernel
def _scatter_h_rows(
  # In:
  ctx_done_in: wp.array[bool],
  h_dof2row_in: wp.array2d[int],
  vec_in: wp.array2d[float],
  # Out:
  vec_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  if ctx_done_in[worldid]:
    return
  row = h_dof2row_in[worldid, dofid]
  vec_out[worldid, dofid] = wp.where(row >= 0, vec_in[worldid, wp.max(row, 0)], 0.0)


def _island_h_order(m: types.Model, d: types.Data, ctx: SolverContext):
  """Lay out each large island's Hessian block in the tile pool."""
  island.island(m, d)
  island.compute_island_mapping(m, d)
  _, nrow = _island_pool_size(m.nv, m.ntree)
  ctx.h_row2dof = wp.full((d.nworld, nrow), -1, dtype=int)
  ctx.h_dof2row = wp.empty((d.nworld, m.nv), dtype=int)
  ctx.h_dof_poolrow = wp.empty((d.nworld, m.nv), dtype=int)
  ctx.h_dof_local = wp.empty((d.nworld, m.nv), dtype=int)
  ctx.island_tilebase = wp.empty((d.nworld, m.ntree), dtype=int)
  ctx.island_ntile = wp.empty((d.nworld, m.ntree), dtype=int)
  ctx.island_vbase = wp.empty((d.nworld, m.ntree), dtype=int)
  ctx.island_sums = wp.empty((d.nworld, m.ntree), dtype=wp.vec2)
  ctx.island_large = wp.zeros(1, dtype=int)
  ctx.pgrad = wp.zeros((d.nworld, nrow), dtype=float)
  ctx.psearch = wp.zeros((d.nworld, nrow), dtype=float)
  wp.launch_tiled(
    _island_pool_layout(m.ntree),
    dim=d.nworld,
    inputs=[m.opt.cone, d.island_nv, d.island_nefc],
    outputs=[ctx.island_tilebase, ctx.island_ntile, ctx.island_vbase, ctx.island_large],
    block_dim=m.block_dim.island_scan_sizes,
  )
  wp.launch(
    _island_pool_maps(types.TILE_SIZE_JTDAJ_DENSE),
    dim=(d.nworld, m.nv),
    inputs=[d.dof_island, d.island_idofadr, d.map_dof2idof, ctx.island_tilebase, ctx.island_ntile, ctx.island_vbase],
    outputs=[ctx.h_row2dof, ctx.h_dof2row, ctx.h_dof_poolrow, ctx.h_dof_local],
  )


def _island_pool_factorize_solve(
  m: types.Model, d: types.Data, ctx: SolverContext, skip_unchanged: bool = False, skip_noflip: bool = False
):
  """Newton step with one block of threads per large island, in island-aligned rows."""
  nrow = ctx.pgrad.shape[1]
  tile = types.TILE_SIZE_JTDAJ_DENSE
  wp.launch(_gather_h_rows, dim=(d.nworld, nrow), inputs=[ctx.done, ctx.h_row2dof, ctx.grad], outputs=[ctx.pgrad])
  wp.launch_tiled(
    _island_pool_cholesky(tile, -(-m.nv // tile) * tile, skip_unchanged, skip_noflip),
    dim=(d.nworld, m.ntree),
    inputs=[
      d.nisland,
      ctx.done,
      ctx.quad_changed_count,
      ctx.state_changed_count,
      ctx.island_tilebase,
      ctx.island_ntile,
      ctx.island_vbase,
      ctx.pgrad.reshape(shape=(d.nworld, nrow, 1)),
      ctx.h,
    ],
    outputs=[ctx.hfactor, ctx.psearch.reshape(shape=(d.nworld, nrow, 1)), ctx.island_sums],
    block_dim=m.block_dim.update_gradient_cholesky_blocked,
  )
  wp.launch(
    _island_pool_sums(skip_noflip),
    dim=d.nworld,
    inputs=[d.nisland, ctx.done, ctx.state_changed_count, ctx.island_ntile, ctx.island_sums],
    outputs=[ctx.search_dot, ctx.newton_decrement],
  )
  wp.launch(_scatter_h_rows, dim=(d.nworld, m.nv), inputs=[ctx.done, ctx.h_dof2row, ctx.psearch], outputs=[ctx.search])


def _solve_small_islands(m: types.Model, d: types.Data) -> wp.array:
  """Solve small and medium islands to convergence; returns their iteration count per world."""
  niter = wp.zeros(d.nworld, dtype=int)
  warmstart = not (m.opt.disableflags & types.DisableBit.WARMSTART)
  inputs = [
    m.opt.tolerance,
    m.opt.ls_tolerance,
    m.opt.cone,
    m.opt.iterations,
    m.M_elemid,
    d.nisland,
    d.qacc_warmstart,
    d.M,
    d.qfrc_smooth,
    d.qacc_smooth,
    d.efc.J_rownnz,
    d.efc.J_rowadr,
    d.efc.J_colind,
    d.efc.J,
    d.efc.D,
    d.efc.aref,
    d.efc.frictionloss,
    d.island_idofadr,
    d.island_nv,
    d.island_nefc,
    d.island_ne,
    d.island_nf,
    d.island_iefcadr,
    d.map_dof2idof,
    d.map_idof2dof,
    d.map_iefc2efc,
  ]
  wp.launch_tiled(
    _small_island_solve(warmstart, m.opt.ls_iterations, int(m.opt.warn_overflow)),
    dim=(d.nworld, m.ntree),
    inputs=inputs,
    outputs=[d.qacc, d.qfrc_constraint, d.efc.force, d.efc.state, d.overflow, niter],
    block_dim=_SMALL_LANES,
  )
  _solve_medium_islands(m, d, niter)
  return niter


@wp.kernel
def _max_niter(
  # In:
  niter_in: wp.array[int],
  # Data out:
  solver_niter_out: wp.array[int],
):
  worldid = wp.tid()
  solver_niter_out[worldid] = wp.max(solver_niter_out[worldid], niter_in[worldid])


def _use_island_solve(m: types.Model) -> bool:
  if not m.opt.island_solve:
    return False
  supported = (
    m.opt.solver == types.SolverType.NEWTON
    and m.is_sparse
    and m.ntree > 1
    and not (m.opt.disableflags & types.DisableBit.ISLAND)
    and m.opt.integrator != types.IntegratorType.DISCRETE
    and not (m.opt.enableflags & types.EnableBit.SLEEP)
  )
  if not supported:
    raise ValueError(
      "opt.island_solve requires the Newton solver, a sparse Jacobian, more than one tree, islands enabled,"
      " a non-discrete integrator, and sleep disabled"
    )
  return True


def _island_solver():
  # imported lazily: island_solver builds on this module
  from mujoco_warp._src import island_solver

  return island_solver


@event_scope
def solve(m: types.Model, d: types.Data):
  if m.opt.enableflags & types.EnableBit.SLEEP:
    island.update_active_dofs(m, d)
    if m.opt.integrator != types.IntegratorType.DISCRETE:
      solve_compact(m, d)
      if m.ntree > 1:
        island.compute_island_mapping(m, d)
      return

  if d.njmax == 0 or m.nv == 0:
    wp.copy(d.qacc, d.qacc_smooth)
    d.solver_niter.fill_(0)
  elif _use_island_solve(m) and _island_solver().supported(m):
    _island_solver().solve(m, d)
  else:
    ctx = _create_solver_context(m, d, island=_use_island_solve(m))
    _solve(m, d, ctx)

  if (m.opt.enableflags & types.EnableBit.SLEEP) and m.opt.integrator == types.IntegratorType.DISCRETE and m.ntree > 1:
    island.compute_island_mapping(m, d)


def _solve(m: types.Model, d: types.Data, ctx: SolverContext, compact: bool = False, nsolving: wp.array | None = None):
  """Finds forces that satisfy constraints.

  nsolving (shape (1,), int) may be preallocated so the solve can run in a conditional graph body.
  """
  if m.opt.integrator == types.IntegratorType.DISCRETE:
    qfrc_smooth_eff = wp.zeros((d.nworld, m.nv_pad), dtype=float, device=d.qacc.device)
    wp.launch(
      _add_qfrc_smooth_eff,
      dim=(d.nworld, m.nv),
      inputs=[m.opt.enableflags, m.dof_treeid, d.tree_awake, d.qfrc_smooth, d.efm_c, d.efm_ca],
      outputs=[qfrc_smooth_eff],
    )
    d = dataclasses.replace(d, qfrc_smooth=qfrc_smooth_eff)

  island_solve = not compact and _use_island_solve(m)
  if island_solve:
    _island_h_order(m, d, ctx)
    # the world solve below may be skipped
    d.solver_niter.zero_()
    small_niter = _solve_small_islands(m, d)

  warmstart = not (m.opt.disableflags & types.DisableBit.WARMSTART)
  wp.launch(
    _solve_init_dof(warmstart, m.is_sparse),
    dim=(d.nworld, m.nv),
    inputs=[d.nefc, d.qacc_warmstart, d.qacc_smooth, d.dof_island, ctx.h_dof2row],
    outputs=[d.qacc, d.qfrc_constraint],
  )
  if m.opt.integrator == types.IntegratorType.DISCRETE and (m.opt.enableflags & types.EnableBit.SLEEP):
    wp.launch(
      derivative._zero_sleeping_dofs,
      dim=(d.nworld, m.nv),
      inputs=[m.dof_treeid, d.tree_awake],
      outputs=[d.qacc],
    )

  # allocated outside the world solve, which may be a conditional graph body
  if nsolving is None:
    nsolving = wp.empty(1, dtype=int)

  def world_solve():
    #  context
    init_context(m, d, ctx, grad=False, compact=compact)
    _update_gradient(m, d, ctx, compact=compact)

    if _use_incremental(m):
      # A new solve computes a new search direction: invalidate the mv/jv reuse
      # left over from the previous solve.
      ctx.search_unchanged.zero_()

    if ctx.nproblem is None:
      nsolving.fill_(d.nworld)
    else:
      wp.copy(nsolving, ctx.nproblem)

    # CG search = -Mgrad
    if m.opt.solver == types.SolverType.CG:
      wp.launch_tiled(
        _solve_init_search_cg_tiled,
        dim=d.nworld,
        inputs=[m.nv, m.opt.tolerance, m.stat.meaninertia, ctx.grad, ctx.Mgrad],
        outputs=[ctx.search, ctx.search_dot, ctx.prev_grad, ctx.prev_Mgrad, ctx.done, nsolving],
        block_dim=m.block_dim.solve_init_search_cg,
      )
    if m.opt.iterations != 0 and m.opt.graph_conditional:
      # Note: the iteration kernel (indicated by while_body) is repeatedly launched
      # as long as condition_iteration is not zero.
      # condition_iteration is a warp array of size 1 and type int, it counts the number
      # of worlds that are not converged, it becomes 0 when all worlds are converged.
      # When the number of iterations reaches m.opt.iterations, solver_niter
      # becomes zero and all worlds are marked as converged to avoid an infinite loop.
      # note: we only launch the iteration kernel if everything is not done
      wp.capture_while(nsolving, while_body=_solver_iteration, m=m, d=d, ctx=ctx, nsolving=nsolving, compact=compact)
    else:
      # This branch is mostly for when JAX is used as it is currently not compatible
      # with CUDA graph conditional.
      # It should be removed when JAX becomes compatible.
      for _ in range(m.opt.iterations):
        _solver_iteration(m, d, ctx, nsolving, compact=compact)

    # Recover qfrc_constraint (the compacted buffer when run under solve_compact):
    # the fast path leaves it stale, and the per-iteration zeroing wiped it for
    # worlds that converged early.
    if _use_incremental(m):
      wp.launch(
        _qfrc_constraint_from_grad,
        dim=(d.nworld, m.nv),
        inputs=[d.qfrc_smooth, d.efc.Ma, ctx.grad, ctx.grad_scale],
        outputs=[d.qfrc_constraint],
      )

  if island_solve and m.opt.graph_conditional:
    # islands that are all small or medium are solved already: only M qacc is left
    wp.capture_if(ctx.island_large, on_true=world_solve, on_false=lambda: support.mul_m(m, d, d.efc.Ma, d.qacc))
  else:
    world_solve()

  if island_solve:
    wp.launch(_max_niter, dim=d.nworld, inputs=[small_niter], outputs=[d.solver_niter])

  if m.opt.integrator == types.IntegratorType.DISCRETE and (m.opt.enableflags & types.EnableBit.SLEEP):
    wp.launch(
      derivative._zero_sleeping_dofs,
      dim=(d.nworld, m.nv),
      inputs=[m.dof_treeid, d.tree_awake],
      outputs=[d.qacc],
    )
    wp.launch(
      derivative._zero_sleeping_dofs,
      dim=(d.nworld, m.nv),
      inputs=[m.dof_treeid, d.tree_awake],
      outputs=[d.qfrc_constraint],
    )


# Active-DOF compaction solve (nvmax < nv).
#
# When fewer than nv DOFs are active, the active set is packed into a contiguous
# [0, ncdof) range (see island.update_active_dofs) and the dense factor/solve runs at
# size nvmax_pad instead of nv. The compacted workspace lives on Data as c* fields
# (mirroring the island-local i* fields). The constrained solve gathers into compacted
# arrays, shallow-replaces (m, d) so the stock dense Newton solver runs at nvmax_pad,
# then scatters qacc/qfrc_constraint back, freezing inactive DOFs to 0.


@wp.kernel
def _init_compact_inertia(
  # Data in:
  ncdof_in: wp.array[int],
  # Out:
  M_c_out: wp.array3d[float],
):
  worldid, i, j = wp.tid()
  val = 0.0
  if i == j and i >= ncdof_in[worldid]:
    val = 1.0
  M_c_out[worldid, i, j] = val


@wp.kernel
def _mul_m_sparse_compact(
  # Model:
  M_mulm_rowadr: wp.array[int],
  M_mulm_col: wp.array[int],
  M_mulm_madr: wp.array[int],
  # Data in:
  M_in: wp.array2d[float],
  dof_cdof_in: wp.array2d[int],
  cdof_dof_in: wp.array2d[int],
  # In:
  vec: wp.array2d[float],
  skip: wp.array[bool],
  # Out:
  res: wp.array2d[float],
):
  """Compacted res = M @ vec via the full-coordinate sparse M (no gathered cM)."""
  worldid, ci = wp.tid()

  if skip[worldid]:
    return

  dof = cdof_dof_in[worldid, ci]
  if dof < 0:
    res[worldid, ci] = 0.0
    return

  acc = float(0.0)
  start = M_mulm_rowadr[dof]
  end = M_mulm_rowadr[dof + 1]
  for k in range(start, end):
    # tree-internal columns are awake with the row; guard like _gather_M_sparse
    # in case M ever carries cross-tree entries (their gathered value is zero)
    cj = dof_cdof_in[worldid, M_mulm_col[k]]
    if cj >= 0:
      acc += M_in[worldid, M_mulm_madr[k]] * vec[worldid, cj]
  res[worldid, ci] = acc


@wp.kernel
def _mul_m_sparse_ragged(
  # Model:
  M_mulm_rowadr: wp.array[int],
  M_mulm_col: wp.array[int],
  M_mulm_madr: wp.array[int],
  # Data in:
  M_in: wp.array2d[float],
  # In:
  ctx_nproblem_in: wp.array[int],
  ctx_efc_world_in: wp.array[int],
  ctx_problem_dof_in: wp.array2d[int],
  vec: wp.array2d[float],
  skip: wp.array[bool],
  # Out:
  res: wp.array2d[float],
):
  """Res = M @ vec per problem via its world's sparse M (identity on padding DOFs).

  A problem's DOFs are whole trees in order, so column col of row dof is local i + col - dof.
  """
  problem, i = wp.tid()

  # empty problems have no DOF maps
  if problem >= ctx_nproblem_in[0] or skip[problem]:
    return

  dof = ctx_problem_dof_in[problem, i]
  if dof < 0:
    res[problem, i] = vec[problem, i]
    return

  worldid = ctx_efc_world_in[problem]
  acc = float(0.0)
  for k in range(M_mulm_rowadr[dof], M_mulm_rowadr[dof + 1]):
    acc += M_in[worldid, M_mulm_madr[k]] * vec[problem, i + M_mulm_col[k] - dof]
  res[problem, i] = acc


def _sparse_compact(ctx: SolverContext | InverseContext) -> bool:
  """Whether this solve is compact over a sparse full model (full J structures exist)."""
  return ctx.compact_d_full is not None and ctx.compact_m_full.is_sparse


def _mul_m_compact_aware(m: types.Model, d: types.Data, ctx: SolverContext | InverseContext, res, vec, skip):
  """M @ vec: full-coordinate sparse walk under compact, support.mul_m natively."""
  if m.opt.integrator == types.IntegratorType.DISCRETE:
    derivative.eff_mul_m(m, d, res, vec, skip=skip)
    return
  if ctx.problem_dof is not None:
    wp.launch(
      _mul_m_sparse_ragged,
      dim=ctx.problem_dof.shape,
      inputs=[
        m.M_mulm_rowadr,
        m.M_mulm_col,
        m.M_mulm_madr,
        d.M,
        ctx.nproblem,
        ctx.efc_world,
        ctx.problem_dof,
        vec,
        skip,
      ],
      outputs=[res],
    )
    return
  dfull = ctx.compact_d_full
  if dfull is not None:
    mfull = ctx.compact_m_full
    wp.launch(
      _mul_m_sparse_compact,
      dim=(d.nworld, m.nv),
      inputs=[
        mfull.M_mulm_rowadr,
        mfull.M_mulm_col,
        mfull.M_mulm_madr,
        dfull.M,
        dfull.dof_cdof,
        dfull.cdof_dof,
        vec,
        skip,
      ],
      outputs=[res],
    )
  else:
    support.mul_m(m, d, res, vec, skip=skip)


@wp.kernel
def _gather_M_sparse(
  # Model:
  M_rownnz: wp.array[int],
  M_rowadr: wp.array[int],
  M_colind: wp.array[int],
  # Data in:
  M_in: wp.array2d[float],
  dof_cdof_in: wp.array2d[int],
  # Out:
  M_c_out: wp.array3d[float],
):
  worldid, i = wp.tid()
  ci = dof_cdof_in[worldid, i]
  if ci < 0:
    return
  rowadr = M_rowadr[i]
  for k in range(M_rownnz[i]):
    adr = rowadr + k
    cj = dof_cdof_in[worldid, M_colind[adr]]
    if cj >= 0:
      val = M_in[worldid, adr]
      M_c_out[worldid, ci, cj] = val
      M_c_out[worldid, cj, ci] = val


@wp.kernel
def _gather_rhs_compact(
  # Data in:
  cdof_dof_in: wp.array2d[int],
  # In:
  vec_in: wp.array2d[float],
  # Out:
  rhs_out: wp.array3d[float],
):
  worldid, ci = wp.tid()
  dof = cdof_dof_in[worldid, ci]
  if dof >= 0:
    rhs_out[worldid, ci, 0] = vec_in[worldid, dof]
  else:
    rhs_out[worldid, ci, 0] = 0.0


@wp.kernel
def _scatter_solution(
  # Data in:
  dof_cdof_in: wp.array2d[int],
  # In:
  x_in: wp.array3d[float],
  # Out:
  vec_out: wp.array2d[float],
):
  worldid, i = wp.tid()
  ci = dof_cdof_in[worldid, i]
  if ci >= 0:
    vec_out[worldid, i] = x_in[worldid, ci, 0]
  else:
    vec_out[worldid, i] = 0.0  # frozen inactive DOF


@event_scope
def smooth_solve_compact(m: types.Model, d: types.Data):
  """Compacted equivalent of solve_m: qacc_smooth[active] = cM^-1 qfrc_smooth[active].

  Inactive DOFs are frozen (qacc_smooth set to 0). Reads the sparse Model inertia.
  """
  wp.launch(
    _init_compact_inertia,
    dim=(d.nworld, d.nvmax_pad, d.nvmax_pad),
    inputs=[d.ncdof],
    outputs=[d.cM],
  )
  wp.launch(
    _gather_M_sparse,
    dim=(d.nworld, m.nv),
    inputs=[m.M_rownnz, m.M_rowadr, m.M_colind, d.M, d.dof_cdof],
    outputs=[d.cM],
  )
  wp.launch(
    _gather_rhs_compact,
    dim=(d.nworld, d.nvmax_pad),
    inputs=[d.cdof_dof, d.qfrc_smooth],
    outputs=[d.crhs],
  )
  wp.launch_tiled(
    _cholesky_factorize_solve_blocked(types.TILE_SIZE_JTDAJ_DENSE, d.nvmax_pad),
    dim=d.nworld,
    inputs=[d.cM, d.crhs],
    outputs=[d.cqLD, d.cx],
    block_dim=m.block_dim.update_gradient_cholesky_blocked,
  )
  wp.launch(_scatter_solution, dim=(d.nworld, m.nv), inputs=[d.dof_cdof, d.cx], outputs=[d.qacc_smooth])


@wp.kernel
def _gather_dof_vecs_compact(
  # Data in:
  qacc_warmstart_in: wp.array2d[float],
  qfrc_smooth_in: wp.array2d[float],
  qacc_smooth_in: wp.array2d[float],
  cdof_dof_in: wp.array2d[int],
  # Out:
  qfrc_smooth_c_out: wp.array2d[float],
  qacc_smooth_c_out: wp.array2d[float],
  qacc_warmstart_c_out: wp.array2d[float],
):
  worldid, ci = wp.tid()
  dof = cdof_dof_in[worldid, ci]
  if dof >= 0:
    qfrc_smooth_c_out[worldid, ci] = qfrc_smooth_in[worldid, dof]
    qacc_smooth_c_out[worldid, ci] = qacc_smooth_in[worldid, dof]
    qacc_warmstart_c_out[worldid, ci] = qacc_warmstart_in[worldid, dof]
  else:
    qfrc_smooth_c_out[worldid, ci] = 0.0
    qacc_smooth_c_out[worldid, ci] = 0.0
    qacc_warmstart_c_out[worldid, ci] = 0.0


@wp.kernel
def _scatter_dof_vecs(
  # Data in:
  dof_cdof_in: wp.array2d[int],
  # In:
  qacc_c_in: wp.array2d[float],
  qfrc_constraint_c_in: wp.array2d[float],
  # Data out:
  qacc_out: wp.array2d[float],
  qfrc_constraint_out: wp.array2d[float],
):
  worldid, i = wp.tid()
  ci = dof_cdof_in[worldid, i]
  if ci >= 0:
    qacc_out[worldid, i] = qacc_c_in[worldid, ci]
    qfrc_constraint_out[worldid, i] = qfrc_constraint_c_in[worldid, ci]
  else:
    qacc_out[worldid, i] = 0.0
    qfrc_constraint_out[worldid, i] = 0.0


@wp.kernel
def _gather_J_sparse(
  # Data in:
  nefc_in: wp.array[int],
  dof_cdof_in: wp.array2d[int],
  # In:
  J_rownnz_in: wp.array2d[int],
  J_rowadr_in: wp.array2d[int],
  J_colind_in: wp.array3d[int],
  J_in: wp.array3d[float],
  # Out:
  J_c_out: wp.array3d[float],
):
  worldid, efcid = wp.tid()
  if efcid >= nefc_in[worldid]:
    return
  rowadr = J_rowadr_in[worldid, efcid]
  for k in range(J_rownnz_in[worldid, efcid]):
    adr = rowadr + k
    cj = dof_cdof_in[worldid, J_colind_in[worldid, 0, adr]]
    if cj >= 0:
      J_c_out[worldid, efcid, cj] = J_in[worldid, 0, adr]


@wp.kernel
def _gather_J_dense(
  # Data in:
  nefc_in: wp.array[int],
  dof_cdof_in: wp.array2d[int],
  # In:
  J_in: wp.array3d[float],
  # Out:
  J_c_out: wp.array3d[float],
):
  worldid, efcid = wp.tid()
  if efcid >= nefc_in[worldid]:
    return
  nv = dof_cdof_in.shape[1]
  for j in range(nv):
    cj = dof_cdof_in[worldid, j]
    if cj >= 0:
      J_c_out[worldid, efcid, cj] = J_in[worldid, efcid, j]


@event_scope
def solve_compact(m: types.Model, d: types.Data):
  """Run the dense Newton constraint solver in compacted DOF space.

  Gathers the active-DOF inertia, constraint Jacobian, and smooth/warmstart vectors
  into nvmax_pad-sized dense workspaces, runs the stock dense Newton solver on a
  shallow-replaced (m, d) at nvmax_pad, then scatters qacc/qfrc_constraint back.
  Inactive DOFs are frozen to 0. On the incremental Newton path the solver
  kernels read the sparse M and J directly through the compaction maps.
  """
  _compact_gather(m, d)

  # shallow-replace (m, d) so the stock dense Newton solver runs at nvmax_pad.
  # Keep graph-conditional early-exit on CUDA (matches baseline: stops at convergence
  # instead of running all iterations); fall back to the plain loop on CPU.
  nvp = d.nvmax_pad
  gc = m.opt.graph_conditional and wp.get_device().is_cuda
  opt2 = dataclasses.replace(m.opt, graph_conditional=gc, tolerance=d.ctol, ls_tolerance=d.cls_tol)
  m2 = dataclasses.replace(
    m, opt=opt2, nv=nvp, nv_pad=nvp, is_sparse=False, dof_tri_row=d.cdof_tri_row, dof_tri_col=d.cdof_tri_col
  )
  efc2 = dataclasses.replace(d.efc, J=d.cJ, Ma=d.cMa)
  d2 = dataclasses.replace(
    d,
    M=d.cM,
    qfrc_smooth=d.cqfrc_smooth,
    qacc_smooth=d.cqacc_smooth,
    qacc_warmstart=d.cqacc_warmstart,
    qacc=d.cqacc,
    qfrc_constraint=d.cqfrc_constraint,
    efc=efc2,
  )

  sctx = _create_solver_context(m2, d2)
  # compact kernels read the full-coordinate sparse structures (M, J) through
  # the compaction maps instead of dense products on gathered blocks
  sctx.compact_m_full = m
  sctx.compact_d_full = d
  _solve(m2, d2, sctx, compact=True)

  _compact_scatter(m, d)


@event_scope
def _compact_gather(m: types.Model, d: types.Data):
  nvp = d.nvmax_pad
  # gather compacted dense inertia and Jacobian only for dense models;
  # sparse models read sparse M and J directly through compaction maps
  if not m.is_sparse:
    wp.launch(
      _init_compact_inertia,
      dim=(d.nworld, nvp, nvp),
      inputs=[d.ncdof],
      outputs=[d.cM],
    )
    wp.launch(
      _gather_M_sparse,
      dim=(d.nworld, m.nv),
      inputs=[m.M_rownnz, m.M_rowadr, m.M_colind, d.M, d.dof_cdof],
      outputs=[d.cM],
    )
    d.cJ.zero_()
    wp.launch(
      _gather_J_dense,
      dim=(d.nworld, d.njmax),
      inputs=[d.nefc, d.dof_cdof, d.efc.J],
      outputs=[d.cJ],
    )
  # gather compacted DOF-space vectors in a single launch
  wp.launch(
    _gather_dof_vecs_compact,
    dim=(d.nworld, nvp),
    inputs=[
      d.qacc_warmstart,
      d.qfrc_smooth,
      d.qacc_smooth,
      d.cdof_dof,
    ],
    outputs=[
      d.cqfrc_smooth,
      d.cqacc_smooth,
      d.cqacc_warmstart,
    ],
  )


@event_scope
def _compact_scatter(m: types.Model, d: types.Data):
  # scatter results back to full DOF space (inactive frozen to 0) in one launch
  wp.launch(
    _scatter_dof_vecs,
    dim=(d.nworld, m.nv),
    inputs=[d.dof_cdof, d.cqacc, d.cqfrc_constraint],
    outputs=[d.qacc, d.qfrc_constraint],
  )

  # Refresh full d.efc.Ma = M @ qacc. The integrators (Euler/implicit damping) use Ma as
  # the RHS; the compact solve only populated the compacted Ma, so recompute it in full
  # space. Inactive DOFs have qacc=0 so their Ma is 0 and they stay frozen.
  support.mul_m(m, d, d.efc.Ma, d.qacc)
