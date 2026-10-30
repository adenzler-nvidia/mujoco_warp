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
"""Tests for the island Newton solver."""

import contextlib
import itertools

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest

import mujoco_warp as mjw
from mujoco_warp import ConeType
from mujoco_warp import SolverType
from mujoco_warp import test_data
from mujoco_warp._src import io
from mujoco_warp._src import island
from mujoco_warp._src import island_solver


def _chain() -> str:
  """An 18-DOF chain on the floor: a group wider than a tile, with a full inertia matrix."""
  chain = '<body name="c0" pos="8 0 .03"><freejoint/><geom type="capsule" fromto="0 0 0 .1 0 0" size=".03"/>'
  for i in range(12):
    axis = "0 1 0" if i % 2 else "0 0 1"
    chain += f'<body pos=".1 0 0"><joint type="hinge" axis="{axis}"/><geom type="capsule" fromto="0 0 0 .1 0 0" size=".03"/>'
  return chain + "</body>" * 13


@contextlib.contextmanager
def _any_size():
  """Island-solve models of any size (island.MIN_NV)."""
  default = island.MIN_NV
  island.MIN_NV = 0
  try:
    yield
  finally:
    island.MIN_NV = default


def _fixture(dense_nv: int = island.DENSE_NV, merge: bool = True, cone: int = ConeType.PYRAMIDAL):
  """Islands of every kind: single boxes, stacks (every factorization), a weld, an arm, a chain."""
  pos = {
    "s0": "0 0 .049",
    "s1": "1 0 .049",
    **{f"k{i}": f"2 0 {0.049 + 0.099 * i:.3f}" for i in range(6)},
    **{f"b{i}": f"5 0 {0.049 + 0.099 * i:.3f}" for i in range(9)},
    **{f"t{i}": f"6 0 {0.049 + 0.099 * i:.3f}" for i in range(12)},
    "fly": "4 0 2",
    "w0": "3 0 1",
    "w1": "3.2 0 1",
  }
  bodies = "".join(
    f'<body name="{n}" pos="{p}"><freejoint/><geom type="box" size=".05 .05 .05"/></body>' for n, p in pos.items()
  )
  # 2-DOF arm with friction loss, its first hinge past its limit
  arm = (
    '<body name="arm" pos="7 0 1"><joint type="hinge" axis="0 1 0" range="-.5 -.1" frictionloss=".5"/>'
    '<geom type="capsule" fromto="0 0 0 .3 0 0" size=".02"/>'
    '<body pos=".3 0 0"><joint type="hinge" axis="0 1 0" frictionloss=".2"/>'
    '<geom type="capsule" fromto="0 0 0 .3 0 0" size=".02"/></body></body>'
  )
  # the second welded box spins, so the weld loads every DOF of the pair; the first single box
  # slides, so its contacts have a velocity
  qvel = np.zeros(18 + 2 + 6 * len(pos))
  qvel[-3:] = [1.0, 2.0, 3.0]
  qvel[20] = 1.0
  xml = f"""
  <mujoco>
    <worldbody><geom type="plane" size="10 10 .1"/>{_chain()}{arm}{bodies}</worldbody>
    <equality><weld body1="w0" body2="w1"/></equality>
    <keyframe><key qvel="{" ".join(map(str, qvel))}"/></keyframe>
  </mujoco>
  """
  mjm, mjd, m, d = test_data.fixture(xml=xml, keyframe=0, overrides={"opt.cone": cone, "opt.solver": SolverType.NEWTON})
  m.opt.island_solve = True
  # rows in island order, dense rows for groups up to dense_nv DOFs; no stale Jacobian entry may be
  # used
  d.efc.J.fill_(np.nan)
  island.slot_buffers(m, d).J.fill_(np.nan)
  default = island.DENSE_NV, island.MERGE
  island.DENSE_NV, island.MERGE = dense_nv, merge
  try:
    mjw.make_constraint(m, d)
  finally:
    island.DENSE_NV, island.MERGE = default
  # start away from the solution so every island iterates
  d.qacc_warmstart.zero_()
  return mjm, mjd, m, d


def _assert_Ma(mjm, mjd, d):
  """efc.Ma is M qacc at the solution, as the world solver leaves it."""
  M = np.zeros((mjm.nv, mjm.nv))
  mujoco.mj_fullM(mjm, mjd, M)
  np.testing.assert_allclose(d.efc.Ma.numpy()[0], M @ d.qacc.numpy()[0], rtol=1e-4, atol=1e-4)


def _cone_hessian(m, d, J, rows, dofs, jaref):
  """Curvature of the elliptic contacts in the cone among rows (MuJoCo's cone Hessian)."""
  H = np.zeros((len(dofs), len(dofs)))
  efc_type, efc_id, state, D = (a.numpy()[0] for a in (d.efc.type, d.efc.id, d.efc.state, d.efc.D))
  address, friction, dim = d.contact.efc_address.numpy(), d.contact.friction.numpy(), d.contact.dim.numpy()
  for r in rows:
    if efc_type[r] != mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC:
      continue
    con = efc_id[r]
    if address[con, 0] != r or state[r] != mujoco.mjtConstraintState.mjCNSTRSTATE_CONE:
      continue
    mu = friction[con, 0] * m.opt.impratio_invsqrt.numpy()[0]
    dm = D[r] / (mu * mu * (1.0 + mu * mu))
    tangent = [address[con, k] for k in range(1, dim[con])]
    scale = friction[con, : len(tangent)]
    u = jaref[tangent] * scale
    t = max(np.linalg.norm(u), mujoco.mjMINVAL)
    z0 = mu * J[r, dofs]
    z = scale[:, None] * J[np.ix_(tangent, dofs)]
    p = u @ z / t
    mu_n_over_t = mu * mu * jaref[r] / t
    H += dm * (
      np.outer(z0, z0)
      - mu * (np.outer(z0, p) + np.outer(p, z0))
      + mu_n_over_t * np.outer(p, p)
      + (mu * mu - mu_n_over_t) * z.T @ z
    )
  return H


def _dense_J(d):
  """World 0's Jacobian, including the rows of islands in slots (kept only there)."""
  nefc = d.nefc.numpy()[0]
  rownnz, rowadr = d.efc.J_rownnz.numpy()[0], d.efc.J_rowadr.numpy()[0]
  colind, J = d.efc.J_colind.numpy()[0, 0], d.efc.J.numpy()[0, 0]
  nv = d.qacc.shape[1]
  out = np.zeros((nefc, nv))
  for r in range(nefc):
    out[r, colind[rowadr[r] : rowadr[r] + rownnz[r]]] = J[rowadr[r] : rowadr[r] + rownnz[r]]
  slots = d.__dict__.get("_island_slots")
  if slots is not None:
    io._island_slot_rows(out, d, slots, 0)
  return out


_ROWS_XML = """
<mujoco>
  <worldbody>
    <geom type="plane" size="10 10 .1"/>
    <body pos="0 0 1"><joint name="a" type="hinge" axis="0 1 0" range="-.2 .2" frictionloss=".1"/>
      <geom type="capsule" fromto="0 0 0 .3 0 0" size=".02"/></body>
    <body pos="1 0 1"><joint name="b" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 .3 0 0" size=".02"/></body>
    <body pos="2 0 1"><joint name="c" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 .3 0 0" size=".02"/></body>
    <body pos="3 0 1"><joint name="d" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 .3 0 0" size=".02"/></body>
    <body name="e" pos="4 0 1"><freejoint/><geom type="box" size=".05 .05 .05"/></body>
    <body name="f" pos="4.2 0 1"><freejoint/><geom type="box" size=".05 .05 .05"/></body>
    <body pos="5 0 .049"><freejoint/><geom type="box" size=".05 .05 .05"/></body>
    <body pos="6 0 .049"><freejoint/><geom type="box" size=".05 .05 .05"/></body>
    <body pos="6 0 .148"><freejoint/><geom type="box" size=".05 .05 .05"/></body>
  </worldbody>
  <tendon>
    <fixed name="ab" limited="true" range="-.01 .01" frictionloss=".2"><joint joint="a" coef="1"/><joint joint="b" coef="-1"/></fixed>
  </tendon>
  <equality>
    <joint joint1="c" joint2="d"/>
    <connect body1="e" body2="f" anchor=".1 0 0"/>
  </equality>
  <keyframe><key qpos=".3 .1 0 0 4 0 1 1 0 0 0 4.2 0 1 1 0 0 0 5 0 .049 1 0 0 0 6 0 .049 1 0 0 0 6 0 .148 1 0 0 0"/></keyframe>
</mujoco>
"""


class IslandSolverTest(absltest.TestCase):
  def test_rows_in_island_order(self):
    """Island-mode constraint rows: the same rows, grouped by island, and the same islands."""
    mjm, mjd, m, d = test_data.fixture(
      xml=_ROWS_XML, keyframe=0, overrides={"opt.cone": ConeType.PYRAMIDAL, "opt.solver": SolverType.NEWTON}
    )
    self.assertTrue(m.is_sparse)

    def rows(island_solve):
      m.opt.island_solve = island_solve
      # without slots, so islands can be rediscovered from the sparse rows
      with island.no_slots():
        mjw.make_constraint(m, d)
      nefc = d.nefc.numpy()[0]
      etype, eid = d.efc.type.numpy()[0, :nefc], d.efc.id.numpy()[0, :nefc]
      order = np.lexsort((np.arange(nefc), eid, etype))
      J = _dense_J(d)
      fields = [etype, eid, d.efc.D.numpy()[0, :nefc], d.efc.aref.numpy()[0, :nefc], J]
      return [f[order] for f in fields]

    world = rows(False)
    island_rows = rows(True)
    for name, a, b in zip(("type", "id", "D", "aref", "J"), island_rows, world):
      np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6, err_msg=name)

    # same islands as discovery from the finished rows
    tree_island, nisland = d.tree_island.numpy().copy(), d.nisland.numpy().copy()
    island.island(m, d)
    np.testing.assert_array_equal(tree_island, d.tree_island.numpy())
    np.testing.assert_array_equal(nisland, d.nisland.numpy())
    # tendon (a, b), joint equality (c, d), connect (e, f), floor box, stack
    self.assertEqual(nisland[0], 5)

    # each island's rows are contiguous and ordered equality, friction, limit, contact
    nefc = d.nefc.numpy()[0]
    row_island = d.efc.island.numpy()[0, :nefc]
    self.assertTrue(np.all(np.diff(row_island) >= 0))
    rank = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3}  # MuJoCo constraint types to classes
    etype = d.efc.type.numpy()[0, :nefc]
    for i in range(nisland[0]):
      cls = [rank[t] for t in etype[row_island == i]]
      self.assertEqual(cls, sorted(cls))
      self.assertEqual(d.island_ne.numpy()[0, i], cls.count(0))
      self.assertEqual(d.island_nf.numpy()[0, i], cls.count(1))
      self.assertEqual(d.island_nefc.numpy()[0, i], len(cls))

  def test_dense_rows(self):
    """Narrow island groups write their rows only as dense rows of whole tiles, local columns."""
    rows_fixture = test_data.fixture(
      xml=_ROWS_XML, keyframe=0, overrides={"opt.cone": ConeType.PYRAMIDAL, "opt.solver": SolverType.NEWTON}
    )
    widths = set()
    default = island.MERGE
    for merge, (_, _, m, d) in itertools.product((True, False), (rows_fixture, _fixture())):
      m.opt.island_solve = True
      island.MERGE = merge

      def rows():
        nefc = d.nefc.numpy()[0]
        etype, eid = d.efc.type.numpy()[0, :nefc], d.efc.id.numpy()[0, :nefc]
        order = np.lexsort((np.arange(nefc), eid, etype))
        return _dense_J(d)[order]

      with island.no_slots():
        mjw.make_constraint(m, d)
      expect = rows()
      # every dense entry must be written (stale values from other steps are not zero)
      island.slot_buffers(m, d).J.fill_(np.nan)
      mjw.make_constraint(m, d)
      np.testing.assert_allclose(rows(), expect, atol=1e-6)

      g = island.slot_buffers(m, d)
      group_slot, J = g.group_slot.numpy()[0], g.J.numpy()[0]
      nv, rownnz = g.group_nv.numpy()[0], d.efc.J_rownnz.numpy()[0]
      ngroup = g.ngroup.numpy()[0]
      # dense rows for single trees up to the dense width
      idof2dof, idofadr = d.map_idof2dof.numpy()[0], g.group_idofadr.numpy()[0]
      tree_nv = m.tree_dofnum.numpy()[m.dof_treeid.numpy()[idof2dof[idofadr[:ngroup]]]]
      wide = np.where((nv[:ngroup] <= island.dense_width(m)) & (tree_nv == nv[:ngroup]), island.dense_width(m), 0)
      np.testing.assert_array_equal(group_slot[:ngroup, 0], wide)
      widths.update(group_slot[:ngroup, 0])
      for i in range(ngroup):
        width, _, nrow, rowadr = group_slot[i]
        if width:
          np.testing.assert_array_equal(rownnz[rowadr : rowadr + nrow], 0)
          np.testing.assert_array_equal(J[rowadr : rowadr + nrow, nv[i] : width], 0.0)
    island.MERGE = default
    # dense rows (the chain, unmerged) and sparse rows (the stacks, merged groups)
    self.assertIn(0, widths)
    self.assertGreater(max(widths), 0)

  def test_dense_width(self):
    """Dense problems are as wide as the model's largest tree of 17 to 32 DOFs (padded to 4).

    The box falls freely, so the chain is the only group: the island solver's own path is skipped.
    """
    self.enterContext(_any_size())
    xml = f"""
    <mujoco>
      <worldbody>
        <geom type="plane" size="10 10 .1"/>{_chain()}
        <body pos="0 0 2"><freejoint/><geom type="box" size=".05 .05 .05"/></body>
      </worldbody>
    </mujoco>
    """
    overrides = {
      "opt.cone": ConeType.PYRAMIDAL,
      "opt.solver": SolverType.NEWTON,
      "opt.jacobian": mujoco.mjtJacobian.mjJAC_SPARSE,
    }
    mjm, mjd, m, d = test_data.fixture(xml=xml, overrides=overrides)
    # trees: the chain (18 DOFs, padded to 20) and the box (6)
    self.assertEqual(m.island_tree_nv, 20)
    m.opt.island_solve = True
    self.assertTrue(island.island_rows(m))
    mjw.make_constraint(m, d)
    d.qacc_warmstart.zero_()
    # every DOF must be written, the box's too
    d.qacc.fill_(np.nan)
    d.qfrc_constraint.fill_(np.nan)
    d.efc.Ma.fill_(np.nan)
    island_solver.solve(m, d)
    g = island.slot_buffers(m, d)
    self.assertEqual(list(g.group_slot.numpy()[0, : g.ngroup.numpy()[0], 0]), [island.dense_width(m)])
    np.testing.assert_allclose(d.qacc.numpy()[0], mjd.qacc, rtol=1e-3, atol=2e-3)
    np.testing.assert_allclose(d.qfrc_constraint.numpy()[0], mjd.qfrc_constraint, rtol=1e-3, atol=1e-3)
    _assert_Ma(mjm, mjd, d)

  def test_solve(self):
    """The island solve reaches MuJoCo's solution, with dense, mixed and sparse islands.

    Single trees (boxes, the chain, the arm) take the dense world solver; several trees (stacks, the
    welded pair, merged groups) the block path.
    """
    for (dense_nv, merge), cone in itertools.product(
      ((32, True), (32, False), (16, True), (0, True)), (ConeType.PYRAMIDAL, ConeType.ELLIPTIC)
    ):
      with self.subTest(dense_nv=dense_nv, merge=merge, cone=cone.name):
        mjm, mjd, m, d = _fixture(dense_nv, merge, cone)
        d.efc.Ma.fill_(np.nan)
        island_solver.solve(m, d)
        # the chain's accelerations end within about 1e-3 of MuJoCo's (atomic sums are unordered)
        np.testing.assert_allclose(d.qacc.numpy()[0], mjd.qacc, rtol=1e-3, atol=2e-3)
        np.testing.assert_allclose(d.qfrc_constraint.numpy()[0], mjd.qfrc_constraint, rtol=1e-3, atol=1e-3)
        _assert_Ma(mjm, mjd, d)
        # from a nonzero start point (M qacc at the start matters)
        mjm, mjd, m, d = _fixture(dense_nv, merge, cone)
        d.qacc_warmstart.assign(0.5 * mjd.qacc[None])
        island_solver.solve(m, d)
        np.testing.assert_allclose(d.qacc.numpy()[0], mjd.qacc, rtol=1e-3, atol=2e-3)

  def test_hessian(self):
    """Each island's H block is M + J'DJ over its quadratic rows, after incremental updates."""
    niter = 0
    for dense_nv, merge, cone in (
      (32, True, ConeType.PYRAMIDAL),
      (32, False, ConeType.PYRAMIDAL),
      (0, True, ConeType.PYRAMIDAL),
      (32, True, ConeType.ELLIPTIC),
      (32, False, ConeType.ELLIPTIC),
      (0, True, ConeType.ELLIPTIC),
    ):
      with self.subTest(dense_nv=dense_nv, merge=merge, cone=cone.name):
        niter = max(niter, self._check_hessian(dense_nv, merge, cone))
    # incremental updates (niter is unwritten for dense groups)
    self.assertGreater(niter, 1)

  def _check_hessian(self, dense_nv, merge, cone):
    mjm, mjd, m, d = _fixture(dense_nv, merge, cone)
    ctx = island_solver.solve(m, d)
    elliptic = cone == ConeType.ELLIPTIC
    # elliptic: H is assembled at the final point (quadratic rows and the cone curvature)
    if elliptic:
      quadratic = d.efc.state.numpy()[0] == mujoco.mjtConstraintState.mjCNSTRSTATE_QUADRATIC

    M = np.zeros((mjm.nv, mjm.nv))
    mujoco.mj_fullM(mjm, mjd, M)
    J = _dense_J(d)
    D = d.efc.D.numpy()[0]
    H, row2dof, row_quad = ctx.H.numpy()[0], ctx.row2dof.numpy()[0], ctx.row_quad.numpy()[0]
    iefc2efc = d.map_iefc2efc.numpy()[0]
    g = island.slot_buffers(m, d)
    island_nv, island_nefc, island_iefcadr = (a.numpy()[0] for a in (g.group_nv, g.group_nefc, g.group_iefcadr))
    ntiles = []
    width = g.group_slot.numpy()[0, :, 0]
    state = d.efc.state.numpy()[0]
    ndense = 0
    for i in range(g.ngroup.numpy()[0]):
      # groups with dense rows (single trees) and sparse ones alike: a block of the pool
      ndense += width[i] > 0
      c, tilebase, vbase = ctx.ntile.numpy()[0, i], ctx.tilebase.numpy()[0, i], ctx.vbase.numpy()[0, i]
      ntiles.append(c)
      dofs = row2dof[vbase : vbase + island_nv[i]]
      rows = range(island_iefcadr[i], island_iefcadr[i] + island_nefc[i])
      Hi = M[np.ix_(dofs, dofs)].copy()
      # convergence is measured relative to the island's inertia trace, as in MuJoCo
      np.testing.assert_allclose(ctx.scale.numpy()[0, i], np.trace(Hi), rtol=1e-5)
      for r in rows:
        if quadratic[r] if elliptic else row_quad[r]:
          Jr = J[iefc2efc[r], dofs]
          Hi += D[iefc2efc[r]] * np.outer(Jr, Jr)
      if elliptic:
        Hi += _cone_hessian(m, d, J, rows, dofs, ctx.row_jaref.numpy()[0])
      for a in range(len(dofs)):
        for b in range(a, len(dofs)):
          T = island_solver._TILE
          entry = H[T * (tilebase + (a // T) * c + b // T) + a % T, b % T]
          np.testing.assert_allclose(entry, Hi[a, b], rtol=1e-4, atol=1e-4, err_msg=f"group {i} ({a}, {b})")
    # a block over one tile per side
    self.assertGreater(max(ntiles), 1)
    # merging joins the single trees with their neighbours
    if dense_nv == 32 and not merge:
      self.assertGreater(ndense, 0)
    return ctx.niter.numpy()[0, : g.ngroup.numpy()[0]].max()

  def test_gradient_scale(self):
    """The stored gradient times grad_scale is the gradient at the final point.

    Islands whose rows keep their states keep their search direction and gradient (scaled), as in
    the world solver's stable-state fast path; the welded pair has only equality rows.
    """
    _, _, m, d = _fixture()
    ctx = island_solver.solve(m, d)
    g = island.slot_buffers(m, d)
    nisland = g.ngroup.numpy()[0]
    grad_true = ctx.Ma.numpy()[0] - ctx.qfrc_smooth.numpy()[0] - ctx.frc.numpy()[0]
    grad, scale = ctx.grad.numpy()[0, :, 0], ctx.grad_scale.numpy()[0, :nisland]
    vbase, nv = ctx.vbase.numpy()[0, :nisland], g.group_nv.numpy()[0, :nisland]
    for i in range(nisland):
      rows = slice(vbase[i], vbase[i] + nv[i])
      # the gradient is a difference of force terms: its float32 noise scales with them, not with
      # the (converged, near zero) gradient
      forces = max(np.abs(t[rows]).max() for t in (ctx.Ma.numpy()[0], ctx.qfrc_smooth.numpy()[0], ctx.frc.numpy()[0]))
      atol = 1e-4 + 1e-5 * forces
      np.testing.assert_allclose(scale[i] * grad[rows], grad_true[rows], rtol=1e-3, atol=atol, err_msg=f"island {i}")
    self.assertTrue(np.any(np.abs(scale - 1.0) > 1e-3))

  def test_islands_converge_independently(self):
    """Islands stop at their own iteration; the world reports the slowest."""
    _, _, m, d = _fixture(merge=False)
    ctx = island_solver.solve(m, d)
    g = island.slot_buffers(m, d)
    niter = ctx.niter.numpy()[0, : g.ngroup.numpy()[0]]
    self.assertLess(niter.min(), niter.max())
    self.assertGreaterEqual(d.solver_niter.numpy()[0], niter.max())

  def test_nefc_overflow(self):
    """On nefc overflow the islands keep the world solver's rows (the last contacts drop)."""
    mjm, mjd, m, d = _fixture()
    nefc = int(d.nefc.numpy()[0])
    njmax = nefc // 2
    kept = []
    for island_solve in (False, True):
      m.opt.island_solve = island_solve
      # put_data checks njmax against mjd's rows: start from fresh data with the same state
      d = mjw.put_data(mjm, mujoco.MjData(mjm), nconmax=d.naconmax, njmax=njmax)
      d.qpos.assign(mjd.qpos[None])
      d.qvel.assign(mjd.qvel[None])
      mjw.forward(m, d)
      kept.append(np.unique(d.efc.type.numpy()[0, :njmax], return_counts=True))
    self.assertGreater(d.nefc.numpy()[0], njmax)
    for k1, k0 in zip(kept[1], kept[0]):
      np.testing.assert_array_equal(k1, k0)
    g = island.slot_buffers(m, d)
    n, ng = d.nisland.numpy()[0], g.ngroup.numpy()[0]
    for adr, nrow in ((d.island_iefcadr, d.island_nefc), (g.group_iefcadr, g.group_nefc)):
      adr, nrow = adr.numpy()[0, : max(n, ng)], nrow.numpy()[0, : max(n, ng)]
      self.assertLessEqual((adr + nrow)[nrow > 0].max(), njmax)
    self.assertTrue(np.isfinite(d.qacc.numpy()).all())

  def test_sleep(self):
    """Sleeping islands are skipped as by the compact solve: same states, qacc = 0, no force."""
    self.enterContext(_any_size())
    xml = """
    <mujoco>
      <option jacobian="sparse"><flag sleep="enable"/></option>
      <worldbody>
        <geom type="plane" size="5 5 .1"/>
        <body pos="0 0 .1"><freejoint/><geom type="box" size=".1 .1 .1"/></body>
        <body pos="0 0 .35"><freejoint/><geom type="box" size=".1 .1 .1"/></body>
        <body pos="1 0 .1"><freejoint/><geom type="box" size=".1 .1 .1"/></body>
        <body pos="1 0 4"><freejoint/><geom type="sphere" size=".08"/></body>
        <body pos="0 2 1">
          <joint type="hinge" axis="0 1 0" range="-10 10" limited="true" damping=".1"/>
          <geom type="capsule" fromto="0 0 0 .5 0 0" size=".05"/>
        </body>
        <body pos="2 2 .5"><joint name="spin" type="hinge" axis="0 0 1"/><geom type="box" size=".3 .05 .05"/></body>
      </worldbody>
      <actuator><motor joint="spin"/></actuator>
    </mujoco>
    """
    mjm = mujoco.MjModel.from_xml_string(xml)
    mjd = mujoco.MjData(mjm)
    mjd.ctrl[0] = 0.5
    mujoco.mj_forward(mjm, mjd)
    hinge = mjm.jnt_dofadr[4]
    runs = []
    for island_solve in (False, True):
      m = mjw.put_model(mjm)
      m.opt.island_solve = island_solve
      d = mjw.put_data(mjm, mjd, nconmax=32, njmax=128)
      self.assertEqual(island.island_rows(m), island_solve)
      states = []
      for i in range(800):
        # trees fall asleep after the solve: check those asleep before and after the step
        awake_before = d.tree_awake.numpy()[0].copy()
        mjw.step(m, d)
        if i % 10 == 9:
          states.append(d.tree_asleep.numpy()[0].copy())
          if island_solve:
            awake = d.tree_awake.numpy()[0] | awake_before
            for t in range(mjm.ntree):
              if not awake[t]:
                dofs = slice(mjm.tree_dofadr[t], mjm.tree_dofadr[t] + mjm.tree_dofnum[t])
                for a in (d.qacc, d.qfrc_constraint, d.efc.Ma):
                  np.testing.assert_array_equal(a.numpy()[0, dofs], 0.0, err_msg=f"step {i} tree {t}")
            # the limit row of the hinge, which falls asleep at its limit
            nefc = d.nefc.numpy()[0]
            limit = (d.efc.type.numpy()[0, :nefc] == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT).nonzero()[0]
            if not awake[mjm.dof_treeid[hinge]] and len(limit):
              np.testing.assert_array_equal(d.efc.force.numpy()[0, limit], 0.0, err_msg=f"step {i}")
      runs.append((np.array(states), d.qpos.numpy()[0]))
    (states0, qpos0), (states1, qpos1) = runs
    asleep0, asleep1 = states0 >= 0, states1 >= 0
    # a ball wakes a sleeping box mid-run; both then fall asleep again (rounding shifts when trees
    # cross the sleep threshold, so compare the settled end)
    self.assertTrue((asleep1[:-1] & ~asleep1[1:]).any())
    np.testing.assert_array_equal(asleep1[-1], asleep0[-1])
    self.assertEqual(asleep1[-1].sum(), mjm.ntree - 1)
    np.testing.assert_allclose(qpos1, qpos0, atol=1e-3)

  def test_graph_capture(self):
    """The island solve replays from a graph capture (its iterations are a conditional loop)."""
    mjm, mjd, m, _ = _fixture()
    self.assertTrue(island.island_rows(m))
    # one step from a cold start, so the islands iterate (atomics make several steps drift apart)
    mjd.qacc_warmstart[:] = 0.0
    qacc, niter = [], []
    for capture in (False, True):
      d = mjw.put_data(mjm, mjd)
      if capture:
        with wp.ScopedCapture() as cap:
          mjw.step(m, d)
        wp.capture_launch(cap.graph)
      else:
        mjw.step(m, d)
      qacc.append(d.qacc.numpy()[0])
      niter.append(d.solver_niter.numpy()[0])
    self.assertGreater(niter[0], 1)
    self.assertEqual(niter[1], niter[0])
    np.testing.assert_allclose(qacc[1], qacc[0], rtol=1e-5, atol=1e-5 * np.abs(qacc[0]).max())

  def test_fallback(self):
    """Models the island solver does not apply to are solved by the world solver."""
    # small worlds (one group), a dense Jacobian, deterministic rows, no qacc_smooth
    cases = ((2, "sparse", 0, True), (8, "dense", 0, True), (8, "sparse", 1, True), (8, "sparse", 0, False))
    for nbox, jacobian, deterministic, run_qacc_smooth in cases:
      boxes = "".join(f'<body pos="{i} 0 .099"><freejoint/><geom type="box" size=".1 .1 .1"/></body>' for i in range(nbox))
      xml = f"""
      <mujoco>
        <option jacobian="{jacobian}"/>
        <worldbody><geom type="plane" size="10 10 .1"/>{boxes}</worldbody>
      </mujoco>
      """
      results = []
      for island_solve in (True, False):
        _, _, m, d = test_data.fixture(xml=xml)
        m.opt.island_solve = island_solve
        if deterministic or not run_qacc_smooth:
          self.assertEqual(island.island_rows(m), island_solve)
          m.opt.deterministic = int(mjw.DeterminismType.CONSTRAINT) if deterministic else 0
          m.opt.run_qacc_smooth = run_qacc_smooth
        self.assertFalse(island.island_rows(m))
        mjw.step(m, d)
        results.append(d.qacc.numpy()[0])
      # the same solve up to the order of atomic sums
      np.testing.assert_allclose(results[0], results[1], rtol=1e-5, atol=1e-6)

  def test_njmax_nnz_overflow(self):
    """Island rows past their sparse storage stay empty: no stale layout, as rows in world order."""
    self.enterContext(_any_size())
    self.enterContext(island.no_slots())
    bodies = "".join(f'<body pos="{i} 0 1"><freejoint/><geom size=".1"/></body>' for i in range(1, 5))
    xml = f"""
    <mujoco>
      <option jacobian="sparse"/>
      <worldbody>
        <geom type="plane" size="10 10 .01"/>
        <body pos="0 0 1"><joint range="-.1 .1"/><geom size=".05" contype="0" conaffinity="0"/></body>
        {bodies}
      </worldbody>
      <keyframe><key qpos=".5  1 0 .09 1 0 0 0  2 0 .09 1 0 0 0  3 0 .09 1 0 0 0  4 0 .09 1 0 0 0"/></keyframe>
    </mujoco>
    """
    for cone, overflow in itertools.product((ConeType.PYRAMIDAL, ConeType.ELLIPTIC), (True, False)):
      mjm, mjd, m, d = test_data.fixture(
        xml=xml, overrides={"opt.cone": cone, "opt.island_solve": True}, njmax_nnz=30 if overflow else None
      )
      self.assertTrue(island.island_rows(m))
      mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
      d.qpos.assign(mjd.qpos[None])
      # stale layout from a previous step: rows covering the whole storage
      d.efc.J_rowadr.fill_(0)
      d.efc.J_rownnz.fill_(mjm.nv)
      d.overflow.zero_()
      mjw.fwd_position(m, d)

      nrow = min(d.nefc.numpy()[0], d.njmax)
      adr, nnz = d.efc.J_rowadr.numpy()[0, :nrow], d.efc.J_rownnz.numpy()[0, :nrow]
      self.assertEqual(bool(d.overflow.numpy()[0] & int(mjw.OverflowType.NJMAX_NNZ)), overflow)
      np.testing.assert_array_less(-1, adr)
      np.testing.assert_array_less(adr + nnz, d.njmax_nnz + 1)
      kept = nnz > 0
      order = np.argsort(adr[kept])
      ends, starts = (adr + nnz)[kept][order], adr[kept][order]
      np.testing.assert_array_less(ends[:-1], starts[1:] + 1, err_msg="overlapping rows")
      contact = d.efc.type.numpy()[0, :nrow] >= int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)
      # contacts keep all their rows: those without storage are empty
      self.assertEqual(set(nnz[contact]), {0, 6} if overflow else {6})
      colind = d.efc.J_colind.numpy()[0, 0]
      for a, n in zip(adr, nnz):
        self.assertTrue(np.all((colind[a : a + n] >= 0) & (colind[a : a + n] < mjm.nv)))

  def test_noslip_solves_worlds(self):
    """Noslip reads the Jacobian rows dense groups leave empty: it keeps the world solver."""
    results = []
    for island_solve in (True, False):
      mjm, mjd, m, d = test_data.fixture(xml=_ROWS_XML, keyframe=0, overrides={"opt.noslip_iterations": 5})
      m.opt.island_solve = island_solve
      self.assertFalse(island.island_rows(m))
      mjw.step(m, d)
      results.append(d.qacc.numpy()[0])
    np.testing.assert_allclose(results[0], results[1], rtol=1e-4, atol=1e-3)

  def test_island_groups(self):
    """Consecutive islands share a group up to MERGE_NV DOFs; the solution is unchanged."""
    mjm, mjd, m, d = _fixture()
    g = island.slot_buffers(m, d)
    nv, group = d.island_nv.numpy()[0], g.island_group.numpy()[0]
    nisland, ngroup = d.nisland.numpy()[0], g.ngroup.numpy()[0]
    self.assertLess(ngroup, nisland)
    group_nv = g.group_nv.numpy()[0, :ngroup]
    for k in range(ngroup):
      members = nv[:nisland][group[:nisland] == k]
      self.assertEqual(members.sum(), group_nv[k])
      if len(members) > 1:
        self.assertLessEqual(group_nv[k], island.MERGE_NV)
    # islands with equality or friction rows join groups too (the arm, the welded pair)
    ne_nf = d.island_ne.numpy()[0, :nisland] + d.island_nf.numpy()[0, :nisland]
    first = np.r_[True, group[1:nisland] != group[: nisland - 1]]
    self.assertTrue(np.any((ne_nf > 0) & ~first))
    island_solver.solve(m, d)
    np.testing.assert_allclose(d.qacc.numpy()[0], mjd.qacc, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
  absltest.main()
