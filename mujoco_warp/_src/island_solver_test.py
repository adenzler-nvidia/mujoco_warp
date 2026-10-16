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

import mujoco
import numpy as np
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


def _fixture(dense_nv: int = island.DENSE_NV, merge: bool = True):
  """Islands of every kind: single boxes, stacks (2, 3 tiles per side), a weld, an arm, a chain."""
  pos = {
    "s0": "0 0 .049",
    "s1": "1 0 .049",
    **{f"k{i}": f"2 0 {0.049 + 0.099 * i:.3f}" for i in range(3)},
    **{f"b{i}": f"5 0 {0.049 + 0.099 * i:.3f}" for i in range(6)},
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
  mjm, mjd, m, d = test_data.fixture(
    xml=xml, keyframe=0, overrides={"opt.cone": ConeType.PYRAMIDAL, "opt.solver": SolverType.NEWTON}
  )
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


def _dense_groups(g, ngroup):
  """Groups the dense world solver takes (dense rows, not the one-tile groups of _solve_small)."""
  width = g.group_slot.numpy()[0, :ngroup, 0]
  return (width > 0) & (width != island.small_width())


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
    for _, _, m, d in (rows_fixture, _fixture()):
      m.opt.island_solve = True

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
      # dense rows for single trees up to the dense width (one-tile groups: their own width)
      idof2dof, idofadr = d.map_idof2dof.numpy()[0], g.group_idofadr.numpy()[0]
      tree_nv = m.tree_dofnum.numpy()[m.dof_treeid.numpy()[idof2dof[idofadr[:ngroup]]]]
      wide = np.where((nv[:ngroup] <= island.dense_width(m)) & (tree_nv == nv[:ngroup]), island.dense_width(m), 0)
      small = island.small_width()
      np.testing.assert_array_equal(group_slot[:ngroup, 0], np.where(nv[:ngroup] <= small, small, wide))
      widths.update(group_slot[:ngroup, 0])
      for i in range(ngroup):
        width, _, nrow, rowadr = group_slot[i]
        if width:
          np.testing.assert_array_equal(rownnz[rowadr : rowadr + nrow], 0)
          np.testing.assert_array_equal(J[rowadr : rowadr + nrow, nv[i] : width], 0.0)
    # dense rows (the chain) and sparse rows (the stacks, merged groups)
    self.assertIn(0, widths)
    self.assertGreater(max(widths), 0)

  def test_dense_width(self):
    """Dense problems are as wide as the model's largest tree of 17 to 32 DOFs (padded to 4).

    The box falls freely, so the chain is the only group: the island solver's own path is skipped.
    """
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
    # islands: the chain (18 DOFs), the box (6), or both (24); trees: the chain
    self.assertEqual(m.island_dense_nv, 24)
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
    for dense_nv, merge in ((32, True), (32, False), (16, True), (0, True)):
      with self.subTest(dense_nv=dense_nv, merge=merge):
        mjm, mjd, m, d = _fixture(dense_nv, merge)
        d.efc.Ma.fill_(np.nan)
        island_solver.solve(m, d)
        # the chain's accelerations end within about 1e-3 of MuJoCo's (atomic sums are unordered)
        np.testing.assert_allclose(d.qacc.numpy()[0], mjd.qacc, rtol=1e-3, atol=2e-3)
        np.testing.assert_allclose(d.qfrc_constraint.numpy()[0], mjd.qfrc_constraint, rtol=1e-3, atol=1e-3)
        _assert_Ma(mjm, mjd, d)
        # from a nonzero start point (M qacc at the start matters)
        mjm, mjd, m, d = _fixture(dense_nv, merge)
        d.qacc_warmstart.assign(0.5 * mjd.qacc[None])
        island_solver.solve(m, d)
        np.testing.assert_allclose(d.qacc.numpy()[0], mjd.qacc, rtol=1e-3, atol=2e-3)

  def test_hessian(self):
    """Each island's H block is M + J'DJ over its quadratic rows, after incremental updates."""
    for dense_nv, merge in ((32, True), (32, False), (0, True)):
      with self.subTest(dense_nv=dense_nv, merge=merge):
        self._check_hessian(dense_nv, merge)

  def _check_hessian(self, dense_nv, merge):
    mjm, mjd, m, d = _fixture(dense_nv, merge)
    ctx = island_solver.solve(m, d)
    self.assertGreater(ctx.niter.numpy()[0].max(), 1)

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
    dense_h, problem = ctx.dense.ctx.h.numpy(), ctx.group_problem.numpy()[0]
    idof2dof, idofadr = d.map_idof2dof.numpy()[0], g.group_idofadr.numpy()[0]
    ndense = 0
    dense = _dense_groups(g, g.ngroup.numpy()[0])
    for i in range(g.ngroup.numpy()[0]):
      if dense[i]:
        # dense groups: the dense world solver's H (upper triangle, identity padding)
        dofs = idof2dof[idofadr[i] : idofadr[i] + island_nv[i]]
        Hi = M[np.ix_(dofs, dofs)].copy()
        # convergence is measured relative to the island's inertia trace (meaninertia * nv)
        meaninertia = ctx.dense.m.stat.meaninertia.numpy()[problem[i]]
        np.testing.assert_allclose(meaninertia * island.dense_width(m), np.trace(Hi), rtol=1e-5)
        for r in range(island_iefcadr[i], island_iefcadr[i] + island_nefc[i]):
          if state[r] == mujoco.mjtConstraintState.mjCNSTRSTATE_QUADRATIC:
            Jr = J[iefc2efc[r], dofs]
            Hi += D[iefc2efc[r]] * np.outer(Jr, Jr)
        Hp = dense_h[problem[i]]
        n = len(dofs)
        np.testing.assert_allclose(np.triu(Hp[:n, :n]), np.triu(Hi), rtol=1e-4, atol=1e-4, err_msg=f"dense group {i}")
        np.testing.assert_allclose(np.diag(Hp)[n:], 1.0)
        ndense += 1
        continue
      c, tilebase, vbase = ctx.ntile.numpy()[0, i], ctx.tilebase.numpy()[0, i], ctx.vbase.numpy()[0, i]
      ntiles.append(c)
      dofs = row2dof[vbase : vbase + island_nv[i]]
      rows = range(island_iefcadr[i], island_iefcadr[i] + island_nefc[i])
      Hi = M[np.ix_(dofs, dofs)].copy()
      # convergence is measured relative to the island's inertia trace, as in MuJoCo
      np.testing.assert_allclose(ctx.scale.numpy()[0, i], np.trace(Hi), rtol=1e-5)
      for r in rows:
        if row_quad[r]:
          Jr = J[iefc2efc[r], dofs]
          Hi += D[iefc2efc[r]] * np.outer(Jr, Jr)
      for a in range(len(dofs)):
        for b in range(a, len(dofs)):
          T = island_solver._TILE
          entry = H[T * (tilebase + (a // T) * c + b // T) + a % T, b % T]
          np.testing.assert_allclose(entry, Hi[a, b], rtol=1e-4, atol=1e-4, err_msg=f"group {i} ({a}, {b})")
    # a block over one tile per side
    self.assertGreater(max(ntiles), 1)
    if dense_nv == 32:
      self.assertGreater(ndense, 0)

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
    dense = _dense_groups(g, nisland)
    for i in range(nisland):
      # dense groups are solved by the dense world solver
      if dense[i]:
        continue
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
    # groups of the island solver (dense groups are solved by the dense world solver)
    ngroup = g.ngroup.numpy()[0]
    # the island solver's iterations, or the dense world solver's of the group's problem
    dense = _dense_groups(g, ngroup)
    problem = ctx.group_problem.numpy()[0, :ngroup]
    niter = np.where(dense, ctx.dense.d.solver_niter.numpy()[np.maximum(problem, 0)], ctx.niter.numpy()[0, :ngroup])
    self.assertLess(niter.min(), niter.max())
    self.assertGreaterEqual(d.solver_niter.numpy()[0], niter.max())

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
    island_solver.solve(m, d)
    np.testing.assert_allclose(d.qacc.numpy()[0], mjd.qacc, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
  absltest.main()
