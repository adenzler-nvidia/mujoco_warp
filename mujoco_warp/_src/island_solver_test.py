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


def _fixture(dense_nv: int = island.DENSE_NV, merge: bool = True):
  """Islands of every kind: single boxes, stacks (2 and 3 tiles per side), a weld, an arm."""
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
  # the second welded box spins, so the weld loads every DOF of the pair
  qvel = np.zeros(2 + 6 * len(pos))
  qvel[-3:] = [1.0, 2.0, 3.0]
  xml = f"""
  <mujoco>
    <worldbody><geom type="plane" size="10 10 .1"/>{arm}{bodies}</worldbody>
    <equality><weld body1="w0" body2="w1"/></equality>
    <keyframe><key qvel="{" ".join(map(str, qvel))}"/></keyframe>
  </mujoco>
  """
  mjm, mjd, m, d = test_data.fixture(
    xml=xml, keyframe=0, overrides={"opt.cone": ConeType.PYRAMIDAL, "opt.solver": SolverType.NEWTON}
  )
  m.opt.island_solve = True
  # rows in island order, dense blocks for islands up to dense_nv DOFs; no stale Jacobian entry
  # may be used
  d.efc.J.fill_(np.nan)
  default = island.DENSE_NV, island.MERGE
  island.DENSE_NV, island.MERGE = dense_nv, merge
  try:
    mjw.make_constraint(m, d)
  finally:
    island.DENSE_NV, island.MERGE = default
  # start away from the solution so every island iterates
  d.qacc_warmstart.zero_()
  return mjm, mjd, m, d


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
    """Narrow island groups store their rows as one dense block of whole tiles, local columns."""
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
      # every entry of a block must be written (stale values from other steps are not zero)
      d.efc.J.fill_(np.nan)
      mjw.make_constraint(m, d)
      np.testing.assert_allclose(rows(), expect, atol=1e-6)

      g = island.slot_buffers(m, d)
      group_slot, J = g.group_slot.numpy()[0], d.efc.J.numpy()[0, 0]
      nv, rownnz = g.group_nv.numpy()[0], d.efc.J_rownnz.numpy()[0]
      ngroup = g.ngroup.numpy()[0]
      np.testing.assert_array_equal(group_slot[:ngroup, 0], (nv[:ngroup] + 15) // 16 * 16)
      widths.update(group_slot[:ngroup, 0])
      for i in range(ngroup):
        width, base, nrow, rowadr = group_slot[i]
        np.testing.assert_array_equal(rownnz[rowadr : rowadr + nrow], 0)
        block = J[base : base + width * nrow].reshape(width // 16, nrow, 16).transpose(1, 0, 2).reshape(nrow, width)
        np.testing.assert_array_equal(block[:, nv[i] :], 0.0)
    # one, two and three tiles
    self.assertEqual(sorted(widths), [16, 32, 48])

  def test_solve(self):
    """The island solve reaches MuJoCo's solution, with dense, mixed and sparse islands.

    Without merging, single boxes, the arm and the welded pair are one-tile groups (_solve_small).
    """
    for dense_nv, merge in ((64, True), (64, False), (32, True), (0, True)):
      with self.subTest(dense_nv=dense_nv, merge=merge):
        mjm, mjd, m, d = _fixture(dense_nv, merge)
        island_solver.solve(m, d)
        np.testing.assert_allclose(d.qacc.numpy()[0], mjd.qacc, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(d.qfrc_constraint.numpy()[0], mjd.qfrc_constraint, rtol=1e-3, atol=1e-3)

  def test_hessian(self):
    """Each island's H block is M + J'DJ over its quadratic rows, after incremental updates."""
    for dense_nv, merge in ((64, True), (64, False), (0, True)):
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
    for i in range(g.ngroup.numpy()[0]):
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
          entry = H[16 * (tilebase + (a // 16) * c + b // 16) + a % 16, b % 16]
          np.testing.assert_allclose(entry, Hi[a, b], rtol=1e-4, atol=1e-4, err_msg=f"group {i} ({a}, {b})")
    self.assertIn(3, ntiles)

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
      np.testing.assert_allclose(scale[i] * grad[rows], grad_true[rows], rtol=1e-3, atol=1e-4, err_msg=f"island {i}")
    self.assertTrue(np.any(np.abs(scale - 1.0) > 1e-3))

  def test_islands_converge_independently(self):
    """Islands stop at their own iteration; the world reports the slowest."""
    _, _, m, d = _fixture(merge=False)
    ctx = island_solver.solve(m, d)
    niter = ctx.niter.numpy()[0, : d.nisland.numpy()[0]]
    self.assertLess(niter.min(), niter.max())
    self.assertEqual(d.solver_niter.numpy()[0], niter.max())

  def test_island_groups(self):
    """Consecutive islands share a group where that saves a tile; the solution is unchanged."""
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
        # merged only where the group takes fewer tiles than its islands
        self.assertLess((group_nv[k] + 15) // 16, ((members + 15) // 16).sum())
    island_solver.solve(m, d)
    np.testing.assert_allclose(d.qacc.numpy()[0], mjd.qacc, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
  absltest.main()
