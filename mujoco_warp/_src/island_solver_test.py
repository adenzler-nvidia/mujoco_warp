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

from mujoco_warp import ConeType
from mujoco_warp import SolverType
from mujoco_warp import test_data
from mujoco_warp._src import island_solver


def _fixture():
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
  # start away from the solution so every island iterates
  d.qacc_warmstart.zero_()
  return mjm, mjd, m, d


def _dense_J(d):
  nefc = d.nefc.numpy()[0]
  rownnz, rowadr = d.efc.J_rownnz.numpy()[0], d.efc.J_rowadr.numpy()[0]
  colind, J = d.efc.J_colind.numpy()[0, 0], d.efc.J.numpy()[0, 0]
  nv = d.qacc.shape[1]
  out = np.zeros((nefc, nv))
  for r in range(nefc):
    out[r, colind[rowadr[r] : rowadr[r] + rownnz[r]]] = J[rowadr[r] : rowadr[r] + rownnz[r]]
  return out


class IslandSolverTest(absltest.TestCase):
  def test_solve(self):
    """The island solve reaches MuJoCo's solution."""
    mjm, mjd, m, d = _fixture()
    island_solver.solve(m, d)
    np.testing.assert_allclose(d.qacc.numpy()[0], mjd.qacc, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(d.qfrc_constraint.numpy()[0], mjd.qfrc_constraint, rtol=1e-3, atol=1e-3)

  def test_hessian(self):
    """Each island's H block is M + J'DJ over its quadratic rows, after incremental updates."""
    mjm, mjd, m, d = _fixture()
    ctx = island_solver.solve(m, d)
    self.assertGreater(ctx.niter.numpy()[0].max(), 1)

    M = np.zeros((mjm.nv, mjm.nv))
    mujoco.mj_fullM(mjm, mjd, M)
    J = _dense_J(d)
    D = d.efc.D.numpy()[0]
    H, row2dof, row_quad = ctx.H.numpy()[0], ctx.row2dof.numpy()[0], ctx.row_quad.numpy()[0]
    iefc2efc = d.map_iefc2efc.numpy()[0]
    island_nv, island_nefc, island_iefcadr = (a.numpy()[0] for a in (d.island_nv, d.island_nefc, d.island_iefcadr))
    ntiles = []
    for i in range(d.nisland.numpy()[0]):
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
          np.testing.assert_allclose(entry, Hi[a, b], rtol=1e-4, atol=1e-4, err_msg=f"island {i} ({a}, {b})")
    self.assertIn(3, ntiles)

  def test_islands_converge_independently(self):
    """Islands stop at their own iteration; the world reports the slowest."""
    _, _, m, d = _fixture()
    ctx = island_solver.solve(m, d)
    niter = ctx.niter.numpy()[0, : d.nisland.numpy()[0]]
    self.assertLess(niter.min(), niter.max())
    self.assertEqual(d.solver_niter.numpy()[0], niter.max())


if __name__ == "__main__":
  absltest.main()
