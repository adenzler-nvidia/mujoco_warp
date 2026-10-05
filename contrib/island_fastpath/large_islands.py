"""Tile-op counts for factoring the large islands left after the small-island fast path.

Layouts: dense (current: one block over all large-island DOFs), skyline (same block, loop bounds
from island boundaries), aligned (each large island padded to whole tiles and factored alone).

Usage: uv run python contrib/island_fastpath/large_islands.py <xml> [--replay npz] [--nstep N]
  [--pyramidal]
"""

import argparse

import mujoco
import numpy as np
from _envelope_ops import TILE
from _envelope_ops import count_ops
from _envelope_ops import tile_top_from_blocks

SMALL_NV, SMALL_NEFC = 8, 64


def ops(o):
  return o["matmul"] + o["chol"] + o["trsm"] + o["backsolve"]


def main():
  p = argparse.ArgumentParser()
  p.add_argument("xml")
  p.add_argument("--replay", default=None)
  p.add_argument("--nstep", type=int, default=200)
  p.add_argument("--nsample", type=int, default=3)
  p.add_argument("--pyramidal", action="store_true", help="small islands go to the fast path")
  args = p.parse_args()

  m = mujoco.MjModel.from_xml_path(args.xml)
  if args.pyramidal:
    m.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
  d = mujoco.MjData(m)
  if m.nkey:
    mujoco.mj_resetDataKeyframe(m, d, 0)
  ctrl = np.load(args.replay)["ctrl"] if args.replay else None
  nstep = min(args.nstep, len(ctrl)) if ctrl is not None else args.nstep
  samples = set(np.linspace(0, nstep - 1, args.nsample).astype(int).tolist())
  name = args.xml.split("/")[-1] + (" (pyramidal)" if args.pyramidal else "")
  print(f"# {name}: nv={m.nv}")
  for step in range(nstep):
    if ctrl is not None:
      d.ctrl[:] = ctrl[step]
    mujoco.mj_step(m, d)
    if step not in samples:
      continue
    mujoco.mj_forward(m, d)
    nv_i = d.island_nv[: d.nisland].astype(int)
    nefc_i = d.island_nefc[: d.nisland].astype(int)
    small = (nv_i <= SMALL_NV) & (nefc_i <= SMALL_NEFC) & args.pyramidal
    large = nv_i[~small].tolist()
    nl = sum(large)
    T = -(-(nl + 1) // TILE)
    dense = ops(count_ops(np.zeros(T, dtype=int)))
    sky = ops(count_ops(tile_top_from_blocks(large, T * TILE)))
    groups = [-(-(n + 1) // TILE) for n in large]
    al = [ops(count_ops(np.zeros(c, dtype=int))) for c in groups]
    print(
      f"  step {step:4d}: islands {len(nv_i)} small {int(small.sum())} large {len(large)} sizes {sorted(large, reverse=True)[:8]}"
      f"\n             large DOFs {nl} -> tiles {T}: dense {dense}  skyline {sky}  aligned {sum(al)} (crit {max(al, default=0)}, rows {sum(groups) * TILE})"
    )


if __name__ == "__main__":
  main()
