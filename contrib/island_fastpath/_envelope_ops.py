"""Count blocked-Cholesky tile ops for dense, skyline and padded-island layouts.

Mirrors the loop structure of the augmented blocked Cholesky in block_cholesky.py (used by
Newton for nv > 32): tile size 16, matrix padded to round_up(nv + 1, 16), with the right-hand side
stored as a dense border in the last tile column. Hessian structure is taken from MuJoCo C:
H = M + J' D J over rows that are not SATISFIED.

Usage: uv run python contrib/island_skyline/envelope.py <xml> [--replay npz] [--nsample 8]
"""

import argparse

import mujoco
import numpy as np

TILE = 16


def hessian(m, d):
  M = np.zeros((m.nv, m.nv))
  mujoco.mj_fullM(m, d, M)
  nefc = d.nefc
  J = np.zeros((nefc, m.nv))
  if mujoco.mj_isSparse(m):
    for r in range(nefc):
      adr, nnz = d.efc_J_rowadr[r], d.efc_J_rownnz[r]
      J[r, d.efc_J_colind[adr : adr + nnz]] = d.efc_J[adr : adr + nnz]
  else:
    J[:] = d.efc_J[: nefc * m.nv].reshape(nefc, m.nv)
  active = d.efc_state[:nefc] != mujoco.mjtConstraintState.mjCNSTRSTATE_SATISFIED
  D = d.efc_D[:nefc] * active
  return M + J.T @ (D[:, None] * J)


def tile_top(H, nv_pad):
  """Per tile column: first tile row of the scalar column envelope (upper triangle)."""
  n = H.shape[0]
  T = nv_pad // TILE
  first = np.arange(nv_pad)  # padded diagonal is identity
  nz = np.abs(np.triu(H)) > 0
  for c in range(n):
    rows = np.flatnonzero(nz[: c + 1, c])
    first[c] = rows[0] if rows.size else c
  top = np.array([first[k * TILE : (k + 1) * TILE].min() // TILE for k in range(T)])
  top[T - 1] = 0  # augmented border column is dense
  return top


def tile_top_from_blocks(block_sizes, nv_pad):
  """Tile envelope implied by contiguous dense blocks (islands, then unconstrained trees)."""
  first = np.arange(nv_pad)
  adr = 0
  for n in block_sizes:
    first[adr : adr + n] = adr
    adr += n
  T = nv_pad // TILE
  top = np.array([first[k * TILE : (k + 1) * TILE].min() // TILE for k in range(T)])
  top[T - 1] = 0
  return top


def count_ops(top):
  """Tile ops of the augmented blocked factor + back solve, restricted to a tile envelope."""
  T = len(top)
  mm = chol = trsm = 0
  for k in range(T):
    mm += k - top[k]  # diagonal update
    chol += 1
    for i in range(k + 1, T):
      if top[i] > k:
        continue
      trsm += 1
      mm += k - max(top[k], top[i])
  back = sum(1 for i in range(T) for j in range(i + 1, T) if top[j] <= i)
  return dict(matmul=mm, chol=chol, trsm=trsm, backsolve=back)


def dense_ops(T, augmented=True):
  return count_ops(np.zeros(T, dtype=int))


def padded_ops(block_sizes):
  """Each block padded to whole tiles and factored densely (non-augmented: +forward solve)."""
  tot = dict(matmul=0, chol=0, trsm=0, backsolve=0)
  slots = 0
  for n in block_sizes:
    t = -(-n // TILE)
    slots += t * TILE
    o = count_ops(np.zeros(t, dtype=int))
    for key in tot:
      tot[key] += o[key]
    tot["backsolve"] += o["backsolve"]  # separate forward substitution
  tot["slots"] = slots
  return tot


def check_fill(H, perm, top, nv_pad):
  """Factor the permuted, identity-padded H and confirm U stays inside the tile envelope."""
  n = H.shape[0]
  A = np.eye(nv_pad)
  A[:n, :n] = H[np.ix_(perm, perm)]
  U = np.linalg.cholesky(A).T
  T = nv_pad // TILE
  for k in range(T):
    for j in range(top[k]):
      blk = U[j * TILE : (j + 1) * TILE, k * TILE : (k + 1) * TILE]
      if np.abs(blk).max() > 1e-9 * np.abs(U).max():
        return False
  return True


def analyze(m, d, verbose=True):
  nv = m.nv
  nv_pad = -(-(nv + 1) // TILE) * TILE
  T = nv_pad // TILE
  H = hessian(m, d)

  # island order: MuJoCo's island-local DOFs first, unconstrained DOFs after
  perm_island = np.asarray(d.map_idof2dof[:nv]).copy()
  perms = {"natural": np.arange(nv), "island": perm_island}

  res = {"dense": dense_ops(T)}
  for name, perm in perms.items():
    Hp = H[np.ix_(perm, perm)]
    top = tile_top(Hp, nv_pad)
    res[f"skyline_{name}"] = count_ops(top)
    res[f"skyline_{name}"]["fill_ok"] = check_fill(H, perm, top, nv_pad)

  # padded: one block per island plus one per unconstrained tree
  blocks = list(d.island_nv[: d.nisland])
  for t in range(m.ntree):
    if d.tree_island[t] < 0:
      blocks.append(int(m.tree_dofnum[t]))
  res["padded"] = padded_ops(blocks)
  res["skyline_blocks"] = count_ops(tile_top_from_blocks(blocks, nv_pad))

  info = dict(nv=nv, nv_pad=nv_pad, T=T, nefc=d.nefc, nisland=d.nisland, max_island_nv=max(d.island_nv[: d.nisland], default=0))
  return info, res


def main():
  p = argparse.ArgumentParser()
  p.add_argument("xml")
  p.add_argument("--replay", default=None)
  p.add_argument("--nsample", type=int, default=6)
  p.add_argument("--nstep", type=int, default=None)
  p.add_argument("--keyframe", type=int, default=0)
  args = p.parse_args()

  m = mujoco.MjModel.from_xml_path(args.xml)
  d = mujoco.MjData(m)
  if m.nkey > args.keyframe:
    mujoco.mj_resetDataKeyframe(m, d, args.keyframe)
  ctrl = np.load(args.replay)["ctrl"] if args.replay else None
  nstep = args.nstep or (len(ctrl) if ctrl is not None else 500)
  samples = set(np.linspace(0, nstep - 1, args.nsample).astype(int).tolist())

  print(f"# {args.xml}: nv={m.nv} ntree={m.ntree} jacobian={'sparse' if mujoco.mj_isSparse(m) else 'dense'}")
  hdr = f"{'step':>5} {'nefc':>5} {'nisl':>4} {'maxnv':>5} {'T':>3} | " + " | ".join(
    f"{n:>27}" for n in ["dense mm/chol/trsm/bs", "skyl-natural", "skyl-island", "skyl-island-extents", "padded (slots)"]
  )
  print(hdr)
  for step in range(nstep):
    if ctrl is not None:
      d.ctrl[:] = ctrl[step]
    mujoco.mj_step(m, d)
    if step in samples:
      mujoco.mj_forward(m, d)
      info, r = analyze(m, d)

      def f(o, extra=""):
        return f"{o['matmul']:>7}/{o['chol']:>4}/{o['trsm']:>5}/{o['backsolve']:>5}{extra}"

      line = f"{step:>5} {info['nefc']:>5} {info['nisland']:>4} {info['max_island_nv']:>5} {info['T']:>3} | "
      line += " | ".join(
        [
          f(r["dense"]) + "   ",
          f(r["skyline_natural"]) + ("   " if r["skyline_natural"]["fill_ok"] else " !!"),
          f(r["skyline_island"]) + ("   " if r["skyline_island"]["fill_ok"] else " !!"),
          f(r["skyline_blocks"]) + "   ",
          f(r["padded"]) + f" ({r['padded']['slots']})",
        ]
      )
      print(line)


if __name__ == "__main__":
  main()
