# Copyright 2026 The Newton Developers
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

"""Reverse A = L.T D L for MuJoCo ancestor-pattern mass matrices.

Ancestor rows are CSR prefixes of descendant rows, so reverse elimination introduces no fill.
Sections place ancestor rows before owned rows. The upward pass eliminates owned rows and sends
only their matrix/RHS contributions upward; original rows are loaded by their owning section.
The root completes its solve, then the downward pass substitutes solved ancestors into each section.
"""

import functools
from typing import NamedTuple

import warp as wp

# Bounds on generated work; planning does not inspect compiled kernel resources.
_FACTOR_SLOTS = 96
_SOLVE_SLOTS = 192
_SECTION_ROWS = 64


@wp.func
def _exchange(value: float, source: int):
  return wp.tile_extract(wp.tile(value), source)


class _RegisterLayout(NamedTuple):
  width: int
  row_chunks: tuple
  slots: tuple
  csr_indices: tuple
  diagonal_slots: tuple
  lower_entries: tuple
  positions: dict
  updates: tuple
  lower_masks: tuple


def _register_layout(rowadr, colind, factorize, block_dim, begin, children):
  rows = tuple(colind[rowadr[i] : rowadr[i + 1]] for i in range(len(rowadr) - 1))
  n = len(rows)
  width = min(4, block_dim)
  if factorize and (begin or children):
    width = min(block_dim, 32, max(4, 1 << (max(map(len, rows)) - 1).bit_length()))
  budget = _FACTOR_SLOTS if factorize else _SOLVE_SLOTS
  limit = min(32 if factorize else 8, block_dim)
  while width < limit and sum(len({j // width for j in row}) for row in (rows if factorize else rows[begin:n])) > budget:
    width *= 2
  indices = {(i, j): s for s, (i, j) in enumerate((i, j) for i, row in enumerate(rows) for j in row)}
  # A register slot holds one aligned CSR row chunk across the lane group.
  chunks = tuple(tuple(sorted({j // width for j in row})) for row in rows)
  slots = tuple((i, c) for i in range(0 if factorize else begin, n) for c in chunks[i])
  pos = {pair: s for s, pair in enumerate(slots)}
  mapping = tuple(tuple(indices.get((i, c * width + lane), -1) for lane in range(width)) for i, c in slots)
  diagonal_slots = tuple(pos.get((k, k // width), -1) for k in range(n))
  lower = tuple(tuple((i, pos[k, i // width]) for i in rows[k][:-1]) if begin <= k < n else () for k in range(n))
  updates = tuple(
    tuple(
      tuple(
        (pos[i, c], pos[k, c], sum(1 << lane for lane in range(width) if c * width + lane in rows[k] and c * width + lane <= i))
        for c in chunks[i]
        if c in chunks[k]
      )
      for i, _ in (lower[k] if factorize else ())
    )
    for k in range(n)
  )
  lower_masks = tuple(
    tuple(sum(1 << lane for lane in range(width) if c * width + lane in row[:-1]) for c in chunks[k])
    for k, row in enumerate(rows)
  )
  return _RegisterLayout(width, chunks, slots, mapping, diagonal_slots, lower, pos, updates, lower_masks)


@functools.cache
def _kernel(rowadr: tuple, colind: tuple, factorize: bool, solve: bool, block_dim: int, *, begin=0, down=False, children=False):
  layout = _register_layout(rowadr, colind, factorize, block_dim, begin, children)
  width = layout.width
  nrows = len(rowadr) - 1
  nslots = len(layout.slots)
  rhs_chunks = (nrows + width - 1) // width
  MatrixRegisters = wp.types.vector(length=nslots, dtype=wp.float32)
  RhsRegisters = wp.types.vector(length=rhs_chunks, dtype=wp.float32)
  row_size = max(map(len, layout.row_chunks[begin:nrows]))
  PivotRegisters = wp.types.vector(length=row_size + 1, dtype=wp.float32)

  def make_rhs_update(k):
    @wp.func
    def update_rhs(matrix_values: MatrixRegisters, rhs_in: RhsRegisters, value: float, lane: int):
      rhs_values = rhs_in
      for ci in range(wp.static(len(layout.row_chunks[k]))):
        c = wp.static(layout.row_chunks[k][ci])
        mask = wp.static(layout.lower_masks[k][ci])
        if ((mask >> lane) & 1) != 0:
          rhs_values[wp.static(c)] -= matrix_values[wp.static(layout.positions[k, c])] * value
      return rhs_values

    return update_rhs

  rhs_updates = tuple(make_rhs_update(k) for k in range(begin, nrows)) if solve else ()

  # Function-local tiles let consecutive pivots reuse shared scratch.
  def make_pivot(k):
    @wp.func
    def pivot(matrix_in: MatrixRegisters, rhs_in: RhsRegisters, tid: int, lane: int):
      matrix_values = matrix_in
      rhs_values = rhs_in
      if wp.static(width == 1):
        diag = matrix_values[wp.static(layout.diagonal_slots[k])]
      else:
        row = PivotRegisters()
        for ci in range(wp.static(len(layout.row_chunks[k]))):
          c = wp.static(layout.row_chunks[k][ci])
          row[ci] = matrix_values[wp.static(layout.positions[k, c])]
        if wp.static(solve):
          row[wp.static(row_size)] = rhs_values[wp.static(k // width)]
        shared_row = wp.tile(row)
        diag = wp.tile_extract(shared_row, wp.static(layout.row_chunks[k].index(k // width)), tid - lane + wp.static(k % width))
      inv = 1.0 / diag
      if wp.static(factorize):
        for ii in range(wp.static(len(layout.lower_entries[k]))):
          i = wp.static(layout.lower_entries[k][ii][0])
          if wp.static(width == 1):
            v = matrix_values[wp.static(layout.lower_entries[k][ii][1])] * inv
          else:
            v = (
              wp.tile_extract(shared_row, wp.static(layout.row_chunks[k].index(i // width)), tid - lane + wp.static(i % width))
              * inv
            )
          for jj in range(wp.static(len(layout.updates[k][ii]))):
            dst = wp.static(layout.updates[k][ii][jj][0])
            src = wp.static(layout.updates[k][ii][jj][1])
            if ((wp.static(layout.updates[k][ii][jj][2]) >> lane) & 1) != 0:
              matrix_values[dst] -= v * matrix_values[src]
        for ci in range(wp.static(len(layout.row_chunks[k]))):
          c = wp.static(layout.row_chunks[k][ci])
          if lane + wp.static(c * width) < wp.static(k):
            matrix_values[wp.static(layout.positions[k, c])] *= inv
      if wp.static(solve):
        if wp.static(width == 1):
          value = rhs_values[wp.static(k // width)]
        else:
          value = wp.tile_extract(shared_row, wp.static(row_size), tid - lane + wp.static(k % width))
        rhs_values = wp.static(rhs_updates[k - begin])(matrix_values, rhs_values, value, lane)
        if wp.static(begin == 0):
          if lane == wp.static(k % width):
            rhs_values[wp.static(k // width)] *= inv
      return matrix_values, rhs_values, inv

    return pivot

  pivots = tuple(make_pivot(k) for k in range(begin, nrows)) if factorize else ()

  def make_substitution(k):
    @wp.func
    def substitute(matrix_values: MatrixRegisters, rhs_in: RhsRegisters, tid: int, lane: int):
      rhs_values = rhs_in
      value = float(0.0)
      for ci in range(wp.static(len(layout.row_chunks[k]))):
        c = wp.static(layout.row_chunks[k][ci])
        mask = wp.static(layout.lower_masks[k][ci])
        if ((mask >> lane) & 1) != 0:
          value += matrix_values[wp.static(layout.positions[k, c])] * rhs_values[wp.static(c)]
      for stage in range(wp.static(width.bit_length() - 1)):
        value += _exchange(value, tid ^ wp.static(1 << stage))
      if lane == wp.static(k % width):
        rhs_values[wp.static(k // width)] -= value
      return rhs_values

    return substitute

  substitutions = tuple(make_substitution(k) for k in range(begin, nrows)) if solve and (begin == 0 or down) else ()

  direct = begin == 0 and not children

  @wp.func
  def dof(dofs: wp.array2d[int], system: int, local: int, dof_base: int):
    if wp.static(direct):
      return dof_base + local
    return dofs[system, local]

  @wp.func
  def address(dofs: wp.array2d[int], system: int, ri: int, csr_index: int, offset: int, base: int, matrix_adr: wp.array[int]):
    if wp.static(direct):
      return base + csr_index
    return matrix_adr[dofs[system, ri]] + csr_index - offset

  @wp.kernel(module="unique", module_options={"enable_backward": False, "max_unroll": 4096})
  def kernel(
    # In:
    dofs: wp.array2d[int],
    contrib_adr: wp.array2d[int],
    contrib_ind: wp.array[int],
    message_adr: wp.array[int],
    matrix_adr: wp.array[int],
    factor_adr: wp.array[int],
    worlds: int,
    matrix: wp.array2d[float],
    rhs: wp.array2d[float],
    # Out:
    factor_out: wp.array2d[float],
    diagonal_out: wp.array2d[float],
    out: wp.array2d[float],
    messages_out: wp.array2d[float],
  ):
    block, tid = wp.tid()
    lane = tid & wp.static(width - 1)
    job = block * wp.static(block_dim // width) + tid // wp.static(width)
    world = job // dofs.shape[0]
    system = job % dofs.shape[0]
    dof_base = dofs[system, 0]
    matrix_base = matrix_adr[dof_base]
    factor_base = factor_adr[dof_base]
    enabled = world < worlds
    matrix_values = MatrixRegisters()
    rhs_values = RhsRegisters()
    for slot in range(wp.static(nslots)):
      if lane + wp.static(layout.slots[slot][1] * width) == wp.static(layout.slots[slot][0]):
        matrix_values[slot] = 1.0
    if enabled:
      for slot in range(wp.static(nslots)):
        row = wp.static(layout.slots[slot][0])
        row_offset = wp.static(rowadr[row])
        csr_index = int(-1)
        for l in range(wp.static(width)):
          if lane == l:
            csr_index = wp.static(layout.csr_indices[slot][l])
        if csr_index >= 0:
          if wp.static(not factorize):
            matrix_values[slot] = factor_out[
              world,
              address(dofs, system, row, csr_index, row_offset, factor_base, matrix_adr),
            ]
          else:
            if wp.static(row >= begin):
              matrix_values[slot] = matrix[
                world,
                address(dofs, system, row, csr_index, row_offset, matrix_base, matrix_adr),
              ]
            else:
              matrix_values[slot] = 0.0
            if wp.static(children):
              for u in range(contrib_adr[system, csr_index], contrib_adr[system, csr_index + 1]):
                matrix_values[slot] += messages_out[world, contrib_ind[u]]
      if wp.static(solve):
        for c in range(wp.static(rhs_chunks)):
          j = wp.static(c * width) + lane
          if j < wp.static(nrows):
            if wp.static(down):
              rhs_values[c] = out[world, dof(dofs, system, j, dof_base)]
            else:
              if j >= wp.static(begin):
                rhs_values[c] = rhs[world, dof(dofs, system, j, dof_base)]
              if wp.static(children):
                for u in range(
                  contrib_adr[system, wp.static(rowadr[-1]) + j], contrib_adr[system, wp.static(rowadr[-1]) + j + 1]
                ):
                  rhs_values[c] += messages_out[world, contrib_ind[u]]
    if wp.static(not factorize and (begin == 0 or down)):
      # Move independent reciprocals before the dependent substitutions.
      for k in range(wp.static(begin), wp.static(nrows)):
        if lane == wp.static(k % width):
          slot = wp.static(layout.diagonal_slots[k])
          matrix_values[slot] = 1.0 / matrix_values[slot]
    if wp.static(not down):
      for rev in range(wp.static(nrows - begin)):
        k = wp.static(nrows - 1 - rev)
        if wp.static(factorize):
          matrix_values, rhs_values, inv = wp.static(pivots[k - begin])(matrix_values, rhs_values, tid, lane)
          if enabled and lane == wp.static(k % width):
            diagonal_out[world, dof(dofs, system, wp.static(k), dof_base)] = inv
        else:
          value = rhs_values[wp.static(k // width)]
          if wp.static(width > 1):
            value = _exchange(value, tid - lane + wp.static(k % width))
          rhs_values = wp.static(rhs_updates[k - begin])(matrix_values, rhs_values, value, lane)
          if wp.static(begin == 0):
            if enabled and lane == wp.static(k % width):
              rhs_values[wp.static(k // width)] *= matrix_values[wp.static(layout.diagonal_slots[k])]
    if wp.static(solve and (begin == 0 or down)):
      for off in range(wp.static(nrows - begin)):
        k = wp.static(begin + off)
        if wp.static(down):
          if lane == wp.static(k % width):
            rhs_values[wp.static(k // width)] *= matrix_values[wp.static(layout.diagonal_slots[k])]
        rhs_values = wp.static(substitutions[k - begin])(matrix_values, rhs_values, tid, lane)
    if wp.static(solve):
      for c in range(wp.static(rhs_chunks)):
        j = wp.static(c * width) + lane
        if enabled and j < wp.static(nrows):
          if j >= wp.static(begin):
            out[world, dof(dofs, system, j, dof_base)] = rhs_values[c]
          elif wp.static(not down):
            messages_out[world, message_adr[system] + wp.static(rowadr[begin]) + j] = rhs_values[c]
    if wp.static(factorize):
      if enabled:
        for slot in range(wp.static(nslots)):
          row = wp.static(layout.slots[slot][0])
          row_offset = wp.static(rowadr[row])
          csr_index = int(-1)
          for l in range(wp.static(width)):
            if lane == l:
              csr_index = wp.static(layout.csr_indices[slot][l])
          if csr_index >= 0:
            if wp.static(row >= begin):
              factor_out[
                world,
                address(dofs, system, row, csr_index, row_offset, factor_base, matrix_adr),
              ] = matrix_values[slot]
            else:
              messages_out[world, message_adr[system] + csr_index] = matrix_values[slot]

  return kernel, width


def launch(m, tile, worlds, matrix, factor, diagonal, rhs=None, out=None, *, factorize=True):
  """Schedule independent blocks together while retaining their CSR factors."""
  block_dim = 32 if factor.device.is_cuda else 1
  rowadr = m.M_sparse_rowadr[tile.pattern_adr : tile.pattern_adr + tile.size + 1]
  colind = m.M_sparse_colind[rowadr[0] : rowadr[-1]]
  kernel, width = _kernel(tuple(a - rowadr[0] for a in rowadr), colind, factorize, rhs is not None, block_dim)
  groups = block_dim // width
  wp.launch_tiled(
    kernel,
    dim=((worlds * tile.adr.size + groups - 1) // groups,),
    inputs=[tile.adr.reshape((tile.adr.size, 1)), None, None, None, m.M_rowadr, m.qLD_block_adr, worlds, matrix, rhs],
    outputs=[factor, diagonal, out, None],
    block_dim=block_dim,
  )


@functools.cache
def _factor_rows(slots: int, width: int, block_dim: int):
  Vec = wp.types.vector(length=slots, dtype=wp.float32)

  @wp.kernel(module="unique", module_options={"enable_backward": False, "max_unroll": 64})
  def kernel(
    # In:
    rows: wp.array[int],
    worlds: int,
    rowadr: wp.array[int],
    rownnz: wp.array[int],
    muladr: wp.array[int],
    mulcol: wp.array[int],
    mulmadr: wp.array[int],
    matrix: wp.array2d[float],
    # Out:
    factor_out: wp.array2d[float],
    diagonal_out: wp.array2d[float],
  ):
    block, tid = wp.tid()
    lane = tid & wp.static(width - 1)
    job = block * wp.static(block_dim // width) + tid // wp.static(width)
    world = job // rows.shape[0]
    i = rows[job % rows.shape[0]]
    base = rowadr[i]
    n = rownnz[i]
    a = Vec()
    if world < worlds:
      for c in range(wp.static(slots)):
        j = lane + wp.static(c * width)
        if j < n:
          a[c] = matrix[world, base + j]
      # Ancestor rows are prefixes of each descendant row in MuJoCo mass storage.
      for u in range(muladr[i + 1] - 1, muladr[i] + n - 1, -1):
        k = mulcol[u]
        ki = factor_out[world, mulmadr[u]]
        gain = ki * factor_out[world, rowadr[k] + rownnz[k] - 1]
        for c in range(wp.static(slots)):
          j = lane + wp.static(c * width)
          if j < n:
            a[c] -= gain * factor_out[world, rowadr[k] + j]
    diag = float(1.0)
    for c in range(wp.static(slots)):
      if (n - 1) // wp.static(width) == wp.static(c):
        diag = a[c]
    diag = _exchange(diag, tid - lane + (n - 1) % wp.static(width))
    if world < worlds:
      for c in range(wp.static(slots)):
        j = lane + wp.static(c * width)
        if j < n:
          value = a[c]
          if j < n - 1:
            value /= diag
          factor_out[world, base + j] = value
      if lane == 0:
        diagonal_out[world, i] = 1.0 / diag

  return kernel


def launch_rows(m, d, matrix, factor, diagonal):
  """Factor ancestor-pattern rows after their descendants, without atomic updates."""
  block_dim = 32 if factor.device.is_cuda else 1
  for tile in reversed(m.qLD_row_tiles):
    width = min(8 if tile.size <= 32 else 32, block_dim)
    groups = block_dim // width
    wp.launch_tiled(
      _factor_rows((tile.size + width - 1) // width, width, block_dim),
      dim=((d.nworld * tile.adr.size + groups - 1) // groups,),
      inputs=[tile.adr, d.nworld, m.M_rowadr, m.M_rownnz, m.M_mulm_rowadr, m.M_mulm_col, m.M_mulm_madr, matrix],
      outputs=[factor, diagonal],
      block_dim=block_dim,
    )


def _partition_tree(rows):
  children = [[] for _ in rows]
  for i, row in enumerate(rows):
    if len(row) > 1:
      children[row[-2]].append(i)
  opened, closed, split = {}, [], {}
  for i in reversed(range(len(rows))):
    branches = [opened.pop(k) for k in children[i]]
    candidate = sorted([i] + [j for branch in branches for j in branch])
    cost = sum((len(rows[j]) + 3) // 4 for j in candidate)
    boundary = any(split[k] for k in children[i])
    # Preserve sibling parallelism once their descendants need separate launches.
    fits = len(candidate) <= _SECTION_ROWS and cost <= _FACTOR_SLOTS and not (len(children[i]) > 1 and boundary)
    split[i] = boundary or not fits
    if fits:
      opened[i] = candidate
    else:
      closed.extend(branches)
      opened[i] = [i]
  return sorted(closed + list(opened.values()), key=lambda ids: ids[0])


def section_layout(rowadr, rownnz, colind, trees):
  """Plan bounded sections and child updates from the ancestor pattern."""
  descriptions, accepted, total = [], [], 0
  for start, size in trees:
    rows = [tuple(int(j) - start for j in colind[rowadr[i] : rowadr[i] + rownnz[i]]) for i in range(start, start + size)]
    parts = _partition_tree(rows)
    owner = {i: s for s, ids in enumerate(parts) for i in ids}
    tree, count = [], 0
    max_scratch = 2 * sum(map(len, rows))
    for ids in parts:
      halo = rows[ids[0]][:-1]
      dofs = list(halo) + ids
      if len(dofs) > _FACTOR_SLOTS:
        break
      local = {i: j for j, i in enumerate(dofs)}
      csr = tuple(tuple(local[j] for j in rows[i]) for i in dofs)
      if sum(len({j // 32 for j in row}) for row in csr) > _FACTOR_SLOTS:
        break
      adr, cols = [0], []
      for row in csr:
        cols.extend(row)
        adr.append(len(cols))
      next_count = count + adr[len(halo)] + len(halo)
      if next_count > max_scratch:
        break
      parent = owner[halo[-1]] if halo else -1
      depth = tree[parent]["depth"] + 1 if parent >= 0 else 0
      tree.append(
        dict(
          dofs=[i + start for i in dofs],
          rowadr=tuple(adr),
          colind=tuple(cols),
          ancestor=len(halo),
          depth=depth,
          children=[],
          message=total + count,
        )
      )
      if parent >= 0:
        tree[parent]["children"].append(tree[-1])
      count = next_count
    if len(tree) != len(parts):
      continue
    descriptions.extend(tree)
    accepted.append((start, size))
    total += count

  groups = {}
  for desc in descriptions:
    dofs, adr, cols = desc["dofs"], desc["rowadr"], desc["colind"]
    local = {v: i for i, v in enumerate(dofs)}
    indices = {(i, cols[a]): a for i in range(len(dofs)) for a in range(adr[i], adr[i + 1])}
    contributions = [[] for _ in range(adr[-1] + len(dofs))]
    for child in desc["children"]:
      halo = child["ancestor"]
      for i in range(halo):
        ri = local[child["dofs"][i]]
        for at in range(child["rowadr"][i], child["rowadr"][i + 1]):
          rj = local[child["dofs"][child["colind"][at]]]
          contributions[indices[ri, rj]].append(child["message"] + at)
        contributions[adr[-1] + ri].append(child["message"] + child["rowadr"][halo] + i)
    key = (desc["depth"], adr, cols, desc["ancestor"], bool(desc["children"]))
    groups.setdefault(key, []).append((dofs, contributions, desc["message"]))
  result = []
  for (depth, adr, cols, ancestor, _), group in sorted(groups.items(), key=lambda item: item[0][0]):
    dofs, starts, offsets, indices = [], [], [], []
    for ids, contributions, message in group:
      dofs.append(ids)
      starts.append(message)
      row = [len(indices)]
      for entries in contributions:
        indices.extend(entries)
        row.append(len(indices))
      offsets.append(row)
    result.append(
      dict(dofs=dofs, rowadr=adr, colind=cols, ancestor=ancestor, update_adr=offsets, update_ind=indices, message_adr=starts)
    )
  return result, total, accepted


def launch_sections(m, d, matrix, factor, diagonal, rhs=None, out=None, *, factorize=True):
  block_dim = 32 if factor.device.is_cuda else 1
  solve = rhs is not None
  for down in (False, True) if solve else (False,):
    sections = m.M_sections if down else reversed(m.M_sections)
    for section in sections:
      if down and section.ancestor == 0:
        continue
      rowadr = m.M_sparse_rowadr[section.pattern_adr : section.pattern_adr + section.dofs.shape[1] + 1]
      colind = m.M_sparse_colind[rowadr[0] : rowadr[-1]]
      kernel, width = _kernel(
        tuple(a - rowadr[0] for a in rowadr),
        colind,
        factorize and not down,
        solve,
        block_dim,
        begin=section.ancestor,
        down=down,
        children=section.update_ind.size > 0,
      )
      groups = block_dim // width
      wp.launch_tiled(
        kernel,
        dim=((d.nworld * section.dofs.shape[0] + groups - 1) // groups,),
        inputs=[
          section.dofs,
          section.update_adr,
          section.update_ind,
          section.message_adr,
          m.M_rowadr,
          m.M_rowadr,
          d.nworld,
          matrix,
          rhs,
        ],
        outputs=[factor, diagonal, out, d.qLD_schur],
        block_dim=block_dim,
      )
