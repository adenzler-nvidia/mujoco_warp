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

Descending elimination uses L[k,i] = A[k,i]/D[k] and A[i,j] -= L[k,i]*A[k,j].
The solve applies L.T y = b in descending order, then L x = D^-1 y in ascending order.

Ancestor rows are CSR prefixes of descendant rows, so reverse elimination introduces no fill.
Sections place ancestor rows before owned rows. The upward pass eliminates owned rows and sends
only their matrix contributions upward; original rows are loaded by their owning section.
"""

import dataclasses
import functools
from typing import NamedTuple

import warp as wp

# Structural limits and lane-width targets; no compiled-resource queries.
_FACTOR_SLOTS = 96
_SOLVE_SLOT_TARGET = 192
_SECTION_ROWS = 64


@wp.func
def _exchange(value: float, source: int):
  return wp.tile_extract(wp.tile(value), source)


class _RegisterLayout(NamedTuple):
  width: int
  rows: tuple
  row_chunks: tuple
  slots: tuple
  csr_indices: tuple
  positions: dict


def _row_mask(columns, chunk, width, last):
  return sum(1 << (j % width) for j in columns if j // width == chunk and j <= last)


def _register_layout(rowadr, colind, factorize, block_dim, section):
  rows = tuple(colind[rowadr[i] : rowadr[i + 1]] for i in range(len(rowadr) - 1))
  n = len(rows)
  width = min(4, block_dim)
  if factorize and section:
    width = min(block_dim, 32, max(4, 1 << (max(map(len, rows)) - 1).bit_length()))
  budget = _FACTOR_SLOTS if factorize else _SOLVE_SLOT_TARGET
  limit = min(32 if factorize else 8, block_dim)
  while width < limit and sum(len({j // width for j in row}) for row in rows) > budget:
    width *= 2
  indices = {(i, j): s for s, (i, j) in enumerate((i, j) for i, row in enumerate(rows) for j in row)}
  # A register slot holds one aligned CSR row chunk across the lane group.
  chunks = tuple(tuple(sorted({j // width for j in row})) for row in rows)
  slots = tuple((i, c) for i in range(n) for c in chunks[i])
  pos = {pair: s for s, pair in enumerate(slots)}
  mapping = tuple(tuple(indices.get((i, c * width + lane), -1) for lane in range(width)) for i, c in slots)
  return _RegisterLayout(width, rows, chunks, slots, mapping, pos)


@functools.cache
def _kernel(rowadr: tuple, colind: tuple, block_dim: int, *, factorize=True, solve=False, nancestor=0, has_children=False):
  section = nancestor > 0 or has_children
  assert factorize or solve
  assert not section or (factorize and not solve)
  layout = _register_layout(rowadr, colind, factorize, block_dim, section)
  width = layout.width
  nrows = len(rowadr) - 1
  nslots = len(layout.slots)
  rhs_chunks = (nrows + width - 1) // width
  MatrixRegisters = wp.types.vector(length=nslots, dtype=wp.float32)
  RhsRegisters = wp.types.vector(length=rhs_chunks, dtype=wp.float32)
  row_size = max(map(len, layout.row_chunks[nancestor:nrows]))
  PivotRegisters = wp.types.vector(length=row_size + 1, dtype=wp.float32)

  def make_transpose_update(k):
    chunks = layout.row_chunks[k]
    slots = tuple(layout.positions[k, c] for c in chunks)

    @wp.func
    def transpose_update(factor: MatrixRegisters, rhs_in: RhsRegisters, y_k: float, lane: int):
      """Apply b[i] -= L[k,i] * y[k] for i < k."""
      rhs_values = rhs_in
      for chunk in range(wp.static(len(chunks))):
        rhs_slot = wp.static(chunks[chunk])
        factor_slot = wp.static(slots[chunk])
        if ((wp.static(_row_mask(layout.rows[k], chunks[chunk], width, k - 1)) >> lane) & 1) != 0:
          rhs_values[rhs_slot] -= factor[factor_slot] * y_k
      return rhs_values

    return transpose_update

  transpose_updates = tuple(make_transpose_update(k) for k in range(nancestor, nrows)) if solve else ()

  # Function-local tiles let consecutive eliminations reuse shared scratch.
  def make_elimination(k):
    chunks = layout.row_chunks[k]
    slots = tuple(layout.positions[k, c] for c in chunks)
    lower = layout.rows[k][:-1]
    diagonal_slot = layout.positions[k, k // width]
    diagonal_chunk = chunks.index(k // width)

    @wp.func
    def eliminate(matrix_in: MatrixRegisters, rhs_in: RhsRegisters, tid: int, lane: int):
      matrix_values = matrix_in
      rhs_values = rhs_in
      if wp.static(width == 1):
        d_k = matrix_values[wp.static(diagonal_slot)]
      else:
        row = PivotRegisters()
        for chunk in range(wp.static(len(chunks))):
          row[chunk] = matrix_values[wp.static(slots[chunk])]
        if wp.static(solve):
          row[wp.static(row_size)] = rhs_values[wp.static(k // width)]
        shared_row = wp.tile(row)
        d_k = wp.tile_extract(shared_row, wp.static(diagonal_chunk), tid - lane + wp.static(k % width))
      inv_d = 1.0 / d_k
      for neighbor in range(wp.static(len(lower))):
        i = wp.static(lower[neighbor])
        if wp.static(width == 1):
          l_ki = matrix_values[wp.static(layout.positions[k, i // width])] * inv_d
        else:
          l_ki = wp.tile_extract(shared_row, wp.static(chunks.index(i // width)), tid - lane + wp.static(i % width)) * inv_d
        # Read unscaled A[k,j] until every Schur update using this row is complete.
        for chunk in range(wp.static(len(layout.row_chunks[i]))):
          column_chunk = wp.static(layout.row_chunks[i][chunk])
          if wp.static((k, column_chunk) in layout.positions):
            a_ij = wp.static(layout.positions[i, column_chunk])
            a_kj = wp.static(layout.positions[k, column_chunk])
            if ((wp.static(_row_mask(layout.rows[k], column_chunk, width, i)) >> lane) & 1) != 0:
              matrix_values[a_ij] -= l_ki * matrix_values[a_kj]
      for chunk in range(wp.static(len(chunks))):
        column = lane + wp.static(chunks[chunk] * width)
        if column < wp.static(k):
          matrix_values[wp.static(slots[chunk])] *= inv_d
      if wp.static(solve):
        if wp.static(width == 1):
          y_k = rhs_values[wp.static(k // width)]
        else:
          y_k = wp.tile_extract(shared_row, wp.static(row_size), tid - lane + wp.static(k % width))
        rhs_values = wp.static(transpose_updates[k - nancestor])(matrix_values, rhs_values, y_k, lane)
        if lane == wp.static(k % width):
          rhs_values[wp.static(k // width)] *= inv_d
      return matrix_values, rhs_values, inv_d

    return eliminate

  eliminations = tuple(make_elimination(k) for k in range(nancestor, nrows)) if factorize else ()

  def make_transpose_solve(k):
    diagonal_slot = layout.positions[k, k // width]

    @wp.func
    def transpose_solve(factor: MatrixRegisters, rhs_in: RhsRegisters, tid: int, lane: int, enabled: bool):
      rhs_values = rhs_in
      y_k = rhs_values[wp.static(k // width)]
      if wp.static(width > 1):
        y_k = _exchange(y_k, tid - lane + wp.static(k % width))
      rhs_values = wp.static(transpose_updates[k - nancestor])(factor, rhs_values, y_k, lane)
      if enabled and lane == wp.static(k % width):
        rhs_values[wp.static(k // width)] *= factor[wp.static(diagonal_slot)]
      return rhs_values

    return transpose_solve

  transpose_steps = tuple(make_transpose_solve(k) for k in range(nancestor, nrows)) if not factorize else ()

  def make_forward_solve(k):
    chunks = layout.row_chunks[k]
    slots = tuple(layout.positions[k, c] for c in chunks)

    @wp.func
    def forward_solve(factor: MatrixRegisters, rhs_in: RhsRegisters, tid: int, lane: int):
      """Finish row k of L x = D^-1 y after diagonal scaling."""
      rhs_values = rhs_in
      dot_lx = float(0.0)
      for chunk in range(wp.static(len(chunks))):
        rhs_slot = wp.static(chunks[chunk])
        factor_slot = wp.static(slots[chunk])
        if ((wp.static(_row_mask(layout.rows[k], chunks[chunk], width, k - 1)) >> lane) & 1) != 0:
          dot_lx += factor[factor_slot] * rhs_values[rhs_slot]
      for stage in range(wp.static(width.bit_length() - 1)):
        dot_lx += _exchange(dot_lx, tid ^ wp.static(1 << stage))
      if lane == wp.static(k % width):
        rhs_values[wp.static(k // width)] -= dot_lx
      return rhs_values

    return forward_solve

  forward_steps = tuple(make_forward_solve(k) for k in range(nancestor, nrows)) if solve else ()

  direct = nancestor == 0 and not has_children

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

  @wp.kernel(module="unique", module_options={"enable_backward": False})
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
            if wp.static(row >= nancestor):
              matrix_values[slot] = matrix[
                world,
                address(dofs, system, row, csr_index, row_offset, matrix_base, matrix_adr),
              ]
            else:
              matrix_values[slot] = 0.0
            if wp.static(has_children):
              for u in range(contrib_adr[system, csr_index], contrib_adr[system, csr_index + 1]):
                matrix_values[slot] += messages_out[world, contrib_ind[u]]
      if wp.static(solve):
        for c in range(wp.static(rhs_chunks)):
          j = wp.static(c * width) + lane
          if j < wp.static(nrows):
            rhs_values[c] = rhs[world, dof_base + j]
    if wp.static(not factorize):
      for k in range(wp.static(nancestor), wp.static(nrows)):
        if lane == wp.static(k % width):
          slot = wp.static(layout.positions[k, k // width])
          matrix_values[slot] = 1.0 / matrix_values[slot]
    for rev in range(wp.static(nrows - nancestor)):
      k = wp.static(nrows - 1 - rev)
      if wp.static(factorize):
        matrix_values, rhs_values, inv_d = wp.static(eliminations[k - nancestor])(matrix_values, rhs_values, tid, lane)
        if enabled and lane == wp.static(k % width):
          diagonal_out[world, dof(dofs, system, wp.static(k), dof_base)] = inv_d
      else:
        rhs_values = wp.static(transpose_steps[k - nancestor])(matrix_values, rhs_values, tid, lane, enabled)
    if wp.static(solve):
      for off in range(wp.static(nrows - nancestor)):
        k = wp.static(nancestor + off)
        rhs_values = wp.static(forward_steps[k - nancestor])(matrix_values, rhs_values, tid, lane)
    if wp.static(solve):
      for c in range(wp.static(rhs_chunks)):
        j = wp.static(c * width) + lane
        if enabled and j < wp.static(nrows):
          out[world, dof_base + j] = rhs_values[c]
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
            if wp.static(row >= nancestor):
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
  kernel, width = _kernel(tuple(a - rowadr[0] for a in rowadr), colind, block_dim, factorize=factorize, solve=rhs is not None)
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


@dataclasses.dataclass
class _Section:
  dofs: tuple[int, ...]
  rowadr: tuple[int, ...]
  colind: tuple[int, ...]
  ancestor: int
  depth: int
  message_adr: int
  children: list["_Section"] = dataclasses.field(default_factory=list)

  @property
  def message_end(self):
    return self.message_adr + self.rowadr[self.ancestor]


class SectionGroup(NamedTuple):
  dofs: list[tuple[int, ...]]
  rowadr: tuple[int, ...]
  colind: tuple[int, ...]
  ancestor: int
  update_adr: list[list[int]]
  update_ind: list[int]
  message_adr: list[int]


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
      parent = owner[halo[-1]] if halo else -1
      section = _Section(
        dofs=tuple(i + start for i in dofs),
        rowadr=tuple(adr),
        colind=tuple(cols),
        ancestor=len(halo),
        depth=tree[parent].depth + 1 if parent >= 0 else 0,
        message_adr=total + count,
      )
      if section.message_end - total > max_scratch:
        break
      tree.append(section)
      if parent >= 0:
        tree[parent].children.append(section)
      count = section.message_end - total
    if len(tree) != len(parts):
      continue
    descriptions.extend(tree)
    accepted.append((start, size))
    total += count

  groups = {}
  for desc in descriptions:
    dofs, adr, cols = desc.dofs, desc.rowadr, desc.colind
    local = {v: i for i, v in enumerate(dofs)}
    indices = {(i, cols[a]): a for i in range(len(dofs)) for a in range(adr[i], adr[i + 1])}
    contributions = [[] for _ in range(adr[-1])]
    for child in desc.children:
      halo = child.ancestor
      for i in range(halo):
        ri = local[child.dofs[i]]
        for at in range(child.rowadr[i], child.rowadr[i + 1]):
          rj = local[child.dofs[child.colind[at]]]
          contributions[indices[ri, rj]].append(child.message_adr + at)
    key = (desc.depth, adr, cols, desc.ancestor, bool(desc.children))
    groups.setdefault(key, []).append((dofs, contributions, desc.message_adr))
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
    result.append(SectionGroup(dofs, adr, cols, ancestor, offsets, indices, starts))
  return result, total, accepted


def launch_sections(m, d, matrix, factor, diagonal):
  block_dim = 32 if factor.device.is_cuda else 1
  for section in reversed(m.M_sections):
    rowadr = m.M_sparse_rowadr[section.pattern_adr : section.pattern_adr + section.dofs.shape[1] + 1]
    colind = m.M_sparse_colind[rowadr[0] : rowadr[-1]]
    kernel, width = _kernel(
      tuple(a - rowadr[0] for a in rowadr),
      colind,
      block_dim,
      nancestor=section.ancestor,
      has_children=section.update_ind.size > 0,
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
        None,
      ],
      outputs=[factor, diagonal, None, d.qLD_schur],
      block_dim=block_dim,
    )
