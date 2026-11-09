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

"""Tests for mass-factor section ownership and dependencies."""

from absl.testing import absltest
from absl.testing import parameterized

from mujoco_warp._src import block_ldl


class BlockLdlTest(parameterized.TestCase):
  @parameterized.parameters(64, 65, 152)
  def test_section_layout(self, size):
    rows = []
    for i in range(size):
      parent = i - 1 if i < 8 or (i - 8) % 8 else 7
      rows.append((rows[parent] if i else ()) + (i,))
    # Include a rejected deep tree after the accepted branching tree.
    rows.extend(tuple(range(size, size + i + 1)) for i in range(100))
    rowadr = [0]
    for row in rows:
      rowadr.append(rowadr[-1] + len(row))
    groups, scratch, accepted = block_ldl.section_layout(
      rowadr, [len(row) for row in rows], tuple(j for row in rows for j in row), [(0, size), (size, 100)]
    )
    self.assertEqual(accepted, [(0, size)])
    owners, messages, consumers = {}, {}, {}
    for group_id, group in enumerate(groups):
      for dofs, message, offsets in zip(group.dofs, group.message_adr, group.update_adr):
        ancestors = dofs[: group.ancestor]
        for row in dofs[group.ancestor :]:
          self.assertNotIn(row, owners)
          owners[row] = group_id
        for row in ancestors:
          self.assertLess(owners[row], group_id)
        for row, dof in enumerate(dofs):
          columns = group.colind[group.rowadr[row] : group.rowadr[row + 1]]
          self.assertEqual(tuple(dofs[col] for col in columns), rows[dof])
          for entry in range(group.rowadr[row], group.rowadr[row + 1]):
            key = (dof, dofs[group.colind[entry]])
            for source in group.update_ind[offsets[entry] : offsets[entry + 1]]:
              self.assertNotIn(source, consumers)
              consumers[source] = (group_id, key)
            if row < group.ancestor:
              address = message + entry
              self.assertNotIn(address, messages)
              messages[address] = (group_id, key)
          rhs_entry = group.rowadr[-1] + row
          for source in group.update_ind[offsets[rhs_entry] : offsets[rhs_entry + 1]]:
            self.assertNotIn(source, consumers)
            consumers[source] = (group_id, (dof,))
          if row < group.ancestor:
            address = message + group.rowadr[group.ancestor] + row
            self.assertNotIn(address, messages)
            messages[address] = (group_id, (dof,))
    self.assertEqual(sorted(owners), list(range(size)))
    self.assertEqual(sorted(messages), list(range(scratch)))
    self.assertEqual(sorted(consumers), sorted(messages))
    for address, (producer, entry) in messages.items():
      consumer, destination = consumers[address]
      self.assertLess(consumer, producer)
      self.assertEqual(entry, destination)


if __name__ == "__main__":
  absltest.main()
