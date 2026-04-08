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

import functools
import inspect
import math
import warnings

import warp as wp

_STACK = None

_WARP_SIZE = 32
_WARP_DEFAULT_BLOCK_DIM = 256

# Cache: (kernel_id, device_ordinal) -> (suggested_block_size, min_grid_size)
_OCCUPANCY_CACHE: dict[tuple, tuple[int, int]] = {}

# Cache: (kernel_id, device_ordinal, dim) -> block_dim
_BLOCK_DIM_CACHE: dict[tuple, int] = {}


def get_block_dim(
  kernel: wp.Kernel,
  dim: int | tuple[int, ...],
  device: wp.DeviceLike = None,
) -> int:
  """Choose block_dim for wp.launch based on CUDA occupancy.

  Uses ``wp.get_suggested_block_size`` to query the optimal block size and
  the minimum grid size to fully utilize all SMs.  When the launch is large
  enough, the suggested block size is returned directly.  For small launches
  where the GPU would be underutilized, a smaller block size is chosen so
  that the work is distributed as evenly as possible across SMs while
  keeping each block as large as possible (so adjacent threads stay on the
  same SM).

  Block sizes are always a multiple of the warp size (32).

  Args:
    kernel: The Warp kernel to launch.
    dim: The launch dimension (scalar or tuple, same as ``wp.launch``).
    device: Target device.  ``None`` = current CUDA device.

  Returns:
    ``block_dim`` value to pass to ``wp.launch``.
  """
  device = wp.get_device(device)
  if device.is_cpu:
    return 256

  # Fast path: exact (kernel, device, dim) seen before.
  dim_key = dim if isinstance(dim, (int, tuple)) else tuple(dim)
  full_key = (id(kernel), device.ordinal, dim_key)
  cached = _BLOCK_DIM_CACHE.get(full_key)
  if cached is not None:
    return cached

  # --- cached occupancy query ---
  occ_key = (id(kernel), device.ordinal)
  occupancy = _OCCUPANCY_CACHE.get(occ_key)
  if occupancy is None:
    occupancy = wp.get_suggested_block_size(kernel, device)
    _OCCUPANCY_CACHE[occ_key] = occupancy
  suggested_block_size, min_grid_size = occupancy

  total_threads = math.prod(dim) if isinstance(dim, (tuple, list)) else int(dim)
  if total_threads == 0:
    _BLOCK_DIM_CACHE[full_key] = suggested_block_size
    return suggested_block_size

  num_blocks = math.ceil(total_threads / suggested_block_size)
  if num_blocks >= min_grid_size:
    _BLOCK_DIM_CACHE[full_key] = suggested_block_size
    return suggested_block_size

  # GPU is underutilized at the suggested block size.  Pick a smaller one
  # that creates enough blocks to fill all SMs.
  sm_count = device.sm_count
  blocks_per_sm = min_grid_size // sm_count

  # Target a total block count that is a multiple of sm_count so every SM
  # gets the same number of blocks.
  target_blocks = sm_count * blocks_per_sm

  # Largest warp-aligned block_size giving >= target_blocks.
  ideal = total_threads // target_blocks
  result = max(_WARP_SIZE, (ideal // _WARP_SIZE) * _WARP_SIZE)

  # Don't increase block_dim beyond the Warp default (256) — the default
  # already gives more blocks (better latency hiding) in the regime where
  # the suggested block size would underutilize the GPU.
  result = min(result, _WARP_DEFAULT_BLOCK_DIM)

  _BLOCK_DIM_CACHE[full_key] = result
  return result


def launch(kernel, dim, **kwargs):
  """Wrapper around ``wp.launch`` that auto-selects ``block_dim``.

  When ``block_dim`` is not given (or is ``None``), it is chosen via
  :func:`get_block_dim`.  All other arguments are forwarded to
  ``wp.launch`` unchanged.
  """
  if kwargs.get("block_dim") is None:
    kwargs["block_dim"] = get_block_dim(kernel, dim, device=kwargs.get("device"))
  wp.launch(kernel, dim=dim, **kwargs)


class EventTracer:
  """Calculates elapsed times of functions annotated with `event_scope`.

  Use as a context manager like so:

    @event_trace
    def my_warp_function(...):
      ...

    with EventTracer() as tracer:
      my_warp_function(...)
      print(tracer.trace())
  """

  def __init__(self, enabled: bool = True):
    global _STACK
    if _STACK is not None:
      raise ValueError("only one EventTracer can run at a time")
    if enabled:
      _STACK = {}

  def __enter__(self):
    return self

  def trace(self) -> dict:
    """Calculates elapsed times for every node of the trace."""
    global _STACK

    if _STACK is None:
      return {}

    ret = {}

    for k, v in _STACK.items():
      events, sub_stack = v
      # push into next level of stack
      saved_stack, _STACK = _STACK, sub_stack
      sub_trace = self.trace()
      # pop!
      _STACK = saved_stack
      events = tuple(wp.get_event_elapsed_time(beg, end) for beg, end in events)
      ret[k] = (events, sub_trace)

    return ret

  def __exit__(self, type, value, traceback):
    global _STACK
    _STACK = None


def _merge(a: dict, b: dict) -> dict:
  """Merges two event trace stacks."""
  ret = {}
  if not a or not b:
    return dict(**a, **b)
  if set(a) != set(b):
    raise ValueError("incompatible stacks")
  for key in a:
    a1_events, a1_substack = a[key]
    a2_events, a2_substack = b[key]
    ret[key] = (a1_events + a2_events, _merge(a1_substack, a2_substack))
  return ret


def event_scope(fn, name: str = ""):
  """Wraps a function and records an event before and after the function invocation."""
  name = name or getattr(fn, "__name__")

  @functools.wraps(fn)
  def wrapper(*args, **kwargs):
    global _STACK
    if _STACK is None:
      return fn(*args, **kwargs)

    for frame_info in inspect.stack():
      if frame_info.function in ("capture_while", "capture_if"):
        return fn(*args, **kwargs)

    # push into next level of stack
    saved_stack, _STACK = _STACK, {}
    beg = wp.Event(enable_timing=True)
    end = wp.Event(enable_timing=True)
    wp.record_event(beg)
    res = fn(*args, **kwargs)
    wp.record_event(end)
    # pop back up to current level
    sub_stack, _STACK = _STACK, saved_stack
    # append events and substack
    prev_events, prev_substack = _STACK.get(name, ((), {}))
    events = prev_events + ((beg, end),)
    sub_stack = _merge(prev_substack, sub_stack)
    _STACK[name] = (events, sub_stack)
    return res

  return wrapper


_KERNEL_CACHE = {}


def cache_kernel(func):
  # caching kernels to avoid crashes in graph_conditional code
  @functools.wraps(func)
  def wrapper(*args):
    def _hash_arg(a):
      if hasattr(a, "size"):
        return a.size
      if isinstance(a, list):
        return hash(tuple(a))
      return hash(a)

    key = tuple(_hash_arg(a) for a in args) + (hash(func.__name__),)
    if key not in _KERNEL_CACHE:
      _KERNEL_CACHE[key] = func(*args)
    return _KERNEL_CACHE[key]

  return wrapper


def check_toolkit_driver():
  wp.init()
  if wp.get_device().is_cuda:
    if not wp.is_conditional_graph_supported():
      warnings.warn(
        """
        CUDA version < 12.4 detected
        - graph capture may be unreliable for < 12.3
        - conditional graph nodes are not available for < 12.4
          Model.opt.graph_conditional should be set to False
        """,
        stacklevel=2,
      )


class scoped_mathdx_gemm_disabled:
  """Temporarily disable Warp MathDx GEMM kernels within this scope."""

  def __init__(self, disable: bool = True):
    self._disable = disable
    self._config = None
    self._prev = None

  def __enter__(self):
    if not self._disable:
      return self
    config = getattr(wp, "config", None)
    if config is None or not hasattr(config, "enable_mathdx_gemm"):
      return self
    self._config = config
    self._prev = config.enable_mathdx_gemm
    self._config.enable_mathdx_gemm = False
    return self

  def __exit__(self, exc_type, exc, tb):
    if self._config is not None:
      self._config.enable_mathdx_gemm = self._prev
    return False
