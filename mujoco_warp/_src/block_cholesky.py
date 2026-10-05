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

from functools import lru_cache

import warp as wp


@wp.func
def solve_search_sums(grad: float, solution: float):
  return wp.vec2(solution * solution, grad * solution)


@lru_cache(maxsize=None)
def _create_newton_decrement_func(matrix_size_static: int, vector_size_static: int):
  @wp.func
  def newton_decrement_func(
    # In:
    solution_tile: wp.tile[float, matrix_size_static, 1],
    b: wp.array2d[float],
    # Out:
    search_out: wp.array2d[float],
  ):
    grad_tile = wp.tile_load(b, shape=(vector_size_static, 1), offset=(0, 0), bounds_check=False)
    active_solution = wp.tile_view(solution_tile, shape=(vector_size_static, 1), offset=(0, 0))
    wp.tile_store(search_out, wp.tile_map(wp.mul, active_solution, -1.0), bounds_check=False)
    return wp.tile_reduce(wp.add, wp.tile_map(solve_search_sums, grad_tile, active_solution))[0]

  return newton_decrement_func


@lru_cache(maxsize=None)
def create_blocked_cholesky_factorize_solve_func(block_size: int, matrix_size_static: int):
  @wp.func
  def blocked_cholesky_factorize_solve_func(
    # In:
    A: wp.array2d[float],
    b: wp.array2d[float],
    matrix_size: int,
    # Out:
    U: wp.array2d[float],
    x: wp.array2d[float],
  ):
    """Block Cholesky factorization and solve while keeping the forward RHS live."""
    rhs_tile = wp.tile_load(b, shape=(matrix_size_static, 1), offset=(0, 0), storage="shared", bounds_check=False)

    for k in range(0, matrix_size, block_size):
      end = k + block_size
      rhs_view = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(k, 0))

      A_kk_tile = wp.tile_load(
        A, shape=(block_size, block_size), offset=(k, k), storage="shared", bounds_check=False, aligned=True
      )

      for j in range(0, k, block_size):
        U_block = wp.tile_load(
          U, shape=(block_size, block_size), offset=(j, k), storage="shared", bounds_check=False, aligned=True
        )
        wp.tile_matmul(wp.tile_transpose(U_block), U_block, A_kk_tile, alpha=-1.0)

        y_block = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(j, 0))
        wp.tile_matmul(wp.tile_transpose(U_block), y_block, rhs_view, alpha=-1.0)

      wp.tile_cholesky_inplace(A_kk_tile, fill_mode="upper")
      wp.tile_store(U, A_kk_tile, offset=(k, k), bounds_check=False, aligned=True)

      wp.tile_lower_solve_inplace(wp.tile_transpose(A_kk_tile), rhs_view)

      for i in range(end, matrix_size, block_size):
        A_ki_tile = wp.tile_load(
          A, shape=(block_size, block_size), offset=(k, i), storage="shared", bounds_check=False, aligned=True
        )

        for j in range(0, k, block_size):
          U_jk_tile = wp.tile_load(
            U, shape=(block_size, block_size), offset=(j, k), storage="shared", bounds_check=False, aligned=True
          )
          U_ji_tile = wp.tile_load(
            U, shape=(block_size, block_size), offset=(j, i), storage="shared", bounds_check=False, aligned=True
          )
          wp.tile_matmul(wp.tile_transpose(U_jk_tile), U_ji_tile, A_ki_tile, alpha=-1.0)

        wp.tile_lower_solve_inplace(wp.tile_transpose(A_kk_tile), A_ki_tile)
        wp.tile_store(U, A_ki_tile, offset=(k, i), bounds_check=False, aligned=True)

    for i in range(matrix_size - block_size, -1, -block_size):
      i_end = i + block_size
      tmp_tile = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(i, 0))
      for j in range(i_end, matrix_size, block_size):
        U_tile = wp.tile_load(
          U, shape=(block_size, block_size), offset=(i, j), storage="shared", bounds_check=False, aligned=True
        )
        x_tile = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(j, 0))
        wp.tile_matmul(U_tile, x_tile, tmp_tile, alpha=-1.0)

      U_tile = wp.tile_load(
        U, shape=(block_size, block_size), offset=(i, i), storage="shared", bounds_check=False, aligned=True
      )
      wp.tile_upper_solve_inplace(U_tile, tmp_tile)

    wp.tile_store(x, rhs_tile, offset=(0, 0), bounds_check=False)

  return blocked_cholesky_factorize_solve_func


@lru_cache(maxsize=None)
def _create_blocked_cholesky_augmented_factorize_solve_func(
  block_size: int,
  matrix_size_static: int,
  with_newton_decrement: bool,
  vector_size_static: int,
):
  WITH_NEWTON_DECREMENT = with_newton_decrement
  border_size = block_size - 1

  @wp.func
  def blocked_cholesky_augmented_factorize_solve_func(
    # In:
    A: wp.array2d[float],
    b: wp.array2d[float],
    top: wp.array[int],
    matrix_size: int,
    # Out:
    U_out: wp.array2d[float],
    result_out: wp.array2d[float],
  ):
    """Factor A with b as an augmented border and reuse it as the forward solution.

    top[t] is the first nonzero row of tile column t (nondecreasing; all zeros for dense).
    Cholesky fill stays inside this envelope, so tiles above it are neither read nor written.
    """
    rhs_tile = wp.tile_zeros(shape=(matrix_size_static, 1), dtype=float, storage="shared")
    last = matrix_size - block_size

    for k in range(0, matrix_size, block_size):
      end = k + block_size
      top_k = top[k // block_size]
      input_rhs = wp.tile_load(b, shape=(block_size, 1), offset=(k, 0), storage="shared", bounds_check=False)
      A_kk_tile = wp.tile_load(
        A, shape=(block_size, block_size), offset=(k, k), storage="shared", bounds_check=False, aligned=True
      )
      input_diagonal_rhs = wp.tile_view(input_rhs, shape=(border_size, 1), offset=(0, 0))
      if end == matrix_size:
        wp.tile_assign(A_kk_tile, input_diagonal_rhs, offset=(0, border_size))
        # Keep the augmented rank-one correction below float32 precision.
        A_kk_tile[border_size, border_size] = 1.0e30

      for j in range(top_k, k, block_size):
        U_block = wp.tile_load(
          U_out, shape=(block_size, block_size), offset=(j, k), storage="shared", bounds_check=False, aligned=True
        )
        wp.tile_matmul(wp.tile_transpose(U_block), U_block, A_kk_tile, alpha=-1.0)

      wp.tile_cholesky_inplace(A_kk_tile, fill_mode="upper")
      diagonal_border = wp.tile_view(A_kk_tile, shape=(border_size, 1), offset=(0, border_size))
      if end == matrix_size:
        wp.tile_assign(rhs_tile, diagonal_border, offset=(k, 0))
      wp.tile_store(U_out, A_kk_tile, offset=(k, k), bounds_check=False, aligned=True)

      # Row k's envelope tiles are contiguous since top is nondecreasing. The last tile always
      # follows: its border column carries the forward solution.
      i_stop = end
      while i_stop < last and top[i_stop // block_size] <= k:
        i_stop += block_size
      i_hi = wp.where(end < matrix_size, i_stop + block_size, end)
      for i_env in range(end, i_hi, block_size):
        i = wp.where(i_env == i_stop, last, i_env)
        A_ki_tile = wp.tile_load(
          A, shape=(block_size, block_size), offset=(k, i), storage="shared", bounds_check=False, aligned=True
        )
        if i + block_size == matrix_size:
          wp.tile_assign(A_ki_tile, input_rhs, offset=(0, border_size))

        j_lo = wp.where(i == last, top_k, wp.max(top_k, top[i // block_size]))
        for j in range(j_lo, k, block_size):
          U_jk_tile = wp.tile_load(
            U_out, shape=(block_size, block_size), offset=(j, k), storage="shared", bounds_check=False, aligned=True
          )
          U_ji_tile = wp.tile_load(
            U_out, shape=(block_size, block_size), offset=(j, i), storage="shared", bounds_check=False, aligned=True
          )
          wp.tile_matmul(wp.tile_transpose(U_jk_tile), U_ji_tile, A_ki_tile, alpha=-1.0)

        wp.tile_lower_solve_inplace(wp.tile_transpose(A_kk_tile), A_ki_tile)
        panel_border = wp.tile_view(A_ki_tile, shape=(block_size, 1), offset=(0, border_size))
        if i + block_size == matrix_size:
          wp.tile_assign(rhs_tile, panel_border, offset=(k, 0))
        wp.tile_store(U_out, A_ki_tile, offset=(k, i), bounds_check=False, aligned=True)

    for i in range(matrix_size - block_size, -1, -block_size):
      i_end = i + block_size
      tmp_tile = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(i, 0))
      # the border entry of x is zero, so U's border column adds nothing outside the envelope
      j_stop = i_end
      while j_stop < matrix_size and top[j_stop // block_size] <= i:
        j_stop += block_size
      for j in range(i_end, j_stop, block_size):
        U_tile = wp.tile_load(
          U_out, shape=(block_size, block_size), offset=(i, j), storage="shared", bounds_check=False, aligned=True
        )
        x_tile = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(j, 0))
        wp.tile_matmul(U_tile, x_tile, tmp_tile, alpha=-1.0)

      U_tile = wp.tile_load(
        U_out, shape=(block_size, block_size), offset=(i, i), storage="shared", bounds_check=False, aligned=True
      )
      wp.tile_upper_solve_inplace(U_tile, tmp_tile)

    sums = wp.vec2(0.0)
    if wp.static(WITH_NEWTON_DECREMENT):
      sums = wp.static(_create_newton_decrement_func(matrix_size_static, vector_size_static))(rhs_tile, b, result_out)
    else:
      wp.tile_store(result_out, rhs_tile, offset=(0, 0), bounds_check=False)

    return sums

  return blocked_cholesky_augmented_factorize_solve_func


@lru_cache(maxsize=None)
def _create_blocked_cholesky_solve_func(
  block_size: int,
  matrix_size_static: int,
  with_newton_decrement: bool,
  vector_size_static: int,
):
  WITH_NEWTON_DECREMENT = with_newton_decrement

  @wp.func
  def blocked_cholesky_solve_func(
    # In:
    U: wp.array2d[float],
    b: wp.array2d[float],
    top: wp.array[int],
    matrix_size: int,
    # Out:
    result_out: wp.array2d[float],
  ):
    """Block Cholesky solve.

    Solves A x = b given the Cholesky factor U (A = U^T U) using blocked forward and backward
    substitution. top is the tile envelope U was factored with.
    """
    rhs_tile = wp.tile_load(b, shape=(matrix_size_static, 1), offset=(0, 0), storage="shared", bounds_check=False)

    # Forward substitution: solve U^T y = b
    for i in range(0, matrix_size, block_size):
      rhs_view = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(i, 0))
      for j in range(top[i // block_size], i, block_size):
        U_block = wp.tile_load(
          U, shape=(block_size, block_size), offset=(j, i), storage="shared", bounds_check=False, aligned=True
        )
        y_block = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(j, 0))
        wp.tile_matmul(wp.tile_transpose(U_block), y_block, rhs_view, alpha=-1.0)

      U_tile = wp.tile_load(
        U, shape=(block_size, block_size), offset=(i, i), storage="shared", bounds_check=False, aligned=True
      )
      wp.tile_lower_solve_inplace(wp.tile_transpose(U_tile), rhs_view)

    # Backward substitution: solve U x = y
    for i in range(matrix_size - block_size, -1, -block_size):
      i_end = i + block_size
      tmp_tile = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(i, 0))
      j_stop = i_end
      while j_stop < matrix_size and top[j_stop // block_size] <= i:
        j_stop += block_size
      for j in range(i_end, j_stop, block_size):
        U_tile = wp.tile_load(
          U, shape=(block_size, block_size), offset=(i, j), storage="shared", bounds_check=False, aligned=True
        )
        x_tile = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(j, 0))
        wp.tile_matmul(U_tile, x_tile, tmp_tile, alpha=-1.0)
      U_tile = wp.tile_load(
        U, shape=(block_size, block_size), offset=(i, i), storage="shared", bounds_check=False, aligned=True
      )

      wp.tile_upper_solve_inplace(U_tile, tmp_tile)

    sums = wp.vec2(0.0)
    if wp.static(WITH_NEWTON_DECREMENT):
      sums = wp.static(_create_newton_decrement_func(matrix_size_static, vector_size_static))(rhs_tile, b, result_out)
    else:
      wp.tile_store(result_out, rhs_tile, offset=(0, 0), bounds_check=False)

    return sums

  return blocked_cholesky_solve_func


def create_blocked_cholesky_augmented_factorize_solve_func(block_size: int, matrix_size_static: int):
  return _create_blocked_cholesky_augmented_factorize_solve_func(block_size, matrix_size_static, False, 0)


def create_blocked_cholesky_augmented_factorize_solve_newton_func(
  block_size: int, matrix_size_static: int, vector_size_static: int
):
  return _create_blocked_cholesky_augmented_factorize_solve_func(block_size, matrix_size_static, True, vector_size_static)


def create_blocked_cholesky_solve_func(block_size: int, matrix_size_static: int):
  return _create_blocked_cholesky_solve_func(block_size, matrix_size_static, False, 0)


def create_blocked_cholesky_solve_newton_func(block_size: int, matrix_size_static: int, vector_size_static: int):
  return _create_blocked_cholesky_solve_func(block_size, matrix_size_static, True, vector_size_static)
