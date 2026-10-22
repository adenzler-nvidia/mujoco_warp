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
    matrix_size: int,
    # Out:
    U_out: wp.array2d[float],
    result_out: wp.array2d[float],
  ):
    """Factor A with b as an augmented border and reuse it as the forward solution."""
    rhs_tile = wp.tile_zeros(shape=(matrix_size_static, 1), dtype=float, storage="shared")

    for k in range(0, matrix_size, block_size):
      end = k + block_size
      input_rhs = wp.tile_load(b, shape=(block_size, 1), offset=(k, 0), storage="shared", bounds_check=False)
      A_kk_tile = wp.tile_load(
        A, shape=(block_size, block_size), offset=(k, k), storage="shared", bounds_check=False, aligned=True
      )
      input_diagonal_rhs = wp.tile_view(input_rhs, shape=(border_size, 1), offset=(0, 0))
      if end == matrix_size:
        wp.tile_assign(A_kk_tile, input_diagonal_rhs, offset=(0, border_size))
        # Keep the augmented rank-one correction below float32 precision.
        A_kk_tile[border_size, border_size] = 1.0e30

      for j in range(0, k, block_size):
        U_block = wp.tile_load(
          U_out, shape=(block_size, block_size), offset=(j, k), storage="shared", bounds_check=False, aligned=True
        )
        wp.tile_matmul(wp.tile_transpose(U_block), U_block, A_kk_tile, alpha=-1.0)

      wp.tile_cholesky_inplace(A_kk_tile, fill_mode="upper")
      diagonal_border = wp.tile_view(A_kk_tile, shape=(border_size, 1), offset=(0, border_size))
      if end == matrix_size:
        wp.tile_assign(rhs_tile, diagonal_border, offset=(k, 0))
      wp.tile_store(U_out, A_kk_tile, offset=(k, k), bounds_check=False, aligned=True)

      for i in range(end, matrix_size, block_size):
        A_ki_tile = wp.tile_load(
          A, shape=(block_size, block_size), offset=(k, i), storage="shared", bounds_check=False, aligned=True
        )
        if i + block_size == matrix_size:
          wp.tile_assign(A_ki_tile, input_rhs, offset=(0, border_size))

        for j in range(0, k, block_size):
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
      for j in range(i_end, matrix_size, block_size):
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
    matrix_size: int,
    # Out:
    result_out: wp.array2d[float],
  ):
    """Block Cholesky solve.

    Solves A x = b given the Cholesky factor U (A = U^T U) using blocked forward and backward
    substitution.
    """
    rhs_tile = wp.tile_load(b, shape=(matrix_size_static, 1), offset=(0, 0), storage="shared", bounds_check=False)

    # Forward substitution: solve U^T y = b
    for i in range(0, matrix_size, block_size):
      rhs_view = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(i, 0))
      for j in range(0, i, block_size):
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


@lru_cache(maxsize=None)
def create_pool_cholesky_newton_func(block_size: int, max_size_static: int, factorize: bool):
  """Blocked Cholesky (factor and) solve of one island block in a tile pool, for Newton.

  The pool is a 2D array of stacked block_size x block_size tiles (tile t is rows
  [t * block_size, (t + 1) * block_size)); an island of ntile x ntile tiles starts at tile tilebase
  and stores tile (i, j) at tilebase + i * ntile + j. Its right-hand side and search direction
  occupy rows [vbase, vbase + ntile * block_size) of b and search_out.
  """
  FACTORIZE = factorize

  @wp.func
  def pool_cholesky_newton_func(
    # In:
    A: wp.array2d[float],
    b: wp.array2d[float],
    tilebase: int,
    ntile: int,
    vbase: int,
    # Out:
    U_out: wp.array2d[float],
    search_out: wp.array2d[float],
  ) -> wp.vec2:
    rhs_tile = wp.tile_load(b, shape=(max_size_static, 1), offset=(vbase, 0), storage="shared", bounds_check=True)

    # factorization fused with the forward solve U^T y = b
    for k in range(ntile):
      rhs_view = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(k * block_size, 0))
      if wp.static(FACTORIZE):
        A_kk = wp.tile_load(
          A,
          shape=(block_size, block_size),
          offset=((tilebase + k * ntile + k) * block_size, 0),
          storage="shared",
          bounds_check=False,
          aligned=True,
        )
      for j in range(k):
        U_jk = wp.tile_load(
          U_out,
          shape=(block_size, block_size),
          offset=((tilebase + j * ntile + k) * block_size, 0),
          storage="shared",
          bounds_check=False,
          aligned=True,
        )
        if wp.static(FACTORIZE):
          wp.tile_matmul(wp.tile_transpose(U_jk), U_jk, A_kk, alpha=-1.0)
        y_j = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(j * block_size, 0))
        wp.tile_matmul(wp.tile_transpose(U_jk), y_j, rhs_view, alpha=-1.0)

      if wp.static(FACTORIZE):
        wp.tile_cholesky_inplace(A_kk, fill_mode="upper")
        wp.tile_store(U_out, A_kk, offset=((tilebase + k * ntile + k) * block_size, 0), bounds_check=False, aligned=True)
        wp.tile_lower_solve_inplace(wp.tile_transpose(A_kk), rhs_view)
        for i in range(k + 1, ntile):
          A_ki = wp.tile_load(
            A,
            shape=(block_size, block_size),
            offset=((tilebase + k * ntile + i) * block_size, 0),
            storage="shared",
            bounds_check=False,
            aligned=True,
          )
          for j in range(k):
            U_jk = wp.tile_load(
              U_out,
              shape=(block_size, block_size),
              offset=((tilebase + j * ntile + k) * block_size, 0),
              storage="shared",
              bounds_check=False,
              aligned=True,
            )
            U_ji = wp.tile_load(
              U_out,
              shape=(block_size, block_size),
              offset=((tilebase + j * ntile + i) * block_size, 0),
              storage="shared",
              bounds_check=False,
              aligned=True,
            )
            wp.tile_matmul(wp.tile_transpose(U_jk), U_ji, A_ki, alpha=-1.0)
          wp.tile_lower_solve_inplace(wp.tile_transpose(A_kk), A_ki)
          wp.tile_store(U_out, A_ki, offset=((tilebase + k * ntile + i) * block_size, 0), bounds_check=False, aligned=True)
      else:
        U_kk = wp.tile_load(
          U_out,
          shape=(block_size, block_size),
          offset=((tilebase + k * ntile + k) * block_size, 0),
          storage="shared",
          bounds_check=False,
          aligned=True,
        )
        wp.tile_lower_solve_inplace(wp.tile_transpose(U_kk), rhs_view)

    # backward solve U x = y
    for kk in range(ntile):
      i = ntile - 1 - kk
      x_i = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(i * block_size, 0))
      for j in range(i + 1, ntile):
        U_ij = wp.tile_load(
          U_out,
          shape=(block_size, block_size),
          offset=((tilebase + i * ntile + j) * block_size, 0),
          storage="shared",
          bounds_check=False,
          aligned=True,
        )
        x_j = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(j * block_size, 0))
        wp.tile_matmul(U_ij, x_j, x_i, alpha=-1.0)
      U_ii = wp.tile_load(
        U_out,
        shape=(block_size, block_size),
        offset=((tilebase + i * ntile + i) * block_size, 0),
        storage="shared",
        bounds_check=False,
        aligned=True,
      )
      wp.tile_upper_solve_inplace(U_ii, x_i)

    # search = -x, and the (|x|^2, b.x) sums of the Newton decrement
    sums = wp.vec2(0.0)
    for a in range(ntile):
      x_a = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(a * block_size, 0))
      b_a = wp.tile_load(b, shape=(block_size, 1), offset=(vbase + a * block_size, 0), bounds_check=False)
      wp.tile_store(search_out, wp.tile_map(wp.mul, x_a, -1.0), offset=(vbase + a * block_size, 0), bounds_check=False)
      sums += wp.tile_reduce(wp.add, wp.tile_map(solve_search_sums, b_a, x_a))[0]
    return sums

  return pool_cholesky_newton_func
