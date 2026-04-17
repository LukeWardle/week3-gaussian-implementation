"""
gaussian_solver.py - Gaussian Elimination Implemented from Scratch

=========================================================================================

Educational implementation exposing every algorithmic detail.
For production use, prefer scipy.linalg.solve() or numpy.linalg.solve().


Functions (in call order):
  swap_rows()                   - Row operation 1: R_i <-> R_j
  multiply_row()                - Row operation 2: R_i <- k·R_i
  add_row_multiple()            - Row operation 3: R_i <- R_i + k·R_j 
  find_pivot()                  - Partial pivoting: find max|value| in column
  forward_elimination()         - Reduce [A|b] to REF, return rank
  back_substitution()           - Solve from REF (Part 2)
  gaussian_solve()              - Top-level public API (Part 2)

Week 3 connection:
  Monday Part 1: theory behind each function
  Wednesday Part 1: geometric interpretation of each function

"""

import numpy as np
from typing import Tuple, Optional

def swap_rows(matrix: np.ndarray, i: int, j: int) -> np.ndarray:
  """
  Swap rows i and j in matrix (in-place, modifies original).

  Geometric meaning: Reorders the equations. Neither line/plane moves;
  only the order in which they appear changes. Intersection unchanged.

  Args:
    matrix: Array to modify (modified in-place)
    i: Index of first row
    j: Index of second row

  Returns:
    Modified matrix (same object - in-place)

  Examples:
    >>> A = np.array([[1, 2], [3, 4]], dtype=float)
    >>> swap-rows(A, 0, 1)
    >>> print(A)
    [[3. 4.]
     [1. 2.]]
  
  """

  matrix[[i, j]] = matrix[[j, i]]
  return matrix

def multiply_row(matrix: np.ndarray, i: int, scalar: float) -> np.ndarray:
  """
  Multiply row i by scalar (in-place).

  Geometric meaning: Scales the equation but leaves the line/plane unchanged.

  2x + y = 6 and x + 0.5y = 3 describe the same set of points.

  Args:
    matrix: Array to modify
    i:      Row index to scale
    scalar: Scale factor. Must be non-zero.

  Raises:
    ValueError: If scalar is zero (destroys equation irreversibly)

  Examples:
    >>> A = np.array([[2, 1, 6]], dtype=float)
    >>> multiply_row(A, 0, 0.5)
    >>> print(A)
    [[1. 0.5 3.]]   <- Same line, simpler equation
  
  """ 

  if abs(scalar) < 1e-15:       # Treat as zero - float comparison
    raise ValueError(
      f"multiply_row() requires non-zero scalar."
      f"Got scalar={scalar}. Multiplying by zero destroys the equation."
    )
  matrix[i] = matrix[i] * scalar
  return matrix

def add_row_multiple(matrix: np.ndarray,
                     target: int,
                     source: int,
                     scalar: float) -> np.ndarray:
  """
  Add scalar x source_row to target_row (in-place).
  Performs: matrix[target] += scalar * matrix[source]

  This is the workhorse of Gaussian elimination. Called O(n²) times:
    - n-1 times to eliminate column 0 (rows 1 through n-1)
    - n-2 times to eliminate column 1 (rows 2 through n-1)
    - ...
    - 1 time to eliminate column n-2
  Total: (n-1) + (n-2) + ... + 1 = n(n-1)/2 ≈ n²/2

  Geometric meaning: Rotates the target line/plane while keeping it 
  anchored at the intersection point (Wednesday Part 1, Section 4).

  Args:
    matrix: Array to modify 
    target: Row to modify (this row changes) 
    source: Row to add from (this row is unchanged)
    scalar: Multiplier for source row. Negative to eliminate.

  Examples:
    >>> A = np.array([[2, 1], [4, 3]], dtype=float)
    >>> # Eliminate A[1,0]: need A[1] = A[1] - 2*A[0]
    >>> add_row_multiple(A, target=1, source=0, scalar=-2)
    >>> print(A)
    [[2. 1.]
     [0. 1.]]   <- A[1, 0] is now 0
  
  """

  matrix[target] = matrix[target] + scalar * matrix[source] 
  return matrix