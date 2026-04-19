"""
test_gaussian_solver.py - Test suite for from-scratch Gaussian elimination

===================================================================================

Tests four categories:
  1. TestUniqueSolution         - well-conditioned, pivoting needed
  2. TestNoSolution             - inconsistent (parallel lines)
  3. TestInfinitieSolutions     - underdetermined (identical rows)
  4. TestComparison             - validate against np.linalg.solve()

Run: pytest test_gaussian_solver.py -v

"""
import numpy as np
import pytest
from gaussian_solver import gaussian_solve

class TestUniqueSolution:
  """
  Tests for systems with exactly one solution
  
  """
  def test_simple_2x2(self):
    """Happy path: well-conditioned 2x2 with known answer."""
    A = np.array([[2, 1], [1, 2]], dtype=float)
    b = np.array([5, 4], dtype=float)
    result = gaussian_solve(A, b)
    assert result["type"] == "unique"
    assert  result["rank"] == 2
    np.testing.assert_allclose(result["solution"], [2, 1], rtol=1e-10)

  def test_needs_pivoting(self):
    """System where first entry is 0 - partial pivoting required."""
    # If partial pivoting absent, first pivot = 0 -> division by zero
    A = np.array([[0, 1], [1, 1]], dtype=float)
    b = np.array([2, 3], dtype=float)
    result = gaussian_solve(A, b)
    assert result["type"] == "unique"
    np.testing.assert_allclose(result["solution"], [1, 2], rtol=1e-10)

  def test_3x3_monday_example(self):
    """Validate against Monday Part 1 worked example."""
    A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float)
    b = np.array([8, -11, -3], dtype=float)
    result = gaussian_solve(A, b)
    assert result["type"] == "unique"
    assert result["rank"] == 3
    # Verify by substitution - more robust than fixed-value check
    x = result["solution"]
    np.testing.assert_allclose(A @ x, b, rtol=1e-10)

class TestNoSolution:
  """
  Tests for inconsistent systems with no solution.
  
  """
  def test_parallel_lines_2x2(self):
    """Parallel lines: same slope, different intercept."""
    A = np.array([[1, 1], [2, 2]], dtype=float)
    b = np.array([3, 7], dtype=float)     # 2 * 3 = 6 != 7
    result = gaussian_solve(A, b)
    assert result["type"] == "none"
    assert result["solution"] is None

  def test_contradictory_3x3(self):
    """Larger system with hidden contradiction in last row."""
    A = np.array([[1, 2, 3], [2, 4, 6], [1, 2, 3]], dtype=float)
    b = np.array([6, 12, 9], dtype=float)     # Third b contradicts first
    result = gaussian_solve(A, b)
    assert result["type"] == "none"

  
class TestInfiniteSolutions:
  """
  Tests for undetermined systems with infinite solutions.

  """
  def test_identical_rows(self):
    """Second row is a multiple of first - redundant equation."""
    A = np.array([[1, 2], [2, 4]], dtype=float)
    b = np.array([3, 6], dtype=float)         # 2 * 3 = 6 -> consistent
    result = gaussian_solve(A, b)
    assert result["type"] == "infinite"
    assert result["rank"] == 1                # Only one independent row
    assert result["solution"] is None         # Cannot give one answer

  def test_underdetermined_3x3(self):
    """3x3 with two identical rows - 2D solution space."""
    A = np.array([[1, 2, 3], [2, 4, 6], [0, 1, 1]], dtype=float)
    b = np.array([6, 12, 2], dtype=float)
    result = gaussian_solve(A, b)
    assert result["type"] == "infinite"
    assert result["rank"] == 2                # Row 1 = 2 * row 0


class TestComparison:
  """
  Validate our implementation against np.linalg.solve().
  
  """
  def test_matches_numpy_3x3(self):
    """Our solution must match NumPy to rtol=1e-9."""
    A = np.array([[3, 2, -1], [2, -2, 4], [-1, 0.5, -1]], dtype=float)
    b = np.array([1, -2, 0], dtype=float)
    our = gaussian_solve(A, b)
    numpy_sol = np.linalg.solve(A, b)
    assert our["type"] == "unique"
    np.testing.assert_allclose(our["solution"], numpy_sol, rtol=1e-9)

  def test_identity_matrix(self):
    """Trivial case: I @ x = b -> x = b."""
    A = np.eye(4)
    b = np.array([1.0, 2.0, 3.0, 4.0])
    result = gaussian_solve(A, b)
    assert result["type"] == "unique"
    np.testing.assert_allclose(result["solution"], b)

  def test_column_vector_b(self):
    """Edge case: b passed as column vector shape (n, 1) not (n,)."""
    A = np.array([[2, 1], [1, 2]], dtype=float)
    b = np.array([[5], [4]], dtype=float)           # shape (2, 1) not (2,)
    result = gaussian_solve(A, b)
    assert result["type"] == "unique"
    np.testing.assert_allclose(result["solution"], [2, 1], rtol=1e-10)