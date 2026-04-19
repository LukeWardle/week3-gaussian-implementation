"""
main.py - Demonstration of Gaussian elimination from scratch

=========================================================================

Three demos covering all solution types + performance context.

"""

import numpy as np
import time
from gaussian_solver import gaussian_solve

def demo_unique_solution():
  """
  Demo 1: System with unique solution - 3 x 3 from Monday Part 1.

  """
  print("=" * 60)
  print("DEMO 1: Unique Solution")
  print("=" * 60)

  A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float)
  b = np.array([8, -11, -3], dtype=float)

  print("System:")
  print("   2x + y - z = 8")
  print("   -3x - y + 2z = -11")
  print("   -2x + y + 2z = -3")

  result = gaussian_solve(A, b)
  x = result["solution"]

  print(f"\nType:     {result['type']}")
  print(f"Rank:     {result['rank']}")
  print("Solution:  x={[0]:.1f},   y={x[1]:.1f},   z={x[2]:.1f}")
  print(f"Verify:     A @ x = {np.round(A @ x, 4)}")
  print(f"Match:      {np.allclose(A @ x, b)}")
  print()

def demo_no_solution():
  """
  Demo 2: Inconsistent system - geometrically: parallel lines.
  
  """
  print("=" * 60)
  print("DEMO 2: No solution (Inconsistent)")
  print("=" * 60)
  
  A = np.array([[1, 1], [2, 2]], dtype=float)
  b = np.array([3, 7], dtype=float)

  print("System:")
  print("  x + y = 3   (Line 1)")
  print("  2x + 2y = 7 (Line 2 — parallel to Line 1)")
  print("\n  Geometrically: parallel lines never intersect.")
  print("  Algebraically: 2×(x+y) = 2×3 = 6 ≠ 7 → contradiction")

  result = gaussian_solve(A, b)
  print(f"\nType:     {result['type']}  (no x satisfies both equations)")
  print(f"Rank:     {result['rank']}")
  print(f"Solution: {result['solution']}")
  print()

def demo_infinite_solutions():
  """
  Demo 3: Underdetermined — geometrically: identical lines.
  
  """
  print("=" * 60)
  print("DEMO 3: Infinite Solutions (Underdetermined)")
  print("=" * 60)

  A = np.array([[1, 2], [2, 4]], dtype=float)
  b = np.array([3, 6], dtype=float)
  
  print("System:")
  print("  x + 2y = 3   (Line 1)")
  print("  2x + 4y = 6  (Line 2 = 2 × Line 1 — identical)")
  print("\n  Geometrically: identical lines → every point is a solution")
  print("  Solution family: x = 3 - 2t,  y = t  for any t ∈ ℝ")

  result = gaussian_solve(A, b)
  print(f"\nType:     {result['type']}")
  print(f"Rank:     {result['rank']}  (only 1 independent equation)")
  print(f"Solution: {result['solution']}  (None — infinite options)")
  print()

def demo_performance_context():
  """
  Demo 4: Performance context — from-scratch vs np.linalg.solve.
  
  """
  print("=" * 60)
  print("DEMO 4: Performance Context")
  print("=" * 60)

  n = 50
  np.random.seed(42)
  A = np.random.randn(n, n) + np.eye(n) * n         # well-conditioned
  b = np.random.randn(n)

  # Our implementation 
  t0 = time.perf_counter()
  result = gaussian_solve(A, b)
  t_scratch = time.perf_counter() - t0

  # NumPy (LAPACK)
  t0 = time.perf_counter()
  x_numpy = np.linalg.solve(A, b)
  t_numpy = time.perf_counter() - t0

  print(f"Matrix size: {n}×{n}")
  print(f"Our solver:  {t_scratch*1000:.2f} ms")
  print(f"np.linalg:  {t_numpy*1000:.2f} ms")
  print(f"Speed ratio: {t_scratch/t_numpy:.0f}× slower (Python vs LAPACK)")
  print()

  # But correctness matches
  match = np.allclose(result["solution"], x_numpy, rtol=1e-8)
  print(f"Correctness: solutions match = {match}")
  print("Conclusion: from-scratch is slower, but mathematically identical.")
  print()

if __name__ == "__main__": 
  demo_unique_solution()
  demo_no_solution()
  demo_infinite_solutions()
  demo_performance_context()
  print("All demos complete. Run pytest test_gaussian_solver.py -v for tests.")