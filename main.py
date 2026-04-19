"""
main.py - Testing Gaussian elimination implementation

"""

import numpy as np
from gaussian_solver import forward_elimination

def test_forward_elimination():
  print("=" * 55)
  print("Testing forward_elimination()")
  print("=" * 55)

  # Test 1: Monday Part 1 example
  # 2x + y - z = 8
  # -3x - y + 2z = -11
  # -2x + y + 2z = -3
  aug1 = np.array([
    [2, 1, -1, 8],
    [-3, -1, 2, -11],
    [-2, 1, 2, -3]], dtype=float)
  
  print("\nTest 1: Monday Part 1 worked example")
  print("Augumented matrix [A|b]:")
  print(aug1)

  ref1, rank1 = forward_elimination(aug1.copy())
  print(f"\nRow Echelon Form (rank = {rank1}):")
  print(np.round(ref1, 4))
  print("Expected rank: 3 - OK if rank1==3, else FAIL")

  # Test 2: Larger 3x3 example with pivoting needed
  aug2 = np.array([
    [2, 1, -1, 8],
    [4, 3, 1, 20],
    [2, 1, 3, 14]], dtype=float)
  
  print("\nTest 2: 3x3 system (partial pivoting needed)")
  ref2, rank2 = forward_elimination(aug2.copy())
  print(f"REF (rank = {rank2}):")
  print(np.round(ref2, 4))


  # Test 3: Rank-deficient matrix
  aug3 = np.array([
    [1, 2, 3],
    [2, 4, 6]], dtype=float)    # Row 2 = 2 x Row 1
  
  print("\nTest 3: Rank-deficient (row 2 = 2 x row 1)")
  ref3, rank3 = forward_elimination(aug3.copy())
  print(f"REF (rank = {rank3}): (expected rank = 1)")
  print(np.round(ref3, 4))

if __name__ == "__main__": 
  test_forward_elimination()