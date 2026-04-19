# Week 3: Gaussian Elimination from Scratch 
 
  Educational implementation of Gaussian elimination without using numpy.linalg.solve(). 
  Built to demonstrate algorithm understanding for technical interviews and deep learning. 
  Companion to [week3-linear-solvers](https://github.com/LukeWardle/week3-linear-solvers) 
  (production-level implementation using NumPy/SciPy). 
 
## What This Implements 
 
  From scratch, without np.linalg.solve(): 
  - `swap_rows()` — NumPy fancy indexing row swap 
  - `multiply_row()` — with zero-scalar validation 
  - `add_row_multiple()` — the O(n²) elimination workhorse 
  - `find_pivot()` — partial pivoting for numerical stability 
  - `forward_elimination()` — produces REF with rank tracking 
  - `back_substitution()` — handles all three REF patterns 
  - `gaussian_solve()` — complete public API returning solution dict 
 
## Installation 
 
  ```bash 
  git clone https://github.com/LukeWardle/week3-gaussian-implementation.git 
  cd week3-gaussian-implementation 
  python -m venv venv 
  venv\Scripts\activate  # Windows 
  pip install -r requirements.txt 
  ``` 
 
## Quick Start 
 
  ```
  python 
  import numpy as np 
  from gaussian_solver import gaussian_solve 
 
  A = np.array([[2, 1], [1, 2]], dtype=float) 
  b = np.array([5, 4], dtype=float) 
  result = gaussian_solve(A, b) 
  print(result['type'])       # 'unique' 
  print(result['solution'])   # [2. 1.] 
  print(result['rank'])       # 2 
  ``` 
 
## Running Demos 
 
  ```
  bash 
  python main.py 
  ``` 
  Shows: unique solution, no solution, infinite solutions, performance vs NumPy. 
 
## Testing 
 
  ```
  bash 
  pytest test_gaussian_solver.py -v 
  ``` 
  10 tests covering all solution types, validated against np.linalg.solve(). 
 
## Algorithm Details 
 
  1. **Forward elimination** with partial pivoting (O(n³)) 
     - find_pivot(): selects max|value| column entry → small multipliers 
     - add_row_multiple(): called O(n²) times — the computational bottleneck 
     - rank tracking: increments only on successful pivots 
 
  2. **Back-substitution** (O(n²)) 
     - Scans for contradiction rows → raises ValueError 
     - rank < n_vars → returns None (infinite solutions) 
     - range(rank-1, -1, -1) → bottom-to-top traversal 
 
## Performance Note 
 
  This implementation uses Python loops → ~40× slower than np.linalg.solve() 
  (which uses LAPACK's DGESV with vectorised C/Fortran routines). 
  For production, always use numpy/scipy. This project is for understanding. 
 
## Skills Demonstrated 
 
  - Algorithm implementation from specification (not library wrapping) 
  - Numerical stability: partial pivoting, zero-scalar validation 
  - Edge case handling: all three linear system outcomes 
  - Testing strategy: validating against established libraries 
 
## For Production 
  Use `numpy.linalg.solve(A, b)` or `scipy.linalg.solve(A, b)`. 
  See companion repo week3-linear-solvers for production-grade API. 