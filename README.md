# FastSSAI3.jl

A high-performance Julia implementation of the SSAI3 (Symmetric Sparse Approximate Inverse) preconditioner.

## Overview

FastSSAI3 constructs a sparse approximate inverse preconditioner M ≈ A⁻¹ for symmetric positive definite matrices. The algorithm was originally developed by Shaked Regev and Michael Saunders at Stanford University.

This high-performance Julia implementation was developed by [Marc A. Tunnell](https://tunnellm.github.io/). See also the [CAPPA benchmark repository](https://tunnellm.github.io/CAPPA-repository/).

**Reference:** [SSAI Technical Report](https://stanford.edu/group/SOL/reports/20SSAI.pdf)

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/tunnellm/FastSSAI3.jl")
```

## Usage

```julia
using FastSSAI3
using SparseArrays, LinearAlgebra

# Create a symmetric positive definite sparse matrix
A = sprand(10000, 10000, 0.001) + 10I
A = (A + A') / 2  # symmetrize

# Compute preconditioner with default fill_factor=1.0
M, work = ssai3(A)

# Compute preconditioner with higher fill (denser M, better approximation)
M, work = ssai3(A; fill_factor=2.0)
```

## Parameters

- `A`: Symmetric positive definite sparse matrix (assumed to have unit diagonal)
- `fill_factor`: Multiplier for average column degree to determine target fill level (default: 1.0)
  - `lfil = ceil(fill_factor * avg_nnz_per_col)`
  - Higher values produce denser M with better approximation quality

## Performance

This implementation achieves significant speedups over MATLAB:

| Matrix | Size | MATLAB | Julia (8 threads) | Speedup | Rel Frobenius Diff† |
|--------|------|--------|-------------------|---------|---------------------|
| ecology2 | 1M × 1M | 320.7s | 0.37s | **862x** | 2.07e-16 |
| crankseg_1 | 53K × 53K | 36.8s | 0.45s | **82x** | 4.11e-16 |

†Relative Frobenius norm difference between Julia and MATLAB implementations. ecology2: Julia `fill_factor=3.0` / MATLAB `lfil=15`. crankseg_1: Julia `fill_factor=0.25` / MATLAB `lfil=51`.

## Features

- **Multi-threaded:** Uses [Polyester.jl](https://github.com/JuliaSIMD/Polyester.jl) for lightweight threading
- **Vectorized:** Uses [LoopVectorization.jl](https://github.com/JuliaSIMD/LoopVectorization.jl) for SIMD operations
- **Direct CSC construction:** Builds the sparse result directly without intermediate triplet storage

## License

MIT License

## Citation

If you use this software, please cite the original SSAI paper:

```
Shaked Regev and Michael Saunders
ICME, Stanford University
https://stanford.edu/group/SOL/reports/20SSAI.pdf
```
