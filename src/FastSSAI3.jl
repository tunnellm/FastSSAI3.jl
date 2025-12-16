# FastSSAI3.jl - Optimized Julia implementation of SSAI3 preconditioner
#
# Algorithm by Shaked Regev and Michael Saunders, ICME, Stanford University.
# See: https://stanford.edu/group/SOL/reports/20SSAI.pdf
#
# Julia implementation by Marc A. Tunnell
# https://tunnellm.github.io/

module FastSSAI3

using SparseArrays
using LinearAlgebra
using Printf
using Polyester
using LoopVectorization
using VectorizedStatistics

export ssai3

include("parallel_sparse.jl")
include("fastssai3_impl.jl")

end # module
