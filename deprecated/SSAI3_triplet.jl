# SSAI3_triplet.jl - Triplet-based Julia implementation of SSAI3 preconditioner
#
# DEPRECATED: This implementation is mathematically equivalent to FastSSAI3.jl
# but ~10% slower. Use FastSSAI3.jl instead.
# Kept for reference as a simpler triplet-based implementation.
#
# Originally implemented by Shaked Regev and Michael Saunders, ICME, Stanford University.
# See: https://stanford.edu/group/SOL/reports/20SSAI.pdf

using SparseArrays, LinearAlgebra, Printf
using Polyester, VectorizedStatistics

include("../src/parallel_sparse.jl")

"""
    M, approx_work = SSAI3_triplet(A, n, lfil)

Constructs a SPAI-type preconditioner M ≈ A⁻¹ for symmetric positive definite A.

Originally implemented by Shaked Regev and Michael Saunders, ICME, Stanford University.
Reference: https://stanford.edu/group/SOL/reports/20SSAI.pdf

Uses triplet storage and Polyester.@batch for threading.
"""
function SSAI3_triplet(A::SparseMatrixCSC{T, Int}, n::Int, lfil::Int) where T <: Real

    nthreads = Threads.nthreads()
    itmax = 2 * lfil

    # Per-thread storage for output triplets (preallocated 1D arrays per thread)
    # ~2x space needed since off-diagonals stored symmetrically
    cols_per_thread = cld(n, nthreads)
    max_triplets = 2 * lfil * cols_per_thread
    I_idx = [Vector{Int}(undef, max_triplets) for _ in 1:nthreads]
    J_idx = [Vector{Int}(undef, max_triplets) for _ in 1:nthreads]
    V_val = [Vector{T}(undef, max_triplets) for _ in 1:nthreads]
    triplet_count = zeros(Int, nthreads)
    approx_work = zeros(Int, nthreads)
    nneg_t = zeros(Int, nthreads)
    nzero_t = zeros(Int, nthreads)

    # Per-thread workspace for m: small arrays since nnz(m) <= lfil
    m_idx = zeros(Int, lfil + 1, nthreads)
    m_val = zeros(T, lfil + 1, nthreads)

    # Per-thread workspace for r: dense values + index tracking
    r_ws = zeros(T, n, nthreads)
    max_col_nnz = vmaximum(@view(A.colptr[2:end]) .- @view(A.colptr[1:end-1]))
    max_r_nnz = 1 + itmax * max_col_nnz
    r_idx = zeros(Int, max_r_nnz, nthreads)

    # Precompute squared column norms of A (parallel)
    Anorms = Vector{T}(undef, n)
    @batch per=core for j in 1:n
        col_start = A.colptr[j]
        col_end = A.colptr[j+1] - 1
        s = zero(T)
        @inbounds for p in col_start:col_end
            s += A.nzval[p]^2
        end
        @inbounds Anorms[j] = s
    end
    approx_work[1] += A.colptr[n+1] - A.colptr[n]  # nnz(A[:,n])

    # Parallel loop over columns using Polyester
    @batch per=core for j in 1:n
        tid = Threads.threadid()

        # Get thread-local workspaces
        m_i = @view m_idx[:, tid]
        m_v = @view m_val[:, tid]
        r = @view r_ws[:, tid]
        r_nz = @view r_idx[:, tid]

        # Initialize: m = 0, r = e_j
        m_nnz = 0
        r_nnz = 1
        @inbounds r[j] = one(T)
        @inbounds r_nz[1] = j

        i1 = 1
        @inbounds ri1 = r[1]
        local_work = 0  # Accumulate locally, write once at end

        @inbounds for k in 1:itmax
            # findmax over r's nonzeros, prefer smaller index for ties
            i = 0
            max_val = zero(T)
            @inbounds for p in 1:r_nnz
                idx = r_nz[p]
                av = abs(r[idx])
                if av > max_val || (av == max_val && av > zero(T) && idx < i)
                    max_val = av
                    i = idx
                end
            end

            r[i1] = ri1

            # Compute delta = A[:,i]' * r / Anorms[i]
            col_start = A.colptr[i]
            col_end = A.colptr[i+1] - 1
            col_nnz = col_end - col_start + 1

            dot_val = zero(T)
            @inbounds for p in col_start:col_end
                dot_val += A.nzval[p] * r[A.rowval[p]]
            end
            delta = dot_val / Anorms[i]
            local_work += col_nnz + 1

            # m[i] += delta using small array storage
            # Search for i in m_i[1:m_nnz]
            found = 0
            @inbounds for p in 1:m_nnz
                if m_i[p] == i
                    found = p
                    break
                end
            end
            if found > 0
                @inbounds m_v[found] += delta
            else
                m_nnz += 1
                @inbounds m_i[m_nnz] = i
                @inbounds m_v[m_nnz] = delta
            end

            if m_nnz >= lfil
                break
            end

            # r -= delta * A[:,i], tracking new nonzeros
            @inbounds for p in col_start:col_end
                row = A.rowval[p]
                if r[row] == zero(T)
                    r_nnz += 1
                    r_nz[r_nnz] = row
                end
                r[row] -= delta * A.nzval[p]
            end
            local_work += col_nnz

            i1 = i
            ri1 = r[i]
            r[i] = zero(T)
        end

        # Get m[j] value
        Mjj = zero(T)
        @inbounds for p in 1:m_nnz
            if m_i[p] == j
                Mjj = m_v[p]
                break
            end
        end

        if Mjj < 0
            nneg_t[tid] += 1
            @inbounds for p in 1:m_nnz
                m_v[p] = -m_v[p]
            end
        end
        if Mjj == 0
            nzero_t[tid] += 1
            m_nnz += 1
            @inbounds m_i[m_nnz] = j
            @inbounds m_v[m_nnz] = one(T)
        end

        # Store column j in thread-local triplets (symmetric: store both (i,j) and (j,i))
        tc = triplet_count[tid]
        @inbounds for p in 1:m_nnz
            i = m_i[p]
            v = m_v[p]
            if i == j
                # Diagonal: store once
                tc += 1
                I_idx[tid][tc] = i
                J_idx[tid][tc] = j
                V_val[tid][tc] = v
            else
                # Off-diagonal: store both (i,j) and (j,i) with half value
                v_half = v * T(0.5)
                tc += 1
                I_idx[tid][tc] = i
                J_idx[tid][tc] = j
                V_val[tid][tc] = v_half
                tc += 1
                I_idx[tid][tc] = j
                J_idx[tid][tc] = i
                V_val[tid][tc] = v_half
            end
        end
        triplet_count[tid] = tc

        # Write accumulated work for this column
        approx_work[tid] += local_work

        # Clear r workspace for next column (m doesn't need clearing - we only read 1:m_nnz)
        @inbounds for p in 1:r_nnz
            r[r_nz[p]] = zero(T)
        end
    end

    # Merge per-thread triplets
    total_triplets = sum(triplet_count)
    I_all = Vector{Int}(undef, total_triplets)
    J_all = Vector{Int}(undef, total_triplets)
    V_all = Vector{T}(undef, total_triplets)
    offset = 0
    for t in 1:nthreads
        tc = triplet_count[t]
        copyto!(I_all, offset + 1, I_idx[t], 1, tc)
        copyto!(J_all, offset + 1, J_idx[t], 1, tc)
        copyto!(V_all, offset + 1, V_val[t], 1, tc)
        offset += tc
    end

    # Build sparse matrix from triplets (already symmetric due to storage strategy)
    # sparse() sums duplicates, giving us (M[i,j] + M[j,i])/2 for off-diagonals
    M = sparse_parallel(I_all, J_all, V_all, n, n)
    total_work = sum(approx_work) + nnz(M)

    nneg = sum(nneg_t)
    nzero = sum(nzero_t)

    println()
    @printf(" Negative diags: %8i\n", nneg)
    @printf(" Zero     diags: %8i\n", nzero)
    @printf(" M symmetrized.  nnz(M) = %d\n", nnz(M))
    @printf(" Polyester cores: %7i\n", nthreads)

    return M, total_work
end

# Convenience method
function SSAI3_polyester(A::SparseMatrixCSC{T, Int}; lfil::Int = ceil(Int, nnz(A) / size(A, 1))) where T <: Real
    n = size(A, 1)
    @assert size(A, 1) == size(A, 2) "Matrix A must be square"
    return SSAI3_polyester(A, n, lfil)
end
