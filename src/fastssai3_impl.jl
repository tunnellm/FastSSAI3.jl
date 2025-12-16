# Core implementation of FastSSAI3 preconditioner
#
# Algorithm by Shaked Regev and Michael Saunders, ICME, Stanford University.
# Julia implementation by Marc A. Tunnell (https://tunnellm.github.io/)

"""
    M, approx_work = ssai3(A; fill_factor=1.0)

Constructs a SPAI-type preconditioner M ≈ A⁻¹ for symmetric positive definite A.

Originally implemented by Shaked Regev and Michael Saunders, ICME, Stanford University.
Reference: https://stanford.edu/group/SOL/reports/20SSAI.pdf

# Arguments
- `A`: Symmetric positive definite sparse matrix (assumed to have unit diagonal)
- `fill_factor`: Multiplier for average column degree to determine target fill level
                 (lfil = ceil(fill_factor * avg_nnz_per_col), default: 1.0)

# Returns
- `M`: Symmetric approximate inverse preconditioner
- `approx_work`: Approximate flop count

# Example
```julia
using FastSSAI3, SparseArrays, LinearAlgebra
A = sprand(1000, 1000, 0.01) + 10I
A = (A + A') / 2  # symmetrize
M, work = ssai3(A; fill_factor=2.0)
```

This implementation uses direct CSC building instead of triplet storage.
Each thread builds sorted columns for its range, then merges into global CSC.
"""
function ssai3(A::SparseMatrixCSC{T, Int}; fill_factor::Real=1.0) where T <: Real

    n = size(A, 1)
    avg_degree = ceil(nnz(A) / n)
    lfil = ceil(Int, fill_factor * avg_degree)

    nthreads = Threads.nthreads()
    itmax = 2 * lfil

    # Determine column ranges per thread
    chunk = cld(n, nthreads)

    # Per-thread storage for LOCAL columns (entries in this thread's column range)
    # Each column can have at most lfil+1 entries (lfil from algorithm + possible diagonal fixup)
    max_per_col = lfil + 1
    local_rows = [Matrix{Int}(undef, max_per_col, min(chunk, n - (t-1)*chunk)) for t in 1:nthreads]
    local_vals = [Matrix{T}(undef, max_per_col, min(chunk, n - (t-1)*chunk)) for t in 1:nthreads]
    local_counts = [Vector{Int}(undef, min(chunk, n - (t-1)*chunk)) for t in 1:nthreads]

    # Per-thread storage for REMOTE entries (symmetric entries going to other threads' columns)
    # Worst case: each column produces lfil off-diagonal entries, each creates one remote entry
    max_remote = lfil * chunk
    remote_I = [Vector{Int}(undef, max_remote) for _ in 1:nthreads]
    remote_J = [Vector{Int}(undef, max_remote) for _ in 1:nthreads]
    remote_V = [Vector{T}(undef, max_remote) for _ in 1:nthreads]
    remote_count = zeros(Int, nthreads)

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

    # Precompute squared column norms of A (parallel + vectorized)
    Anorms = Vector{T}(undef, n)
    nzval = A.nzval
    colptr = A.colptr
    @batch per=core for j in 1:n
        col_start = colptr[j]
        col_end = colptr[j+1] - 1
        s = zero(T)
        @turbo for p in col_start:col_end
            s += nzval[p]^2
        end
        @inbounds Anorms[j] = s
    end

    # Parallel loop - explicitly assign column ranges to threads
    @batch per=core for tid in 1:nthreads
        col_start_tid = (tid - 1) * chunk + 1
        col_end_tid = min(tid * chunk, n)

        # Get thread-local workspaces
        m_i = @view m_idx[:, tid]
        m_v = @view m_val[:, tid]
        r = @view r_ws[:, tid]
        r_nz = @view r_idx[:, tid]

        lrows = local_rows[tid]
        lvals = local_vals[tid]
        lcounts = local_counts[tid]
        rI = remote_I[tid]
        rJ = remote_J[tid]
        rV = remote_V[tid]
        rc = 0  # remote count for this thread

        local_work = 0

        for j in col_start_tid:col_end_tid
            local_j = j - col_start_tid + 1

            # Initialize: m = 0, r = e_j
            m_nnz = 0
            r_nnz = 1
            @inbounds r[j] = one(T)
            @inbounds r_nz[1] = j

            i1 = 1
            @inbounds ri1 = r[1]

            @inbounds for kk in 1:itmax
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
                cs = A.colptr[i]
                ce = A.colptr[i+1] - 1
                col_nnz = ce - cs + 1

                dot_val = zero(T)
                @turbo for p in cs:ce
                    dot_val += A.nzval[p] * r[A.rowval[p]]
                end
                delta = dot_val / Anorms[i]
                local_work += col_nnz + 1

                # m[i] += delta using small array storage
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
                @inbounds for p in cs:ce
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

            # Store entries: local column j storage + remote entries for symmetric pairs
            k = 0  # count for local column j
            @inbounds for p in 1:m_nnz
                i = m_i[p]
                v = m_v[p]
                if i == j
                    # Diagonal: store locally
                    k += 1
                    lrows[k, local_j] = i
                    lvals[k, local_j] = v
                else
                    v_half = v * T(0.5)
                    # Entry (i, j) - column j is local
                    k += 1
                    lrows[k, local_j] = i
                    lvals[k, local_j] = v_half
                    # Entry (j, i) - column i might be in different thread's range
                    # Store as remote (we'll merge later)
                    rc += 1
                    rI[rc] = j  # row j in column i
                    rJ[rc] = i  # column i
                    rV[rc] = v_half
                end
            end
            @inbounds lcounts[local_j] = k

            # Sort local column j by row index (insertion sort for small arrays)
            if k > 1
                col_rows = @view lrows[1:k, local_j]
                col_vals = @view lvals[1:k, local_j]
                @inbounds for ii in 2:k
                    key_row = col_rows[ii]
                    key_val = col_vals[ii]
                    jj = ii - 1
                    while jj >= 1 && col_rows[jj] > key_row
                        col_rows[jj + 1] = col_rows[jj]
                        col_vals[jj + 1] = col_vals[jj]
                        jj -= 1
                    end
                    col_rows[jj + 1] = key_row
                    col_vals[jj + 1] = key_val
                end
            end

            # Clear r workspace
            @inbounds for p in 1:r_nnz
                r[r_nz[p]] = zero(T)
            end
        end

        remote_count[tid] = rc
        approx_work[tid] += local_work
    end

    # Phase 2: Build CSC from local columns + remote entries
    # Count local entries per column (parallel)
    col_counts_local = zeros(Int, n)
    @batch per=core for tid in 1:nthreads
        col_start_tid = (tid - 1) * chunk + 1
        col_end_tid = min(tid * chunk, n)
        counts = local_counts[tid]
        for local_j in 1:(col_end_tid - col_start_tid + 1)
            j = col_start_tid + local_j - 1
            @inbounds col_counts_local[j] = counts[local_j]
        end
    end

    # Count remote entries per column per thread (parallel)
    remote_counts_per_col = zeros(Int, n, nthreads)
    @batch per=core for tid in 1:nthreads
        rc = remote_count[tid]
        rJ = remote_J[tid]
        counts = @view remote_counts_per_col[:, tid]
        @inbounds for p in 1:rc
            counts[rJ[p]] += 1
        end
    end

    # Compute total counts per column
    col_counts = Vector{Int}(undef, n)
    @batch per=core for j in 1:n
        s = col_counts_local[j]
        @inbounds for tid in 1:nthreads
            s += remote_counts_per_col[j, tid]
        end
        @inbounds col_counts[j] = s
    end

    # Build column pointers
    colptr = Vector{Int}(undef, n + 1)
    colptr[1] = 1
    @inbounds for j in 1:n
        colptr[j + 1] = colptr[j] + col_counts[j]
    end
    total_nnz = colptr[n + 1] - 1

    # Allocate CSC arrays
    rowval = Vector{Int}(undef, total_nnz)
    nzval_out = Vector{T}(undef, total_nnz)

    # Copy local entries to CSC (parallel)
    @batch per=core for tid in 1:nthreads
        col_start_tid = (tid - 1) * chunk + 1
        col_end_tid = min(tid * chunk, n)
        lrows = local_rows[tid]
        lvals = local_vals[tid]
        counts = local_counts[tid]

        for local_j in 1:(col_end_tid - col_start_tid + 1)
            j = col_start_tid + local_j - 1
            cnt = counts[local_j]
            pos = colptr[j]  # Start position for column j
            @inbounds for p in 1:cnt
                rowval[pos + p - 1] = lrows[p, local_j]
                nzval_out[pos + p - 1] = lvals[p, local_j]
            end
        end
    end

    # Compute write offsets for remote entries per thread per column
    # For each column j, compute cumulative offset for each thread's remote entries
    remote_write_offsets = zeros(Int, n, nthreads + 1)
    @batch per=core for j in 1:n
        base = colptr[j] + col_counts_local[j]  # After local entries
        @inbounds remote_write_offsets[j, 1] = base
        @inbounds for tid in 1:nthreads
            remote_write_offsets[j, tid + 1] = remote_write_offsets[j, tid] + remote_counts_per_col[j, tid]
        end
    end

    # Insert remote entries (parallel per thread)
    # Pre-allocate workspaces for counting sort
    max_remote_per_thread = maximum(remote_count)
    sort_prefix = [Vector{Int}(undef, n + 1) for _ in 1:nthreads]
    sort_cursor = [Vector{Int}(undef, n) for _ in 1:nthreads]
    sort_temp_I = [Vector{Int}(undef, max_remote_per_thread) for _ in 1:nthreads]
    sort_temp_V = [Vector{T}(undef, max_remote_per_thread) for _ in 1:nthreads]

    @batch per=core for tid in 1:nthreads
        rc = remote_count[tid]
        if rc == 0
            continue
        end

        rI = remote_I[tid]
        rJ = remote_J[tid]
        rV = remote_V[tid]
        counts = @view remote_counts_per_col[:, tid]

        # Use pre-allocated workspaces
        prefix = sort_prefix[tid]
        cursor = sort_cursor[tid]
        temp_I = sort_temp_I[tid]
        temp_V = sort_temp_V[tid]

        # Compute prefix sums
        prefix[1] = 0
        @inbounds for j in 1:n
            prefix[j + 1] = prefix[j] + counts[j]
            cursor[j] = prefix[j]
        end

        # Sort entries by column into temp arrays using counting sort
        @inbounds for p in 1:rc
            j = rJ[p]
            pos = cursor[j] + 1
            cursor[j] += 1
            temp_I[pos] = rI[p]
            temp_V[pos] = rV[p]
        end

        # Place sorted entries into final CSC arrays
        @inbounds for j in 1:n
            col_start = prefix[j] + 1
            col_end = prefix[j + 1]
            base = remote_write_offsets[j, tid]
            for p in col_start:col_end
                rowval[base + p - col_start] = temp_I[p]
                nzval_out[base + p - col_start] = temp_V[p]
            end
        end
    end

    # Sort each column (parallel) - hybrid: insertion sort for small, sortperm for large
    @batch per=core for j in 1:n
        col_start_j = colptr[j]
        col_end_j = colptr[j + 1] - 1
        col_len = col_end_j - col_start_j + 1

        col_len <= 1 && continue

        col_rows = @view rowval[col_start_j:col_end_j]
        col_vals = @view nzval_out[col_start_j:col_end_j]

        if col_len <= 64
            # Insertion sort for small columns
            @inbounds for ii in 2:col_len
                key_row = col_rows[ii]
                key_val = col_vals[ii]
                jj = ii - 1
                while jj >= 1 && col_rows[jj] > key_row
                    col_rows[jj + 1] = col_rows[jj]
                    col_vals[jj + 1] = col_vals[jj]
                    jj -= 1
                end
                col_rows[jj + 1] = key_row
                col_vals[jj + 1] = key_val
            end
        else
            # Sortperm for larger columns
            perm = sortperm(col_rows)
            col_rows[:] = col_rows[perm]
            col_vals[:] = col_vals[perm]
        end
    end

    # Compress duplicates (parallel per column)
    unique_counts = Vector{Int}(undef, n)
    @batch per=core for j in 1:n
        col_start_j = colptr[j]
        col_end_j = colptr[j + 1] - 1
        if col_start_j > col_end_j
            @inbounds unique_counts[j] = 0
            continue
        end

        count = 1
        @inbounds for p in col_start_j+1:col_end_j
            if rowval[p] != rowval[p - 1]
                count += 1
            end
        end
        @inbounds unique_counts[j] = count
    end

    # Build new column pointers
    new_colptr = Vector{Int}(undef, n + 1)
    new_colptr[1] = 1
    @inbounds for j in 1:n
        new_colptr[j + 1] = new_colptr[j] + unique_counts[j]
    end
    new_nnz = new_colptr[n + 1] - 1

    # Allocate compressed arrays
    new_rowval = Vector{Int}(undef, new_nnz)
    new_nzval = Vector{T}(undef, new_nnz)

    # Compress: sum duplicates (parallel)
    @batch per=core for j in 1:n
        old_start = colptr[j]
        old_end = colptr[j + 1] - 1
        new_start = new_colptr[j]

        if old_start > old_end
            continue
        end

        new_pos = new_start
        @inbounds new_rowval[new_pos] = rowval[old_start]
        @inbounds new_nzval[new_pos] = nzval_out[old_start]

        @inbounds for p in old_start+1:old_end
            if rowval[p] == rowval[p - 1]
                new_nzval[new_pos] += nzval_out[p]
            else
                new_pos += 1
                new_rowval[new_pos] = rowval[p]
                new_nzval[new_pos] = nzval_out[p]
            end
        end
    end

    M = SparseMatrixCSC(n, n, new_colptr, new_rowval, new_nzval)
    total_work = sum(approx_work) + nnz(M)

    nneg = sum(nneg_t)
    nzero = sum(nzero_t)

    println()
    @printf(" Negative diags: %8i\n", nneg)
    @printf(" Zero     diags: %8i\n", nzero)
    @printf(" M symmetrized.  nnz(M) = %d\n", nnz(M))
    @printf(" ssai3: fill_factor=%.2f, lfil=%d, %d cores\n", fill_factor, lfil, nthreads)

    return M, total_work
end
