# Parallel sparse matrix construction utilities

"""
    cumsum_parallel!(result, x)

Parallel inclusive cumsum using Polyester (2-phase algorithm).
result[i] = sum(x[1:i])
"""
function cumsum_parallel!(result::AbstractVector{T}, x::AbstractVector{T}) where T
    n = length(x)
    nthreads = Threads.nthreads()

    if n < 1000  # Fall back to sequential for small arrays
        result[1] = x[1]
        @inbounds for i in 2:n
            result[i] = result[i-1] + x[i]
        end
        return result
    end

    chunk_size = cld(n, nthreads)
    chunk_sums = zeros(T, nthreads)

    # Phase 1: Each thread computes local prefix sums and stores chunk total
    @batch per=core for tid in 1:nthreads
        start_idx = (tid - 1) * chunk_size + 1
        end_idx = min(tid * chunk_size, n)

        if start_idx <= n
            @inbounds result[start_idx] = x[start_idx]
            @inbounds for i in start_idx+1:end_idx
                result[i] = result[i-1] + x[i]
            end
            @inbounds chunk_sums[tid] = result[end_idx]
        end
    end

    # Phase 2: Each thread computes its offset and adds to elements
    @batch per=core for tid in 2:nthreads
        start_idx = (tid - 1) * chunk_size + 1
        end_idx = min(tid * chunk_size, n)

        # Compute offset = sum of all previous chunk totals
        offset = zero(T)
        @inbounds for t in 1:tid-1
            offset += chunk_sums[t]
        end

        @turbo for i in start_idx:end_idx
            result[i] += offset
        end
    end

    return result
end

"""
    sparse_parallel(I, J, V, m, n) -> SparseMatrixCSC

Build a CSC sparse matrix from COO triplets in parallel.
Assumes entries may have duplicates (which get summed).
"""
function sparse_parallel(I::Vector{Int}, J::Vector{Int}, V::Vector{T},
                         m::Int, n::Int) where T

    nnz_coo = length(I)
    nthreads = Threads.nthreads()

    # Step 1: Count entries per column (parallel)
    col_counts_local = zeros(Int, n, nthreads)

    chunk_size = cld(nnz_coo, nthreads)
    @batch per=core for tid in 1:nthreads
        start_idx = (tid - 1) * chunk_size + 1
        end_idx = min(tid * chunk_size, nnz_coo)
        counts = @view col_counts_local[:, tid]
        @inbounds for k in start_idx:end_idx
            counts[J[k]] += 1
        end
    end

    # Merge counts (parallel over columns)
    col_counts = Vector{Int}(undef, n)
    @batch per=core for j in 1:n
        s = 0
        @inbounds for tid in 1:nthreads
            s += col_counts_local[j, tid]
        end
        @inbounds col_counts[j] = s
    end

    # Step 2: Build column pointers (parallel prefix sum)
    colptr = Vector{Int}(undef, n + 1)
    colptr[1] = 1
    cumsum_parallel!(@view(colptr[2:end]), col_counts)
    @tturbo for j in 2:n+1
        colptr[j] += 1  # Shift by 1 for 1-based indexing
    end
    nnz_total = colptr[n + 1] - 1

    # Step 3: Allocate CSC arrays
    rowval = Vector{Int}(undef, nnz_total)
    nzval = Vector{T}(undef, nnz_total)

    # Step 4: Place entries in parallel
    # Each thread processes its chunk and writes to appropriate column positions
    # Use per-thread cursors to avoid conflicts

    # Per-thread cursor offsets for each column
    col_cursor_offsets = zeros(Int, n, nthreads + 1)

    # First pass: compute how many entries each thread will place in each column
    @batch per=core for tid in 1:nthreads
        start_idx = (tid - 1) * chunk_size + 1
        end_idx = min(tid * chunk_size, nnz_coo)
        offsets = @view col_cursor_offsets[:, tid + 1]
        @inbounds for k in start_idx:end_idx
            offsets[J[k]] += 1
        end
    end

    # Cumsum to get write positions for each thread in each column (parallel over columns)
    @batch per=core for j in 1:n
        base = colptr[j]
        @inbounds for tid in 1:nthreads
            col_cursor_offsets[j, tid + 1] += col_cursor_offsets[j, tid]
        end
        # Add base offset
        @inbounds for tid in 1:nthreads + 1
            col_cursor_offsets[j, tid] += base - 1
        end
    end

    # Second pass: place entries (parallel, no conflicts)
    col_cursors_local = zeros(Int, n, nthreads)
    @batch per=core for tid in 1:nthreads
        start_idx = (tid - 1) * chunk_size + 1
        end_idx = min(tid * chunk_size, nnz_coo)
        cursors = @view col_cursors_local[:, tid]
        @inbounds for k in start_idx:end_idx
            j = J[k]
            pos = col_cursor_offsets[j, tid] + cursors[j] + 1
            cursors[j] += 1
            rowval[pos] = I[k]
            nzval[pos] = V[k]
        end
    end

    # Step 5: Sort each column by row index and sum duplicates (parallel)
    @batch per=core for j in 1:n
        col_start = colptr[j]
        col_end = colptr[j + 1] - 1
        col_len = col_end - col_start + 1

        if col_len <= 1
            continue
        end

        # Sort by row index (in-place)
        col_rows = @view rowval[col_start:col_end]
        col_vals = @view nzval[col_start:col_end]

        # Simple insertion sort for small columns, otherwise use sortperm
        if col_len <= 32
            # Insertion sort
            @inbounds for i in 2:col_len
                key_row = col_rows[i]
                key_val = col_vals[i]
                k = i - 1
                while k >= 1 && col_rows[k] > key_row
                    col_rows[k + 1] = col_rows[k]
                    col_vals[k + 1] = col_vals[k]
                    k -= 1
                end
                col_rows[k + 1] = key_row
                col_vals[k + 1] = key_val
            end
        else
            # Use sortperm for larger columns
            perm = sortperm(col_rows)
            col_rows[:] = col_rows[perm]
            col_vals[:] = col_vals[perm]
        end
    end

    # Step 6: Compress duplicates (parallel per column)
    # Count unique entries per column first
    unique_counts = Vector{Int}(undef, n)
    @batch per=core for j in 1:n
        col_start = colptr[j]
        col_end = colptr[j + 1] - 1
        if col_start > col_end
            @inbounds unique_counts[j] = 0
            continue
        end

        count = 1
        @inbounds for p in col_start+1:col_end
            if rowval[p] != rowval[p - 1]
                count += 1
            end
        end
        unique_counts[j] = count
    end

    # Build new colptr (parallel prefix sum)
    new_colptr = Vector{Int}(undef, n + 1)
    new_colptr[1] = 1
    cumsum_parallel!(@view(new_colptr[2:end]), unique_counts)
    @tturbo for j in 2:n+1
        new_colptr[j] += 1  # Shift by 1 for 1-based indexing
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
        @inbounds new_nzval[new_pos] = nzval[old_start]

        @inbounds for p in old_start+1:old_end
            if rowval[p] == rowval[p - 1]
                # Duplicate: sum values
                new_nzval[new_pos] += nzval[p]
            else
                # New row: advance position
                new_pos += 1
                new_rowval[new_pos] = rowval[p]
                new_nzval[new_pos] = nzval[p]
            end
        end
    end

    return SparseMatrixCSC(m, n, new_colptr, new_rowval, new_nzval)
end

# Test function
function test_sparse_parallel()
    # Create test COO data with duplicates
    I = [1, 2, 3, 1, 2, 3, 1]
    J = [1, 1, 1, 2, 2, 2, 1]  # Note: (1,1) appears twice
    V = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5]

    M_ref = sparse(I, J, V, 3, 3)
    M_par = sparse_parallel(I, J, V, 3, 3)

    println("Reference:")
    display(Matrix(M_ref))
    println("\nParallel:")
    display(Matrix(M_par))
    println("\nMatch: ", M_ref == M_par)
end

function bench_sparse_parallel(n, nnz_per_col; nruns=3)
    # Simulate SSAI-like triplet data: symmetric storage
    println("Generating test data: n=$n, ~$(nnz_per_col) entries per column")

    # Generate random triplets (simulating symmetric off-diagonal storage)
    total_triplets = 2 * nnz_per_col * n  # ~2x for symmetric
    I = Vector{Int}(undef, total_triplets)
    J = Vector{Int}(undef, total_triplets)
    V = Vector{Float64}(undef, total_triplets)

    k = 0
    for j in 1:n
        # Diagonal
        k += 1
        I[k] = j
        J[k] = j
        V[k] = rand()

        # Off-diagonals (symmetric pairs)
        for _ in 1:nnz_per_col-1
            i = rand(1:n)
            if i != j
                v = rand() * 0.5
                k += 1
                I[k] = i
                J[k] = j
                V[k] = v
                k += 1
                I[k] = j
                J[k] = i
                V[k] = v
            end
        end
    end
    resize!(I, k)
    resize!(J, k)
    resize!(V, k)

    println("Total triplets: $k")

    # Warmup
    _ = sparse(I, J, V, n, n)
    _ = sparse_parallel(I, J, V, n, n)

    # Benchmark built-in
    t_builtin = minimum([@elapsed sparse(I, J, V, n, n) for _ in 1:nruns])

    # Benchmark parallel
    t_parallel = minimum([@elapsed sparse_parallel(I, J, V, n, n) for _ in 1:nruns])

    # Verify correctness
    M_ref = sparse(I, J, V, n, n)
    M_par = sparse_parallel(I, J, V, n, n)
    match = M_ref == M_par

    println("Built-in sparse(): $(round(t_builtin, digits=4))s")
    println("Parallel sparse:   $(round(t_parallel, digits=4))s")
    println("Speedup: $(round(t_builtin / t_parallel, digits=2))x")
    println("Match: $match")

    return t_builtin, t_parallel, match
end

if abspath(PROGRAM_FILE) == @__FILE__
    test_sparse_parallel()
    println("\n" * "="^50)
    bench_sparse_parallel(100_000, 15)
end
