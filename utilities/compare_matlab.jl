using SparseArrays, Printf, MatrixMarket, MATLAB, LinearAlgebra

# Load the module from parent directory
push!(LOAD_PATH, dirname(@__DIR__))
using FastSSAI3

"""
Compare Julia FastSSAI3 with MATLAB SSAI3 on a matrix.
"""
function compare_with_matlab(matrix_path, name; fill_factor=1.0)
    println("="^60)
    println("Comparing Julia vs MATLAB on: $name")
    println("fill_factor = $fill_factor")
    println("="^60)

    # Load matrix
    A = MatrixMarket.mmread(matrix_path)
    n = size(A, 1)
    avg_nnz = nnz(A) / n

    # Compute lfil from fill_factor (for MATLAB compatibility)
    lfil = ceil(Int, fill_factor * ceil(avg_nnz))

    println("\nMatrix: $n × $n, nnz=$(nnz(A)), avg nnz/col=$(round(avg_nnz, digits=1))")
    println("lfil = $lfil")

    # Run Julia
    println("\nRunning Julia FastSSAI3...")
    M_julia, _ = ssai3(A; fill_factor=fill_factor)
    julia_nnz_per_col = nnz(M_julia) / n
    println("Julia M: nnz=$(nnz(M_julia)), nnz/col=$(round(julia_nnz_per_col, digits=2))")

    # Run MATLAB
    println("\nRunning MATLAB SSAI3...")
    mat"addpath('/data1/SSAI')"
    # Convert to Float64 for MATLAB compatibility
    n_dbl = Float64(n)
    lfil_dbl = Float64(lfil)
    @mput A n_dbl lfil_dbl
    mat"[M_mat, work] = SSAI3(A, n_dbl, lfil_dbl);"
    @mget M_mat

    M_matlab = M_mat
    matlab_nnz_per_col = nnz(M_matlab) / n
    println("MATLAB M: nnz=$(nnz(M_matlab)), nnz/col=$(round(matlab_nnz_per_col, digits=2))")

    # Compare
    println("\n" * "-"^40)
    println("Comparison:")
    println("-"^40)

    # Check dimensions
    if size(M_julia) != size(M_matlab)
        println("ERROR: Dimensions differ!")
        println("  Julia:  $(size(M_julia))")
        println("  MATLAB: $(size(M_matlab))")
        return
    end

    # Check nnz
    nnz_diff = abs(nnz(M_julia) - nnz(M_matlab))
    @printf("nnz difference: %d\n", nnz_diff)

    # Numerical difference
    diff = norm(M_julia - M_matlab)
    rel_diff = diff / norm(M_matlab)
    @printf("Frobenius norm of difference: %.2e\n", diff)
    @printf("Relative difference: %.2e\n", rel_diff)

    # Check if they're numerically equal
    if rel_diff < 1e-10
        println("\n✓ Results match!")
    else
        println("\n✗ Results differ significantly")

        # Find where they differ
        D = M_julia - M_matlab
        max_diff = maximum(abs.(D))
        @printf("Max element difference: %.2e\n", max_diff)

        # Check a few columns
        println("\nSample column comparison:")
        for j in [1, n÷2, n]
            julia_col = M_julia[:, j]
            matlab_col = M_matlab[:, j]
            col_diff = norm(julia_col - matlab_col)
            @printf("  Column %d: Julia nnz=%d, MATLAB nnz=%d, diff=%.2e\n",
                    j, nnz(julia_col), nnz(matlab_col), col_diff)
        end
    end

    return M_julia, M_matlab
end

# Run comparison on crankseg_1 with high fill_factor
println("Testing with fill_factor = 3.0")
compare_with_matlab("/data1/matrices/crankseg_1/crankseg_1.mtx", "crankseg_1", fill_factor=3.0)
