using SparseArrays, Printf, MatrixMarket, MATLAB, LinearAlgebra

# Load the module from parent directory
push!(LOAD_PATH, dirname(@__DIR__))
using FastSSAI3

"""
Benchmark MATLAB vs Julia SSAI3 on a matrix.
Times only the computation, not data transfer.
"""
function benchmark_comparison(matrix_path, name; fill_factor=1.0, nruns=3)
    println("="^60)
    println("Benchmarking: $name")
    println("="^60)

    # Load matrix
    A = MatrixMarket.mmread(matrix_path)
    n = size(A, 1)
    avg_nnz = nnz(A) / n

    # Compute lfil from fill_factor (for MATLAB compatibility)
    lfil = ceil(Int, fill_factor * ceil(avg_nnz))

    println("Matrix: $n × $n, nnz=$(nnz(A)), avg nnz/col=$(round(avg_nnz, digits=1))")
    println("fill_factor = $fill_factor, lfil = $lfil")
    println()

    # Transfer data to MATLAB once
    mat"addpath('/data1/SSAI')"
    n_dbl = Float64(n)
    lfil_dbl = Float64(lfil)
    @mput A n_dbl lfil_dbl

    # Warmup MATLAB
    mat"[M_mat, ~] = SSAI3(A, n_dbl, lfil_dbl);"

    # Benchmark MATLAB (time only inside MATLAB)
    println("MATLAB timings ($nruns runs):")
    matlab_times = Float64[]
    for i in 1:nruns
        mat"tic; [M_mat, ~] = SSAI3(A, n_dbl, lfil_dbl); matlab_time = toc;"
        @mget matlab_time
        push!(matlab_times, matlab_time)
        @printf("  Run %d: %.3f s\n", i, matlab_time)
    end
    matlab_best = minimum(matlab_times)
    @printf("  Best: %.3f s\n\n", matlab_best)

    # Warmup Julia
    ssai3(A; fill_factor=fill_factor)

    # Benchmark Julia
    println("Julia timings ($nruns runs):")
    julia_times = Float64[]
    for i in 1:nruns
        t = @elapsed M_julia, _ = ssai3(A; fill_factor=fill_factor)
        push!(julia_times, t)
        @printf("  Run %d: %.3f s\n", i, t)
    end
    julia_best = minimum(julia_times)
    @printf("  Best: %.3f s\n\n", julia_best)

    # Summary
    println("-"^40)
    @printf("MATLAB best: %.3f s\n", matlab_best)
    @printf("Julia best:  %.3f s\n", julia_best)
    @printf("Speedup:     %.2fx\n", matlab_best / julia_best)
    println("-"^40)

    return matlab_best, julia_best
end

# Run benchmarks
println("\n" * "="^60)
println("ECOLOGY2")
println("="^60 * "\n")
eco_mat, eco_jul = benchmark_comparison(
    "/data1/matrices/ecology2/ecology2.mtx",
    "ecology2",
    fill_factor=3.0
)

println("\n\n")

println("="^60)
println("CRANKSEG_1")
println("="^60 * "\n")
crank_mat, crank_jul = benchmark_comparison(
    "/data1/matrices/crankseg_1/crankseg_1.mtx",
    "crankseg_1",
    fill_factor=0.25
)

# Final summary
println("\n\n")
println("="^60)
println("FINAL SUMMARY")
println("="^60)
@printf("\n%-15s %12s %12s %10s\n", "Matrix", "MATLAB (s)", "Julia (s)", "Speedup")
println("-"^50)
@printf("%-15s %12.3f %12.3f %10.2fx\n", "ecology2", eco_mat, eco_jul, eco_mat/eco_jul)
@printf("%-15s %12.3f %12.3f %10.2fx\n", "crankseg_1", crank_mat, crank_jul, crank_mat/crank_jul)
