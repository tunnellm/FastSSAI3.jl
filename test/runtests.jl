using Test
using FastSSAI3
using SparseArrays, LinearAlgebra

@testset "FastSSAI3" begin
    @testset "Basic functionality" begin
        # Create a simple SPD matrix
        n = 100
        A = sprand(n, n, 0.05) + 10I
        A = (A + A') / 2

        # Test default fill_factor
        M, work = ssai3(A)
        @test size(M) == (n, n)
        @test issymmetric(M)
        @test nnz(M) > 0
        @test work > 0

        # Test custom fill_factor
        M2, work2 = ssai3(A; fill_factor=2.0)
        @test size(M2) == (n, n)
        @test nnz(M2) >= nnz(M)  # Higher fill_factor should give denser M
    end

    @testset "Preconditioner quality" begin
        # M*A should be closer to I than A alone
        n = 50
        A = sprand(n, n, 0.1) + 5I
        A = (A + A') / 2

        M, _ = ssai3(A; fill_factor=3.0)
        MA = M * A

        # Check diagonal is close to 1
        @test all(abs.(diag(MA) .- 1) .< 0.5)
    end
end
