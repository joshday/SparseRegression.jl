module SparseRegressionTests
using SparseRegression, Base.Test; S = SparseRegression
include("datagenerator.jl")

@testset "SparseReg Constructors" begin
    SparseReg(5)
    @testset "one arg" begin
        SparseReg(5, LinearRegression())
        SparseReg(5, L1Penalty())
        SparseReg(5, ProxGrad())
        SparseReg(5, .1)
        SparseReg(5, rand(5))
    end
end

end
