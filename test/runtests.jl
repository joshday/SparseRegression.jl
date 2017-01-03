module SparseRegressionTests
using SparseRegression, Base.Test; S = SparseRegression
include("datagenerator.jl")

@testset "Constructors" begin
    o = SparseReg(5)
    @test coef(o) == zeros(5)

    x, y, β = linregdata(1000, 10)
    o = SparseReg(x, y)
    @test length(coef(o)) == 10
end
end
