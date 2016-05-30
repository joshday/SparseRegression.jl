module AlgorithmTest
using SparseRegression
if VERSION >= v"0.5.0-dev+7720"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end
include("../datagenerator.jl")

n, p = 10_000, 50
x, y, β = linregdata(n, p)

@testset "Algorithms" begin
@testset "Sweep" begin
	o = SparseReg(x, y, algorithm = Sweep(), intercept = false)
	@test coef(o) ≈ x\y

	wts = rand(n)
	W = Diagonal(sqrt(wts))
	o = SparseReg(x, y, wts, algorithm = Sweep(), intercept = false)
	@test coef(o) ≈ (W * x) \ (W * y)
end
@testset "Fista" begin
	o = SparseReg(x, y, algorithm = Fista(standardize = false), intercept = false)
	@test_approx_eq_eps coef(o)[:, 1] x\y .001
end
end






end
