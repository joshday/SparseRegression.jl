module SparseRegTest
using SparseRegression; sp = SparseRegression
if VERSION >= v"0.5.0-dev+7720"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

include("../datagenerator.jl")

o = SparseReg(10, penalty = RidgePenalty(), lambda = .1:.1:1)
o2 = SparseReg(10, model = LogisticRegression(), penalty = RidgePenalty(), lambda = .1:.1:1)
x = rand(100, 10)
y = randn(100)

@testset "SparseReg" begin
    @test_throws Exception sp.λindex(o, .123)
    @test sp.λindex(o, .2) == 2
    @test coef(o, .2) == zeros(11)


    o.β0[2] = 1.0
    @test predict(o, x, .2) == ones(100)

    o2.β0[2] = 1.0
    @test classify(o2, x, .2) == ones(100)

    @test loss(o, x, y, .2) == 0.5 * mean(abs2(y - 1.0))
    @test sp.cost(o, x, y, .2) == loss(o, x, y, .2)
end



end
