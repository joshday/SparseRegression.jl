module SparseRegressionTests
using SparseRegression, PenaltyFunctions, Base.Test
include("datagenerator.jl")


losses = [LinearRegression(), L1Regression(), LogisticRegression(), PoissonRegression(),
          HuberRegression(), SVMLike(), DWDLike(1.0), QuantileRegression(.7)]
penalties = [NoPenalty(), L1Penalty(), L2Penalty(), ElasticNetPenalty(.5), LogPenalty(),
          SCADPenalty(), MCPPenalty()]

#------------------------------------------------------------# Show methods
info("Show Methods")
o = Obs(randn(100,5), randn(100))
show(o)
println()
show(SparseReg(o))

#------------------------------------------------------------# Tests Here
println("\n\n\n")
info("Tests Start Here")
data(::Loss, n, p) = DataGenerator.linregdata(n, p)
data(::MarginLoss, n, p) = DataGenerator.logregdata(n, p)

function _test(l::Loss, p::Penalty, a::LearningStrategy)
    x, y, β = data(l, 1000, 5)
    o = SparseReg(Obs(x, y), l, p)
    learn!(o, a, MaxIter(10))
    coef(o)
    predict(o, x)
end

@testset "ProxGrad Sanity Check" begin
    for l in losses, p in penalties
        isa(p, PenaltyFunctions.ConvexElementPenalty) && _test(l, p, ProxGrad())
    end
end
@testset "Sweep Sanity Check" begin
    for l in [L2DistLoss(), LinearRegression()], p in [NoPenalty(), L2Penalty()]
        _test(l, p, Sweep())
    end
end
@testset "GradientDescent Sanity Check" begin
    for l in losses, p in penalties
        _test(l, p, GradientDescent())
    end
end

@testset "Ones" begin
    o = Ones(10)
    @test length(o) == 10
    for oi in o
        @test oi == 1.0
    end
    @test o[1:4] == ones(4)
end
@testset "Obs" begin
    x, y, β = DataGenerator.linregdata(100, 5)
    o = Obs(x, y)
    @test size(o) == (100, 5)
    @test size(o, 1) == 100
    @test size(o, 2) == 5
end
@testset "SparseReg" begin
    x, y, β = DataGenerator.linregdata(100, 5)
    o = Obs(x, y)
    @testset "Constructor inference" begin
        @inferred SparseReg(o)

        @inferred SparseReg(o, L2DistLoss())
        @inferred SparseReg(o, L2Penalty())
        @inferred SparseReg(o, rand(5))

        @inferred SparseReg(o, L2DistLoss(), L2Penalty())
        @inferred SparseReg(o, L2DistLoss(), rand(5))
        @inferred SparseReg(o, L2Penalty(), L2DistLoss())
        @inferred SparseReg(o, L2Penalty(), rand(5))
        @inferred SparseReg(o, rand(5), L2DistLoss())
        @inferred SparseReg(o, rand(5), L2Penalty())

        @inferred SparseReg(o, L2DistLoss(), L2Penalty(), rand(5))
        @inferred SparseReg(o, L2DistLoss(), rand(5), L2Penalty())
        @inferred SparseReg(o, L2Penalty(), L2DistLoss(), rand(5))
        @inferred SparseReg(o, L2Penalty(), rand(5), L2DistLoss())
        @inferred SparseReg(o, rand(5), L2DistLoss(), L2Penalty())
        @inferred SparseReg(o, rand(5), L2Penalty(), L2DistLoss())
    end
    @testset "predict" begin
        s = SparseReg(o, L2DistLoss())
        @test predict(s) == predict(s, x)
        s = SparseReg(o, LogitMarginLoss())
        @test predict(s) == predict(s, x)
    end
end



end
