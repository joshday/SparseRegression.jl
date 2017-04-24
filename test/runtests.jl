module SparseRegressionTests
using SparseRegression, PenaltyFunctions, Base.Test
include("datagenerator.jl")


losses = [LinearRegression(), L1Regression(), LogisticRegression(), PoissonRegression(),
          HuberRegression(), SVMLike(), DWDLike(1.0), QuantileRegression(.7)]
penalties = [NoPenalty(), L1Penalty(), L2Penalty(), ElasticNetPenalty(.5), LogPenalty(),
          SCADPenalty(), MCPPenalty()]

#------------------------------------------------------------#
println("\n")
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



end
