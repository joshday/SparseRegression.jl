module SparseRegressionTests
using SparseRegression, PenaltyFunctions, Base.Test
include("datagenerator.jl")


losses = [LinearRegression(), L1Regression(), LogisticRegression(), PoissonRegression(),
          HuberRegression(), SVMLike(), DWDLike(1.0), QuantileRegression(.7)]
penalties = [NoPenalty(), L1Penalty(), L2Penalty(), ElasticNetPenalty(.5), LogPenalty(),
          SCADPenalty(), MCPPenalty()]

#------------------------------------------------------------# 
println("\n")
info("Fitting Every Loss for ProximalGradientModel/StochasticModel")
n, p = 1000, 5
data(::Loss, n, p) = DataGenerator.linregdata(n, p)
data(::MarginLoss, n, p) = DataGenerator.logregdata(n, p)
for l in losses
    print_with_color(:blue, "$l\n")
    x, y, β = data(l, n, p)

    ProximalGradientModel(Obs(x, y), loss = l, step = .1, maxit=1000)
    StochasticModel(Obs(x, y), loss = l)
end


end
