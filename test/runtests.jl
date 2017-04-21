module SparseRegressionTests
using SparseRegression, PenaltyFunctions, Base.Test
include("datagenerator.jl")


losses = [LinearRegression(), L1Regression(), LogisticRegression(), PoissonRegression(),
          HuberRegression(), SVMLike(), DWDLike(1.0), QuantileRegression(.7)]
penalties = [NoPenalty(), L1Penalty(), L2Penalty(), ElasticNetPenalty(.5), LogPenalty(),
          SCADPenalty(), MCPPenalty()]

#------------------------------------------------------------#
println("\n")
info("Fitting Every Loss for ProxGrad")
n, p = 1000, 5
data(::Loss, n, p) = DataGenerator.linregdata(n, p)
data(::MarginLoss, n, p) = DataGenerator.logregdata(n, p)
for l in losses
    print_with_color(:blue, "$l\n")
    x, y, β = data(l, n, p)

    s = SparseReg(Obs(x, y), l)
    learn!(s, ProxGrad(), MaxIter(100))
end


end
