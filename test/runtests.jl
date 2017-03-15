module SparseRegressionTests
reload("SparseRegression")
using SparseRegression, Base.Test; S = SparseRegression
include("datagenerator.jl")


losses = [LinearRegression(), L1Regression(), LogisticRegression(), PoissonRegression(),
          HuberRegression(), SVMLike(), DWDLike(1.0), QuantileRegression(.7)]
penalties = [NoPenalty(), L1Penalty(), L2Penalty(), ElasticNetPenalty(.5), LogPenalty(),
          SCADPenalty(), MCPPenalty()]


@testset "ProxGrad with each Loss/Penalty combination" begin
    # setup
    data(::Loss, n, p) = linregdata(n, p)
    data(::MarginLoss, n, p) = logregdata(n, p)
    n, p = 1000, 10
    for l in losses
        print_with_color(:blue, "$l\n")
        for r in penalties
            x, y, β = data(l, n, p)
            o = SparseReg(x, y, l, r)
            print_with_color(:red, "  > $r\n")
        end
    end
end

end
