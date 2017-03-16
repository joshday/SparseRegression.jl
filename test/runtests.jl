module SparseRegressionTests
using SparseRegression, PenaltyFunctions, Base.Test
include("datagenerator.jl")


losses = [LinearRegression(), L1Regression(), LogisticRegression(), PoissonRegression(),
          HuberRegression(), SVMLike(), DWDLike(1.0), QuantileRegression(.7)]
penalties = [NoPenalty(), L1Penalty(), L2Penalty(), ElasticNetPenalty(.5), LogPenalty(),
          SCADPenalty(), MCPPenalty()]
n, p = 1000, 10

@testset "SparseReg Constructor Type Stability" begin
    @testset "One Argument" begin
        @inferred SparseReg(p)
        for l in losses
            @inferred SparseReg(p, l)
        end
        for r in penalties
            @inferred SparseReg(p, r)
        end
        @inferred SparseReg(p, ProxGrad())
        @inferred SparseReg(p, .1)
        @inferred SparseReg(p, rand(p))
    end



    for l in losses, r in penalties
        @inferred SparseReg(p, l, r)
        @inferred SparseReg(p, r, l)
    end
end




#-----------------------------------------------------------------------# ProxGrad fitting
println("\n")
info("Fitting Every Loss/Penalty combination that ProxGrad can handle")
# setup
data(::Loss, n, p) = linregdata(n, p)
data(::MarginLoss, n, p) = logregdata(n, p)


for l in losses
    print_with_color(:blue, "$l\n")
    for r in [NoPenalty(), L1Penalty(), L2Penalty(), ElasticNetPenalty()]
        x, y, β = data(l, n, p)
        o = SparseReg(x, y, l, r, ProxGrad(step=.1, maxit=500, tol = 1e-3))
        print_with_color(:red, "  > $r\n")
    end
end


end
