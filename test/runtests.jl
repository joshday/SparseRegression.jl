module SparseRegressionTests
using SparseRegression, PenaltyFunctions, Base.Test
include("datagenerator.jl")


losses = [LinearRegression(), L1Regression(), LogisticRegression(), PoissonRegression(),
          HuberRegression(), SVMLike(), DWDLike(1.0), QuantileRegression(.7)]
penalties = [NoPenalty(), L1Penalty(), L2Penalty(), ElasticNetPenalty(.5), LogPenalty(),
          SCADPenalty(), MCPPenalty()]

n, p = 1000, 10
possible_args = vcat(losses, penalties, .1, rand(p), ProxGrad())


@testset "SparseReg Constructor Type Stability" begin
    @testset "One Argument" begin
        @inferred SparseReg(n, p)
        for t in possible_args
            @inferred SparseReg(n, p, t)
        end
    end
    @testset "Two Arguments" begin
        for t1 in possible_args, t2 in possible_args
            @inferred SparseReg(n, p, t1, t2)
        end
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
        o = SparseReg(x, y, rand(n), l, r, ProxGrad(step=.1, maxit=500, tol = 1e-3))
        print_with_color(:red, "  > $r\n")
    end
end


end
