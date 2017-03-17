module SparseRegressionTests
using SparseRegression, PenaltyFunctions, Base.Test
include("datagenerator.jl")


losses = [LinearRegression(), L1Regression(), LogisticRegression(), PoissonRegression(),
          HuberRegression(), SVMLike(), DWDLike(1.0), QuantileRegression(.7)]
penalties = [NoPenalty(), L1Penalty(), L2Penalty(), ElasticNetPenalty(.5), LogPenalty(),
          SCADPenalty(), MCPPenalty()]

#------------------------------------------------------------# ProxGrad/Sweep fitting
println("\n")
info("Fitting Every Loss/Penalty combination that ProxGrad/Sweep can handle")
n, p = 1000, 5
data(::Loss, n, p) = linregdata(n, p)
data(::MarginLoss, n, p) = logregdata(n, p)
for l in losses
    print_with_color(:blue, "$l\n")
    for r in [NoPenalty(), L1Penalty(), L2Penalty(), ElasticNetPenalty()]
        x, y, β = data(l, n, p)
        w = rand(n)

        o = SparseReg(p, l, r)
        alg = ProxGrad(Obs(x, y); maxit=500, tol=1e-3, step=.1)
        fit!(o, alg)

        o2 = SparseReg(p, r, l)
        alg2 = ProxGrad(Obs(x, y, w); maxit=500, tol=1e-3, step=.1)
        fit!(o2, alg2)

        @test coef(o) != coef(o2)

        if isa(l, LinearRegression) && isa(r, SparseRegression.Sweepable)
            o3 = SparseReg(p, r, l)
            alg3 = Sweep(Obs(x, y))
            fit!(o3, alg3)

            o4 = SparseReg(p, l, r)
            alg4 = Sweep(Obs(x, y, w))
            fit!(o4, alg4)

            @test coef(o3) != coef(o4)
        end
        print_with_color(:red, "  > $r\n")
    end
end


end
