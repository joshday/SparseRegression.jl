module SanityCheck
using SparseRegression, Distributions
if VERSION >= v"0.5.0-dev+7720"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end
include("../datagenerator.jl")


# Ensure no errors using Fista with any Model
@testset "Fista Sanity Check" begin
    n, p = 1000, 11
    x, y, β = linregdata(n, p)
    w = rand(n)

    o = SparseReg(x, y, w, algorithm = Fista(crit = :coef))
    o = SparseReg(x, y, model = L1Regression(), algorithm = Fista())
    o = SparseReg(x, y, model = QuantileRegression(.7), algorithm = Fista())
    o = SparseReg(x, y, model = HuberRegression(.7), algorithm = Fista())

    x, y, β = logregdata(n, p, false)
    o = SparseReg(x, y, model = LogisticRegression(), algorithm = Fista())

    x, y, β = logregdata(n, p, true)
    o = SparseReg(x, y, model = SVMLike(), algorithm = Fista())

    x, y, β = poissonregdata(n, p)
    o = SparseReg(x, y, model = PoissonRegression(), algorithm = Fista(step = .01))
end
end  # module
