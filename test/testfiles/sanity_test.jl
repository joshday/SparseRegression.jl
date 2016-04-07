module SanityCheck
using SparseRegression, FactCheck, Distributions

facts("SanityCheck") do
    n, p = 1000, 11
    x = randn(n, p)
    β = collect(linspace(-.5, .5, p))
    y1 = x*β + randn(n)
    y2 = 2.0 * [rand(Bernoulli(1. / (1. + exp(-η)))) for η in x*β] - 1.0
    y3 = Float64[rand(Poisson(exp(η))) for η in x*β]
    tol = 1e-4

    context("L2Regression") do
        o = SparseReg(x, y1, tol = tol)
    end
    context("L1Regression") do
        o = SparseReg(x, y1, model = L1Regression(), tol = tol)
    end
    context("LogisticRegression") do
        o = SparseReg(x, y2, model = LogisticRegression(), tol = tol)
    end
    context("PoissonRegression") do
        o = SparseReg(x, y3, model = PoissonRegression(), tol = tol)
    end
    context("SVMLike") do
        o = SparseReg(x, y2, model = SVMLike(), tol = tol)
    end
    context("QuantileRegression") do
        o = SparseReg(x, y1, model = QuantileRegression(.7), tol = tol)
    end
    context("HuberRegression") do
        o = SparseReg(x, y1, model = HuberRegression(.7), tol = tol)
    end
end
end
