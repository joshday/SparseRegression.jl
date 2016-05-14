module SanityCheck
using SparseRegression, FactCheck, Distributions

facts("Performance") do
    n, p = 1000, 11
    x = randn(n, p)
    β = collect(linspace(-.5, .5, p))
    y1 = x*β + randn(n)
    y2 = 2.0 * [rand(Bernoulli(1. / (1. + exp(-η)))) for η in x*β] - 1.0
    y3 = Float64[rand(Poisson(exp(η))) for η in x*β]
    tol = 1e-4

end #facts
end #module
