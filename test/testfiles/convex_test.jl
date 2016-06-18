# Do not include in runtests.jl
#
# This file compare numbers against Convex.jl

module ConvexTest
reload("SparseRegression")
reload("OnlineStats")
using SparseRegression, StatsBase, Convex, Mosek, FactCheck, DataFrames
using Lasso, Distributions
srand(1234)

n, p = 10_000, 10
x = randn(n, p) * 2

y = x * collect(1:p) + randn(n)
λ = 1.

# SparseRegression
o = SparseReg(x, y,
    penalty = LassoPenalty(),
    intercept = false,
    lambda = [λ],
    model = LinearRegression(),
    algorithm = Fista(step = .5, tol = 1e-10, standardize = false)
)
b1 = coef(o)[:, 1]

# Lasso
o2 = fit(LassoPath, x, y, λ = [λ], intercept = false, standardize = false)
b2 = Matrix(coef(o2))[:, 1]

# Convex
β = Variable(p)
problem = minimize(0.5 * sumsquares(y - x * β) + n * λ * sumabs(β))
solve!(problem, MosekSolver(LOG = 0))
b3 = β.value[:, 1]

# Compare coefficients
@show DataFrame(
    SparseReg   = b1,
    Lasso       = b2,
    Convex      = b3,
)

info("SparseRegression")
@show mean(abs2(y - x * b1))
info("Lasso")
@show mean(abs2(y - x * b2))
info("Convex")
@show mean(abs2(y - x * b3))


end
