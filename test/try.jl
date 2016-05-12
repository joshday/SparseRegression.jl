module Try
using StatsBase, DataGenerator, Benchmarks, Lasso
import SparseRegression
S = SparseRegression

x, y, β = linregdata(100_000, 100)

@time @show S.SparseReg(x, y, lambda = 0:.1:1, penalty = S.LassoPenalty())
@time @show fit(LassoPath, x, y, λ = collect(0:.1:1), α = 1.)
end
