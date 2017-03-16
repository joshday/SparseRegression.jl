module AdaptiveLassoExample
using SparseRegression, DataGenerator

x, y, β = linregdata(10_000, 100; β = vcat(ones(5), zeros(95)))

lm = SparseReg(x, y, Sweep())

o = SparseReg(x, y, L1Penalty(), .1, inv.(abs.(coef(lm))))
@show coef(o)
end
