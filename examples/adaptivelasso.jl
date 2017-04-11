# Adaptive LASSO is a method which uses elementwise regularization.
#
# Rather than `λ * abs(βj)`, the elementwise penalty is:
#  - `λ * inv(abs(βols_j)) * abs(β_j)`,
#    where `βols_j` is the ordinary least square coefficient.

module AdaptiveLassoExample
using SparseRegression, DataGenerator

x, y, β = linregdata(10_000, 100; β = vcat(ones(5), zeros(95)))

lm = SweepModel(Obs(x,y); penalty = NoPenalty(), λ = [0.])
βols = coef(lm, 1)

o = ProximalGradientModel(Obs(x, y), penalty = L1Penalty(), factor = inv.(abs.(βols)))
end
