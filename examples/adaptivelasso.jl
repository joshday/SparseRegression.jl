# Adaptive LASSO is a method which uses elementwise regularization.
#
# Rather than `λ * abs(βj)`, the elementwise penalty is:
#  - `λ * inv(abs(βols_j)) * abs(β_j)`,
#    where `βols_j` is the ordinary least square coefficient.

module AdaptiveLassoExample
using SparseRegression, DataGenerator

x, y, β = linregdata(10_000, 100; β = vcat(ones(5), zeros(95)))


λ = .1 * inv.(abs.(x \ y))  # λj = .1 * inv(abs(βj))
o = SparseReg(Obs(x, y), L1Penalty(), λ)
fit!(o, ProxGrad(), MaxIter(100))

println()
println("The true β is $β")
println()
println("The adaptive lasso estimated β is $(coef(o))")
end
