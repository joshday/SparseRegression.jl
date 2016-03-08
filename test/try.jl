# Sandbox for testing code
module Try
using StatsBase, Distributions, Plots
plotly()
import SparseRegression
S = SparseRegression


macro display(expr) :(display($expr)) end


n, p = 10000, 20
x = randn(n, p)
xtest = randn(n, p)
# for j in 1:p
#     x[:, j] *= j
# end
β = collect(linspace(-.5, .5, p))
pfac = ones(p)
# pfac[3] = 0.0

# Linear Regression
y = x*β + randn(n)
ytest = xtest * β + randn(n)
λs = collect(0.:.1:1)

print_with_color(:green, "L2egression\n")
@time o = S.SparseReg(x, y, penalty = S.LassoPenalty())
@display o
@display plot(o, xtest, ytest)

# print_with_color(:green, "L1Regression\n")
# @time o = S.SparseReg(x, y, penalty = S.LassoPenalty(), model = S.L1Regression())
# @display coef(o)
# @display plot(o, x, y)
#
# print_with_color(:green, "QuantileRegression\n")
# @time o = S.SparseReg(x, y, penalty = S.LassoPenalty(), model = S.QuantileRegression(.7))
# @display coef(o)
# @display plot(o, x, y)
#
# print_with_color(:green, "HuberRegression\n")
# @time o = S.SparseReg(x, y, penalty = S.LassoPenalty(), model = S.HuberRegression(.7))
# @display coef(o)
# @display plot(o, x, y)
#
#
#
# # Logistic Regression
# y2 = 2.0 * [rand(Bernoulli(1 / (1 + exp(-η)))) for η in x*β] - 1.0
# print_with_color(:green, "LogisticRegression\n")
# @time o = S.SparseReg(x, y2, penalty = S.LassoPenalty(), model = S.LogisticRegression())
# @display coef(o)
# @display plot(o, x, y2)
#
# print_with_color(:green, "SVMLike\n")
# @time o = S.SparseReg(x, y2, penalty = S.LassoPenalty(), model = S.SVMLike())
# @display coef(o)
# @display plot(o, x, y2)

end
