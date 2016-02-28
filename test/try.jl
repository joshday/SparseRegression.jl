# Sandbox for testing code

module Try
import StatisticalLearning
S = StatisticalLearning
using GLMNet
using Distributions

macro display(expr) :(display($expr)) end


n, p = 10000, 100
x = randn(n, p)
β = collect(linspace(-.5, .5, p))

print_with_color(:green, "\nL2Regression\n")
y = x*β + randn(n)
o = S.GLMPath(x, y, model = S.L2Regression(), penalty = S.L1Penalty(), λs = collect(.1:.1:.4))
@time o = S.GLMPath(x, y, model = S.L2Regression(), penalty = S.L1Penalty(), λs = collect(.1:.1:.4), weights = rand(n))
@display o
@time gnet = glmnet(x, y, lambda = collect(.1:.1:.4))
# @display o.β


print_with_color(:green, "\nLogisticRegression\n")
y = Float64[rand(Bernoulli(1 / (1 + exp(-η)))) for η in x*β]

S.GLMPath(x, y, model = S.LogisticRegression(), penalty = S.L1Penalty(), λs = collect(.01:.01:.04))
@time o = S.GLMPath(x, y, model = S.LogisticRegression(), penalty = S.L1Penalty(), λs = collect(.01:.01:.04))
@display o
y2 = Matrix{Float64}(hcat(y .== 0, y .== 1))
@time glmnet(x, y2, Binomial(), lambda = collect(.01:.01:.04))
# @display o.β


print_with_color(:green, "\nProbitRegression\n")
S.GLMPath(x, y, model = S.ProbitRegression(), penalty = S.L1Penalty(), λs = collect(.01:.01:.04))
@time o = S.GLMPath(x, y, model = S.ProbitRegression(), penalty = S.L1Penalty(), λs = collect(.01:.01:.04))
@display o
# @time glmnet(x, y2, Binomial(), lambda = collect(.01:.01:.04))
# @display o.β


print_with_color(:green, "\nPoissonRegression\n")
y = Float64[rand(Poisson(exp(η))) for η in x*β]
S.GLMPath(x, y, model = S.PoissonRegression(), penalty = S.L1Penalty(), λs = collect(1.:4))
@time o = S.GLMPath(x, y, model = S.PoissonRegression(), penalty = S.L1Penalty(), λs = collect(1.:4))
@display o
# @display o.β



end
