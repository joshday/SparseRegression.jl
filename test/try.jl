# Sandbox for testing code

module Try
using StatsBase
import StatisticalLearning
S = StatisticalLearning


macro display(expr) :(display($expr)) end


n, p = 1000, 10
x = randn(n, p)
β = collect(linspace(-.5, .5, p))

print_with_color(:green, "\nL2Regression\n")
y = x*β + randn(n)
λs = collect(0.:.1:1)

o = S.StatLearnPath(x, y,
    lambdas = λs, penalty = S.LassoPenalty(), verbose = false)
@time S.StatLearnPath(x, y,
    lambdas = λs, penalty = S.LassoPenalty())

o2 = S.StatLearnPath(x, y, loss = S.AbsoluteErrorLoss(),
    lambdas = λs, penalty = S.LassoPenalty(), verbose = false)
@time S.StatLearnPath(x, y, loss = S.AbsoluteErrorLoss(),
    lambdas = λs, penalty = S.LassoPenalty())

o3 = S.StatLearnPath(x, y, loss = S.QuantileErrorLoss(.5),
    lambdas = λs, penalty = S.LassoPenalty(), verbose = false)
@time S.StatLearnPath(x, y, loss = S.QuantileErrorLoss(.5),
    lambdas = λs, penalty = S.LassoPenalty())


@display coef(o)
@display coef(o2)
@display coef(o3)

end
