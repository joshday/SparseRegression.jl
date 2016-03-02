# Sandbox for testing code

module Try
import StatisticalLearning
S = StatisticalLearning


macro display(expr) :(display($expr)) end


n, p = 1000, 10
x = randn(n, p)
β = collect(linspace(-.5, .5, p))

print_with_color(:green, "\nL2Regression\n")
y = x*β + randn(n)
λs = collect(.1:.1:1)

o = S.StatLearnPath(x, y,
    lambdas = λs, penalty = S.LassoPenalty())
@time S.StatLearnPath(x, y,
    lambdas = λs, penalty = S.LassoPenalty())

o2 = S.StatLearnPath(x, y, loss = S.AbsoluteErrorLoss(),
    lambdas = λs, penalty = S.LassoPenalty())
@time S.StatLearnPath(x, y, loss = S.AbsoluteErrorLoss(),
    lambdas = λs, penalty = S.LassoPenalty())

o3 = S.StatLearnPath(x, y, loss = S.QuantileErrorLoss(.7),
    lambdas = λs, penalty = S.LassoPenalty())
@time S.StatLearnPath(x, y, loss = S.QuantileErrorLoss(.7),
    lambdas = λs, penalty = S.LassoPenalty())




end
