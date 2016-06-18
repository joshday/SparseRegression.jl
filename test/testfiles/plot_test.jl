module TestPlots
reload("SparseRegression")
using SparseRegression, Plots

include("../datagenerator.jl")
x, y, β = logregdata(10_000, 10, false, V = [.6^abs(i-j) for i in 1:10, j in 1:10])

o = SparseReg(x, y, penalty = LassoPenalty(), model = LogisticRegression(),
    algorithm = Fista(step = .1), lambda = .01:.01:.1)

plotlyjs()

display(plot(o))
display(plot(o, x, y))
end
