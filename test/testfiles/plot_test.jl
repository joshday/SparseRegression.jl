module TestPlots
reload("SparseRegression")
using SparseRegression, Plots; plotlyjs()

include("../datagenerator.jl")

x, y, β = logregdata(10_000, 10, false, V = [.6 ^ abs(i-j) for i in 1:10, j in 1:10])
o = SparseReg(x, y, penalty = LassoPenalty(), model = LogisticRegression(),
    algorithm = Fista(step = .1), lambda = .01:.01:.1)
display(plot(o))
display(plot(o, x, y))


x, y, β = linregdata(10_000, 10, V = [.6 ^ abs(i-j) for i in 1:10, j in 1:10])
o = SparseReg(x, y, penalty = RidgePenalty(), lambda = .1:.1:1)
display(plot(o))
display(plot(o, x, y))
end
