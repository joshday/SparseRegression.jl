module Try
using StatsBase, DataGenerator, SparseRegression
x, y, β = linregdata(100_000, 100)

s = SparseReg(Obs(x, y))
@time @show learn!(s, Sweep())
end
