module SweepTest
using SparseRegression, FactCheck

facts("SweepTest") do
    n, p = 1000, 11
    x = randn(n, p)
    β = collect(linspace(-.5, .5, p))
    y = x*β + randn(n)

    context("L2Regression") do
        o = SparseReg(x, y, algorithm = Sweep())
		o2 = SparseReg(x, y, ones(n), algorithm = Sweep())
		@fact coef(o) --> roughly(coef(o2))
    end
end
end
