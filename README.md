| Documentation | Master Build | Test Coverage |
|---------------|---------------|---------------|
| [![](https://img.shields.io/badge/docs-latest-blue.svg)](https://joshday.github.io/SparseRegression.jl/latest) | [![Build Status](https://travis-ci.org/joshday/SparseRegression.jl.svg?branch=master)](https://travis-ci.org/joshday/SparseRegression.jl) [![Build status](https://ci.appveyor.com/api/projects/status/qs7pa6m3tx6ivyq7?svg=true)](https://ci.appveyor.com/project/joshday/sparseregression-jl) | [![codecov](https://codecov.io/gh/joshday/SparseRegression.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/joshday/SparseRegression.jl)

# SparseRegression

This package relies on primitives defined in the [JuliaML](https://github.com/JuliaML) ecosystem to implement high-performance algorithms for linear models which often produce sparsity in the coefficients.  

```julia
x = randn(10_000, 50)
y = x * linspace(-1, 1, 50) + randn(10_000)

s = SModel(x, y, L2DistLoss(), L2Penalty())
@time learn!(s)
s
```