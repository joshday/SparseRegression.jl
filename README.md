[![Build Status](https://travis-ci.org/joshday/SparseRegression.jl.svg?branch=master)](https://travis-ci.org/joshday/SparseRegression.jl)
[![codecov](https://codecov.io/gh/joshday/SparseRegression.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/joshday/SparseRegression.jl)

# SparseRegression

This package relies on primitives defined in the JuliaML ecosystem to implement high-performance algorithms for linear models which often produce sparsity in the coefficients.  SparseRegression aims to solve statistical learning problems of the form:

<img width=300 src="https://cloud.githubusercontent.com/assets/8075494/25072239/5d85db30-2297-11e7-817e-e7bebaf056cd.png">

That is, we want to minimize a loss, subject to element-wise regularization of the coefficients.  

# Installation

SparseRegression requires Julia 0.6

```julia
Pkg.clone("https://github.com/joshday/SparseRegression.jl")
```

# Example

```julia
using SparseRegression
include(Pkg.dir("SparseRegression", "test", "datagenerator.jl"))

x, y, b = DataGenerator.linregdata(10_000, 10)


s = SparseReg(Obs(x, y), LinearRegression(), L2Penalty())
fit!(s, ProxGrad(), MaxIter(50), Converged(coef))

s = SparseReg(Obs(x, y), LinearRegression(), L2Penalty())
fit!(s, Sweep())
```

# Low-level details
The core iterations are performed with [JuliaML/LearningStrategies.jl](https://github.com/JuliaML/LearningStrategies.jl).  

SparseRegression.jl provides:
1. `SparseReg`, a type for organizing a model
2. New `LearningStrategy` subtypes for minimizing the objective function of a model
   - `ProxGrad(stepsize)`: Proximal Gradient Method
   - `Fista(stepsize)`: Faster iterative shrinkage and thresholding algorithm
   - `Sweep()`:  Linear/Ridge regression via the sweep operator
