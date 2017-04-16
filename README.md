[![Build Status](https://travis-ci.org/joshday/SparseRegression.jl.svg?branch=master)](https://travis-ci.org/joshday/SparseRegression.jl)

[![codecov](https://codecov.io/gh/joshday/SparseRegression.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/joshday/SparseRegression.jl)

# SparseRegression

This package relies on primitives defined in the JuliaML ecosystem to implement high-performance algorithms for linear models which often produce sparsity in the coefficients.

SparseRegression aims to solve statistical learning problems of the form:

![](https://cloud.githubusercontent.com/assets/8075494/25072239/5d85db30-2297-11e7-817e-e7bebaf056cd.png)

# Installation

Note: SparseRegression requires Julia 0.6

```julia
Pkg.clone("https://github.com/joshday/SparseRegression.jl")
```

# Example

```julia
using SparseRegression
include(Pkg.dir("SparseRegression", "test", "datagenerator.jl"))

x, y, b = DataGenerator.linregdata(10_000, 10)

observations = Obs(x, y)

SweepModel(observations; penalty = L2Penalty(), λ = collect(0:.01:.1))

ProximalGradientModel(observations; loss = HuberLoss(2.0), penalty = L1Penalty())

StochasticModel(observations, ADAGRAD(); loss = L1Regression(), penalty = ElasticNetPenalty(.1))
```

# Notes on Design


#### Observations
Observations are held in a type: `Obs(x, y)` (or `Obs(x, y, w)` for weighted observations)
