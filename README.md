[![Build Status](https://travis-ci.org/joshday/SparseRegression.jl.svg?branch=master)](https://travis-ci.org/joshday/SparseRegression.jl)
[![codecov](https://codecov.io/gh/joshday/SparseRegression.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/joshday/SparseRegression.jl)

# SparseRegression

This package relies on primitives defined in the JuliaML ecosystem to implement high-performance algorithms for linear models which often produce sparsity in the coefficients.   


# Readme Contents
1. [What Can SparseRegression Do?](#what-can-sparseregression-do)
2. [Installation](#installation)
3. [Toy Example](#toy-example)
4. [Algorithms](#algorithms)


# What Can SparseRegression Do?

SparseRegression aims to solve statistical learning problems of the form:

<img width=300 src="https://cloud.githubusercontent.com/assets/8075494/25072239/5d85db30-2297-11e7-817e-e7bebaf056cd.png">

That is, we want to minimize a loss, subject to element-wise regularization of the coefficients.

- With few exceptions, SparseRegression can handle:
  - any Loss from [LossFunctions.jl](https://github.com/JuliaML/LossFunctions.jl#available-losses)
  - any ElementPenalty from [PenaltyFunctions.jl](https://github.com/JuliaML/PenaltyFunctions.jl#available-penalties)

# Installation

SparseRegression requires Julia 0.6

```julia
Pkg.clone("https://github.com/joshday/SparseRegression.jl")
```

# Toy Example

```julia
using SparseRegression

include(Pkg.dir("SparseRegression", "test", "datagenerator.jl"))

x, y, b = DataGenerator.linregdata(10_000, 10)

s = SparseReg(Obs(x, y), LinearRegression(), L2Penalty())

learn!(s, ProxGrad(), MaxIter(50), Converged(coef))
```

# Algorithms

### `ProxGrad(s)`

### `Fista(s)`

### `Sweep()`
