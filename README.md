[![Build Status](https://travis-ci.org/joshday/SparseRegression.jl.svg?branch=master)](https://travis-ci.org/joshday/SparseRegression.jl)
[![codecov](https://codecov.io/gh/joshday/SparseRegression.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/joshday/SparseRegression.jl)

# SparseRegression

This package relies on primitives defined in the JuliaML ecosystem to implement high-performance algorithms for linear models which often produce sparsity in the coefficients.   

Install on Julia 0.6 with
```julia
Pkg.clone("https://github.com/joshday/SparseRegression.jl")
```

---

# Readme Contents
1. [What Can SparseRegression Do?](#what-can-sparseregression-do)
1. [SparseReg](#sparsereg)
1. [Algorithms](#algorithms)

---


# What Can SparseRegression Do?

SparseRegression aims to solve statistical learning problems of the form:

<img width=300 src="https://cloud.githubusercontent.com/assets/8075494/25072239/5d85db30-2297-11e7-817e-e7bebaf056cd.png">

That is, we want to minimize a loss, subject to element-wise regularization of the coefficients.

- With few exceptions, SparseRegression can handle:
  - any Loss from [LossFunctions.jl](https://github.com/JuliaML/LossFunctions.jl#available-losses)
  - any ElementPenalty from [PenaltyFunctions.jl](https://github.com/JuliaML/PenaltyFunctions.jl#available-penalties)


# SparseReg
The main struct exported by SparseRegression is `SparseReg`.



```julia
o = SparseReg(Obs(x, y), loss, penalty, λfactor)

o = SparseReg(Obs(x, y, w), loss, penalty, λfactor)
```
- `x` is a design matrix (observations in rows)
- `y` is the response vector
  - Either continuous or \{-1, 1\} for binary response
- `w` is an optional vector for weighted observations
- `loss` is a Loss from LossFunctions.jl
  - default is `LinearRegression() == scaled(L2DistLoss(), Val{.5})`
- `penalty` is an ElementPenalty from PenaltyFunctions.jl
  - default is `L2Penalty()`
- `λfactor` is the element-wise regularization parameters
  - `β[j]` receives penalty `λfactor[j] * value(penalty, β[j])`  
  - default is `fill(.1, size(x, 2))`

After creating a model, it must then be "learned" by dispatching on an algorithm and "learning strategies" from [LearningStrategies.jl](https://github.com/JuliaML/LearningStrategies.jl).  

```julia
using SparseRegression

x, y = randn(1000, 10), randn(1000)

s = SparseReg(Obs(x, y))

# Fit the model using Proximal Gradient Method
# - maximum of 50 iterations
# - convergence criteria: norm(β - βold) < 1e-6
learn!(s, ProxGrad(), MaxIter(50), Converged(coef))
```



# Algorithms

### `ProxGrad(s)`
Proximal Gradient Method with step size `s`.

### `Fista(s)`
Fast Iterative Shrinkage-Thresholding Algorithm (accelerated proximal gradient) with step size `s`.

### `Sweep()`
Linear or Ridge regression via the [sweep operator](https://github.com/joshday/SweepOperator.jl)
