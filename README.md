| Build Status |
|:------------:|
|[![Build Status](https://travis-ci.org/joshday/SparseRegression.jl.svg?branch=master)](https://travis-ci.org/joshday/SparseRegression.jl) [![codecov](https://codecov.io/gh/joshday/SparseRegression.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/joshday/SparseRegression.jl)


# SparseRegression

This package relies on primitives defined in the JuliaML ecosystem to implement high-performance algorithms for linear models which often produce sparsity in the coefficients.   


___
Install on Julia 0.6 with
```julia
Pkg.clone("https://github.com/joshday/SparseRegression.jl")
```


---

# Readme Contents
1. [What Can SparseRegression Do?](#what-can-sparseregression-do)
1. [SModel](#smodel)
1. [Algorithms](#algorithms)

---


# What Can SparseRegression Do?

SparseRegression aims to solve statistical learning problems of the form:

<img width=300 src="https://cloud.githubusercontent.com/assets/8075494/25072239/5d85db30-2297-11e7-817e-e7bebaf056cd.png">

That is, we want to minimize a loss, subject to element-wise regularization of the coefficients.

- With few exceptions, SparseRegression can handle:
  - any Loss from [LossFunctions.jl](https://github.com/JuliaML/LossFunctions.jl#available-losses)
  - any ElementPenalty from [PenaltyFunctions.jl](https://github.com/JuliaML/PenaltyFunctions.jl#available-penalties)


# `SModel`
- The main struct exported by SparseRegression is `SModel`:

```julia
struct SModel{L <: Loss, P <: Penalty}
    β::Vector{Float64}
    λfactor::Vector{Float64}
    loss::L
    penalty::P
end
```

# Observations
- Observations are wrapped in a lightweight `Obs` type
```julia
julia> x, y = randn(1000, 10), randn(1000);

julia> Obs(x,y)
SparseRegression.Obs{Void,Float64,Array{Float64,2},Array{Float64,1}}
  > x: 1000×10 Array{Float64,2}
  > y: 1000-element Array{Float64,1}
  > w: Void
```
- Optionally, the observations can be given a weight vector
```julia
julia> Obs(x, y, rand(1000))
SparseRegression.Obs{Array{Float64,1},Float64,Array{Float64,2},Array{Float64,1}}
  > x: 1000×10 Array{Float64,2}
  > y: 1000-element Array{Float64,1}
  > w: 1000-element Array{Float64,1}
```
- This allows algorithms to dispatch on whether or not observations are weighted.


### Toy Example
```julia
using SparseRegression

x, y = randn(1000, 10), randn(1000)

obs = Obs(x, y)

s = SModel(obs)

# Learn the model using Proximal Gradient Method
# - maximum of 50 iterations
# - convergence criteria: norm(β - βold) < 1e-6
learn!(s, ProxGrad(obs), MaxIter(50), Converged(coef))
```



# Algorithms

### `ProxGrad(obs, s)`
Proximal Gradient Method with step size `s`.  Handles any loss and convex penalty.

### `Fista(obs, s)`
Fast Iterative Shrinkage-Thresholding Algorithm (accelerated proximal gradient) with step size `s`.  Handles any loss and convex penalty.

### `GradientDescent(obs, s)`
Gradient Descent with step size `s`.  Handles any loss and penalty.

### `Sweep(obs)`
Linear or Ridge regression via the [sweep operator](https://github.com/joshday/SweepOperator.jl).

### `LinRegCholesky(obs)`
Linear or Ridge regression via Cholesky decomposition
