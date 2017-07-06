| Build Status |
|:------------:|
|[![Build Status](https://travis-ci.org/joshday/SparseRegression.jl.svg?branch=master)](https://travis-ci.org/joshday/SparseRegression.jl) [![codecov](https://codecov.io/gh/joshday/SparseRegression.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/joshday/SparseRegression.jl)


# SparseRegression

This package relies on primitives defined in the [JuliaML](https://github.com/JuliaML) ecosystem to implement high-performance algorithms for linear models which often produce sparsity in the coefficients.  

<img width=400 src="https://user-images.githubusercontent.com/8075494/27926282-d99e2792-6255-11e7-91ec-a4421a8bfd75.png">

The three core JuliaML packages that SparseRegression brings together are:

- [LossFunctions](https://github.com/JuliaML/LossFunctions.jl)
  - "grammar of losses": <img width=15 src="https://user-images.githubusercontent.com/8075494/27926340-148c700c-6256-11e7-8a6c-f6aa7ae796b7.png">
- [PenaltyFunctions](https://github.com/JuliaML/PenaltyFunctions.jl)
  - "grammar of regularization": <img width=12 src = "https://user-images.githubusercontent.com/8075494/27926360-2855af7c-6256-11e7-90c1-0f924d5131bf.png">
- [LearningStrategies](https://github.com/JuliaML/LearningStrategies.jl)
  - "grammar of iterative learning algorithms"

With few exceptions, SparseRegression can handle:
  - any Loss from [LossFunctions.jl](https://github.com/JuliaML/LossFunctions.jl#available-losses)
  - any ElementPenalty from [PenaltyFunctions.jl](https://github.com/JuliaML/PenaltyFunctions.jl#available-penalties)


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

# `SModel`
The main struct exported by SparseRegression is `SModel`:

```julia
struct SModel{L <: Loss, P <: Penalty}
    β::Vector{Float64}
    λfactor::Vector{Float64}
    loss::L
    penalty::P
end
```
An `SModel` is constructed with the number of predictors (or `Obs`), as well as a loss, penalty, and λfactor in any order (and it's type stable).
```julia
SModel(5)  # default: LinearRegression, L2Penalty(), fill(.1, 5)
SModel(5, LogisticRegression(), L1Penalty())
SModel(5, L2Penalty(), L1HingeLoss())
SModel(obs, NoPenalty(), QuantileRegression(.7))
```

After creating an SModel, it must then be learned with an `Algorithm` and any other number of learning strategies.

# Example
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

An `Algorithm` contains `Obs`, parameters for the algorithm, and storage buffers.  Some algorithms only work with specific loss/penalty combinations.

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
