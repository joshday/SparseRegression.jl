# SparseRegression

[![Build Status](https://travis-ci.org/joshday/SparseRegression.jl.svg?branch=master)](https://travis-ci.org/joshday/SparseRegression.jl)


Solution paths for penalized regression: `ℓ(β) = f(β) + J(β)`.  The main type exported by this package is `SparseReg`.


# `SparseReg(x, y; kw...)`
# `SparseReg(x, y, weights; kw...)`

### Keyword arguments:

| keyword          | type              | description                                       |
|:-----------------|:------------------|:--------------------------------------------------|
| `intercept`      | `Bool`            | Should an intercept be included in the model?     |
| `model`          | `Model`           | The type of model.  Default = `L2Regression()`    |
| `penalty`        | `Penalty`         | The type of penalty. Default = `NoPenalty()`      |
| `penalty_factor` | `Vector{Float64}` | Weights applied to the penalty.  Default is ones. |
| `lambda`         | `Vector{Float64}` | The lambdas for which to get a solution path      |
| `algorithm`      | `Algorithm`       | The algorithm/settings for the fitting procedure  |


# Algorithms
### `FISTA(;kw...)` (Fast Iterative Shrinkage-Thresholding Algorithm)

| keyword       | type              | description                                                |
|:--------------|:------------------|:-----------------------------------------------------------|
| `standardize` | `Bool`            | Should `x` values be standardized? Default = `true`        |
| `weights`     | `Vector{Float64}` | Weights for each observation                               |
| `tol`         | `Float64`         | tolerance for convergence.  Default = `1e-7`               |
| `maxit`       | `Int`             | Maximum number of iterations.  Default = `100`             |
| `stepsize`    | `Float64`         | Step size for gradient descent part of algorithm           |
| `verbose`     | `Bool`            | Whether to print information.  Default = `true`            |
| `crit`        | `Symbol`          | Convergence criteria: `:coef` or `:obj`.  Default = `true` |


### `CD(;kw...)` (Coordinate Descent)
- not yet implemented


### `Sweep()` (Only for `L2Regression` and `NoPenalty`)
- not yet implemented


# Models
- `L2Regression()`
- `L1Regression()`
- `LogisticRegression()`
- `PoissonRegression()`
- `SVMLike()`
- `HuberRegression(δ)`
- `QuantileRegression(τ)`


# Penalties
- `NoPenalty()`
- `RidgePenalty()`
- `LassoPenalty()`
- `ElasticNetPenalty(α = .5)`
- `SCADPenalty(a = 3.7)`


# Example

```julia
using SparseRegression, Plots; plotly()
n, p = 10000, 11
x = randn(n, p)
β = collect(linspace(-.5, .5, p))
y = x * β + randn(n)

o = SparseReg(x, y, penalty = LassoPenalty(), model = L2Regression())
plot(o)
```
