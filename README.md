# WARNING:
**Development is unstable and features may change at any time**

# SparseRegression

[![Build Status](https://travis-ci.org/joshday/SparseRegression.jl.svg?branch=master)](https://travis-ci.org/joshday/SparseRegression.jl)
[![codecov](https://codecov.io/gh/joshday/SparseRegression.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/joshday/SparseRegression.jl)

Solution paths for penalized regression: `f(Θ) + λ * g(Θ)`


 The main type exported by this package is `SparseReg`.


# `SparseReg(x, y; kw...)`
# `SparseReg(x, y, weights; kw...)`

### Keyword arguments:

| keyword          | type              | description                                        |
|:-----------------|:------------------|:---------------------------------------------------|
| `intercept`      | `Bool`            | Should an intercept be included in the model?      |
| `model`          | `Model`           | The type of model.  Default = `LinearRegression()` |
| `penalty`        | `Penalty`         | The type of penalty. Default = `NoPenalty()`       |
| `penalty_factor` | `Vector{Float64}` | Weights applied to the penalty.  Default is ones.  |
| `lambda`         | `Vector{Float64}` | The lambdas for which to get a solution path       |
| `algorithm`      | `Algorithm`       | The algorithm/settings for the fitting procedure   |


# Algorithms
### `Fista(;kw...)` (Fast Iterative Shrinkage-Thresholding Algorithm)

**Accelerated Proximal Gradient Method**

| keyword       | type              | description                                                |
|:--------------|:------------------|:-----------------------------------------------------------|
| `standardize` | `Bool`            | Should `x` values be standardized? Default = `true`        |
| `weights`     | `Vector{Float64}` | Weights for each observation                               |
| `tol`         | `Float64`         | tolerance for convergence.  Default = `1e-7`               |
| `maxit`       | `Int`             | Maximum number of iterations.  Default = `100`             |
| `stepsize`    | `Float64`         | Step size for gradient descent part of algorithm           |
| `verbose`     | `Bool`            | Whether to print information.  Default = `true`            |
| `crit`        | `Symbol`          | Convergence criteria: `:coef` or `:obj`.  Default = `:obj` |


### `Sweep()` (Only for `LinearRegression` with `NoPenalty`)

**Efficient linear regression using the sweep operator.**

| keyword       | type   | description                                         |
|:--------------|:-------|:----------------------------------------------------|
| `standardize` | `Bool` | Should `x` values be standardized? Default = `true` |

```julia
using SparseRegression, GLM, BenchmarkTools
n, p = 1_000_000, 20
x = randn(n, p)
β = collect(1.0:p)
y = x * β + randn(n)

@benchmark SparseReg(x, y, algorithm = Sweep(), intercept = false)
# median time:      81.62 ms (0.00% GC)
# memory estimate:  20.19 kb

@benchmark x \ y
# median time:      355.34 ms (3.39% GC)
# memory estimate:  160.25 mb

@benchmark lm(x, y)
# median time:      414.89 ms (3.14% GC)
# memory estimate:  175.49 mb
```


# Models
- `LinearRegression()`
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
using SparseRegression, Plots
n, p = 10000, 11
x = randn(n, p)
β = collect(linspace(-.5, .5, p))
y = x * β + randn(n)

o = SparseReg(x, y, penalty = LassoPenalty(), model = LinearRegression(), algorithm = Fista(tol=1e-4))
plot(o)
plot(o, x, y)
```
