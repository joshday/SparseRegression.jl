# StatisticalLearning

[![Build Status](https://travis-ci.org/joshday/StatisticalLearning.jl.svg?branch=master)](https://travis-ci.org/joshday/StatisticalLearning.jl)


Solution paths for statistical learning problems of the form `f(Xβ) + λ * g(β)`.  


# `StatLearnPath(x, y; kw...)`

### Keyword arguments:

| keyword       | type              | description                                                                                      |
|:--------------|:------------------|:-------------------------------------------------------------------------------------------------|
| `intercept`   | `Bool`            | Should an intercept be included in the model?                                                    |
| `model`       | `Model`           | The type of model.  Default = `L2Regression()`                                                   |
| `penalty`     | `Penalty`         | The type of penalty. Default = `NoPenalty()`                                                     |
| `lambdas`     | `Vector{Float64}` | The lambdas for which to get a solution path                                                     |
| `standardize` | `Bool`            | Should `x` values be standardized? Coefficients are returned in original scale. default = `true` |
| `weights`     | `Vector{Float64}` | Weights for each observation.                                                                    |


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
using StatisticalLearning, Plots; plotly()
n, p = 10000, 11
x = randn(n, p)
β = collect(linspace(-5, 5, p))
y = x*β + randn(n)

o = StatLearnPath(x, y, penalty = LassoPenalty(), lambdas = 0:.1:6)
plot(o)
```
