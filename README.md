# StatisticalLearning

[![Build Status](https://travis-ci.org/joshday/StatisticalLearning.jl.svg?branch=master)](https://travis-ci.org/joshday/StatisticalLearning.jl)


FISTA algorithm for solving statistical learning problems of the form `f(Θ) + λ * g(Θ)`.  


# `StatLearn(x, y; kw...)`

### Keyword arguments:

| keyword       | type              | description                                         |
|:--------------|:------------------|:----------------------------------------------------|
| `intercept`   | `Bool`            | Should an intercept be included in the model?       |
| `model`       | `Model`           | The type of model.  default = `L2Regression()`      |
| `penalty`     | `Penalty`         | The type of penalty. default = `NoPenalty()`        |
| `lambdas`     | `Vector{Float64}` | The lambdas for which to get a solution path        |
| `standardize` | `Bool`            | Should `x` values be standardized? default = `true` |


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
