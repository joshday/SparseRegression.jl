# SparseRegression

This package relies on primitives defined in the JuliaML ecosystem to implement high-performance algorithms for linear models which often produce sparsity in the coefficients.

# Installation

Note: SparseRegression requires Julia 0.6

```julia
Pkg.clone("https://github.com/JuliaML/PenaltyFunctions.jl")
Pkg.clone("https://github.com/joshday/SparseRegression.jl")
Pkg.checkout("LossFunctions")
```


# Quick Example

```julia
# For creating data for generalized linear models
Pkg.clone("https://github.com/joshday/DataGenerator.jl")
```

```julia
using SparseRegression, DataGenerator

x, y, b = linregdata(1000, 10)

# Order of arguments after x, y does not matter (and it's type stable!)
SparseReg(x, y, LinearRegression(), L1Penalty(), .1, ProxGrad(verbose=true), ones(10))
```

# Design

SparseRegression fits statistical models of the form `f(β) + λ * g(β)` where `f` is the loss and `g` is a penalty/regularization term.

## SparseReg

**The `SparseReg` type is the "organizer" for a model**.  It keeps track of everything but the observations.

```julia
immutable SparseReg{A <: Algorithm, L <: Loss, P <: Penalty}
    β::VecF       # coefficient vector
    loss::L       # loss f(β)
    penalty::P    # penalty g(β)
    algorithm::A  # algorithm used to calculate β, given other fields
    λ::Float64    # regularization parameter
    factor::VecF  # elementwise-adjustment to λ (default is ones)
end
```

### SparseReg Constructors

- Without Data
```julia
o = SparseReg(n_predictors, args...)
```

- With Data
```julia
o = SparseReg(x, y, args...)
```

- With Weighted data
```julia
o = SparseReg(x, y, w, args...)
```

Here, `args...` will get automatically mapped to the correct field:

Argument type      | Gets mapped to
-------------------|---------------
 `Loss`            | `o.loss`
 `Penalty`         | `o.penalty`
 `Algorithm`       | `o.algorithm`
 `Float64`         | `o.λ`
 `Vector{Float64}` | `o.factor`


## [Losses](https://github.com/JuliaML/LossFunctions.jl)
- Continuous response
  - `LinearRegression()`
  - `L1Regression()`
  - `PoissonRegression()`
  - `QuantileRegression(q)`
  - `HuberRegression(q = 1.0)`
- Bivariate response (assumes each y is in [-1, 1])
  - `LogisticRegression()`
  - `SVMLike()`
  - `DWDLike(q)`

## [Penalties](https://github.com/JuliaML/PenaltyFunctions.jl)
- `NoPenalty()`
- `L1Penalty()`
- `L2Penalty()`
- `ElasticNetPenalty(a = .5)`
- `SCADPenalty(a = 3.7, γ = 1.0)`
- `MCPPenalty(γ = 2.0)`
- `LogPenalty(η = 1.0)`

## Algorithms
### Offline Algorithms
- Proximal Gradient Method
  - Works with convex penalties (`NoPenalty`, `L1Penalty`, `L2Penalty`, `ElasticNetPenalty`)
  - `ProxGrad(;maxit=100, tol=1e-6, verbose=false, step=1.0)`
### Online Algorithms
- Stochastic Gradient Descent
  - `SGD(wt::OnlineStats.Weight, η::Float64)`
