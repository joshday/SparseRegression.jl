# SparseRegression

This package relies on primitives defined in the JuliaML ecosystem to implement high-performance algorithms for linear models which often produce sparsity in the coefficients.

## Install
```julia
Pkg.clone("https://github.com/joshday/SparseRegression.jl")
Pkg.checkout("LossFunctions")
Pkg.checkout("PenaltyFunctions")
```


## Example

```julia
Pkg.clone("https://github.com/joshday/DataGenerator.jl")
```

```julia
using SparseRegression, DataGenerator

x, y, b = linregdata(1000, 10)

# Order of arguments after x, y does not matter (and it's type stable!)
SparseReg(x, y, LinearRegression(), L1Penalty(), .1, ProxGrad(verbose=true))
```

## Design

SparseRegression fits statistical models of the form `f(β) + g(β)` where `f` is the "loss" and `g` is a penalty/regularization term.

The `SparseReg` type is the "organizer" for a model.  It holds the:
- parameter vector `β`
- loss function (from LossFunctions.jl)
- penalty function (from PenaltyFunctions.jl)
- regularization parameter `λ`
- penalty factor `factor`
  - penalties `g(β)` are applied elementwise to β.  
  - For example, `L1Penalty()` (LASSO) will result in `g(β) = λ * sum(factor .* abs.(β))`
  - This allows for "adaptive" penalties or removing penalties completely from good predictors
- algorithm used to calculate `β`, given the other fields


```julia
immutable SparseReg{A <: Algorithm, L <: Loss, P <: Penalty}
    β::VecF
    loss::L
    penalty::P
    algorithm::A
    λ::Float64
    factor::VecF
end
```

### [Losses](https://github.com/JuliaML/LossFunctions.jl)
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

### [Penalties](https://github.com/JuliaML/PenaltyFunctions.jl)
- Penalties
  - `NoPenalty()`
  - `L1Penalty()`
  - `L2Penalty()`
  - `ElasticNetPenalty(a = .5)`
  - `SCADPenalty(a = 3.7, γ = 1.0)`
  - `MCPPenalty(γ = 2.0)`
  - `LogPenalty(η = 1.0)`

### Algorithms
- Offline Algorithms
  - Proximal Gradient Method
    - Works with convex penalties (`NoPenalty`, `L1Penalty`, `L2Penalty`, `ElasticNetPenalty`)
    - `ProxGrad(;maxit=100, tol=1e-6, verbose=false, step=1.0)`
- Online Algorithms
  - Stochastic Gradient Descent
  - `SGD(wt::Weight, step::Float64)`
