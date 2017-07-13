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
