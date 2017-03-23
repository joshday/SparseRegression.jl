# SparseRegression

This package relies on primitives defined in the JuliaML ecosystem to implement high-performance algorithms for linear models which often produce sparsity in the coefficients.

# Installation

Note: SparseRegression requires Julia 0.6

```julia
Pkg.clone("https://github.com/JuliaML/PenaltyFunctions.jl")
Pkg.clone("https://github.com/joshday/SparseRegression.jl")
Pkg.checkout("LossFunctions")
```

# This package is a work in progress

## Example

```julia
using SparseRegression
include(Pkg.dir("SparseRegression", "test", "datagenerator.jl"))

x, y, b = DataGenerator.linregdata(10_000, 10)

observations = Obs(x, y)

SweepModel(observations; penalty = L2Penalty(), λ = collect(0:.01:.1))
# fit(SweepModel, observations; penalty = L2Penalty, λ = collect(0:.01:.1))

ProximalGradientModel(observations; loss = HuberLoss(2.0), penalty = L1Penalty())
# fit(ProximalGradientModel, x, y; loss = HuberLoss(2.0), penalty = L1Penalty())

StochasticModel(observations, ADAGRAD(); loss = L1Regression(), penalty = ElasticNetPenalty(.1))
```
