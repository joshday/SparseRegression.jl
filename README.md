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

x, y, b = linregdata(1000, 5)

# Order of arguments after x, y does not matter
args = LinearRegression(), L1Penalty(), .1
fitmodel(x, y, args...; verbose = true)

args = L2Penalty(), HuberRegression(2.)
fitpath(x, y, args...; λs = .01:.01:.1)
```
