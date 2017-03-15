# SparseRegression

This package relies on primitives defined in the JuliaML ecosystem to implement high-performance algorithms for linear models which often produce sparsity in the coefficients.

# This package is a work in progress

# Install
```julia
Pkg.clone("https://github.com/joshday/SparseRegression.jl")
Pkg.checkout("LossFunctions")
Pkg.checkout("PenaltyFunctions")
```

# Example

```julia
Pkg.clone("https://github.com/joshday/DataGenerator.jl")
```

```julia
using SparseRegression, DataGenerator

x, y, b = linregdata(1000, 10)

SparseReg(x, y, LinearRegression(), L1Penalty(), .1, ProxGrad(verbose=true))
```
