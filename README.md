# SparseRegression

This package relies on primitives defined in the JuliaML ecosystem to implement high-performance algorithms for linear models which often produce sparsity in the coefficients.

# This package is a work in progress

# Install
```julia
Pkg.clone("https://github.com/joshday/SparseRegression.jl")
Pkg.checkout("LossFunctions")
Pkg.checkout("PenaltyFunctions")  # SparseRegression is broken until PR gets merged here
```
