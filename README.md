# SparseRegression

This package relies on primitives defined in the JuliaML ecosystem to implement high-performance algorithms for linear models which often produce sparsity in the coefficients.

# Low-level notes
### The `SparseReg` type contains details of the "model"

```julia
immutable SparseReg{A <: Algorithm, L <: Loss, P <: Penalty}
    β::VecF       # coefficients
    loss::L       # loss (negative log-likelihood in some cases)
    penalty::P    # type of regularization
    algorithm::A  # algorithm used to create β, given all other fields
    λ::Float64    # regularization parameter
    factor::VecF  # element-wise adjustments to λ
end
```

### To fit a model, `SparseReg` needs `Observations`
```julia
immutable Observations{X <: AMat, Y <: AVec, W <: AVec}
    x::X  # x matrix
    y::Y  # y vector
    w::W  # weights for observations, length(w) == length(y)
end
```
