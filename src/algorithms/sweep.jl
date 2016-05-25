immutable Sweep <: Algorithm
    standardize::Bool

    function Sweep(;
        standardize::Bool = true
        )
        new(standardize)
    end
end

#-----------------------------------------------------------------------------# Sweep
# TODO: intercept, standardization
function fit!(o::SparseReg{L2Regression, NoPenalty, Sweep}, x::AMat, y::AVec, wts::AVec)
    n, p = size(x)
    A = zeros(p + 1, p + 1)
    useweights = length(wts) == n
    if useweights
        BLAS.syrk!('U', 'T', 1 / n, Diagonal(sqrt(wts)) * x, 0.0, sub(A, 1:p, 1:p))
        BLAS.gemv!('T', 1 / n, Diagonal(wts) * x, y, 0.0, slice(A, 1:p, p+1))
    else
        BLAS.syrk!('U', 'T', 1 / n, x, 0.0, sub(A, 1:p, 1:p))
        BLAS.gemv!('T', 1 / n, x, y, 0.0, slice(A, 1:p, p+1))
    end
    A[end, end] = sumabs2(y) / n
    sweep!(A, 1:p)
    o.β[:, 1] = sub(A, 1:p, p + 1)
    o
end



#--------------------------------------------------------------------# sweep! methods
"""
`sweep!(A, k, inv = false)`, `sweep!(A, k, v, inv = false)`

Symmetric sweep operator of the matrix `A` on element `k`.  `A` is overwritten.
`inv = true` will perform the inverse sweep.  Only the upper triangle is read and swept.

An optional vector `v` can be provided to avoid memory allocation.
This requires `length(v) == size(A, 1)`.  Both `A` and `v`
will be overwritten.

```julia
x = randn(100, 10)
xtx = x'x
sweep!(xtx, 1)
sweep!(xtx, 1, true)
```
"""
function sweep!{T<:Real}(A::AMat{T}, k::Integer, inv::Bool = false)
    n, p = size(A)
    # ensure @inbounds is safe
    @assert n == p "A must be square"
    @assert k <= p "pivot element not within range"
    @inbounds d = 1.0 / A[k, k]  # pivot
    # get column A[:, k] (hack because only upper triangle is available)
    akk = zeros(p)
    for j in 1:p
        if j <= k
            @inbounds akk[j] = A[j, k]
        else
            @inbounds akk[j] = A[k, j]
        end
    end
    BLAS.syrk!('U', 'N', -d, akk, 1.0, A)  # everything not in col/row k
    scale!(akk, d * (-1) ^ inv)
    for i in 1:k-1  # col k
        @inbounds A[i, k] = akk[i]
    end
    for j in k+1:p  # row k
        @inbounds A[k, j] = akk[j]
    end
    A[k, k] = -d  # pivot element
    A
end

function sweep!{T<:Real, I<:Integer}(A::AMat{T}, ks::AVec{I}, inv::Bool = false)
    for k in ks
        sweep!(A, k, inv)
    end
    A
end
