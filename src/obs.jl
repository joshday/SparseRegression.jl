"""
    Obs(x, y)
    Obs(x, y, w)

Simple structure for holding observations with (optional) user-defined weights.
It is assumed that observations are stored in rows.
"""
struct Obs{W, T <: Number, X <: AbstractArray{T}, Y <: AbstractArray{T}}
    w::W    # weights
    x::X    # predictors
    y::Y    # response
end
#--------------------------------------------------------------------# Constructor without weights
function Obs(x::AbstractArray, y::AbstractArray)
    n1 = size(x, 1)
    n2 = size(y, 1)
    n1 == n2 || throw(DimensionMismatch("number of rows do not match: $n1, $n2"))
    Obs{Void, eltype(x), typeof(x), typeof(y)}(nothing, x, y)
end
#-----------------------------------------------------------------------# Constructor with weights
function Obs(x::AbstractArray, y::AbstractArray, w::AbstractArray)
    n1, n2, n3 = size(x, 1), size(y, 1), size(w, 1)
    n1 == n2 == n3 || throw(DimensionMismatch("number of rows do not match: $n1, $n2, $n3"))
    all(x -> x>=0, w) || throw(ArgumentError("weight vector must be nonnegative"))
    Obs{typeof(w), eltype(x), typeof(x), typeof(y)}(w, x, y)
end
#-----------------------------------------------------------------------# show
function Base.show(io::IO, o::Obs)
    println(io, typeof(o))
    println(io, "  > x: ", summary(o.x))
    println(io, "  > y: ", summary(o.y))
    print(io,   "  > w: ", summary(o.w))
end
#-----------------------------------------------------------------------# methods
nobs(o::Obs) = size(o, 1)
Base.size(o::Obs) = size(o.x)
Base.size(o::Obs, i) = size(o.x, i)



#------------------------------------------------------------------------# gradient!
# To calculate a gradient, we need two storage buffers (nvec, pvec)
#  - length n: stores derivatives with respect to x * β
#  - length p: stores x' * derivatives
function gradient!(nvec, pvec, β::Vector, L::Loss, O::Obs)
    A_mul_B!(nvec, O.x, β)            # nvec ← x * β
    deriv!(nvec, L, O.y, nvec)        # nvec ← deriv(L, y, x * β)
    multiply_by_weights!(nvec, O.w)   # nvec *.= w
    At_mul_B!(pvec, O.x, nvec)        # pvec ← x'nvec
end

# normalize (divide by n) and multiply weights
function multiply_by_weights!(nvec, w::Void)
    wt = inv(length(nvec))
    for i in eachindex(nvec)
        @inbounds nvec[i] *= wt
    end
end
function multiply_by_weights!(nvec, w)
    wt = inv(length(nvec))
    for i in eachindex(nvec)
        @inbounds nvec[i] *= w[i] * wt
    end
end
