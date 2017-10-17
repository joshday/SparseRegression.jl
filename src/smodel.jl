#-----------------------------------------------------------------------# SModel
struct SModel{ # A L P X Y W D
        A <: Algorithm, L <: Loss, P <: Penalty, X <: AbstractMatrix, Y <: AbstractVector,
        W <: Union{Void, AbstractWeights}, D <: ObsPlacement}
    x::X
    y::Y
    w::W
    β::Vector{Float64}
    λ::Vector{Float64}
    loss::L
    penalty::P
    algorithm::A
    dim::D

    function SModel(x::X, y::Y, w::W, λ::Vector{Float64}, loss::L, penalty::P, a::A, dim::D) where
            {
                A<:Algorithm,
                L<:Loss,
                P<:Penalty,
                X<:AbstractMatrix,
                Y<:AbstractVector,
                W,
                D<:ObsPlacement
            }
        n, p = _nobs(x, dim), nparams(x, dim)
        _nobs(y, dim) == n || throw(DimensionMismatch("x and y have different nobs"))
        if w != nothing
            _nobs(w, dim) == n || throw(DimensionMismatch("weights are incorrect length"))
        end
        all(x -> x>=0, λ) || throw(ArgumentError("Regularization parameters must be >= 0"))
        new{A,L,P,X,Y,W,D}(x, y, w, zeros(p), λ, loss, penalty, a, dim)
    end
end

_nobs(y::AbstractVector, dim) = length(y)
_nobs(x::AbstractMatrix, ::Rows) = size(x, 1)
_nobs(x::AbstractMatrix, ::Cols) = size(x, 2)

nparams(x::AbstractMatrix, ::Rows) = size(x, 2)
nparams(x::AbstractMatrix, ::Cols) = size(x, 1)

# hacks for type-stable arbitrary argument order
# Default args(6): weight, λfactor, loss, penalty, algorithm, dim
const LinearRegression = LossFunctions.ScaledDistanceLoss{L2DistLoss, .5}
function d(x::AbstractMatrix)
    nothing, fill(.1, size(x,2)), LinearRegression(), L2Penalty(), ProxGrad(), Rows()
end

a(argu::AbstractWeights, t::Tuple)  = (argu, t[2], t[3], t[4], t[5], t[6])
a(argu::Vector{Float64}, t::Tuple)  = (t[1], argu, t[3], t[4], t[5], t[6])
a(argu::Loss, t::Tuple)             = (t[1], t[2], argu, t[4], t[5], t[6])
a(argu::Penalty, t::Tuple)          = (t[1], t[2], t[3], argu, t[5], t[6])
a(argu::Algorithm, t::Tuple)        = (t[1], t[2], t[3], t[4], argu, t[6])
a(argu::ObsPlacement, t::Tuple)     = (t[1], t[2], t[3], t[4], t[5], argu)

SModel(x::AbstractMatrix, y::AbstractVector, t::Tuple) = SModel(x, y, t...)
SModel(x::AbstractMatrix, y::AbstractVector) = SModel(x, y, d(x))
function SModel(x::AbstractMatrix, y::AbstractVector, a1)
    SModel(x, y, a(a1, d(x)))
end
function SModel(x::AbstractMatrix, y::AbstractVector, a1, a2)
    SModel(x, y, a(a2, a(a1, d(x))))
end
function SModel(x::AbstractMatrix, y::AbstractVector, a1, a2, a3)
    SModel(x, y, a(a3, a(a2, a(a1, d(x)))))
end
function SModel(x::AbstractMatrix, y::AbstractVector, a1, a2, a3, a4)
    SModel(x, y, a(a4, a(a3, a(a2, a(a1, d(x))))))
end
function SModel(x::AbstractMatrix, y::AbstractVector, a1, a2, a3, a4, a5)
    SModel(x, y, a(a5, a(a4, a(a3, a(a2, a(a1, d(x)))))))
end
function SModel(x::AbstractMatrix, y::AbstractVector, a1, a2, a3, a4, a5, a6)
    SModel(x, y, a(a6, a(a5, a(a4, a(a3, a(a2, a(a1, d(x))))))))
end

#-----------------------------------------------------------------------# show
const ticks = ['▁','▂','▃','▄','▅','▆','▇','█']
function Base.show(io::IO, o::SModel)
    println(io, "█ SModel: ncoefficients = $(length(o.β)), nobs = $(_nobs(o.x, o.dim))")
    println(io, "  > β        : ", o.β')
    println(io, "  > λ factor : ", o.λ')
    println(io, "  > Loss     : ", o.loss)
    println(io, "  > Penalty  : ", o.penalty)
    println(io, "  > Data")
    println(io, "    - x : ", summary(o.x))
    println(io, "    - y : ", summary(o.y))
    print(io,   "    - w : ", summary(o.w))
end

coef(o::SModel) = o.β


# assumes o.algorithm has fields nvec, pvec
function gradient!(o::SModel{A}) where {A <: GradientAlgorithm}

end
