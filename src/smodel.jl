#-----------------------------------------------------------------------# SModel
struct SModel{ # L P X Y W
        L <: Loss, P <: Penalty, X <: AbstractMatrix, Y <: AbstractVector,
        W <: Union{Void, AbstractWeights}}
    x::X
    y::Y
    w::W
    β::Vector{Float64}
    λ::Vector{Float64}
    loss::L
    penalty::P

    function SModel(x::X, y::Y, w::W, λ::Vector{Float64}, loss::L, penalty::P) where
            {L<:Loss, P<:Penalty, X<:AbstractMatrix, Y<:AbstractVector,
                W<:Union{Void,AbstractWeights}}
        n, p = size(x)
        length(y) == n || throw(DimensionMismatch("x and y have different nobs"))
        if w != nothing
            length(w) == n || throw(DimensionMismatch("weights are incorrect length"))
        end
        all(x -> x>=0, λ) || throw(ArgumentError("Regularization parameters must be >= 0"))
        new{L,P,X,Y,W}(x, y, w, zeros(p), λ, loss, penalty)
    end
end

# hacks for type-stable arbitrary argument order
# Default args(6): weight, λfactor, loss, penalty
const LinearRegression = LossFunctions.ScaledDistanceLoss{L2DistLoss, .5}
function d(x::AbstractMatrix)
    nothing, fill(.1, size(x,2)), LinearRegression(), L2Penalty()
end

a(argu::AbstractWeights, t::Tuple)  = (argu, t[2], t[3], t[4])
a(argu::Vector{Float64}, t::Tuple)  = (t[1], argu, t[3], t[4])
a(argu::Loss, t::Tuple)             = (t[1], t[2], argu, t[4])
a(argu::Penalty, t::Tuple)          = (t[1], t[2], t[3], argu)

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


#-----------------------------------------------------------------------# show
const ticks = ['▁','▂','▃','▄','▅','▆','▇','█']
function Base.show(io::IO, o::SModel)
    println(io, "█ SModel")
    println(io, "  > β        : ", o.β')
    println(io, "  > λ factor : ", o.λ')
    println(io, "  > Loss     : ", o.loss)
    println(io, "  > Penalty  : ", o.penalty)
    println(io, "  > Data")
    println(io, "    - x : ", summary(o.x))
    println(io, "    - y : ", summary(o.y))
    print(io,   "    - w : ", summary(o.w))
end


#-----------------------------------------------------------------------# Methods
coef(o::SModel) = o.β
predict(o::SModel, x::AbstractMatrix) = x * o.β
predict(o::SModel, x::AbstractVector) = x'o.β
