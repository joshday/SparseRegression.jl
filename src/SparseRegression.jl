module SparseRegression

using Reexport, Parameters
@reexport using LearnBase
@reexport using LossFunctions
@reexport using PenaltyFunctions
importall LossFunctions
import SweepOperator
import StatsBase: predict, coef

export
    SparseReg, SolutionPath, predict, coef,
    # algorithms
    PROXGRAD,
    # Model typealiases
    LinearRegression, L1Regression, LogisticRegression, PoissonRegression, HuberRegression, SVMLike, DWDLike, QuantileRegression


#---------------------------------------------------------------------------------# types
AVec        = AbstractVector
AMat        = AbstractMatrix
AVecF       = AbstractVector{Float64}
AMatF       = AbstractMatrix{Float64}
VecF        = Vector{Float64}
MatF        = Matrix{Float64}
AverageMode = LossFunctions.AverageMode

LinearRegression      = LossFunctions.ScaledDistanceLoss{L2DistLoss,0.5}
L1Regression          = L1DistLoss
LogisticRegression    = LogitMarginLoss
PoissonRegression     = PoissonLoss
HuberRegression       = HuberLoss
SVMLike               = L1HingeLoss
QuantileRegression    = QuantileLoss
DWDLike               = DWDMarginLoss


abstract Algorithm
abstract OfflineAlgorithm   <: Algorithm
abstract OnlineAlgorithm    <: Algorithm


name(a) = replace(string(typeof(a)), "SparseRegression.", "")
Base.show(io::IO, a::Algorithm) = print(io, name(a))

immutable Zeros <: AVecF end
Base.size(v::Zeros) = (0, )
Base.getindex(v::Zeros, i) = 0.0
Base.show(io::IO, v::Zeros) = print(io, "Constant Vector of 0.0")

immutable CVec <: AVecF
    c::Float64
end
Base.size(v::CVec) = (0, )
Base.getindex(v::CVec, i) = v.c
Base.show(io::IO, v::CVec) = print(io, "Constant Vector of $(v.c)")


#----------------------------------------------------------------------# SparseReg
immutable SparseReg{A <: Algorithm, L <: Loss, P <: Penalty, T <: AVecF}
    β::VecF
    loss::L
    penalty::P
    algorithm::A
    λ::T
end
function SparseReg(p::Integer, l::Loss, r::Penalty, a::Algorithm, λ::AVecF)
    SparseReg(zeros(p), l, r, a, λ)
end


# TODO: add type stable version
function SparseReg(p::Integer, args...)
    l = LinearRegression()
    r = NoPenalty()
    a = default(Algorithm)
    λ = Zeros()
    for arg in args
        if typeof(arg) <: Loss
            l = arg
        elseif typeof(arg) <: Penalty
            r = arg
        elseif typeof(arg) <: Algorithm
            a = arg
        elseif typeof(arg) <: AVecF
            λ = arg
        else
            throw(ArumentError("Argument $arg is invalid"))
        end
    end
    SparseReg(p, l, r, a, λ)
end


function SparseReg(x::AMatF, y::AVecF, args...)
    o = SparseReg(size(x, 2), args...)
    fit!(o, x, y)
    o
end


function Base.show(io::IO, o::SparseReg)
    println(io, "Sparse Regression Model")
    print_item(io, "β", o.β)
    print_item(io, "Loss", o.loss)
    print_item(io, "Penalty", o.penalty)
    print_item(io, "λ", o.λ)
end

#-------------------------------------------------------------------------------# helpers
function print_item(io::IO, name::AbstractString, value)
    print(io, "  >" * @sprintf("%13s", name * ":  "))
    println(io, value)
end
coef(o::SparseReg) = o.β
logistic(x::Float64) = 1.0 / (1.0 + exp(-x))
xβ(o::SparseReg, x::AMat) = x * o.β
xβ(o::SparseReg, x::AVec) = dot(x, o.β)
predict(o::SparseReg, x::AVec) = _predict(o.loss, xβ(o, x))
predict(o::SparseReg, x::AMat) = _predict.(o.loss, xβ(o, x))

# scary names so that nobody uses them
predict_from_xβ(l::Loss, xβ::Real) = xβ
predict_from_xβ(l::LogitMarginLoss, xβ::Real) = logistic(xβ)
predict_from_xβ(l::PoissonLoss, xβ::Real) = exp(xβ)
function xβ_to_ŷ!(l::Union{LogitMarginLoss, PoissonLoss}, xβ::AVec)
    for i in eachindex(xβ)
        @inbounds xβ[i] = predict_from_xβ(l, xβ[i])
    end
    xβ
end
xβ_to_ŷ!(l::Loss, xβ::AVec) = xβ;  # no-op if linear predictor == ŷ


loss(o::SparseReg, x::AMat, y::AVec, mode) = value(o.loss, y, predict(o, x), mode)


#-------------------------------------------------------------------------------# algorithms
include("algorithms/proxgrad.jl")

end
