module SparseRegression

using Reexport, ArgCheck
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
typealias AVec AbstractVector
typealias AMat AbstractMatrix
typealias AVecF AbstractVector{Float64}
typealias AMatF AbstractMatrix{Float64}
typealias VecF Vector{Float64}
typealias MatF Matrix{Float64}

typealias AverageMode LossFunctions.AverageMode

abstract Algorithm
abstract OfflineAlgorithm   <: Algorithm
abstract OnlineAlgorithm    <: Algorithm
function Base.print(io::IO, a::Algorithm)
    print(io, replace(string(typeof(a)), "SparseRegression.", ""))
end

abstract AbstractSparseReg

const ϵ = 1e-8  # constant to avoid dividing by zero, etc.

#-----------------------------------------------------------------------------# typealias
typealias LinearRegression      LossFunctions.ScaledDistanceLoss{L2DistLoss,0.5}
typealias L1Regression          L1DistLoss
typealias LogisticRegression    LogitMarginLoss
typealias PoissonRegression     PoissonLoss
typealias HuberRegression       HuberLoss
typealias SVMLike               L1HingeLoss
typealias QuantileRegression    QuantileLoss
typealias DWDLike               DWDMarginLoss


#-----------------------------------------------------------------------------# SparseReg
type SparseReg{L <: Loss, P <: Penalty} <: AbstractSparseReg
    β::VecF
    loss::L
    penalty::P
    λ::VecF     # element-wise penalties
end
SparseReg(p::Integer, loss::Loss, pen::Penalty, λ::VecF) = SparseReg(zeros(p), loss, pen, λ)
function SparseReg(p::Integer; loss::Loss = LinearRegression(), penalty::Penalty = NoPenalty(),
                   λ::VecF = fill(.1, p))
    SparseReg(p, loss, penalty, λ)
end

# function SparseReg(x::AMat, y::AVec; kw...)
#     o = SparseReg(size(x, 2); kw...)
#     fit!(o, x, y)
#     o
# end
function print_item(io::IO, name::AbstractString, value, ln::Bool = true)
    print(io, "  >" * @sprintf("%13s", name * ":  "))
    ln ? println(io, value) : print(io, value)
end
function Base.show(io::IO, o::SparseReg)
    println(io, "Sparse Regression Model")
    print_item(io, "β", o.β)
    print_item(io, "Loss", o.loss)
    print_item(io, "Penalty", o.penalty, false)
    println(io, " with λ = $(o.λ)")
end

#-------------------------------------------------------------------------------# helpers
coef(o::SparseReg) = o.β

logistic(x::Float64) = 1.0 / (1.0 + exp(-x))
xβ(o::SparseReg, x::AMat) = x * o.β
xβ(o::SparseReg, x::AVec) = dot(x, o.β)

predict(o::SparseReg, x::AVec) = _predict(o.loss, xβ(o, x))
predict(o::SparseReg, x::AMat) = _predict.(o.loss, xβ(o, x))

_predict(l::Loss, xβ::Real) = xβ
_predict(l::LogitMarginLoss, xβ::Real) = logistic(xβ)
_predict(l::PoissonLoss, xβ::Real) = exp(xβ)
function _predict!(l::Loss, xβ::AVec)
    for i in eachindex(xβ)
        @inbounds xβ[i] = _predict(l, xβ[i])
    end
    xβ
end

# loss(o::SparseReg, x::AMat, y::AVec) = value(o.loss, y, predict(o, x), o.avg)





#----------------------------------------------------------------------------# Algorithms
include("algorithms/proxgrad.jl")
defaultalg(loss::Loss, pen::Penalty) = PROXGRAD()
# include("algorithms/sgdlike.jl")
#
# include("solutionpath.jl")

end
