module SparseRegression

using Reexport
@reexport using LearnBase
@reexport using LossFunctions
@reexport using PenaltyFunctions
importall LossFunctions
import SweepOperator
import StatsBase: predict, coef

export
    SparseReg, SolutionPath, predict, coef,
    # algorithms
    ProxGrad,
    # Model typealiases
    LinearRegression, L1Regression, LogisticRegression, PoissonRegression, HuberRegression, SVMLike, DWDLike, QuantileRegression


#---------------------------------------------------------------------------------# types
const AVec        = AbstractVector
const AMat        = AbstractMatrix
const AVecF       = AbstractVector{Float64}
const AMatF       = AbstractMatrix{Float64}
const VecF        = Vector{Float64}
const MatF        = Matrix{Float64}
const AverageMode = LossFunctions.AverageMode

const LinearRegression      = LossFunctions.ScaledDistanceLoss{L2DistLoss,0.5}
const L1Regression          = L1DistLoss
const LogisticRegression    = LogitMarginLoss
const PoissonRegression     = PoissonLoss
const HuberRegression       = HuberLoss
const SVMLike               = L1HingeLoss
const QuantileRegression    = QuantileLoss
const DWDLike               = DWDMarginLoss


abstract Algorithm
abstract OfflineAlgorithm   <: Algorithm
abstract OnlineAlgorithm    <: Algorithm
Base.show(io::IO, a::Algorithm) = print(io, name(a))

#-------------# constant vector of ones
immutable Ones <: AVecF n::Int end
Ones(y::AVec) = Ones(length(y))
Base.size(o::Ones) = (o.n, )
Base.getindex(o::Ones, i) = 1.0

#-------------# observations
immutable Observations{X <: AMat, Y <: AVec, W <: AVec}
    x::X
    y::Y
    w::W
end
Observations(x::AMat, y::AVec, w::AVec = Ones(y)) = Observations(x, y, w)

#----------------------------------------------------------------------# SparseReg
immutable SparseReg{A <: Algorithm, L <: Loss, P <: Penalty}
    β::VecF
    loss::L
    penalty::P
    algorithm::A
    λ::Float64
    factor::VecF
end
function SparseReg(p::Integer, l::Loss, r::Penalty, a::Algorithm, λ::Float64, factor::VecF)
    SparseReg(zeros(p), l, r, a, λ, factor)
end

# TODO: make type stable
function SparseReg(p::Integer, args...)
    l = LinearRegression()
    r = NoPenalty()
    a = default(Algorithm)
    λ = 0.01
    f = ones(p)
    for arg in args
        l, r, a, λ, f = _arg(l, r, a, λ, f, arg)
    end
    SparseReg(p, l, r, a, λ, f)
end

_arg(l::Loss, r::Penalty, a::Algorithm, λ::Float64, f::VecF, t::Loss)       = (t, r, a, λ, f)
_arg(l::Loss, r::Penalty, a::Algorithm, λ::Float64, f::VecF, t::Penalty)    = (l, t, a, λ, f)
_arg(l::Loss, r::Penalty, a::Algorithm, λ::Float64, f::VecF, t::Algorithm)  = (l, r, t, λ, f)
_arg(l::Loss, r::Penalty, a::Algorithm, λ::Float64, f::VecF, t::Float64)    = (l, r, a, t, f)
_arg(l::Loss, r::Penalty, a::Algorithm, λ::Float64, f::VecF, t::VecF)       = (l, r, a, λ, t)

function SparseReg(x::AMatF, y::AVecF, args...)
    o = SparseReg(size(x, 2), args...)
    fit!(o, Observations(x, y))
    o
end
function SparseReg(x::AMatF, y::AVecF, w::AVecF, args...)
    o = SparseReg(size(x, 2), args...)
    fit!(o, Observations(x, y, w))
    o
end

function Base.show(io::IO, o::SparseReg)
    println(io, "Sparse Regression Model")
    print_item(io, "β", o.β)
    print_item(io, "Loss", o.loss)
    print_item(io, "Penalty", o.penalty)
    print_item(io, "λ", o.λ)
    any(x -> x != 1.0, o.factor) && print_item(io, "λ scaling", o.factor)
    print_item(io, "Algorithm", o.algorithm)
end

coef(o::SparseReg) = o.β
logistic(x::Float64) = 1.0 / (1.0 + exp(-x))
xβ(o::SparseReg, x::AMat) = x * o.β
xβ(o::SparseReg, x::AVec) = dot(x, o.β)
predict(o::SparseReg, x::AVec) = _predict(o.loss, xβ(o, x))
predict(o::SparseReg, x::AMat) = _predict.(o.loss, xβ(o, x))
loss(o::SparseReg, x::AMat, y::AVec, args...) = value(o.loss, y, predict(o, x), args...)

#-------------------------------------------------------------------------------# helpers
name(a) = replace(string(typeof(a)), "SparseRegression.", "")
function print_item(io::IO, name::AbstractString, value)
    print(io, "  >" * @sprintf("%13s", name * ":  "))
    println(io, value)
end

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

function objective_value(o::SparseReg, obs::Observations, ŷ::AVec)
    value(o.loss, obs.y, ŷ, AvgMode.Mean()) + value(o.penalty, o.β)
end

#-------------------------------------------------------------------------------# algorithms
include("algorithms/proxgrad.jl")
# include("solutionpath.jl")

end
