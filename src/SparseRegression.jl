module SparseRegression

import SweepOperator
import StatsBase: predict, coef
using LearnBase
using LossFunctions
using PenaltyFunctions

# Reexports
eval(Expr(:toplevel, Expr(:export, setdiff(names(LearnBase), [:LearnBase])...)))
eval(Expr(:toplevel, Expr(:export, setdiff(names(LossFunctions), [:LossFunctions])...)))
eval(Expr(:toplevel, Expr(:export, setdiff(names(PenaltyFunctions), [:PenaltyFunctions])...)))

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
# const AverageMode = LossFunctions.AverageMode

const LinearRegression      = LossFunctions.ScaledDistanceLoss{L2DistLoss,0.5}
const L1Regression          = L1DistLoss
const LogisticRegression    = LogitMarginLoss
const PoissonRegression     = PoissonLoss
const HuberRegression       = HuberLoss
const SVMLike               = L1HingeLoss
const QuantileRegression    = QuantileLoss
const DWDLike               = DWDMarginLoss


abstract type Algorithm end
abstract type OfflineAlgorithm   <: Algorithm end
abstract type OnlineAlgorithm    <: Algorithm end
Base.show(io::IO, a::Algorithm) = print(io, name(a))


#----------------------------------------------------------------------#  observations
# constant vector of ones (default observation weights)
immutable Ones <: AVecF n::Int end
Ones(y::AVec) = Ones(length(y))
Base.size(o::Ones) = (o.n, )
Base.getindex(o::Ones, i::Integer) = 1.
Base.getindex{I <: Integer}(o::Ones, rng::AVec{I}) = Ones(length(rng))

immutable Obs{W, X, Y}
    w::W
    x::X
    y::Y
end
Obs(x::AMat, y::AVec, w::AVec = Ones(y)) = Obs{typeof(w), typeof(x), typeof(y)}(w, x, y)

#----------------------------------------------------------------------# SparseReg
immutable SparseReg{A <: Algorithm, L <: Loss, P <: Penalty}
    β::VecF
    loss::L
    penalty::P
    algorithm::A
    λ::Float64
    factor::VecF
end

# Type stable constructor with arbitrary argument order!
function SparseReg(p::Integer, args...)
    l = getarg(p, Loss, args)
    r = getarg(p, Penalty, args)
    a = getarg(p, Algorithm, args)
    λ = getarg(p, Float64, args)
    f = getarg(p, VecF, args)
    SparseReg{typeof(a), typeof(l), typeof(r)}(zeros(p), l, r, a, λ, f)
end
@generated function getarg(p, dt::Type, args...)
    i = findfirst(x -> x == dt, args)
    if i == 0
        return :(_default_arg(p, dt))
    else
        return args[i]
    end
end

function SparseReg(x::AMatF, y::AVecF, args...)
    o = SparseReg(size(x, 2), args...)
    fit!(o, Obs(x, y))
    o
end
function SparseReg(x::AMatF, y::AVecF, w::AVecF, args...)
    o = SparseReg(size(x, 2), args...)
    fit!(o, Obs(x, y, w))
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

function objective_value(o::SparseReg, obs::Obs{Ones}, ŷ::AVec)
    value(o.loss, obs.y, ŷ, AvgMode.Mean()) + value(o.penalty, o.β)
end
function objective_value(o::SparseReg, obs::Obs, ŷ::AVec)
    value(o.loss, obs.y, ŷ, AvgMode.WeightedMean(obs.w)) + value(o.penalty, o.β)
end

#-------------------------------------------------------------------------------# algorithms
include("algorithms/proxgrad.jl")
include("algorithms/sweep.jl")
# include("algorithms/sgdlike.jl")
# include("solutionpath.jl")

# Defaults for SparseReg
_default_arg(p::Integer, ::Type{Loss})       = LinearRegression()
_default_arg(p::Integer, ::Type{Penalty})    = NoPenalty()
_default_arg(p::Integer, ::Type{Algorithm})  = ProxGrad()
_default_arg(p::Integer, ::Type{Float64})    = 0.01
_default_arg(p::Integer, ::Type{VecF})       = ones(p)
end
