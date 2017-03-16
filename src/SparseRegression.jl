module SparseRegression

import SweepOperator
import StatsBase: predict, coef
using LearnBase
using LossFunctions
using PenaltyFunctions
using OnlineStats

# Reexports
for pkg in [:LearnBase, :LossFunctions, :PenaltyFunctions, :OnlineStats]
    eval(Expr(:toplevel, Expr(:export, setdiff(names(eval(pkg)), [pkg])...)))
end


export
    SparseReg, SolutionPath, predict, classify, coef,
    # algorithms
    ProxGrad, Sweep, SGD,
    # Model typealiases
    LinearRegression, L1Regression, LogisticRegression, PoissonRegression, HuberRegression, SVMLike, DWDLike, QuantileRegression


#---------------------------------------------------------------------------------# types
const AVec        = AbstractVector
const AMat        = AbstractMatrix
const AVecF       = AbstractVector{Float64}
const AMatF       = AbstractMatrix{Float64}
const VecF        = Vector{Float64}
const MatF        = Matrix{Float64}

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

immutable Obs{W <: AVec, X <: AMat, Y <: AVec}
    w::W
    x::X
    y::Y
end
function Obs(x::AMat, y::AVec, w::AVec = Ones(y))
    n1 = size(x, 1)
    n2 = length(y)
    n3 = length(w)
    n1 == n2 == n3 || throw(DimensionMismatch("number of rows should match: $n1, $n2, $n3"))
    Obs(w, x, y)
end

#----------------------------------------------------------------------# SparseReg
immutable SparseReg{A <: Algorithm, L <: Loss, P <: Penalty}
    β::VecF
    loss::L
    penalty::P
    algorithm::A
    λ::Float64
    factor::VecF
end

_defaults(p::Integer) = LinearRegression(), NoPenalty(), ProxGrad(), 0.1, ones(p)

# Type-stable constructor with arbitrary order of arguments.
# There must be a better way to do this
# generated functions?
SparseReg(p::Integer) = SparseReg(zeros(p), _defaults(p)...)
function SparseReg(p::Integer, a)
    args = _defaults(p)
    args2 = _a(args..., a)
    SparseReg(zeros(p), args2...)
end
function SparseReg(p::Integer, a1, a2)
    args = _defaults(p)
    args2 = _a(args..., a1)
    args3 = _a(args2..., a2)
    SparseReg(zeros(p), args3...)
end
function SparseReg(p::Integer, a1, a2, a3)
    args = _defaults(p)
    args2 = _a(args..., a1)
    args3 = _a(args2..., a2)
    args4 = _a(args3..., a3)
    SparseReg(zeros(p), args4...)
end
function SparseReg(p::Integer, a1, a2, a3, a4)
    args = _defaults(p)
    args2 = _a(args..., a1)
    args3 = _a(args2..., a2)
    args4 = _a(args3..., a3)
    args5 = _a(args4..., a4)
    SparseReg(zeros(p), args5...)
end
function SparseReg(p::Integer, a1, a2, a3, a4, a5)
    args = _defaults(p)
    args2 = _a(args..., a1)
    args3 = _a(args2..., a2)
    args4 = _a(args3..., a3)
    args5 = _a(args4..., a4)
    args6 = _a(args5..., a5)
    SparseReg(zeros(p), args6...)
end

# "overwrite" one argument in a tuple based on type
_a(l::Loss,r::Penalty,a::Algorithm,λ::Float64,f::VecF,t::Loss)      = t,r,a,λ,f
_a(l::Loss,r::Penalty,a::Algorithm,λ::Float64,f::VecF,t::Penalty)   = l,t,a,λ,f
_a(l::Loss,r::Penalty,a::Algorithm,λ::Float64,f::VecF,t::Algorithm) = l,r,t,λ,f
_a(l::Loss,r::Penalty,a::Algorithm,λ::Float64,f::VecF,t::Float64)   = l,r,a,t,f
_a(l::Loss,r::Penalty,a::Algorithm,λ::Float64,f::VecF,t::VecF)      = l,r,a,λ,t


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
    typeof(o.penalty) != NoPenalty && print_item(io, "λ", o.λ)
    any(x -> x != 1.0, o.factor) && print_item(io, "λ scaling", o.factor)
    print_item(io, "Algorithm", o.algorithm)
end

coef(o::SparseReg) = o.β
logistic(x::Float64) = 1.0 / (1.0 + exp(-x))
xβ(o::SparseReg, x::AMat) = x * o.β
xβ(o::SparseReg, x::AVec) = dot(x, o.β)
predict(o::SparseReg, x::AVec) = predict_from_xβ(o.loss, xβ(o, x))
predict(o::SparseReg, x::AMat) = predict_from_xβ.(o.loss, xβ(o, x))
classify{A, L<:MarginLoss}(o::SparseReg{A,L}, x::AVec) = sign(xβ(o, x))
classify{A, L<:MarginLoss}(o::SparseReg{A,L}, x::AMat) = sign.(xβ(o, x))
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
include("algorithms/sgdlike.jl")
# include("solutionpath.jl")

end
