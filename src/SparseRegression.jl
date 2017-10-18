module SparseRegression

import SweepOperator: sweep!
import LearnBase: learn!, ObsDim, value, predict
import LearningStrategies: strategy, setup!, update!, finished, cleanup!
import StatsBase: coef, AbstractWeights, Weights


using Reexport, RecipesBase
@reexport using LossFunctions, PenaltyFunctions, LearningStrategies

export
    SModel, ProxGrad, Fista, GradientDescent, Sweep, LinRegCholesky,
    Weights,
    coef, predict

#-----------------------------------------------------------------------# Types
abstract type Algorithm <: LearningStrategy end
abstract type GradientAlgorithm <: Algorithm end
abstract type OneIterAlgorithm <: Algorithm end
finished(a::OneIterAlgorithm, model, i) = true

include("smodel.jl")
include("algorithms.jl")

#-----------------------------------------------------------------------# Auto learn!
function learn!(o::SModel; verbose::Bool = true)
    s = strategy(o)
    verbose ? learn!(o, Verbose(s)) : learn!(o, s)
    s
end

strategy(o::SModel) = strategy(ProxGrad(o), MaxIter(), Converged(coef))

const ScaledL2 = Union{L2DistLoss, LossFunctions.ScaledDistanceLoss{L2DistLoss}}
strategy(o::SModel{<:ScaledL2, <:Union{NoPenalty, L2Penalty}}) = Sweep(o)

#
# #-----------------------------------------------------------------------# Obs
# struct Obs{W, X <: AbstractMatrix, Y <: AbstractArray, T <: Dimension}
#     x::X
#     y::Y
#     w::W
#     dim::T
# end
#
# Obs(x::AbstractArray, y::AbstractArray, dim::Dimension = Rows(), w = nothing) = Obs(x, y, w, dim)
#
# function Base.show(io::IO, o::Obs)
#     println(io, typeof(o))
#     println(io, "  > x:   ", summary(o.x))
#     println(io, "  > y:   ", summary(o.y))
#     println(io, "  > w:   ", summary(o.w))
#     print(  io, "  > dim: ", o.dim)
# end
#
# function nobs(o::Obs)
#     n = _nobs(o.x, o.dim)
#     _nobs(o.x, o.dim) == n ? n : throw(DimensionMismatch("x and y have different nobs."))
# end
# _nobs(x::AbstractVector, dim) = length(x)
# _nobs(x::AbstractMatrix, ::Rows) = size(x, 1)
# _nobs(x::AbstractMatrix, ::Cols) = size(x, 2)
#
# nparams{W,X,Y}(o::Obs{W,X,Y,Rows}) = size(o.x, 2)
# nparams{W,X,Y}(o::Obs{W,X,Y,Cols}) = size(o.x, 1)
#
# Base.start(o::Obs) = 1
# Base.next(o::Obs, i) = (o, i + 1)
# Base.done(o::Obs, i) = false
#
# #-----------------------------------------------------------------------# SModel
# """
#     SModel(p::Int, args...)
#
# Create a SparseRegression model of `p` coefficients.  Additional arguments can be given in any
# order (and is still type stable):
#
# | argument  | type              | default             |
# |-----------|-------------------|---------------------|
# | `λfactor` | `Vector{Float64}` | `fill(.1, p)`       |
# | `loss`    | `Loss`            | `.5 * L2DistLoss()` |
# | `penalty` | `Penalty`         | `L2Penalty()`       |
#
# # Example
#
#     SModel(10, L1Penalty(), vcat(0.0, ones(9)), LogitMarginLoss())
# """
# struct SModel{L <: Loss, P <: Penalty}
#     β::Vector{Float64}
#     λfactor::Vector{Float64}
#     loss::L
#     penalty::P
# end
#
# # hacks for type-stable arbitrary argument order
# d(p::Integer) = (fill(.1, p), .5 * L2DistLoss(), L2Penalty())
# a(argu::Vector{Float64}, t::Tuple)  = (argu, t[2], t[3])
# a(argu::Loss, t::Tuple)             = (t[1], argu, t[3])
# a(argu::Penalty, t::Tuple)          = (t[1], t[2], argu)
#
# SModel(p::Integer, t::Tuple)     = SModel(zeros(p), t...)
# SModel(p::Integer)               = SModel(p, d(p))
# SModel(p::Integer, a1)           = SModel(p, a(a1, d(p)))
# SModel(p::Integer, a1, a2)       = SModel(p, a(a2, a(a1, d(p))))
# SModel(p::Integer, a1, a2, a3)   = SModel(p, a(a3, a(a2, a(a1, d(p)))))
# SModel(obs::Obs, args...)        = SModel(nparams(obs), args...)
#
# function Base.show(io::IO, o::SModel)
#     println(io, typeof(o))
#     println(io, "  > β        : ", o.β')
#     println(io, "  > λ factor : ", o.λfactor')
#     println(io, "  > Loss     : ", o.loss)
#     print(io,   "  > Penalty  : ", o.penalty)
# end
#
# coef(o::SModel) = o.β
# factor(o::SModel) = o.λfactor
# loss(o::SModel) = o.loss
# penalty(o::SModel) = o.penalty
# value(o::SModel) = o.β
# predict(o::SModel, x::AbstractVector) = At_mul_B(x, o.β)
# predict(o::SModel, x::AbstractMatrix) = x * o.β
#
# #-----------------------------------------------------------------------# GradientBuffer
# struct GradientBuffer
#     nvec::Vector{Float64}
#     pvec::Vector{Float64}
# end
# GradientBuffer(o::Obs) = GradientBuffer(zeros(nobs(o)), zeros(nparams(o)))
#
# gradient!(o::GradientBuffer, m::SModel, obs::Obs) = gradient!(o.nvec, o.pvec, m.β, m.loss, obs)
#
# function gradient!(nvec::Vector, pvec::Vector, β::Vector, L::Loss, O::Obs)
#     A_mul_B!(nvec, O.x, β)            # nvec ← x * β
#     deriv!(nvec, L, O.y, nvec)        # nvec ← deriv(L, y, x * β)
#     multiply_by_weights!(nvec, O.w)
#     At_mul_B!(pvec, O.x, nvec)        # pvec ← x'nvec
# end
# multiply_by_weights!(nvec, w::Void) = scale!(nvec, 1 / length(nvec))
# function multiply_by_weights!(nvec, w)
#     wt = inv(length(nvec))
#     for i in eachindex(nvec)
#         @inbounds nvec[i] *= w[i] * wt
#     end
# end
#
# #-----------------------------------------------------------------------# Algorithms
# include("algorithms.jl")




end
