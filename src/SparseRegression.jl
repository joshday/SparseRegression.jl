module SparseRegression

import SweepOperator
using LearnBase
importall LearnBase
import StatsBase: coef
using LossFunctions
using PenaltyFunctions
using OnlineStats

# Reexports
for pkg in [:LearnBase, :LossFunctions, :PenaltyFunctions, :OnlineStats]
    eval(Expr(:toplevel, Expr(:export, setdiff(names(eval(pkg)), [pkg])...)))
end

export
    SparseReg, Obs, SolutionPath, classify, fitmodel, fitpath, coef,
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

abstract type AbstractSparseReg end

abstract type Algorithm end
abstract type OfflineAlgorithm   <: Algorithm end
abstract type OnlineAlgorithm    <: Algorithm end

#-------------------------------------------------------------------------------# includes
include("obs.jl")
include("sparsereg.jl")
include("fittedmodel.jl")
include("common.jl")

include("algorithms/proxgrad.jl")
include("algorithms/sweep.jl")
include("solutionpath.jl")
# include("algorithms/sgdlike.jl")


#-------------------------------------------------------------------------------# fit
default_algorithm{L, P}(o::SparseReg{L, P}, obs; kw...) = ProxGrad(size(obs.x)...; kw...)
default_algorithm(o::SparseReg{LinearRegression, NoPenalty}, obs; kw...) = Sweep(obs)
default_algorithm(o::SparseReg{LinearRegression, L2Penalty}, obs; kw...) = Sweep(obs)



function fit(s::Type{SparseReg}, obs::Obs, args...; kw...)
    n, p = size(obs)
    o = s(p, args...)
    alg = ProxGrad(obs; kw...)
    fit!(o, alg, obs)
    FittedModel(o, alg, obs)
end

function fit{A <: Algorithm}(s::Type{SparseReg}, a::Type{A}, obs::Obs, args...; kw...)
    n, p = size(obs)
    o = s(p, args...)
    alg = a(obs; kw...)
    fit!(o, alg, obs)
    FittedModel(o, alg, obs)
end


# function fitmodel(x::AMat, y::AVec, args...; kw...)
#     n, p = size(x)
#     o = SparseReg(p, args...)
#     obs = Obs(x, y)
#     alg = default_algorithm(o, obs; kw...)
#     fit!(o, alg, obs)
#     FittedModel(o, alg, obs)
# end
# function fitmodel(x::AMat, y::AVec, w::AVec, args...; kw...)
#     n, p = size(x)
#     o = SparseReg(p, args...)
#     obs = Obs(x, y, w)
#     alg = default_algorithm(o, obs; kw...)
#     fit!(o, alg, obs)
#     FittedModel(o, alg, obs)
# end

end #module
