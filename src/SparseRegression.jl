module SparseRegression

import SweepOperator
using LearnBase
importall LearnBase
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

abstract type AbstractSparseReg end

abstract type Algorithm end
abstract type OfflineAlgorithm   <: Algorithm end
abstract type OnlineAlgorithm    <: Algorithm end

default_color = :light_cyan

#-------------------------------------------------------------------------------# includes
include("obs.jl")
include("sparsereg.jl")
include("common.jl")

include("algorithms/proxgrad.jl")
include("algorithms/sweep.jl")
# include("algorithms/sgdlike.jl")
# include("solutionpath.jl")

#-------------------------------------------------------------------------------# fit
default_algorithm{L, P}(o::SparseReg{L, P}, obs; kw...) = ProxGrad(obs; kw...)
default_algorithm(o::SparseReg{LinearRegression, NoPenalty}, obs; kw...) = Sweep(obs)
default_algorithm(o::SparseReg{LinearRegression, L2Penalty}, obs; kw...) = Sweep(obs)

function fitmodel(x::AMat, y::AVec, args...; kw...)
    n, p = size(x)
    o = SparseReg(p, args...)
    alg = default_algorithm(o, Obs(x, y); kw...)
    fit!(o, alg)
    FittedModel(o, alg)
end
function fitmodel(x::AMat, y::AVec, w::AVec, args...; kw...)
    n, p = size(x)
    o = SparseReg(p, args...)
    alg = default_algorithm(o, Obs(x, y, w); kw...)
    fit!(o, alg)
    FittedModel(o, alg)
end

end #module
