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

abstract type AbstractSparseReg end

abstract type Algorithm end
abstract type OfflineAlgorithm   <: Algorithm end
abstract type OnlineAlgorithm    <: Algorithm end

#-------------------------------------------------------------------------------# includes
include("obs.jl")
include("sparsereg.jl")
include("common.jl")

include("algorithms/proxgrad.jl")
# include("algorithms/sweep.jl")
# include("algorithms/sgdlike.jl")
# include("solutionpath.jl")

end
