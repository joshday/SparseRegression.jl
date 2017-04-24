module SparseRegression

import SweepOperator
importall LearnBase
importall StatsBase
importall LearningStrategies
using LearnBase
using LossFunctions
using PenaltyFunctions
using LearningStrategies

# Reexports
for pkg in [:LearnBase, :LossFunctions, :PenaltyFunctions, :LearningStrategies]
    eval(Expr(:toplevel, Expr(:export, setdiff(names(eval(pkg)), [pkg])...)))
end

export
    SparseReg, Obs, ProxGrad, Fista, Sweep, GradientDescent,
    # Model typealiases
    LinearRegression, L1Regression, LogisticRegression, PoissonRegression, HuberRegression, SVMLike, DWDLike, QuantileRegression,
    # functions
    coef, predict

#---------------------------------------------------------------------------------# types
const AVec  = AbstractVector
const AMat  = AbstractMatrix
const AVecF = AbstractVector{Float64}
const AMatF = AbstractMatrix{Float64}
const VecF  = Vector{Float64}
const MatF  = Matrix{Float64}

const LinearRegression      = LossFunctions.ScaledDistanceLoss{L2DistLoss,0.5}
const L1Regression          = L1DistLoss
const LogisticRegression    = LogitMarginLoss
const PoissonRegression     = PoissonLoss
const HuberRegression       = HuberLoss
const SVMLike               = L1HingeLoss
const QuantileRegression    = QuantileLoss
const DWDLike               = DWDMarginLoss

abstract type AlgorithmStrategy <: LearningStrategy end

const ϵ = 1e-5

#-------------------------------------------------------------------------------# includes
include("obs.jl")
include("printing.jl")
include("sparsereg.jl")
include("algorithms/gradientdescent.jl")
include("algorithms/proxgrad.jl")
include("algorithms/fista.jl")
include("algorithms/sweep.jl")



end #module
