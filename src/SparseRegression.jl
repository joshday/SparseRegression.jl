module SparseRegression

import SweepOperator: sweep!
importall LearnBase, StatsBase, LearningStrategies
using LossFunctions, PenaltyFunctions, RecipesBase, OnlineStatsBase


# Reexports
for pkg in [:LearnBase, :LossFunctions, :PenaltyFunctions, :LearningStrategies]
    eval(Expr(:toplevel, Expr(:export, setdiff(names(eval(pkg)), [pkg])...)))
end

export
    SModel, Obs, Path,
    # algorithms
    ProxGrad, GradientDescent, Fista, Sweep, LinRegCholesky,
    # aliases
    LinearRegression, L1Regression, LogisticRegression, PoissonRegression, HuberRegression,
    SVMLike, QuantileRegression, DWDLike,
    # functions
    coef, predict, fitted, residuals


abstract type Algorithm <: LearningStrategy end

const LinearRegression      = LossFunctions.ScaledDistanceLoss{L2DistLoss,0.5}
const L1Regression          = L1DistLoss
const LogisticRegression    = LogitMarginLoss
const PoissonRegression     = PoissonLoss
const HuberRegression       = HuberLoss
const SVMLike               = L1HingeLoss
const QuantileRegression    = QuantileLoss
const DWDLike               = DWDMarginLoss

include("obs.jl")
include("model.jl")
include("algorithms.jl")




end
