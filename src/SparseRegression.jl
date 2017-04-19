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
    SparseReg, Obs, ProxGrad, Sweep,
    # Model typealiases
    LinearRegression, L1Regression, LogisticRegression, PoissonRegression, HuberRegression, SVMLike, DWDLike, QuantileRegression,
    # functions
    coef

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

include("obs.jl")
include("printing.jl")
include("sparsereg.jl")

#---------------------------------------------------------------------------# random helpers

# AlgorithmStrategy needs constructor with method: MyAlg(a::MyAlg, o::Obs)
abstract type AlgorithmStrategy <: LearningStrategy end

function fit!(o::SparseReg, a::AlgorithmStrategy, m::MaxIter = MaxIter(1), args...)
    a2 = typeof(a)(a, o.obs)
    ml = MetaLearner(a2, m, args...)
    learn!(o, ml)
    o
end

const ϵ = 1e-5

#-------------------------------------------------------------------------------# includes

include("algorithms/proxgrad.jl")
include("algorithms/sweep.jl")



end #module
