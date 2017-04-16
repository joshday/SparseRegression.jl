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
for pkg in [:LearnBase, :LossFunctions, :PenaltyFunctions]
    eval(Expr(:toplevel, Expr(:export, setdiff(names(eval(pkg)), [pkg])...)))
end

export
    SparseReg, ProxGrad,
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

#---------------------------------------------------------------------------# random helpers
const ϵ = 1e-5

#-------------------------------------------------------------------------------# includes
include("obs.jl")
include("printing.jl")
include("sparsereg.jl")
include("algorithms/proxgrad.jl")
# include("algorithms/fista.jl")
# include("algorithms/sweep.jl")
# include("algorithms/stochastic.jl")


#-------------------------------------------------------------------------------# fit
# for m in [:ProximalGradientModel, :SweepModel]
#     @eval fit(::Type{$(m)}, args...; kw...) = $(m)(Obs(args...); kw...)
# end






end #module
