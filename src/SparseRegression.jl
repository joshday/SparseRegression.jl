module SparseRegression

import SweepOperator
using LearnBase
importall LearnBase
import StatsBase: coef, fit, fit!
using LossFunctions
using PenaltyFunctions
using OnlineStats

# Reexports
for pkg in [:LearnBase, :LossFunctions, :PenaltyFunctions, :OnlineStats]
    eval(Expr(:toplevel, Expr(:export, setdiff(names(eval(pkg)), [pkg])...)))
end

export
    ProximalGradientModel, SweepModel, Obs, classify, coef,
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

#-----------------------------------------------------------------------# AbstractSparseReg
Base.show(io::IO, o::AbstractSparseReg) = showmodel(io, o)
function showmodel(io::IO, o::AbstractSparseReg)
    header(io, name(o))
    show(io, o.θ)
    header(io, "Model Specification")
    print_item(io, "Loss", o.loss)
    print_item(io, "Penalty", o.penalty)
    print_item(io, "λ factor", o.factor')
end
coef(o::AbstractSparseReg) = o.θ
coef(o::AbstractSparseReg, i) = @view o.θ.β[:, i]
nparams(o::AbstractSparseReg) = size(o.θ.β, 1)


#---------------------------------------------------------------------------# random helpers
defaultλ()          =  collect(linspace(0, 1, 10))
defaultloss()       = LinearRegression()
defaultpenalty()    = L2Penalty()

#-------------------------------------------------------------------------------# includes
include("obs_coefs.jl")
include("printing.jl")
include("algorithms/proxgrad.jl")
include("algorithms/sweep.jl")
include("algorithms/stochastic.jl")


#-------------------------------------------------------------------------------# fit
for m in [:ProximalGradientModel, :SweepModel]
    @eval fit(::Type{$(m)}, args...; kw...) = $(m)(Obs(args...); kw...)
end






end #module
