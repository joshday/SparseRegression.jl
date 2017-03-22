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
    SparseReg, Obs, SolutionPath, classify, coef, fitmodel,
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
include("printing.jl")

include("algorithms/proxgrad.jl")
# include("algorithms/sweep.jl")
# include("solutionpath.jl")

#----------------------------------------------------------------------# FittedModel
immutable FittedModel{S <: SparseReg, A <: Algorithm}
    model::S
    algorithm::A
end
function Base.show(io::IO, o::FittedModel)
    print_with_color(:light_cyan, io, "■■■■ FittedModel\n")
    show(io, o.model); println(io)
    show(io, o.algorithm)
end

function fitmodel(A::Algorithm, args...)
    n, p = size(A.obs.x)
    o = SparseReg(p, args...)
    fit!(o, A)
    FittedModel(o, A)
end

function fitpath(A::Algorithm, args...)
    n, p = size(A.obs.x)
    init = SparseReg(p, args...)

end


end #module
