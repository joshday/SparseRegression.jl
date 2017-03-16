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
Base.show(io::IO, a::Algorithm) = print(io, name(a))


#----------------------------------------------------------------------#  observations
# constant vector of ones (default observation weights)
immutable Ones <: AVecF n::Int end
Ones(y::AVec) = Ones(length(y))
Base.size(o::Ones) = (o.n, )
Base.getindex(o::Ones, i::Integer) = 1.
Base.getindex{I <: Integer}(o::Ones, rng::AVec{I}) = Ones(length(rng))


immutable Obs{W <: AVec, X <: AMat, Y <: AVec}
    w::W
    x::X
    y::Y
end
function Obs(x::AMat, y::AVec, w::AVec = Ones(y))
    n1 = size(x, 1)
    n2 = length(y)
    n3 = length(w)
    n1 == n2 == n3 || throw(DimensionMismatch("number of rows should match: $n1, $n2, $n3"))
    Obs(w, x, y)
end

include("offline.jl")
include("online.jl")
include("common.jl")

#-------------------------------------------------------------------------------# algorithms
include("algorithms/proxgrad.jl")
include("algorithms/sweep.jl")
include("algorithms/sgdlike.jl")
include("solutionpath.jl")

end
