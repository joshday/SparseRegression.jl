module SparseRegression

using Reexport
@reexport using LearnBase
@reexport using LossFunctions
@reexport using PenaltyFunctions
@reexport using OnlineStats
importall LossFunctions
import SweepOperator
import StatsBase: predict, coef

export
    SparseReg, SolutionPath, predict, coef,
    # algorithms
    PROXGRAD,
    SGD, MOMENTUM, FOBOS, ADAGRAD,
    # Model typealiases
    LinearRegression, L1Regression, LogisticRegression, PoissonRegression, HuberRegression,
    SVMLike, DWDLike, QuantileRegression


#-----------------------------------------------------------------------------# types
typealias AVec AbstractVector
typealias AMat AbstractMatrix
typealias AVecF AbstractVector{Float64}
typealias AMatF AbstractMatrix{Float64}
typealias VecF Vector{Float64}
typealias MatF Matrix{Float64}

typealias AverageMode LossFunctions.AverageMode

abstract Algorithm
abstract OfflineAlgorithm   <: Algorithm
abstract OnlineAlgorithm    <: Algorithm
function Base.print(io::IO, a::Algorithm)
    print(io, replace(string(typeof(a)), "SparseRegression.", ""))
end

const ϵ = 1e-8  # constant to avoid dividing by zero, etc.

#----------------------------------------------------------------------------------# typealias
typealias LinearRegression      LossFunctions.ScaledDistanceLoss{L2DistLoss,0.5}
typealias L1Regression          L1DistLoss
typealias LogisticRegression    LogitMarginLoss
typealias PoissonRegression     PoissonLoss
typealias HuberRegression       HuberLoss
typealias SVMLike               L1HingeLoss
typealias QuantileRegression    QuantileLoss
typealias DWDLike               DWDMarginLoss

#----------------------------------------------------------------------------------# SparseReg
type SparseReg{A <: Algorithm, L <: Loss, P <: Penalty, M <: AverageMode}
    β::VecF
    loss::L
    penalty::P
    algorithm::A
    avg::M
    penaltyfactor::VecF
end
function SparseReg(p::Integer, n::Integer = 0;
                   loss::Loss = LinearRegression(),
                   penalty::Penalty = NoPenalty(),
                   algorithm::Algorithm = PROXGRAD(),
                   avgmode::AverageMode = AvgMode.Mean(),
                   penaltyfactor::VecF = ones(p)
                   )
    @assert length(penaltyfactor) == p
    SparseReg(zeros(p), loss, penalty, init(algorithm, n, p), avgmode, penaltyfactor)
end
function SparseReg(x::AMat, y::AVec;
                   loss::Loss = LinearRegression(),
                   penalty::Penalty = NoPenalty(),
                   algorithm::Algorithm = PROXGRAD(),
                   avgmode::AverageMode = AvgMode.Mean(),
                   penaltyfactor::VecF = ones(size(x, 2))
                   )
    o = SparseReg(size(x, 2), size(x, 1); loss=loss, penalty=penalty, algorithm=algorithm,
                  avgmode = avgmode, penaltyfactor = penaltyfactor)
    fit!(o, x, y)
end
function print_item(io::IO, name::AbstractString, value)
    print(io, "  >" * @sprintf("%18s", name * ":  "))
    println(io, value)
end
function Base.show(io::IO, o::SparseReg)
    println(io, "Sparse Regression Model")
    print_item(io, "β", o.β)
    print_item(io, "Loss", o.loss)
    print_item(io, "Penalty", o.penalty)
    print_item(io, "Algorithm", o.algorithm)
end

coef(o::SparseReg) = o.β

logistic(x::Float64) = 1.0 / (1.0 + exp(-x))
xβ(o::SparseReg, x::AMat) = x * o.β
xβ(o::SparseReg, x::AVec) = dot(x, o.β)

predict(o::SparseReg, x::AVec) = _predict(o.loss, xβ(o, x))
predict(o::SparseReg, x::AMat) = _predict.(o.loss, xβ(o, x))

_predict(l::Loss, xβ::Real) = xβ
_predict(l::LogitMarginLoss, xβ::Real) = logistic(xβ)
_predict(l::PoissonLoss, xβ::Real) = exp(xβ)
function _predict!(l::Loss, xβ::AVec)
    for i in eachindex(xβ)
        @inbounds xβ[i] = _predict(l, xβ[i])
    end
    xβ
end

penaltyfactor!(o::SparseReg, v::VecF) = (@assert length(v) == length(o.β); o.penfact[:] = v)

isoffline(o::SparseReg) = typeof(o.algorithm) <: OfflineAlgorithm

smooth(a, b, γ) = a + γ * (b - a)



#------------------------------------------------------------------------# Algorithms
include("algorithms/proxgrad.jl")
include("algorithms/sgdlike.jl")

include("solutionpath.jl")

end
