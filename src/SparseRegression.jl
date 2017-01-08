module SparseRegression

using Reexport
@reexport using LearnBase
@reexport using LossFunctions
@reexport using PenaltyFunctions
importall LossFunctions
import SweepOperator
import StatsBase: predict, coef

export
    SparseReg, SolutionPath, predict, coef,
    # algorithms
    PROXGRAD


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


#-------------------------------------------------------------------------# SparseReg
type SparseReg{A <: Algorithm, L <: Loss, P <: Penalty, M <: AverageMode}
    β::Vector{Float64}
    loss::L
    penalty::P
    algorithm::A
    avg::M
end
function _SparseReg(p::Integer, n::Integer, loss::Loss, pen::Penalty, alg::Algorithm,
                    avg::AverageMode)
    SparseReg(zeros(p), loss, pen, init(alg, n, p), avg)
end
function SparseReg(p::Integer, n::Integer = 0, args...)
    loss = scaledloss(L2DistLoss(), .5)
    pen = NoPenalty()
    alg = PROXGRAD()
    avg  = AvgMode.Mean()
    for arg in args
        T = typeof(arg)
        if T <: Loss
            loss = arg
        elseif T <: Penalty
            pen = arg
        elseif T <: Algorithm
            alg = arg
        elseif T <: AverageMode
            avg = arg
        else
            warn("At least one unused argument!!!")
        end
    end
    _SparseReg(p, n, loss, pen, alg, avg)
end
function SparseReg(x::AMat, y::AVec, args...)
    o = SparseReg(size(x, 2), size(x, 1), args...)
    fit!(o, x, y)
end
function SparseReg(x::AMat, y::AVec, w::AVec, args...)
    o = SparseReg(size(x, 2), size(x, 1), AvgMode.WeightedMean(w), args...)
    fit!(o, x, y)
end
default_penalty_factor(p::Integer) = (v = ones(p); v[end] = 0.0; v)
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

#------------------------------------------------------------------------# Algorithms
include("algorithms/proxgrad.jl")

include("solutionpath.jl")

end
