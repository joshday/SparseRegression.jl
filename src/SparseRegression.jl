module SparseRegression

using Reexport
@reexport using OnlineStats
@reexport using LearnBase
@reexport using LossFunctions
@reexport using PenaltyFunctions
import SweepOperator
import Sparklines
importall LearnBase
import LearnBase: Loss, deriv, meanvalue, value

export SparseReg

#-----------------------------------------------------------------------# typealiases
typealias AVec AbstractVector
typealias AMat AbstractMatrix
typealias AVecF AbstractVector{Float64}
typealias AMatF AbstractMatrix{Float64}

#------------------------------------------------------------------# available models
typealias L1Regression          L1DistLoss
typealias LogisticRegression    LogitMarginLoss
typealias PoissonRegression     PoissonLoss
typealias HuberRegression       HuberLoss
typealias SVMLike               L1HingeLoss
typealias QuantileRegression    QuantileLoss
LinearRegression() = ScaledLoss(L2DistLoss(), .5)

#------------------------------------------------------------------------# Algorithms
abstract Algorithm
abstract OfflineAlgorithm   <: Algorithm
abstract OnlineAlgorithm    <: Algorithm
Base.print(io::IO, a::Algorithm) = print(io, replace(string(typeof(a)), "SparseRegression.", ""))

immutable SWEEP <: OfflineAlgorithm
    S::Matrix{Float64}
end
SWEEP() = SWEEP(zeros(0, 0))
SWEEP(p::Integer) = SWEEP(zeros(p + 1, p + 1))

#---------------------------------------------------------------------# Is supported?
is_supported(loss::Loss, pen::Penalty, alg::Algorithm) = false
is_supported(loss::ScaledLoss, pen::Union{NoPenalty, L2Penalty}, alg::SWEEP) = true

#-------------------------------------------------------------------------# SparseReg
immutable SparseReg{A <: Algorithm, L <: Loss, P <: Penalty}
    β::Vector{Float64}
    loss::L
    penalty::P
    algorithm::A
    penalty_factor::Vector{Float64}
end
function _SparseReg(p::Integer, loss::Loss, pen::Penalty, alg::Algorithm, penalty_factor::Vector{Float64})
    @assert all(x -> (x>=0), penalty_factor)
    @assert length(penalty_factor) == p
    @assert is_supported(loss, pen, alg) "Unsupported combination of Loss, Penalty, and Algorithm"
    SparseReg(zeros(p), loss, pen, typeof(alg)(p), penalty_factor)
end
function SparseReg(p::Integer, args...)
    loss = LinearRegression()
    pen = NoPenalty()
    penalty_factor = default_penalty_factor(p)
    alg = SWEEP()
    for arg in args
        T = typeof(arg)
        if T <: Loss
            loss = arg
        elseif T <: Penalty
            pen = arg
        elseif T <: Algorithm
            alg = arg
        elseif T == VecF
            penalty_factor = arg
        end
    end
    _SparseReg(p, loss, pen, alg, penalty_factor)
end
default_penalty_factor(p::Integer) = (v = ones(p); v[end] = 0.0; v)
function print_item(io::IO, name::AbstractString, value)
    print(io, "  >" * @sprintf("%18s", name * ":  "))
    println(io, value)
end
function Base.show(io::IO, o::SparseReg)
    println(io, "SparseReg Model")
    print_item(io, "β", o.β)
    print_item(io, "Loss", o.loss)
    print_item(io, "Penalty", o.penalty)
    print_item(io, "Algorithm", o.algorithm)
    print_item(io, "Penalty Factor", o.penalty_factor)
end



end
