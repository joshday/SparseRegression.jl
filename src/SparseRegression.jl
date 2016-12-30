module SparseRegression

using Reexport
@reexport using OnlineStats
@reexport using LearnBase
@reexport using LossFunctions
@reexport using PenaltyFunctions
@reexport using StatsBase
import SweepOperator
importall LearnBase
import StatsBase: predict
export SparseReg

#-----------------------------------------------------------------------------# types
typealias AVec AbstractVector
typealias AMat AbstractMatrix
typealias AVecF AbstractVector{Float64}
typealias AMatF AbstractMatrix{Float64}

abstract Algorithm
abstract OfflineAlgorithm   <: Algorithm
abstract OnlineAlgorithm    <: Algorithm
Base.print(io::IO, a::Algorithm) = print(io, replace(string(typeof(a)), "SparseRegression.", ""))

#------------------------------------------------------------------# available models
typealias L1Regression          L1DistLoss
typealias LogisticRegression    LogitMarginLoss
typealias PoissonRegression     PoissonLoss
typealias HuberRegression       HuberLoss
typealias SVMLike               L1HingeLoss
typealias QuantileRegression    QuantileLoss

immutable LinearRegression <: DistanceLoss end
deriv(::LinearRegression, y::Number, yhat::Number) = 0.5 * deriv(L2DistLoss(), y, yhat)
value(::LinearRegression, y::Number, yhat::Number) = 0.5 * value(L2DistLoss(), y, yhat)


#-------------------------------------------------------------------------# SparseReg
type SparseReg{A <: Algorithm, L <: Loss, P <: Penalty}
    β::Vector{Float64}
    loss::L
    penalty::P
    algorithm::A
end
function _SparseReg(p::Integer, loss::Loss, pen::Penalty, alg::Algorithm)
    @assert is_supported(loss, pen, alg) "($loss, $pen, $alg) is unsupported"
    SparseReg(zeros(p), loss, pen, init(alg, p))
end
function SparseReg(p::Integer, args...)
    loss = LinearRegression()
    pen = NoPenalty()
    alg = PROXGRAD()
    for arg in args
        T = typeof(arg)
        if T <: Loss
            loss = arg
        elseif T <: Penalty
            pen = arg
        elseif T <: Algorithm
            alg = arg
        end
    end
    _SparseReg(p, loss, pen, alg)
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

logistic(x::Float64) = 1.0 / (1.0 + exp(-x))
predict(o::SparseReg, x::AMat) = x * o.β
predict(o::SparseReg, x::AVec) = dot(x, o.β)
predict{A<:Algorithm}(o::SparseReg{A, LogisticRegression}, x::AMat) = logistic.(x * o.β)
predict{A<:Algorithm}(o::SparseReg{A, LogisticRegression}, x::AVec) = logistic(dot(x, o.β))
predict{A<:Algorithm}(o::SparseReg{A, PoissonRegression}, x::AMat) = exp.(x * o.β)
predict{A<:Algorithm}(o::SparseReg{A, PoissonRegression}, x::AVec) = exp(dot(x, o.β))

#------------------------------------------------------------------------# Algorithms
include("algorithms/sweep.jl")
include("algorithms/proxgrad.jl")

#---------------------------------------------------------------------# Is supported?
is_supported(loss::Loss, pen::Penalty, alg::Algorithm) = false
end
