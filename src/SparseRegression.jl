module SparseRegression

using Reexport
@reexport using LearnBase
@reexport using LossFunctions
@reexport using PenaltyFunctions
importall LearnBase
import SweepOperator
import StatsBase: predict, coef

export SparseReg, predict, coef, PROXGRAD, SWEEP

#-----------------------------------------------------------------------------# types
typealias AVec AbstractVector
typealias AMat AbstractMatrix
typealias AVecF AbstractVector{Float64}
typealias AMatF AbstractMatrix{Float64}

abstract Algorithm
abstract OfflineAlgorithm   <: Algorithm
abstract OnlineAlgorithm    <: Algorithm
Base.print(io::IO, a::Algorithm) = print(io, replace(string(typeof(a)), "SparseRegression.", ""))




#-------------------------------------------------------------------------# SparseReg
type SparseReg{A <: Algorithm, L <: Loss, P <: Penalty}
    β::Vector{Float64}
    loss::L
    penalty::P
    algorithm::A
end
function _SparseReg(p::Integer, loss::Loss, pen::Penalty, alg::Algorithm)
    # is_supported(loss, pen, alg)
    SparseReg(zeros(p), loss, pen, init(alg, p))
end
function SparseReg(p::Integer, args...)
    loss = ScaledLoss(L2DistLoss(), .5)
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
function SparseReg(x::AMat, y::AVec, args...)
    o = SparseReg(size(x, 2), args...)
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
include("algorithms/sweep.jl")
include("algorithms/proxgrad.jl")

end
