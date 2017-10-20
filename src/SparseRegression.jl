module SparseRegression

import SweepOperator: sweep!
import LearnBase: learn!, ObsDim, value, predict
import LearningStrategies: strategy, setup!, update!, finished, cleanup!
import StatsBase: coef, AbstractWeights, Weights

using LossFunctions, PenaltyFunctions, LearningStrategies, RecipesBase

# Reexports
for pkg in [:LossFunctions, :PenaltyFunctions, :LearningStrategies]
    eval(Expr(:toplevel, Expr(:export, setdiff(names(eval(pkg)), [pkg])...)))
end

export
    SModel, ProxGrad, Fista, GradientDescent, Sweep, LinRegCholesky,
    Weights,
    coef, predict

#-----------------------------------------------------------------------# Types
abstract type Algorithm <: LearningStrategy end
abstract type GradientAlgorithm <: Algorithm end
abstract type OneIterAlgorithm <: Algorithm end
finished(a::OneIterAlgorithm, model, i) = true

include("smodel.jl")
include("algorithms.jl")

#-----------------------------------------------------------------------# Auto learn!
function learn!(o::SModel; verbose::Bool = true)
    s = strategy(o)
    verbose ? learn!(o, Verbose(s)) : learn!(o, s)
    s
end

strategy(o::SModel) = strategy(ProxGrad(o), MaxIter(), Converged(coef))

const ScaledL2 = Union{L2DistLoss, LossFunctions.ScaledDistanceLoss{L2DistLoss}}
strategy(o::SModel{<:ScaledL2, <:Union{NoPenalty, L2Penalty}}) = Sweep(o)


#-----------------------------------------------------------------------# Fake data
function fakedata(::DistanceLoss, n, p; β = linspace(-1, 1, p))
    x = randn(n, p)
    y = x*β + randn(n)
    x, y, collect(β)
end

function fakedata(::MarginLoss, n, p; β = linspace(-1, 1, p))
    x = randn(n, p)
    y = Float64[2 * (1 / (1+exp(-η)) > rand()) - 1 for η in x * β]
    x, y, collect(β)
end

end
