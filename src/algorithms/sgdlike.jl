abstract type SGDLike <: OnlineAlgorithm end

function fit!{ALG <: SGDLike}(o::StreamReg{ALG}, obs::Obs)
    for i in eachindex(obs.y)
        OnlineStats.updatecounter!(o.weight)
        xi = @view obs.x[i, :]
        yi = obs.y[i]
        g = deriv(o.loss, yi, predict_from_xβ(o.loss, xi'o.β))
        γ = OnlineStats.weight(o.weight)
        for j in eachindex(o.β)
            updateβj!(o, j, γ, g, xi[j])
        end
    end
    o
end


#-----------------------------------------------------------------------------------# SGD
"""
Stochastic Gradient Descent
    SGD(wt::W, η = 1.0) where W <: Weight
"""
immutable SGD <: SGDLike end
function updateβj!(o::StreamReg, j::Integer, γ::Float64, g::Float64, xj::Float64)
    s = o.η * γ
    λ = o.λ * o.factor[j]
    o.β[j] -= s * (g * xj + λ * deriv(o.penalty, o.β[j]))
end

#------------------------------------------------------------------------------# MOMENTUM
# "SGD with MOMENTUM"
# immutable MOMENTUM <: SGDLike
#     α::Float64
#     H::VecF
# end
# MOMENTUM(p::Integer, α = .1) = MOMENTUM(α, zeros(p))
# init(a::MOMENTUM, n, p) = MOMENTUM(a.weight, a.η, a.α, zeros(p))
# function updateβj(A::MOMENTUM, γ, ηγ, gx, βj, P, j, s)
#     @inbounds A.H[j] = OnlineStats.smooth(A.H[j], gx, A.α)
#     prox(P, βj - ηγ * A.H[j], ηγ * s)
# end
#
# #---------------------------------------------------------------------------------# FOBOS
# "Proximal Stochastic Gradient Descent"
# immutable FOBOS{W <: Weight} <: SGDLike
#     weight::W
#     η::Float64
# end
# FOBOS(wt::Weight = LearningRate(), η::Number = 1.0) = FOBOS(wt, η)
# init(alg::FOBOS, n, p) = alg
# updateβj(A::FOBOS, γ, ηγ, gx, βj, P, j, s) = prox(P, βj - ηγ * gx, ηγ * s)
#
# #-------------------------------------------------------------------------------# ADAGRAD
# "ADAGRAD"
# type ADAGRAD{W <: Weight} <: SGDLike
#     weight::W
#     η::Float64
#     H::VecF
# end
# ADAGRAD(wt::Weight = LearningRate(), η::Number = 1.0) = ADAGRAD(wt, η, zeros(0))
# init(a::ADAGRAD, n, p) = (a.H = zeros(p); a)
# function updateβj(A::ADAGRAD, γ, ηγ, gx, βj, P, j, s)
#     @inbounds A.H[j] = OnlineStats.smooth(A.H[j], gx * gx, 1 / A.weight.nups)
#     @inbounds step = ηγ / (sqrt(A.H[j]) + ϵ)
#     prox(P, βj - step * gx, step * s)
# end
#
# #----------------------------------------------------------------------------------# ADAM
# "ADAM"
# type ADAM{W <: Weight} <: SGDLike
#     weight::W
#     η::Float64
#     m1::Float64
#     m2::Float64
#     H::VecF
#     G::VecF
# end
# function ADAM(wt::Weight = LearningRate(), η::Float64 = 1.0, m1 = .1, m2 = .1)
#     ADAM(wt, η, m1, m2, zeros(0), zeros(0))
# end
# init(a::ADAM, n, p) = ADAM(a.weight, a.η, a.m1, a.m2, zeros(p), zeros(p))
# @inline function updateβj(A::ADAM, γ, ηγ, gx, βj, P, j, s)
#     m1 = A.m1
#     m2 = A.m2
#     nups = A.weight.nups
#     ratio = sqrt(1.0 - m2 ^ nups) / (1.0 - m1 ^ nups)  # this line faster in OnlineStatsModels?
#     A.H[j] = smooth(A.H[j], gx, m1)
#     A.G[j] = smooth(A.G[j], gx * gx, m2)
#     step = ηγ * ratio / (sqrt(A.G[j]) + ϵ)
#     prox(P, βj - step * A.H[j], step * s)
# end
