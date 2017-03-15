abstract SGDLike <: OnlineAlgorithm
makebuffer{A <: SGDLike}(o::SparseReg{A}) = nothing

function fit!{ALG <: SGDLike}(o::SparseReg{ALG}, x::AMat, y::AVec)
    η = o.algorithm.η
    w = o.algorithm.weight
    A = o.algorithm
    β = o.β
    P = o.penalty
    n, p = size(x)
    @assert n == length(y)
    @inbounds for i in eachindex(y)
        OnlineStats.updatecounter!(w)
        xi = view(x, i, :)
        yi = y[i]
        g = deriv(o.loss, yi, _predict(o.loss, dot(xi, β)))
        γ = OnlineStats.weight(w)
        for j in eachindex(β)
            β[j] = updateβj(A, γ, γ * η, g * xi[j], β[j], P, j, o.penaltyfactor[j])
        end
    end
    o
end
# function fit!{ALG <: SGDLike}(o::SparseReg{ALG}, x::AMat, y::AVec, b::Int)
#     η = o.algorithm.η
#     w = o.algorithm.weight
#     A = o.algorithm
#     L = o.loss
#     β = o.β
#     P = o.penalty
#     n, p = size(x)
#     @assert n == length(y)
#     i = 1
#     while i <= n
#         rng = i:min(i + b - 1, n)
#         bsize = length(rng)
#         OnlineStats.updatecounter!(w, bsize)
#         xi = @view x[rng, :]
#         yi = @view y[rng]
#         _fitbatch!(o, xi, yi, OnlineStats.weight(w, bsize), A, L, β, P)
#         i += b
#     end
#     o
# end


# Singleton updater
# function _fit!{ALG <: SGDLike}(o::SparseReg{ALG}, x::AVec, y::Real, γ, η, A, L, β, P)
#     g = deriv(o.loss, y, _predict(L, dot(x, β)))
#     for j in eachindex(β)
#         β[j] = updateβj(A, γ, γ * η, g * x[j], β[j], P, j, o.penaltyfactor[j])
#     end
# end

# minibatch updater
# function _fitbatch!{ALG <: SGDLike}(o::SparseReg{ALG}, x::AMat, y::AVec, γ, A, L, β, P)
#     ηγ = γ * A.η
#     g = deriv(o.loss, y, xβ(o, x))
#     @inbounds for j in eachindex(β)
#         gx = mean(g .* x[:, j])
#         β[j] = updateβj(A, γ, ηγ, gx, β[j], P, j, o.penaltyfactor[j])
#     end
# end


#-----------------------------------------------------------------------------------# SGD
"Stochastic Gradient Descent"
immutable SGD{W <: Weight} <: SGDLike
    weight::W
    η::Float64
end
SGD(wt::Weight = LearningRate(), η::Number = 1.0) = SGD(wt, η)


init(alg::SGD, n, p) = alg
updateβj(A::SGD, γ, ηγ, gx, βj, P, j, s) = βj - ηγ * (gx + s * deriv(P, βj))
#
# #------------------------------------------------------------------------------# MOMENTUM
# "SGD with MOMENTUM"
# immutable MOMENTUM{W <: Weight} <: SGDLike
#     weight::W
#     η::Float64
#     α::Float64
#     H::VecF
# end
# MOMENTUM(wt::Weight = LearningRate(), η::Number = 1.0, α = .1) = MOMENTUM(wt, η, α, zeros(0))
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
