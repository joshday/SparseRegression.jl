abstract type StochasticUpdater end
Base.show(io::IO, o::StochasticUpdater) = print(io, name(o))

immutable StochasticModel{
        U <: StochasticUpdater,
        L <: Loss,
        P <: Penalty,
        W <: Weight
    } <: AbstractSparseReg
    # AbstractSparseReg
    θ::Coefficients
    loss::L
    penalty::P
    factor::VecF
    # Weighting
    weight::W
    η::Float64
    updater::U
end
function StochasticModel(updater::StochasticUpdater;
        λ::VecF = defaultλ(),
        loss::Loss = defaultloss(),
        penalty::Penalty = defaultpenalty(),
        factor::VecF = ones(updater.p),
        weight::Weight = LearningRate(),
        η::Float64 = 1.0
    )
    StochasticModel(Coefficients(updater.p, λ), loss, penalty, factor, weight, η, updater)
end
function Base.show(io::IO, o::StochasticModel)
    showmodel(io, o)
    header(io, "Learning Rate")
    print_item(io, "weight", o.weight)
    print_item(io, "η", o.η)
    print_item(io, "updater", o.updater)
end

# -----------------------------------------------------------------------------------# fit!
fit!(o::StochasticModel, args...) = fit!(o, Obs(args...))

function fit!(o::StochasticModel, obs::Obs)
    for (k, λ) in enumerate(o.θ.λ)
        β = @view o.θ.β[:, k]
        for i in eachindex(obs.y)
            OnlineStats.updatecounter!(o.weight)
            xi = @view obs.x[i, :]
            yi = obs.y[i]
            g = deriv(o.loss, yi, predict_from_xβ(o.loss, dot(xi, β)))
            g = weight_g(g, obs, i)
            γ = OnlineStats.weight(o.weight)
            for j in eachindex(β)
                updateβj!(o, γ, g, β, xi, yi, j, λ)
            end
        end
    end
    o
end
weight_g(g, obs::Obs{Ones}, i) = g
weight_g(g, obs::Obs, i) = g * obs.w[i]



immutable SGD <: StochasticUpdater p::Int end
function updateβj!(o::StochasticModel{SGD}, γ, g, β, xi, yi, j, λ)
    β[j] -= o.η * γ * (g * xi[j] + λ * o.factor[j] * deriv(o.penalty, β[j]))
end


#
# # -----------------------------------------------------------------------------------# SGD
# """
# Stochastic Gradient Descent
#     SGD(wt::W, η = 1.0) where W <: Weight
# """
# immutable SGD{W <: Weight} <: SGDLike
#     η::Float64
#     weight::W
# end
#
# function updateβj!(o, A::SGrad{SGD}, j, )
# end

# ------------------------------------------------------------------------------# Momentum
# "SGD with Momentum"
# immutable Momentum <: SGDLike
#     α::Float64
#     H::VecF
# end
# Momentum(p::Integer, α = .1) = Momentum(α, zeros(p))
# init(a::Momentum, n, p) = Momentum(a.weight, a.η, a.α, zeros(p))
# function updateβj(A::Momentum, γ, ηγ, gx, βj, P, j, s)
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
