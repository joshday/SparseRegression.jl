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
    # Buffer
    xβ::VecF
    g::VecF
end
function StochasticModel(p::Integer, updater::StochasticUpdater = SGD();
        λ::VecF = defaultλ(),
        loss::Loss = defaultloss(),
        penalty::Penalty = defaultpenalty(),
        factor::VecF = ones(p),
        weight::Weight = LearningRate(),
        η::Float64 = 1.0)
    d = length(λ)
    c = Coefficients(p, λ)
    u = init(updater, p , d)
    o = StochasticModel(c, loss, penalty, factor, weight, η, u, zeros(d), zeros(d))
    init!(o)  # Inialize βs to something nonzero
    o
end
function StochasticModel(obs::Obs, updater = SGD(); kw...)
    o = StochasticModel(nparams(obs), updater; kw...)
    fit!(o, obs)
    o
end
function Base.show(io::IO, o::StochasticModel)
    showmodel(io, o)
    header(io, "Update Specification")
    print_item(io, "weight", o.weight)
    print_item(io, "η", o.η)
    print_item(io, "updater", o.updater)
end

function init!(o::StochasticModel)
    for i in eachindex(o.θ.β)
        @inbounds o.θ.β[i] = randn()
    end
end

# -----------------------------------------------------------------------------------# fit!
fit!(o::StochasticModel, args...) = fit!(o, Obs(args...))

function fit!(o::StochasticModel, obs::Obs)
    for i in eachindex(obs.y)
        OnlineStats.updatecounter!(o.weight)
        xi = @view obs.x[i, :]
        yi = obs.y[i]
        At_mul_B!(o.xβ, o.θ.β, xi)
        update_g!(o, xi, yi)
        !isa(obs, Obs{Ones}) && scale!(o.g, obs.w[i])
        γ = OnlineStats.weight(o.weight)
        ηγ = o.η * γ
        for (k, λ) in enumerate(o.θ.λ)
            for j in 1:nparams(o)
                updateβj!(o, γ, ηγ, xi, yi, j, k, λ)
            end
        end
    end
    o
end

function update_g!(o::StochasticModel, xi, yi)
    for k in eachindex(o.g)
        o.g[k] = deriv(o.loss, yi, predict_from_xβ(o.loss, o.xβ[k]))
    end
end

# -----------------------------------------------------------------------------------# SGD
"Stochastic Gradient Descent"
immutable SGD <: StochasticUpdater end
init(o::SGD, p, d) = o
function updateβj!(o::StochasticModel{SGD}, γ, ηγ, xi, yi, j, k, λ)
    λj = λ * o.factor[j]
    o.θ.β[j, k] -= ηγ * (o.g[k] * xi[j] + λj * deriv(o.penalty, o.θ.β[j, k]))
end

#----------------------------------------------------------------------------------# Momentum
"Stochastic Gradient Descent with Momentum"
immutable Momentum <: StochasticUpdater
    α::Float64
    H::MatF
end
Momentum(α = .1) = Momentum(α, zeros(0, 0))
init(o::Momentum, p, d) = Momentum(o.α, fill(ϵ, p, d))
function updateβj!(o::StochasticModel{Momentum}, γ, ηγ, xi, yi, j, k, λ)
    U = o.updater
    ∇ = o.g[k] * xi[j] + λ * o.factor[j] * deriv(o.penalty, o.θ.β[j, k])
    U.H[j, k] = OnlineStats.smooth(U.H[j, k], ∇, U.α)
    o.θ.β[j, k] -= ηγ * U.H[j, k]
end


#---------------------------------------------------------------------------------# FOBOS
"Stochastic Proximal Gradient"
immutable SPG <: StochasticUpdater end
init(o::SPG, p, d) = o
function updateβj!(o::StochasticModel{SPG}, γ, ηγ, xi, yi, j, k, λ)
    λj = λ * o.factor[j]
    gx = o.g[k] * xi[j]
    o.θ.β[j, k] = prox(o.penalty, o.θ.β[j,k] - ηγ * gx, ηγ * λj)
end

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
