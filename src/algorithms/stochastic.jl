abstract type StochasticUpdater end
Base.show(io::IO, o::StochasticUpdater) = print(io, name(o))

mutable struct StochasticModel{
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

updatecounter!(o::StochasticModel, n2::Int = 1) = OnlineStats.updatecounter!(o.weight, n2)
weight(o::StochasticModel, n2::Int = 1) = OnlineStats.weight(o.weight, n2)

# -----------------------------------------------------------------------------------# fit!
fit!(o::StochasticModel, args...) = fit!(o, Obs(args...))

function fit!(o::StochasticModel, obs::Obs)
    for i in eachindex(obs.y)
        updatecounter!(o)
        xi = @view obs.x[i, :]
        yi = obs.y[i]
        At_mul_B!(o.xβ, o.θ.β, xi)
        update_g!(o, xi, yi)
        !isa(obs, Obs{Ones}) && scale!(o.g, obs.w[i])
        γ = weight(o)
        ηγ = o.η * γ
        for (k, λ) in enumerate(o.θ.λ)
            for j in 1:nparams(o)
                # updateβj!(o, γ, ηγ, xi, yi, j, k, λ * o.factor[j])
                gx = o.g[k] * xi[j]
                updateβj!(o, j, k, γ, ηγ, gx, λ * o.factor[j])
            end
        end
    end
    o
end

function update_g!(o::StochasticModel, xi, yi)
    for k in eachindex(o.g)
        o.g[k] = deriv(o.loss, yi, o.xβ[k])
    end
end

# -----------------------------------------------------------------------------------# SGD
"Stochastic Gradient Descent"
struct SGD <: StochasticUpdater end
init(o::SGD, p, d) = o
function updateβj!(o::StochasticModel{SGD}, j, k, γ, ηγ, gx, λj)
    o.θ.β[j, k] -= ηγ * (gx + λj * deriv(o.penalty, o.θ.β[j, k]))
end

#----------------------------------------------------------------------------------# Momentum
"Stochastic Gradient Descent with Momentum"
struct Momentum <: StochasticUpdater
    α::Float64
    H::MatF
end
Momentum(α = .1) = Momentum(α, zeros(0, 0))
init(o::Momentum, p, d) = Momentum(o.α, fill(ϵ, p, d))
function updateβj!(o::StochasticModel{Momentum}, j, k, γ, ηγ, gx, λj)
    U = o.updater
    @inbounds ∇ = gx + λj * deriv(o.penalty, o.θ.β[j, k])
    @inbounds U.H[j, k] = OnlineStats.smooth(U.H[j, k], ∇, U.α)
    @inbounds o.θ.β[j, k] -= ηγ * U.H[j, k]
end


#---------------------------------------------------------------------------------# SPGD
"Stochastic Proximal Gradient Descent"
struct SPGD <: StochasticUpdater end
init(o::SPGD, p, d) = o
function updateβj!(o::StochasticModel{SPGD}, j, k, γ, ηγ, gx, λj)
    @inbounds o.θ.β[j, k] = prox(o.penalty, o.θ.β[j,k] - ηγ * gx, ηγ * λj)
end

#-------------------------------------------------------------------------------# ADAGRAD
"Adaptive Gradient"
struct ADAGRAD <: StochasticUpdater
    H::MatF
end
ADAGRAD() = ADAGRAD(zeros(0, 0))
init(o::ADAGRAD, p, d) = ADAGRAD(fill(ϵ, p, d))
function updateβj!(o::StochasticModel{ADAGRAD}, j, k, γ, ηγ, gx, λj)
    U = o.updater
    @inbounds U.H[j, k] = OnlineStats.smooth(U.H[j, k], gx * gx, γ)
    @inbounds s = ηγ * inv(sqrt(U.H[j, k]) + ϵ)
    @inbounds o.θ.β[j, k] = prox(o.penalty, o.θ.β[j, k] - s * gx, s * λj)
end

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
