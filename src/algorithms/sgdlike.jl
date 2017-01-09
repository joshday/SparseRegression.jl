abstract SGDLike <: OnlineAlgorithm

function fit!{ALG <: SGDLike}(o::SparseReg{ALG}, x::AMat, y::AVec)
    η = o.algorithm.η
    w = o.algorithm.weight
    A, L, β, P = o.algorithm, o.loss, o.β, o.penalty
    n, p = size(x)
    @assert n == length(y)
    for i in eachindex(y)
        OnlineStats.updatecounter!(w)
        _fit!(o, @view(x[i, :]), y[i], OnlineStats.weight(w), A, L, β, P)
    end
    o
end
function fit!{ALG <: SGDLike}(o::SparseReg{ALG}, x::AMat, y::AVec, b::Int)
    η = o.algorithm.η
    w = o.algorithm.weight
    A, L, β, P = o.algorithm, o.loss, o.β, o.penalty
    n, p = size(x)
    i = 1
    while i <= n
        rng = i:min(i + b - 1, n)
        bsize = length(rng)
        OnlineStats.updatecounter!(w, bsize)
        xi = @view x[rng, :]
        yi = @view y[rng]
        _fitbatch!(o, xi, yi, OnlineStats.weight(w, bsize), A, L, β, P)
        i += b
    end
    o
end


# Singleton updater
function _fit!{ALG <: SGDLike}(o::SparseReg{ALG}, x::AVec, y::Real, γ, A, L, β, P)
    ηγ = γ * A.η
    g = deriv(o.loss, y, _predict(L, dot(x, β)))
    for j in eachindex(β)
        β[j] = updateβj(A, γ, ηγ, g * x[j], β[j], P, j, o.penfact[j])
    end
end

# minibatch updater
function _fitbatch!{ALG <: SGDLike}(o::SparseReg{ALG}, x::AMat, y::AVec, γ, A, L, β, P)
    ηγ = γ * A.η
    g = deriv(o.loss, y, xβ(o, x))
    @inbounds for j in eachindex(β)
        gx = mean(g .* x[:, j])
        β[j] = updateβj(A, γ, ηγ, gx, β[j], P, j, o.penfact[j])
    end
end


#-----------------------------------------------------------------------------------------# SGD
"Stochastic Gradient Descent"
immutable SGD{W <: Weight} <: SGDLike
    weight::W
    η::Float64
end
SGD(wt::Weight = LearningRate(), η::Number = 1.0) = SGD(wt, η)
init(alg::SGD, n, p) = alg
updateβj(A::SGD, γ, ηγ, gx, βj, P, j, s) = βj - ηγ * (gx + s * deriv(P, βj))

#------------------------------------------------------------------------------------# MOMENTUM
"SGD with MOMENTUM"
immutable MOMENTUM{W <: Weight} <: SGDLike
    weight::W
    η::Float64
    α::Float64
    H::VecF
end
MOMENTUM(wt::Weight = LearningRate(), η::Number = 1.0, α = .1) = MOMENTUM(wt, η, α, zeros(0))
init(a::MOMENTUM, n, p) = MOMENTUM(a.weight, a.η, a.α, zeros(p))
function updateβj(A::MOMENTUM, γ, ηγ, gx, βj, P, j, s)
    @inbounds A.H[j] = OnlineStats.smooth(A.H[j], gx, A.α)
    prox(P, βj - ηγ * A.H[j], ηγ * s)
end

#---------------------------------------------------------------------------------------# FOBOS
"Proximal Stochastic Gradient Descent"
immutable FOBOS{W <: Weight} <: SGDLike
    weight::W
    η::Float64
end
FOBOS(wt::Weight = LearningRate(), η::Number = 1.0) = FOBOS(wt, η)
init(alg::FOBOS, n, p) = alg
updateβj(A::FOBOS, γ, ηγ, gx, βj, P, j, s) = prox(P, βj - ηγ * gx, ηγ * s)

#-------------------------------------------------------------------------------------# ADAGRAD
"ADAGRAD"
type ADAGRAD{W <: Weight} <: SGDLike
    weight::W
    η::Float64
    H::VecF
end
ADAGRAD(wt::Weight = LearningRate(), η::Number = 1.0) = ADAGRAD(wt, η, zeros(0))
init(a::ADAGRAD, n, p) = (a.H = zeros(p); a)
function updateβj(A::ADAGRAD, γ, ηγ, gx, βj, P, j, s)
    @inbounds A.H[j] = OnlineStats.smooth(A.H[j], gx * gx, 1 / A.weight.nups)
    @inbounds step = ηγ / (sqrt(A.H[j]) + ϵ)
    prox(P, βj - step * gx, step * s)
end
