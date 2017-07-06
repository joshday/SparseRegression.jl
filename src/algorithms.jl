#============================================================================= notes
Here is what each Algorithm will end up using:

function learn!(model, meta::MetaLearner, data)
    pre_hook(meta, model)
    for (i, item) in enumerate(data)
        for mgr in meta.managers
            learn!(model, mgr, item)
        end
        iter_hook(meta, model, i)
        finished(meta, model, i) && break
    end
    post_hook(meta, model)
end

where:
    - data is Void.
    - The actual data (Obs) are held by the algorithm and are part of the MetaLearner


An Algorithm should only implement:

- pre_hook(alg, model)
    - Only if the algorithm needs some setup (copying data into a buffer, etc.)

- update!(model, alg, item::Void)
    - For example, a single gradient step.  The algorithm contain Obs, so the
      `item` is nothing.
==============================================================================#

#-----------------------------------------------------------------------# ProxGrad
"""
    ProxGrad(obs, step)
Proximal Gradient algorithm.  Works for any loss and convex penalty.
"""
struct ProxGrad{O <: Obs} <: Algorithm
    obs::O
    step::Float64
    derivs::Vector{Float64}
    ∇::Vector{Float64}
end
function ProxGrad(obs::Obs, step::Float64 = 1.0)
    n, p = size(obs)
    ProxGrad(obs, step, zeros(n), zeros(p))
end
function update!(o::SModel, a::ProxGrad, item::Void)
    gradient!(a.derivs, a.∇, o.β, o.loss, a.obs)
    s = a.step
    for j in eachindex(o.β)
        @inbounds o.β[j] = prox(o.penalty, o.β[j] - s * a.∇[j], s * o.λfactor[j])
    end
    o
end

#-----------------------------------------------------------------------# Fista
"""
    Fista(obs, step)
Accelerated Proximal Gradient.  Works for any loss and convex penalty.
"""
mutable struct Fista{O <: Obs} <: Algorithm
    obs::O
    step::Float64
    derivs::Vector{Float64}
    ∇::Vector{Float64}
    β1::Vector{Float64}
    β2::Vector{Float64}
    t::Int
end
function Fista(obs::Obs, step::Float64 = 1.0)
    n, p = size(obs)
    Fista(obs, step, zeros(n), zeros(p), zeros(p), zeros(p), 0)
end
pre_hook(a::Fista, o::SModel) = (a.t = 0)
function update!(o::SModel, a::Fista, item::Void)
    copy!(a.β2, a.β1)
    copy!(a.β1, o.β)
    a.t += 1
    if a.t > 2
        γ = (a.t - 2) / (a.t + 1)
        for j in eachindex(o.β)
            @inbounds o.β[j] = a.β1[j] + γ * (a.β1[j] - a.β2[j])
        end
    end
    gradient!(a.derivs, a.∇, o.β, o.loss, a.obs)
    s = a.step
    for j in eachindex(o.β)
        @inbounds o.β[j] = prox(o.penalty, o.β[j] - s * a.∇[j], s * o.λfactor[j])
    end
end


#-----------------------------------------------------------------------# GradientDescent
"""
    GradientDescent(obs, step)
Gradient Descent.  Works for any loss and any penalty.
"""
struct GradientDescent{O <: Obs} <: Algorithm
    obs::O
    step::Float64
    derivs::Vector{Float64}
    ∇::Vector{Float64}
end
function GradientDescent(obs::Obs, step::Float64 = 1.0)
    n, p = size(obs)
    GradientDescent(obs, step, zeros(n), zeros(p))
end
function update!(o::SModel, a::GradientDescent, item::Void)
    gradient!(a.derivs, a.∇, o.β, o.loss, a.obs)
    s = a.step
    for j in eachindex(o.β)
        @inbounds o.β[j] -= s * (a.∇[j] + o.λfactor[j] * deriv(o.penalty, o.β[j]))
    end
    o
end


# For algorithms that only need a single iteration to solve
abstract type OneIterAlgorithm <: Algorithm end
finished(a::OneIterAlgorithm, o::SModel, i::Void) = true
#-----------------------------------------------------------------------# Sweep
"""
    Sweep(obs)
Linear/ridge regression via sweep operator.  Works for LinearRegression/L2DistLoss
with NoPenalty or L2Penalty.
"""
struct Sweep{O <: Obs} <: OneIterAlgorithm
    obs::O
    A::Matrix{Float64}  # [x y]'[x y]
    S::Matrix{Float64}  # sweep!(A, 1:p)
end
Sweep(obs::Obs) = (A = make_A(obs); Sweep(obs, make_A(obs), zeros(A)))

function make_A(obs::Obs)
    n, p = size(obs.x)
    a = zeros(p + 1, p + 1)
    b = zeros(p + 1, p + 1)
    BLAS.syrk!('U', 'T', 1 / n, Diagonal(sqrt.(obs.w)) * obs.x, 0.0, view(a, 1:p, 1:p)) # x'wx
    BLAS.gemv!('T', 1 / n, Diagonal(obs.w) * obs.x, obs.y, 0.0, @view(a[1:p, end]))     # x'wy
    a[end, end] = dot(obs.y, Diagonal(obs.w) * obs.y) / n                               # y'wy
    a
end
function make_A(obs::Obs{Void})
    n, p = size(obs.x)
    a = zeros(p + 1, p + 1)
    BLAS.syrk!('U', 'T', 1 / n, obs.x, 0.0, view(a, 1:p, 1:p))      # x'x
    BLAS.gemv!('T', 1 / n, obs.x, obs.y, 0.0, @view(a[1:p, end]))   # x'y
    a[end, end] = dot(obs.y, obs.y) / n                             # y'y
    a
end
function update!(o::SModel, a::Sweep, item::Void)
    n, p = size(a.obs)
    copy!(a.S, a.A)
    add_ridge!(o, a, o.λfactor)
    sweep!(a.S, 1:p)
    for j in eachindex(o.β)
        @inbounds o.β[j] = a.S[j, end]
    end
end
function add_ridge!{L}(o::SModel{L, NoPenalty}, a::Sweep, λf::Vector{Float64}) end
function add_ridge!{L}(o::SModel{L, L2Penalty}, a::Sweep, λf::Vector{Float64})
    for i in eachindex(o.β)
        @inbounds a.S[i, i] += λf[i]
    end
end


#-----------------------------------------------------------------------# LinRegCholesky
"""
    LinRegCholesky(obs)
Linear/ridge regression via cholesky decomposition.  Works for LinearRegression/L2DistLoss
with NoPenalty or L2Penalty.
"""
struct LinRegCholesky{O <: Obs} <: OneIterAlgorithm
    obs::O
    A::Matrix{Float64}
    S::Matrix{Float64}
end
LinRegCholesky(obs::Obs) = (A = make_A(obs); LinRegCholesky(obs, make_A(obs), zeros(A)))

function update!(o::SModel, a::LinRegCholesky, item::Void)
    copy!(a.S, a.A)
    cholfact!(Symmetric(a.S))
    o.β[:] = UpperTriangular(@view(a.S[1:end-1, 1:end-1])) \ @view(a.S[1:end-1, end])
end
function add_ridge!{L}(o::SModel{L, NoPenalty}, a::LinRegCholesky, λf::Vector{Float64}) end
function add_ridge!{L}(o::SModel{L, L2Penalty}, a::LinRegCholesky, λf::Vector{Float64})
    for i in eachindex(o.β)
        @inbounds a.S[i, i] += λf[i]
    end
end
