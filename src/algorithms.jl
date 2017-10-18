#-----------------------------------------------------------------------# GradientAlgorithm
function gradient!(a::GradientAlgorithm, o::SModel)
    A_mul_B!(a.nvec, o.x, o.β)          # nvec ← x * β
    deriv!(a.nvec, o.loss, o.y, a.nvec) # nvec ← deriv(L, y, x * β)
    multiply_by_weights!(a.nvec, o.w)   # nvec .*= w ./ n
    At_mul_B!(a.pvec, o.x, a.nvec)      # pvec ← x'nvec
end
multiply_by_weights!(nvec, w::Void) = scale!(nvec, 1 / length(nvec))
function multiply_by_weights!(nvec, w)
    wt = inv(length(nvec))
    for i in eachindex(nvec)
        @inbounds nvec[i] *= w[i] * wt
    end
end

#-----------------------------------------------------------------------# ProxGrad
"""
    ProxGrad(model, step = 1.0)

Proximal gradient method with step size `step`.
"""
struct ProxGrad <: GradientAlgorithm
    nvec::Vector{Float64}
    pvec::Vector{Float64}
    step::Float64
end
ProxGrad(o::SModel, s::Float64 = 1.0) = ProxGrad(zeros(size(o.x, 1)), zeros(size(o.x, 2)), s)
Base.show(io::IO, a::ProxGrad) = print(io, "ProxGrad(step = $(a.step))")
function update!(o::SModel, a::ProxGrad)
    gradient!(a, o)
    ∇ = a.pvec
    s = a.step
    for j in eachindex(o.β)
        o.β[j] = prox(o.penalty, o.β[j] - s * ∇[j], s * o.λ[j])
    end
end

#
# #-----------------------------------------------------------------------# Fista
# """
#     Fista(obs, step)
# Accelerated Proximal Gradient.  Works for any loss and convex penalty.
# """
# mutable struct Fista{O <: Obs} <: Algorithm
#     obs::O
#     step::Float64
#     derivs::Vector{Float64}
#     ∇::Vector{Float64}
#     β1::Vector{Float64}
#     β2::Vector{Float64}
#     t::Int
# end
# function Fista(obs::Obs, step::Float64 = 1.0)
#     n, p = size(obs)
#     Fista(obs, step, zeros(n), zeros(p), zeros(p), zeros(p), 0)
# end
# pre_hook(a::Fista, o::SModel) = (a.t = 0)
# function update!(o::SModel, a::Fista, item::Void)
#     copy!(a.β2, a.β1)
#     copy!(a.β1, o.β)
#     a.t += 1
#     if a.t > 2
#         γ = (a.t - 2) / (a.t + 1)
#         for j in eachindex(o.β)
#             @inbounds o.β[j] = a.β1[j] + γ * (a.β1[j] - a.β2[j])
#         end
#     end
#     gradient!(a.derivs, a.∇, o.β, o.loss, a.obs)
#     s = a.step
#     for j in eachindex(o.β)
#         @inbounds o.β[j] = prox(o.penalty, o.β[j] - s * a.∇[j], s * o.λfactor[j])
#     end
# end
#
#
# #-----------------------------------------------------------------------# GradientDescent
# """
#     GradientDescent(obs, step)
# Gradient Descent.  Works for any loss and any penalty.
# """
# struct GradientDescent{O <: Obs} <: Algorithm
#     obs::O
#     step::Float64
#     derivs::Vector{Float64}
#     ∇::Vector{Float64}
# end
# function GradientDescent(obs::Obs, step::Float64 = 1.0)
#     n, p = size(obs)
#     GradientDescent(obs, step, zeros(n), zeros(p))
# end
# function update!(o::SModel, a::GradientDescent, item::Void)
#     gradient!(a.derivs, a.∇, o.β, o.loss, a.obs)
#     s = a.step
#     for j in eachindex(o.β)
#         @inbounds o.β[j] -= s * (a.∇[j] + o.λfactor[j] * deriv(o.penalty, o.β[j]))
#     end
#     o
# end
#
#

#-----------------------------------------------------------------------# Sweep
"""
    Sweep(obs)
Linear/ridge regression via sweep operator.  Works for LinearRegression/L2DistLoss
with NoPenalty or L2Penalty.
"""
struct Sweep <: OneIterAlgorithm
    A::Matrix{Float64}  # [x y]'[x y]
    S::Matrix{Float64}  # sweep!(A, 1:p)
end
Sweep(o::SModel) = (A = make_A(o); Sweep(A, similar(A)))

function make_A(o::SModel{L,P,X,Y,W}) where {L,P,X,Y,W<:AbstractWeights}
    n, p = size(o.x)
    a = zeros(p + 1, p + 1)
    # b = zeros(p + 1, p + 1)
    BLAS.syrk!('U', 'T', 1 / n, Diagonal(sqrt.(o.w)) * o.x, 0.0, view(a, 1:p, 1:p)) # x'wx
    BLAS.gemv!('T', 1 / n, Diagonal(o.w) * o.x, o.y, 0.0, @view(a[1:p, end]))       # x'wy
    a[end, end] = dot(o.y, Diagonal(o.w) * o.y) / n                                 # y'wy
    a
end
function make_A(o::SModel{L,P,X,Y,Void}) where {L,P,X,Y}
    n, p = size(o.x)
    a = zeros(p + 1, p + 1)
    BLAS.syrk!('U', 'T', 1 / n, o.x, 0.0, view(a, 1:p, 1:p))    # x'x
    BLAS.gemv!('T', 1 / n, o.x, o.y, 0.0, @view(a[1:p, end]))   # x'y
    a[end, end] = dot(o.y, o.y) / n                             # y'y
    a
end
function update!(o::SModel, a::Sweep)
    n, p = size(o.x)
    copy!(a.S, a.A)
    add_ridge!(o, a)
    sweep!(a.S, 1:p)
    for j in eachindex(o.β)
        @inbounds o.β[j] = a.S[j, end]
    end
end
function add_ridge!(o::SModel{L, NoPenalty}, a::Sweep) where {L} end
function add_ridge!(o::SModel{L, L2Penalty}, a::Sweep) where {L}
    for i in eachindex(o.β)
        @inbounds a.S[i, i] += o.λ[i]
    end
end
#
#
# #-----------------------------------------------------------------------# LinRegCholesky
# """
#     LinRegCholesky(obs)
# Linear/ridge regression via cholesky decomposition.  Works for LinearRegression/L2DistLoss
# with NoPenalty or L2Penalty.
# """
# struct LinRegCholesky{O <: Obs} <: OneIterAlgorithm
#     obs::O
#     A::Matrix{Float64}
#     S::Matrix{Float64}
# end
# LinRegCholesky(obs::Obs) = (A = make_A(obs); LinRegCholesky(obs, make_A(obs), zeros(A)))
#
# function update!(o::SModel, a::LinRegCholesky, item::Void)
#     copy!(a.S, a.A)
#     cholfact!(Symmetric(a.S))
#     o.β[:] = UpperTriangular(@view(a.S[1:end-1, 1:end-1])) \ @view(a.S[1:end-1, end])
# end
# function add_ridge!{L}(o::SModel{L, NoPenalty}, a::LinRegCholesky, λf::Vector{Float64}) end
# function add_ridge!{L}(o::SModel{L, L2Penalty}, a::LinRegCholesky, λf::Vector{Float64})
#     for i in eachindex(o.β)
#         @inbounds a.S[i, i] += λf[i]
#     end
# end
