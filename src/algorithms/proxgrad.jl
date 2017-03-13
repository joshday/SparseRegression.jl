"""
Proximal Gradient Method
"""
immutable ProxGrad <: OfflineAlgorithm
    maxit::Int
    tol::Float64
    verbose::Bool
    step::Float64
end
function ProxGrad(;maxit::Int=100, tol::Float64=1e-6, verbose::Bool=false, step::Float64=1.0)
    ProxGrad(maxit, tol, verbose, step)
end
function Base.show(io::IO, o::ProxGrad)
    print(io,"ProxGrad(maxit=$(o.maxit), tol=$(o.tol), verbose=$(o.verbose), step=$(o.step))")
end

default(::Type{Algorithm}) = ProxGrad()

immutable ProxGradBuffer
    ∇::VecF
    ŷ::VecF
    deriv_vec::VecF
end
function makebuffer(o::SparseReg{ProxGrad}, obs::Obs)
    n, p = size(obs.x)
    ProxGradBuffer(zeros(p), zeros(n), zeros(n))
end


# TODOs:
# - line search?
# - Estimate Lipschitz constant for step size?
# - Use FISTA acceleration?
function fit!(o::SparseReg{ProxGrad}, obs::Obs, buffer = makebuffer(o, obs))
    n, p = size(obs.x)
    p == length(o.β) || throw(ArgumentError("x dimension does not match β"))

    oldcost = -Inf
    newcost = objective_value(o, obs, buffer.ŷ)
    niters = 0
    for k in 1:o.algorithm.maxit
        oldcost = newcost
        niters += 1

        get_gradient!(o, obs, buffer)
        update_β!(o, buffer)
        update_ŷ!(o, obs, buffer)

        newcost = objective_value(o, obs, buffer.ŷ)
        converged(oldcost, newcost, niters, o.algorithm) && break
    end

    if niters == o.algorithm.maxit
        tolerance = abs(newcost - oldcost) / min(abs(newcost), abs(oldcost))
        warn("DID NOT CONVERGE in $niters iterations, Relative Tolerance = $tolerance")
    end
    o
end

#--------------------------------------------------------------# components of loop
function get_gradient!(o, obs::Obs{Ones}, buffer)
    for i in eachindex(obs.y)
        @inbounds buffer.deriv_vec[i] = deriv(o.loss, obs.y[i], buffer.ŷ[i])
    end
    At_mul_B!(buffer.∇, obs.x, buffer.deriv_vec)
    scale!(buffer.∇, 1 / length(obs.y))
end
# weighted version
function get_gradient!(o, obs, buffer)
    for i in eachindex(obs.y)
        @inbounds buffer.deriv_vec[i] = deriv(o.loss, obs.y[i], buffer.ŷ[i]) * obs.w[i]
    end
    At_mul_B!(buffer.∇, obs.x, buffer.deriv_vec)
    scale!(buffer.∇, 1 / length(obs.y))
end

function update_β!(o, buffer)
    s = o.algorithm.step
    @simd for j in eachindex(o.β)
        @inbounds λj = o.λ * o.factor[j]
        @inbounds o.β[j] = prox(o.penalty, o.β[j] - s * buffer.∇[j], s * λj)
    end
end

function update_ŷ!(o, obs, buffer)
    A_mul_B!(buffer.ŷ, obs.x, o.β)
    xβ_to_ŷ!(o.loss, buffer.ŷ)
end

@inline function converged(oldcost, newcost, niters, alg)
    tolerance = abs(newcost - oldcost) / min(abs(newcost), abs(oldcost))
    isconverged = tolerance < alg.tol
    isconverged ?
        alg.verbose && info("CONVERGED: $niters, Relative Tolerance = $tolerance") :
        alg.verbose && info("Iteration: $niters, Relative Tolerance = $tolerance")
    isconverged
end
