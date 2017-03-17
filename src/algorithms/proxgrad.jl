"""
Proximal Gradient Method
"""
immutable ProxGrad <: OfflineAlgorithm
    maxit::Int
    tol::Float64
    verbose::Bool
    step::Float64
    # buffers
    ∇::VecF
    ŷ::VecF
    deriv_vec::VecF
end

# constructor with "empty" buffers
function ProxGrad(;maxit::Int=100, tol::Float64=1e-6, verbose::Bool=false, step::Float64=1.0)
    ProxGrad(maxit, tol, verbose, step, zeros(0), zeros(0), zeros(0))
end

# add buffers
function init(n::Integer, p::Integer, a::ProxGrad)
    ProxGrad(a.maxit, a.tol, a.verbose, a.step, zeros(p), zeros(n), zeros(n))
end

showme(o::ProxGrad) = [:maxit, :tol, :verbose, :step]




# TODOs:
# - line search?
# - Estimate Lipschitz constant for step size?
# - FISTA acceleration?
function fit!(o::SparseReg{ProxGrad}, obs::Obs)
    n, p = size(obs.x)
    p == length(o.β) || throw(ArgumentError("x dimension does not match β"))

    oldcost = -Inf
    newcost = objective_value(o, obs, o.algorithm.ŷ)
    niters = 0
    for k in 1:o.algorithm.maxit
        oldcost = newcost
        niters += 1

        get_gradient!(o, obs)
        update_β!(o)
        update_ŷ!(o, obs)

        newcost = objective_value(o, obs, o.algorithm.ŷ)
        converged(oldcost, newcost, niters, o.algorithm) && break
    end

    if niters == o.algorithm.maxit
        tolerance = abs(newcost - oldcost) / min(abs(newcost), abs(oldcost))
        warn("DID NOT CONVERGE in $niters iterations, Relative Tolerance = $tolerance")
    end
    o
end

#--------------------------------------------------------------# components of loop
function get_gradient!(o, obs::Obs{Ones})
    A = o.algorithm
    for i in eachindex(obs.y)
        @inbounds A.deriv_vec[i] = deriv(o.loss, obs.y[i], A.ŷ[i])
    end
    At_mul_B!(A.∇, obs.x, A.deriv_vec)
    scale!(A.∇, 1 / length(obs.y))
end
# weighted version
function get_gradient!(o, obs)
    A = o.algorithm
    for i in eachindex(obs.y)
        @inbounds A.deriv_vec[i] = deriv(o.loss, obs.y[i], A.ŷ[i]) * obs.w[i]
    end
    At_mul_B!(A.∇, obs.x, A.deriv_vec)
    scale!(A.∇, 1 / length(obs.y))
end

function update_β!(o)
    A = o.algorithm
    s = A.step
    @simd for j in eachindex(o.β)
        @inbounds λj = o.λ * o.factor[j]
        @inbounds o.β[j] = prox(o.penalty, o.β[j] - s * A.∇[j], s * λj)
    end
end

function update_ŷ!(o, obs)
    A_mul_B!(o.algorithm.ŷ, obs.x, o.β)
    xβ_to_ŷ!(o.loss, o.algorithm.ŷ)
end

@inline function converged(oldcost, newcost, niters, alg)
    tolerance = abs(newcost - oldcost) / min(abs(newcost), abs(oldcost))
    isconverged = tolerance < alg.tol
    isconverged ?
        alg.verbose && info("CONVERGED: $niters, Relative Tolerance = $tolerance") :
        alg.verbose && info("Iteration: $niters, Relative Tolerance = $tolerance")
    isconverged
end
