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

function ProxGrad(n::Integer, p::Integer; maxit::Int=100, tol::Float64=1e-6,
                  verbose::Bool=false, step::Float64=1.0, adaptivestep::Bool = true)
    ProxGrad(maxit, tol, verbose, step, zeros(p), zeros(n), zeros(n))
end

showme(a::ProxGrad) = [:maxit, :tol, :verbose, :step]



# TODOs:
# - Estimate Lipschitz constant for step size?
# - FISTA acceleration?
function fit!(o::SparseReg, A::ProxGrad, obs::Obs)
    n, p = size(obs.x)
    p == length(o.β) || throw(ArgumentError("x dimension does not match β"))

    oldcost = -Inf
    newcost = objective_value(o, obs, A.ŷ)
    niters = 0
    for k in 1:A.maxit
        oldcost = newcost
        niters += 1

        get_gradient!(o, A, obs)
        update_β!(o, A, obs)
        update_ŷ!(o, A, obs)

        newcost = objective_value(o, obs, A.ŷ)
        converged(oldcost, newcost, niters, A) && break
    end
    o
end

#--------------------------------------------------------------# components of loop
function get_gradient!(o, A::ProxGrad, obs::Obs{Ones})
    for i in eachindex(obs.y)
        @inbounds A.deriv_vec[i] = deriv(o.loss, obs.y[i], A.ŷ[i])
    end
    At_mul_B!(A.∇, obs.x, A.deriv_vec)
    scale!(A.∇, 1 / length(obs.y))
end
# weighted version
function get_gradient!(o, A::ProxGrad, obs::Obs)
    for i in eachindex(obs.y)
        @inbounds A.deriv_vec[i] = deriv(o.loss, obs.y[i], A.ŷ[i]) * obs.w[i]
    end
    At_mul_B!(A.∇, obs.x, A.deriv_vec)
    scale!(A.∇, 1 / length(obs.y))
end

function update_β!(o, A, obs)
    s = A.step
    @simd for j in eachindex(o.β)
        @inbounds λj = o.λ * o.factor[j]
        @inbounds o.β[j] = prox(o.penalty, o.β[j] - s * A.∇[j], s * λj)
    end
end

function update_ŷ!(o, A, obs)
    A_mul_B!(A.ŷ, obs.x, o.β)
    xβ_to_ŷ!(o.loss, A.ŷ)
end

@inline function converged(oldcost, newcost, niters, A)
    tolerance = abs(newcost - oldcost) / min(abs(newcost), abs(oldcost))
    isconverged = tolerance < A.tol
    isconverged ?
        A.verbose && info("CONVERGED: $niters, Relative Tolerance = $tolerance") :
        A.verbose && info("Iteration: $niters, Relative Tolerance = $tolerance")
        niters == A.maxit && warn("DID NOT CONVERGE in $niters iterations, Tol = $tolerance")
    isconverged
end
