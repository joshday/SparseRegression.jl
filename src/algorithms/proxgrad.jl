"""
Proximal Gradient Method
"""
immutable ProxGrad{O <: Obs} <: OfflineAlgorithm
    obs::O
    maxit::Int
    tol::Float64
    verbose::Bool
    step::Float64
    crit::Symbol    # objective, gradient
    # buffers
    ∇::VecF
    ŷ::VecF
    deriv_vec::VecF
end

function ProxGrad(o::Obs; maxit::Int=100, tol::Float64=1e-6, verbose::Bool=false,
                  step::Float64=1.0, crit::Symbol = :objective, adaptivestep::Bool = true)
    n, p = size(o.x)
    ProxGrad(o, maxit, tol, verbose, step, crit, zeros(p), zeros(n), zeros(n))
end

showme(a::ProxGrad) = [:maxit, :tol, :verbose, :step, :crit]




# TODOs:
# - Estimate Lipschitz constant for step size?
# - FISTA acceleration?
function fit!(o::SparseReg, A::ProxGrad)
    n, p = size(A.obs.x)
    p == length(o.β) || throw(ArgumentError("x dimension does not match β"))

    oldcost = -Inf
    newcost = objective_value(o, A.obs, A.ŷ)
    niters = 0
    for k in 1:A.maxit
        oldcost = newcost
        niters += 1

        get_gradient!(o, A)
        update_β!(o, A)
        update_ŷ!(o, A)

        if A.crit == :objective
            newcost = objective_value(o, A.obs, A.ŷ)
            converged(oldcost, newcost, niters, A) && break
        elseif A.crit == :gradient
            newcost = vecnorm(A.∇)
            vecnorm(A.∇) < A.tol && break
        end
    end

    if niters == A.maxit
        if A.crit == :objective
            tolerance = abs(newcost - oldcost) / min(abs(newcost), abs(oldcost))
            warn("DID NOT CONVERGE in $niters iterations, Relative Tolerance = $tolerance")
        elseif A.crit == :gradient
            warn("DID NOT CONVERGE IN $niters iterations, gradient norm = $newcost")
        end
    end
    o
end

#--------------------------------------------------------------# components of loop
function get_gradient!(o, A::ProxGrad{Obs{Ones}})
    for i in eachindex(A.obs.y)
        @inbounds A.deriv_vec[i] = deriv(o.loss, A.obs.y[i], A.ŷ[i])
    end
    At_mul_B!(A.∇, A.obs.x, A.deriv_vec)
    scale!(A.∇, 1 / length(A.obs.y))
end
# weighted version
function get_gradient!(o, A::ProxGrad)
    for i in eachindex(A.obs.y)
        @inbounds A.deriv_vec[i] = deriv(o.loss, A.obs.y[i], A.ŷ[i]) * A.obs.w[i]
    end
    At_mul_B!(A.∇, A.obs.x, A.deriv_vec)
    scale!(A.∇, 1 / length(A.obs.y))
end

function update_β!(o, A)
    s = A.step
    @simd for j in eachindex(o.β)
        @inbounds λj = o.λ * o.factor[j]
        @inbounds o.β[j] = prox(o.penalty, o.β[j] - s * A.∇[j], s * λj)
    end
end

function update_ŷ!(o, A)
    A_mul_B!(A.ŷ, A.obs.x, o.β)
    xβ_to_ŷ!(o.loss, A.ŷ)
end

@inline function converged(oldcost, newcost, niters, A)
    tolerance = abs(newcost - oldcost) / min(abs(newcost), abs(oldcost))
    isconverged = tolerance < A.tol
    isconverged ?
        A.verbose && info("CONVERGED: $niters, Relative Tolerance = $tolerance") :
        A.verbose && info("Iteration: $niters, Relative Tolerance = $tolerance")
    isconverged
end
