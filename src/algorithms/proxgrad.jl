"""
Proximal Gradient Method
"""
immutable ProxGrad{O <: Obs} <: OfflineAlgorithm
    obs::O
    maxit::Int
    tol::Float64
    verbose::Bool
    step::Float64
    # buffers
    ∇::VecF
    ŷ::VecF
    deriv_vec::VecF
end

function ProxGrad(o::Obs; maxit::Int=100, tol::Float64=1e-6, verbose::Bool=false,
                  step::Float64=1.0)
    n, p = size(o.x)
    ProxGrad(o, maxit, tol, verbose, step, zeros(p), zeros(n), zeros(n))
end

function Base.show(io::IO, A::ProxGrad)
    print(io, "■ ProxGrad")
    showfields(io, A, [:maxit, :tol, :verbose, :step])
    println(io, "")
    show(io, A.obs)
end




# TODOs:
# - line search?
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

        newcost = objective_value(o, A.obs, A.ŷ)
        converged(oldcost, newcost, niters, A) && break
    end

    if niters == A.maxit
        tolerance = abs(newcost - oldcost) / min(abs(newcost), abs(oldcost))
        warn("DID NOT CONVERGE in $niters iterations, Relative Tolerance = $tolerance")
    end
    o
end

#--------------------------------------------------------------# components of loop
function get_gradient!(o, A::ProxGrad)
    for i in eachindex(A.obs.y)
        @inbounds A.deriv_vec[i] = deriv(o.loss, A.obs.y[i], A.ŷ[i])
    end
    At_mul_B!(A.∇, A.obs.x, A.deriv_vec)
    scale!(A.∇, 1 / length(A.obs.y))
end
# weighted version
function get_gradient!(o, A::ProxGrad{Obs{Ones}})
    for i in eachindex(A.obs.y)
        @inbounds A.deriv_vec[i] = deriv(o.loss, A.obs.y[i], A.ŷ[i]) * A.obs.w[i]
    end
    At_mul_B!(A.∇, obs.x, A.deriv_vec)
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

@inline function converged(oldcost, newcost, niters, alg)
    tolerance = abs(newcost - oldcost) / min(abs(newcost), abs(oldcost))
    isconverged = tolerance < alg.tol
    isconverged ?
        alg.verbose && info("CONVERGED: $niters, Relative Tolerance = $tolerance") :
        alg.verbose && info("Iteration: $niters, Relative Tolerance = $tolerance")
    isconverged
end
