"""
Proximal Gradient Method
"""
@with_kw immutable ProxGrad <: OfflineAlgorithm
    maxit::Int      = 100
    tol::Float64    = 1e-6
    verbose::Bool   = false
    step::Float64   = 1.0
end
default(::Type{Algorithm}) = ProxGrad()

immutable ProxGradBuffer
    ∇::VecF
    ŷ::VecF
    deriv_buffer::VecF
end
function buffer(a::ProxGrad, x, y)
    n = size(x, 1)
    p = size(x, 2)
    ProxGradBuffer(zeros(p), zeros(n), zeros(n))
end


# TODOs:
# - line search
# - Estimate Lipschitz constant for step size?
# - Use FISTA acceleration?
# - Other criteria for convergence?
function fit!(o::SparseReg{ProxGrad}, x::AMat, y::AVec, buffer = buffer(o.algorithm, x, y))
    # error handling and setup
    n, p = size(x)
    p == length(o.β) || throw(ArgumentError("x dimension does not match β"))
    β = o.β
    L = o.loss
    P = o.penalty
    A = o.algorithm
    s = A.step

    # iterations
    oldcost = -Inf
    newcost = value(L, y, buffer.ŷ, AvgMode.Mean()) + value(P, o.β)
    niters = 0
    for k in 1:A.maxit
        oldcost = newcost
        niters += 1

        get_gradient!(L, y, buffer.ŷ, buffer.∇, x, buffer.deriv_buffer, 1/n)
        update_β!(β, P, s, buffer.∇, o.λ)
        update_ŷ!(buffer.ŷ, x, β, L)

        newcost = value(L, y, buffer.ŷ, AvgMode.Mean()) + value(P, β)
        converged(L, y, buffer.ŷ, P, β, oldcost, A.tol, A.verbose, niters, newcost) && break
    end

    tolerance = abs(newcost - oldcost) / min(abs(newcost), abs(oldcost))
    if niters == A.maxit
        warn("DID NOT CONVERGE in $niters iterations, Relative Tolerance = $tolerance")
    end
    o
end

#--------------------------------------------------------------# components of loop
function get_gradient!(L, y, ŷ, ∇, x, deriv_buffer, n_inv)
    for i in eachindex(y)
        @inbounds deriv_buffer[i] = deriv(L, y[i], ŷ[i])
    end
    At_mul_B!(∇, x, deriv_buffer)
    scale!(∇, n_inv)
end

function update_β!(β, P, s, ∇, λ)
    @simd for j in eachindex(β)
        @inbounds β[j] = prox(P, β[j] - s * ∇[j], s * λ[j])
    end
end
function update_β!(β, P, s, ∇, λ::Zeros)
    @simd for j in eachindex(β)
        @inbounds β[j] -= s * ∇[j]
    end
end

update_ŷ!(ŷ, x, β, L) = (A_mul_B!(ŷ, x, β); xβ_to_ŷ!(L, ŷ))

@inline function converged(L, y, ŷ, P, β, oldcost, tol, verbose, niters, newcost)
    tolerance = abs(newcost - oldcost) / min(abs(newcost), abs(oldcost))
    isconverged = tolerance < tol
    isconverged ?
        verbose && info("CONVERGED: $niters, Relative Tolerance = $tolerance") :
        verbose && info("Iteration: $niters, Relative Tolerance = $tolerance")
    isconverged
end
