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
    function ProxGradBuffer(x::Matrix, y::Vector)
        n, p = size(x)
        new(zeros(p), zeros(n), zeros(n))
    end
end


# TODOs:
# - line search
# - Estimate Lipschitz constant for step size?
# - Use FISTA acceleration?
# - Other criteria for convergence?
# - Convergence doesn't use Penalty yet!!!!!
function fit!(o::SparseReg{ProxGrad}, x::AMat, y::AVec,
              buffer::ProxGradBuffer = ProxGradBuffer(x, y))
    # error handling and setup
    n, p = size(x)
    A = buffer
    p == length(o.β) || throw(ArgumentError("x dimension does not match β"))
    β = o.β
    L = o.loss
    P = o.penalty
    s = o.algorithm.step
    λ = o.λ

    # iterations
    oldloss = -Inf
    newloss = value(L, y, A.ŷ, AvgMode.Mean())
    niters = 0
    for k in 1:o.algorithm.maxit
        oldloss = newloss
        niters += 1
        # calculate the gradient
        A.deriv_buffer .= deriv.(L, y, A.ŷ)
        At_mul_B!(A.∇, x, A.deriv_buffer)
        scale!(A.∇, 1 / n)
        # update parameters
        @simd for j in eachindex(β)
            @inbounds β[j] = prox(P, β[j] - s * A.∇[j], λ[j])
        end
        # update ŷ
        A_mul_B!(A.ŷ, x, β)  # Overwrite ŷ with linear predictor x * β
        A.ŷ .= _predict.(L, A.ŷ)  # turn linear predictor into prediction
        # check for convergence
        newloss = value(L, y, A.ŷ, AvgMode.Mean())  # needs weighted version
        abs(newloss - oldloss) < min(abs(newloss), abs(oldloss)) * o.algorithm.tol && break
        if o.algorithm.verbose
            tolerance = abs(newloss - oldloss) / min(abs(newloss), abs(oldloss))
            info("Iteration: $niters, Relative Tolerance = $tolerance")
        end
    end

    tolerance = abs(newloss - oldloss) / min(abs(newloss), abs(oldloss))
    if tolerance < o.algorithm.tol
        o.algorithm.verbose && info("CONVERGED: $niters, Relative Tolerance = $tolerance")
    else
        warn("DID NOT CONVERGE in $niters iterations, Relative Tolerance = $tolerance")
    end
    o
end
