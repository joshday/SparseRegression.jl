"""
Proximal Gradient Method
"""
immutable PROXGRAD <: OfflineAlgorithm
    maxit::Int
    tol::Float64
    verbose::Bool
    step::Float64
    # buffers
    ∇::VecF
    yhat::VecF
    deriv_buffer::VecF
end
function PROXGRAD(n::Integer = 0, p::Integer = 0; maxit::Integer = 100, tol::Float64 = 1e-6,
                  verbose::Bool = false, step::Float64 = 1.0)
    PROXGRAD(maxit, tol, verbose, step, zeros(p), zeros(n), zeros(n))
end
function init(a::PROXGRAD, n, p)
    typeof(a)(n, p; maxit = a.maxit, tol = a.tol, verbose = a.verbose, step = a.step)
end

# TODOs:
# - line search
# - Estimate Lipschitz constant for step size?
# - Use FISTA acceleration?
# - Other criteria for convergence?
# - Convergence doesn't use Penalty yet!!!!!
function fit!(o::SparseReg, x::AMat, y::AVec, alg::Algorithm = PROXGRAD())
    # error handling and setup
    n, p = size(x)
    A = init(alg, n, p)
    p == length(o.β) || throw(ArgumentError("x dimension does not match β"))
    β = o.β
    L = o.loss
    P = o.penalty
    s = A.step
    λ = o.λ

    # iterations
    oldloss = -Inf
    newloss = value(L, y, A.yhat, AvgMode.Mean())
    niters = 0
    for k in 1:A.maxit
        oldloss = newloss
        niters += 1
        # calculate the gradient
        A.deriv_buffer .= deriv.(L, y, A.yhat)
        At_mul_B!(A.∇, x, A.deriv_buffer)
        scale!(A.∇, 1 / n)
        # update parameters
        @simd for j in eachindex(β)
            @inbounds β[j] = prox(P, β[j] - s * A.∇[j], λ[j])
        end
        # update yhat
        A_mul_B!(A.yhat, x, β)  # Overwrite yhat with linear predictor x * β
        A.yhat .= _predict.(L, A.yhat)  # turn linear predictor into prediction
        # check for convergence
        newloss = value(L, y, A.yhat, AvgMode.Mean())  # needs weighted version
        abs(newloss - oldloss) < min(abs(newloss), abs(oldloss)) * A.tol && break
        if A.verbose
            tolerance = abs(newloss - oldloss) / min(abs(newloss), abs(oldloss))
            info("Iteration: $niters, Relative Tolerance = $tolerance")
        end
    end

    tolerance = abs(newloss - oldloss) / min(abs(newloss), abs(oldloss))
    if tolerance < A.tol
        A.verbose && info("CONVERGED: $niters, Relative Tolerance = $tolerance")
    else
        warn("DID NOT CONVERGE in $niters iterations, Relative Tolerance = $tolerance")
    end
    o
end
