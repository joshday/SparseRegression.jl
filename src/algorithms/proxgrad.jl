"""
Proximal Gradient Method
"""
immutable PROXGRAD <: OfflineAlgorithm
    maxit::Int
    tol::Float64
    verbose::Bool
    # buffers
    ∇::VecF
    yhat::VecF
    deriv_buffer::VecF
end
function PROXGRAD(n::Integer = 0, p::Integer = 0;
                  maxit::Integer = 100, tol::Float64 = 1e-6, verbose::Bool = false)
    PROXGRAD(maxit, tol, verbose, zeros(p), zeros(n), zeros(n))
end
function init(alg::PROXGRAD, n, p)
    typeof(alg)(n, p; maxit = alg.maxit, tol = alg.tol, verbose = alg.verbose)
end

# TODOs:
# - line search
# - Estimate Lipschitz constant for step size?
# - Use FISTA acceleration?
# - Other criteria for convergence?
function fit!(o::SparseReg{PROXGRAD}, x::AMat, y::AVec)
    # error handling and setup
    n, p = size(x)
    p == length(o.β) || throw(ArgumentError("x dimension does not match β"))
    use_weights = typeof(o.avg) <: AvgMode.WeightedMean
    !use_weights ||
        length(o.avg.weights) == n ||
            throw(ArgumentError("`weights` must have length $n"))
    β = o.β
    A = o.algorithm
    L = o.loss
    P = o.penalty
    AVG = o.avg
    penfact = o.penaltyfactor

    # iterations
    oldloss = -Inf
    newloss = value(L, y, A.yhat, AVG)
    niters = 0
    for k in 1:A.maxit
        oldloss = newloss
        niters += 1
        # calculate the gradient
        if use_weights
            A.deriv_buffer .= deriv.(L, y, A.yhat) .* AVG.weights.values
        else
            A.deriv_buffer .= deriv.(L, y, A.yhat)
        end
        At_mul_B!(A.∇, x, A.deriv_buffer)
        scale!(A.∇, 1 / n)
        # update parameters
        @simd for j in eachindex(β)
            @inbounds β[j] = prox(P, β[j] - A.∇[j], penfact[j])
        end
        # update yhat
        A_mul_B!(A.yhat, x, β)  # Overwrite yhat with linear predictor x * β
        A.yhat .= _predict.(L, A.yhat)  # turn linear predictor into prediction
        # check for convergence
        newloss = value(L, y, A.yhat, AVG)  # needs weighted version
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
