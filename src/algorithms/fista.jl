# TODO: For parity with glmnet:
#   - contraints
#   - automatic choice of λs
#       - dfmax
#       - nlambda
#       - lambda_min_ratio
#       - pmax
# TODO: Issues
#   - Lasso p > n behaves strangely

immutable FISTA <: Algorithm end
default_alg(m::LinPredModel) = FISTA()

#-----------------------------------------------------------------------------# FISTA
function fit!{M <: LinPredModel}(alg::FISTA, o::SparseReg{M};
        maxit::Integer = 100,
        tol::Float64 = 1e-7,
        verbose::Bool = true,
        stepsize::Float64 = 1.0
    )
    # setup
    n, p = size(o.x)
    β0 = 0.0
    β = zeros(p)
    β1 = zeros(p)       # last iteration
    β2 = zeros(p)       # two iterations ago
    Δ = zeros(p)            # Δ = x' * deriv_vec
    deriv_vec = zeros(n)    # derivative of loss with respect to η
    ŷ = zeros(n)
    η = zeros(n)
    lossvec = zeros(n)
    intercept = o.intercept
    useweights = length(o.weights) == n

    # main loop
    for k in eachindex(o.λs)
        ######## Check to see if βs have been zeroed out
        if k > 1
            if β == zeros(p)
                # fill in intercept values since we don't need to estimate coefs
                for j in k:length(o.β0)
                    o.β0[j] = β0
                end
                verbose && info("All coefficients zero at λ = $(o.λs[k-1])\n")
                break
            end
        end
        iters = 0
        newcost = Inf
        oldcost = Inf
        @inbounds λ = o.λs[k]
        s = stepsize
        for rep in 1:maxit
            iters += 1
            oldcost = newcost
            ############ FISTA momentum
            copy!(β2, β1)
            copy!(β1, β)
            if rep > 2
                β = β1 + ((rep - 2) / (rep + 1)) * (β1 - β2)
            end
            ############ η
            BLAS.gemv!('N', 1.0, o.x, β, 0.0, η)
            if intercept
                for i in eachindex(η)
                    @inbounds η[i] += β0
                end
            end
            ############ ŷ
            predict!(o.model, ŷ, η)
            ############ derivative vector
            for i in eachindex(deriv_vec)
                @inbounds deriv_vec[i] = lossderiv(o.model, o.y[i], η[i])
            end
            if useweights
                for i in eachindex(deriv_vec)
                    @inbounds deriv_vec[i] *= o.weights[i]
                end
            end
            ############ gradient
            BLAS.gemv!('T', 1/n, o.x, deriv_vec, 0.0, Δ)
            ############ gradient descent
            if intercept
                β0 -= s * mean(deriv_vec)
            end
            β -= s * Δ
            ############ prox operator
            prox!(o.penalty, β, λ * o.penalty_factor, s)
            ############ check for convergence
            lossvector!(o.model, lossvec, o.y, η)
            if useweights
                for i in eachindex(lossvec)
                    @inbounds lossvec[i] *= o.weights[i]
                end
            end
            newcost = mean(lossvec) + penalty(o.penalty, β, λ)
            if abs(newcost - oldcost) < tol * abs(oldcost)
                break
            end
            ############ decrease step size
            if newcost > oldcost
                s *= .5
                copy!(β, β1)
            else
                s = stepsize
            end
        end
        ############ Did the algorithm reach convergence?
        if maxit == iters
            reltol = abs(newcost - oldcost) / abs(oldcost)
            warn("Not converged for λ = $(o.λs[k]).  Tolerance = $(round(reltol, 8))")
        end
        ############ Get maximum relative difference
        ############ update parameters
        if intercept
            o.β0[k] = β0
        end
        o.β[:, k] = β
    end  # end main loop
    o
end
