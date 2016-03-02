function fit!{L <: ErrorLoss}(o::StatLearnPath{L};
        maxit::Integer = 100,
        tol::Float64 = 1e-6,
        verbose::Bool = true,
        stepsize::Float64 = 1.0
    )
    # setup
    n, p = size(o.x)
    T = typeof(o.β[1])
    β0 = zero(T)
    β = zeros(T, p)
    β1 = zeros(T, p)       # last iteration
    β2 = zeros(T, p)       # two iterations ago
    Δ = zeros(T, p)
    grad = zeros(T, n)   # gradient = x'deriv(o.loss, resids)
    ŷ = zeros(T, n)
    η = zeros(T, n)
    lossvec = zeros(T, n)
    converged = true
    intercept = o.intercept
    useweights = length(o.weights) == n

    # main loop
    for k in eachindex(o.λs)
        iters = 0
        newcost = Inf
        @inbounds λ = o.λs[k]
        s = stepsize / sqrt(k)
        for rep in 1:maxit
            iters += 1
            oldcost = newcost
            ############ FISTA momentum
            copy!(β2, β1)
            copy!(β1, β)
            if rep > 2
                β += T((rep - 2) / (rep + 1)) * (β1 - β2)
            end
            ############ η
            BLAS.gemv!('N', 1.0, o.x, β, 0.0, η)
            if intercept
                for i in eachindex(η)
                    @inbounds η[i] += β0
                end
            end
            ############ ŷ
            predict!(o.link, ŷ, η)
            ############ residuals
            for j in eachindex(grad)
                grad[j] = deriv(o.loss, o.y[j] - ŷ[j])
            end
            ############ gradient
            BLAS.gemv!('T', -T(1/n), o.x, grad, zero(T), Δ)
            ############ gradient descent
            if intercept
                β0 += mean(grad)
            end
            β -= s * Δ
            ############ prox operator
            prox!(o.penalty, β, λ, s)
            ############ check for convergence
            lossvector!(o.loss, lossvec, grad)
            newcost = mean(lossvec) + penalty(o.penalty, β, λ)
            abs(newcost - oldcost) < tol * abs(oldcost) && break


            if iters == maxit
                converged = false
            end
        end
        ############ update parameters
        if intercept
            o.β0[k] = β0
        end
        o.β[:, k] = β
    end
    converged || warn("Not converged for at least one λ")
    o
end
