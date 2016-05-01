#-----------------------------------------------------------------------------# FISTA
immutable FISTA <: Algorithm end

"""
Fast Iterative Shrinkage and Thresholding Algorithm

`fista!(o::SparseReg, X, y; kw...)`

While FISTA works for every model/penalty pair, it may not be the most efficient.
"""
function fit!{M <: Model}(o::SparseReg{M}, alg::FISTA, x::AMatF, y::AVecF;
        maxit::Integer      = 100,
        tol::Float64        = 1e-7,
        verbose::Bool       = true,
        step::Float64       = 0.5,
        weights::AVecF      = ones(0),
        standardize::Bool   = true
    )
    #-------------------------------------------------------------------------# setup
    n, p = size(x)
    @assert size(o.β, 1) == p "Columns of `x` don't match columns in `β`"
    β0 = 0.0
    β = zeros(p)
    Θ1 = zeros(p)           # last iteration
    Θ2 = zeros(p)           # two iterations ago
    Δ = zeros(p)            # Δ = x' * deriv_vec
    deriv_vec = zeros(n)    # derivative of loss with respect to η
    ŷ = zeros(n)            # vector of predicted values
    η = zeros(n)            # linear predictor
    if o.crit == :obj       # need lossvec if using objective as convergence criteria
        lossvec = zeros(n)
    elseif o.crit == :coef
        lossvec = zeros(0)
    end
    if standardize          # standardize columns of x
        x = zscore(x, 1)
    end
    intercept = o.intercept
    useweights = length(weights) > 0  # use weights if they are provided
    @assert !useweights || length(weights) == n "`weights` must have length $n"

    # main loop
    for k in eachindex(o.λ)
        #-----------------------------------# Check to see if βs have been zeroed out
        if k > 1
            if β == zeros(p)
                # fill in intercept values since we don't need to estimate coefs
                for j in k:length(o.β0)
                    o.β0[j] = β0
                end
                verbose && info("All coefficients zero at λ = $(o.λ[k-1])\n")
                break
            end
        end
        #----------------------------------------------------------# setup for next λ
        iters = 0
        newcost = Inf
        oldcost = Inf
        @inbounds λ = o.λ[k]
        s = step
        for rep in 1:maxit
            iters += 1
            oldcost = newcost
            #--------------------------------------------------------# FISTA momentum
            copy!(Θ2, Θ1)
            copy!(Θ1, β)
            if rep > 2
                ratio = (rep - 2) / (rep + 1)
                for j in eachindex(β)
                    @inbounds β[j] = Θ1[j] + ratio * (Θ1[j] - Θ2[j])
                end
            end
            #--------------------------------------------# linear predictor η = x * β
            BLAS.gemv!('N', 1.0, x, β, 0.0, η)
            if intercept
                for i in eachindex(η)
                    @inbounds η[i] += β0
                end
            end
            # ŷ
            predict!(o.model, ŷ, η)
            #-----------------------------------------------------# derivative vector
            for i in eachindex(deriv_vec)
                @inbounds deriv_vec[i] = lossderiv(o.model, y[i], η[i])
            end
            if useweights
                for i in eachindex(deriv_vec)
                    @inbounds deriv_vec[i] *= weights[i]
                end
            end
            #-------------------------------------# calculate gradient from deriv_vec
            BLAS.gemv!('T', 1 / n, x, deriv_vec, 0.0, Δ)
            #---------------------------------------------# gradient descent and prox
            if intercept
                β0 -= s * mean(deriv_vec)
            end
            β -= s * Δ
            prox!(o.penalty, β, λ * o.penalty_factor, s)
            #-------------------------------------------------# check for convergence
            if o.crit == :obj
                lossvector!(o.model, lossvec, y, η)
                if useweights
                    for i in eachindex(lossvec)
                        @inbounds lossvec[i] *= weights[i]
                    end
                end
                newcost = mean(lossvec) + penalty(o.penalty, β, λ)
            elseif o.crit == :coef
                newcost = maxabs(β - Θ1)
            end
            if abs(newcost - oldcost) < tol * (min(abs(oldcost), abs(newcost)) + 1.0)
                break
            end
        end
        #--------------------------------------# Did the algorithm reach convergence?
        reltol = abs(newcost - oldcost) / min(abs(oldcost), abs(newcost))
        if maxit == iters
            warn("Not converged for λ = $(o.λ[k]).  Tolerance = $(round(reltol, 12))")
        end
        #---------------------------------------------------------# update parameters
        if intercept
            o.β0[k] = β0
        end
        o.β[:, k] = β
    end  # end main loop
    standardize && scaled_to_original!(o, x)
    o
end



function scaled_to_original!(o::SparseReg, x)
    p, d = size(o.β)
    σx = vec(std(x, 1))
    μx = vec(mean(x, 1))
    scale!(1.0 ./ σx, o.β)
    if o.intercept
        for j in 1:d
            o.β[j] = o.β[j] - dot(μx, o.β[:, j])
        end
    end
end
