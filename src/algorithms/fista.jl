#-----------------------------------------------------------------------------# FISTA
immutable FISTA <: Algorithm
    maxit::Int
    tol::Float64
    verbose::Bool
    step::Float64
    criteria::Symbol
    standardize::Bool
end
function FISTA(;
        maxit::Integer      = 100,
        tol::Real           = 1e-7,
        verbose::Bool       = true,
        step::Real          = 0.5,
        criteria::Symbol    = :obj,
        standardize::Bool   = false
    )
    FISTA(maxit, tol, verbose, step, criteria, standardize)
end
Base.show(io::IO, o::FISTA) = println(io, "FISTA")


"""
Fast Iterative Shrinkage and Thresholding Algorithm

While FISTA works for every model/penalty pair, it may not be the most efficient.
"""
function fit!{M <: Model, P <: Penalty}(
        o::SparseReg{M, P, FISTA}, x::AMatF, y::AVecF, wts::AVecF = ones(0)
    )
    #------------------------------------------------------------------------# checks
    n, p = size(x)
    @assert size(o.β, 1) == p "Columns of `x` don't match columns in `β`"
    useweights = length(wts) > 0  # use weights if they are provided
    @assert !useweights || length(wts) == n "`weights` must have length $n"
    @assert !(o.intercept == false && alg.standardize == true) "standardizing implies an intercept"

    #-------------------------------------------------------------------------# setup
    alg = o.algorithm
    β0 = 0.0
    β = zeros(p)
    Θ1 = zeros(p)           # last iteration
    Θ2 = zeros(p)           # two iterations ago
    Δ = zeros(p)            # Δ = x' * deriv_vec
    deriv_vec = zeros(n)    # derivative of loss with respect to η
    # ŷ = zeros(n)            # vector of predicted values
    η = zeros(n)            # linear predictor
    if alg.criteria == :obj       # need lossvec if using objective as convergence criteria
        lossvec = zeros(n)
    elseif alg.criteria == :coef
        lossvec = zeros(0)
    end
    if alg.standardize          # standardize columns of x
        x_original = x
        x = zscore(x, 1)
    end
    intercept = o.intercept

    # main loop
    for k in eachindex(o.λ)
        #-----------------------------------# Check to see if βs have been zeroed out
        if k > 1
            if β == zeros(p)
                # fill in intercept values since we don't need to estimate coefs
                for j in k:length(o.β0)
                    o.β0[j] = β0
                end
                o.verbose && info("All coefficients zero at λ = $(o.λ[k-1])\n")
                break
            end
        end
        #----------------------------------------------------------# setup for next λ
        iters = 0
        newcost = Inf
        oldcost = Inf
        @inbounds λ = o.λ[k]
        s = alg.step
        for rep in 1:alg.maxit
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
            # predict!(o.model, ŷ, η)
            #-----------------------------------------------------# derivative vector
            for i in eachindex(deriv_vec)
                @inbounds deriv_vec[i] = lossderiv(o.model, y[i], η[i])
            end
            if useweights
                for i in eachindex(deriv_vec)
                    @inbounds deriv_vec[i] *= wts[i]
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
            if alg.criteria == :obj
                lossvector!(o.model, lossvec, y, η)
                if useweights
                    for i in eachindex(lossvec)
                        @inbounds lossvec[i] *= wts[i]
                    end
                end
                newcost = mean(lossvec) + penalty(o.penalty, β, λ)
            elseif alg.criteria == :coef
                newcost = maxabs(β - Θ1)
            end
            if abs(newcost - oldcost) < alg.tol * (min(abs(oldcost), abs(newcost)) + 1.0)
                break
            end
            if (alg.criteria == :obj) && (newcost > oldcost)  # step-"halving"
                s *= .7
                copy!(β, Θ1)
            end
            if any(isnan(β))
                error("NaNs in β after iteration $iters")
            end
        end
        #--------------------------------------# Did the algorithm reach convergence?
        reltol = abs(newcost - oldcost) / min(abs(oldcost), abs(newcost))
        if alg.maxit == iters
            warn("Not converged for λ = $(o.λ[k]).  Tolerance = $(round(reltol, 12))")
        end
        #---------------------------------------------------------# update parameters
        if intercept
            o.β0[k] = β0
        end
        o.β[:, k] = β
    end  # end main loop
    if alg.standardize
        scaled_to_original!(o, x_original)
    end
    o
end



function scaled_to_original!(o::SparseReg, x)
    p, d = size(o.β)
    σx = vec(std(x, 1))
    μx = vec(mean(x, 1))
    scale!(1 ./ σx, o.β)
    for j in eachindex(o.β0)
        o.β0[j] = o.β0[j] - dot(μx, o.β[:, j])
    end
end
