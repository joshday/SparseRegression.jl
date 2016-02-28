#---------------------------------------------------------------------------# GLMPath
type GLMPath{M <: GLMModel, P <: Penalty} <: ModelPath
    β0::VecF            # intercepts
    β::MatF             # coefficients
    intercept::Bool     # should intercept be estimated?
    model::M            # link and loss
    penalty::P          # regularization
    x::MatF             # design matrix
    y::VecF             # response vector
    weights::VecF       # weights
    λs::VecF            # regularization parameters
end
function GLMPath(x::Matrix, y::Vector;
        intercept::Bool     = true,
        model::GLMModel     = L2Regression(),
        penalty::Penalty    = NoPenalty(),
        weights::Vector     = ones(0),
        λs::Vector          = zeros(1),
        standardize::Bool   = true,  # TODO
        algkw...
    )
    n, p = size(x)
    @assert length(y) == n "size(x, 2) != length(y)"
    nλ = length(λs)
    o = GLMPath(zeros(nλ), zeros(p, nλ), intercept, model, penalty, x, y, weights, λs)
    fista!(o; algkw...)
end
function Base.show(io::IO, o::GLMPath)
    print_header(io, "GLMPath")
    print_item(io, "Model", o.model)
    print_item(io, "Penalty", o.penalty)
    print_item(io, "Intercept", o.intercept)
    print_item(io, "λs", "$(length(o.λs))")
end
function coef(o::GLMPath, λ::Real = o.λs[1])
    i = findfirst(o.λs, λ)
    i != 0 ?
        vcat(o.β0[i], o.β[:, i]) :
        error("λ = $λ was not fit by GLMPath")
end



#--------------------------------------------------------------------# main algorithm
# Fast Iterative Shrinkage-Thresholding Algorithm
function fista!(o::GLMPath;
        maxit::Integer = 100,
        eps::Float64 = 1e-4,
        verbose::Bool = true,
        stepsize::Float64 = 1.0
    )
    # setup
    n, p = size(o.x)
    β = zeros(p)
    β1 = zeros(p)       # last iteration
    β2 = zeros(p)       # two iterations ago
    Δ = zeros(p)
    resids = zeros(n)
    ŷ = zeros(n)
    η = zeros(n)
    lossvec = zeros(n)
    converged = true
    useweights = length(o.weights) == n

    # main loop
    for k in eachindex(o.λs)
        iters = 0
        newcost = Inf
        @inbounds λ = o.λs[k]
        s = stepsize / sqrt(k)        # step size for FISTA
        for rep in 1:maxit
            iters += 1
            oldcost = newcost
        # FISTA momentum
            copy!(β2, β1)
            copy!(β1, β)
            if rep > 2
                β += ((rep - 2) / (rep + 1)) * (β1 - β2)
            end
        # get η
            BLAS.gemv!('N', 1.0, o.x, β, 0.0, η)
            @inbounds β0 = o.β0[k]
            for i in eachindex(η)
                @inbounds η[i] += β0
            end
        # get ŷ
            predict!(o.model, ŷ, η)
        # get residuals
            for j in eachindex(resids)
                @inbounds resids[j] = o.y[j] - ŷ[j]
            end
        # get gradient
            useweights ?
                weightedgradient!(o.model, Δ, o.x, resids, η, o.weights) :
                gradient!(o.model, Δ, o.x, resids, η)
        # take step
            if o.intercept
                @inbounds o.β0[k] += mean(resids)
            end
            β -= s * Δ / n
        # evaluate prox operator
            prox!(o.penalty, λ, β, s)
        # check for convergence
            useweights ?
                weightedlossvector!(o.model, lossvec, o.y, η, o.weights) :
                lossvector!(o.model, lossvec, o.y, η)
            newcost = mean(lossvec) + penalty(o.penalty, λ, β)
            if newcost > oldcost  # decrease step size if cost doesn't decrease
                s *= 0.75
            end
            abs(newcost - oldcost) < eps * abs(oldcost + 1.0) && break
        end
        if iters == maxit
            converged = false
        end
        @inbounds o.β[:, k] = β
    end
    converged || warn("Not converged for at least one λ")
    o
end
