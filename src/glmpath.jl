#----------------------------------------------------------------# Models for GLMPath
abstract GLMModel <: StatisticalModel
function gradient!(m::GLMModel, Δ::VecF, x::MatF, resids::VecF, η::VecF)
    BLAS.gemv!('T', -1.0, x, resids, 0.0, Δ)
end

# loss vector, proportional to negative loglikelihood
# log f(y | β) ∝ y * Θ + b(Θ)
function lossvector!(m::GLMModel, storage::VecF, y::VecF, η::VecF)
    for j in eachindex(storage)
        @inbounds Θ = _Θ(m, η[j])
        @inbounds storage[j] = -y[j] * Θ + _b(m, Θ)
    end
end


immutable L2Regression <: GLMModel end
predict!(m::L2Regression, storage::VecF, η::VecF) = copy!(storage, η)
predict(m::L2Regression, η::VecF) = η
_Θ(m::L2Regression, η::Float64) = η
_b(m::L2Regression, Θ::Float64) = 0.5 * Θ * Θ


immutable LogisticRegression <: GLMModel end
function  predict!(m::LogisticRegression, storage::VecF, η::VecF)
    for i in eachindex(storage)
        @inbounds storage[i] = 1.0 / (1.0 + exp(-η[i]))
    end
end
predict(m::LogisticRegression, η::VecF) = 1.0 ./ (1.0 + exp(-η))
_Θ(m::LogisticRegression, η::Float64) = η
_b(m::LogisticRegression, Θ::Float64) = log(1.0 + exp(Θ))


immutable ProbitRegression <: GLMModel end
function predict!(m::ProbitRegression, storage::VecF, η::VecF)
    for i in eachindex(storage)
        @inbounds storage[i] = Distributions.cdf(Normal(), η[i])
    end
end
predict(m::ProbitRegression, η::VecF) = Distributions.cdf(Normal(), η)
_Θ(m::ProbitRegression, η::Float64) = Distributions.logcdf(Normal(), η)
_b(m::ProbitRegression, Θ::Float64) = log(1.0 + exp(Θ))


immutable PoissonRegression <: GLMModel end
function predict!(m::PoissonRegression, storage::VecF, η::VecF)
    for i in eachindex(storage)
        @inbounds storage[i] = exp(η[i])
    end
end
predict(m::PoissonRegression, η::VecF) = exp(η)
_Θ(m::PoissonRegression, η::Float64) = η
_b(m::PoissonRegression, Θ::Float64) = exp(Θ)


#---------------------------------------------------------------------------# GLMPath
type GLMPath{M <: GLMModel, P <: Penalty} <: ModelPath
    β0::VecF            # intercepts
    β::MatF             # coefficients
    intercept::Bool     # should intercept be estimated?
    model::M            # link and loss
    penalty::P
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
        kw...
    )
    n, p = size(x)
    @assert length(y) == n "size(x, 2) != length(y)"
    nλ = length(λs)
    o = GLMPath(zeros(nλ), zeros(p, nλ), intercept, model, penalty, x, y, weights, λs)
    fista!(o; kw...)
end
function Base.show(io::IO, o::GLMPath)
    print_header(io, "GLMPath")
    print_item(io, "Model", o.model)
    print_item(io, "Penalty", o.penalty)
    print_item(io, "λs", "$(length(o.λs))")
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

    # main loop
    for ki in 1:length(o.λs)
        iters = 0
        newcost = Inf
        @inbounds λ = o.λs[ki]
        s = stepsize / sqrt(ki)        # step size for FISTA
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
            BLAS.gemv!('N', 1.0, x, β, 0.0, η)
            @inbounds β0 = o.β0[ki]
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
            gradient!(o.model, Δ, o.x, resids, η)
            # take step
            if o.intercept
                @inbounds o.β0[ki] += mean(resids)
            end
            β -= s * Δ / n
            # evaluate prox operator
            prox!(o.penalty, λ, β, s)
            # check for convergence
            lossvector!(o.model, lossvec, o.y, η)
            newcost = mean(lossvec) + penalty(o.penalty, λ, β)
            if newcost < oldcost  # decrease step size if cost doesn't decrease
                s *= 0.75
            end
            abs(newcost - oldcost) < eps * abs(oldcost + 1.0) && break
            if iters == maxit
                converged = false
            end
        end
        @inbounds o.β[:, ki] = β
    end
    converged || warn("Not converged for at least one λ")
    o
end






# TEST
using GLMNet
using Distributions
n, p = 10000, 100
x = randn(n, p)
β = collect(linspace(-.5, .5, p))

print_with_color(:green, "\nL2Regression\n")
y = x*β + randn(n)
o = GLMPath(x, y, model = L2Regression(), penalty = L1Penalty(), λs = collect(.1:.1:.4))
@time o = GLMPath(x, y, model = L2Regression(), penalty = L1Penalty(), λs = collect(.1:.1:.4))
@display o
@time glmnet(x, y, lambda = collect(.1:.1:.4))
# @display o.β


print_with_color(:green, "\nLogisticRegression\n")
y = Float64[rand(Bernoulli(1 / (1 + exp(-η)))) for η in x*β]
@time o = GLMPath(x, y, model = LogisticRegression(), penalty = L1Penalty(), λs = collect(.01:.01:.04))
@display o
# @display o.β


print_with_color(:green, "\nProbitRegression\n")
y = Float64[rand(Bernoulli(1 / (1 + exp(-η)))) for η in x*β]
@time o = GLMPath(x, y, model = ProbitRegression(), penalty = L1Penalty(), λs = collect(.01:.01:.04))
@display o
# @display o.β


print_with_color(:green, "\nPoissonRegression\n")
y = Float64[rand(Poisson(exp(η))) for η in x*β]
@time o = GLMPath(x, y, model = PoissonRegression(), penalty = L1Penalty(), λs = collect(1.:4))
@display o
# @display o.β
