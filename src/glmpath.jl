#----------------------------------------------------------------# Models for GLMPath
abstract GLMModel <: StatisticalModel
function gradient!(m::GLMModel, Δ::VecF, x::MatF, resids::VecF, η::VecF)
    BLAS.gemv!('T', -1.0, x, resids, 0.0, Δ)
end

# loss vector, proportional to negative loglikelihood
# log f(y | β) ∝ y * Θ + b(Θ)
function lossvector(m::GLMModel, y::VecF, η::VecF)
    Θ = _Θ(m, η)
    -y .* Θ + _b(m, Θ)
end

immutable L2Regression <: GLMModel end
predict(m::L2Regression, η::VecF) = η
_Θ(m::L2Regression, η::VecF) = η
_b(m::L2Regression, Θ::VecF) = 0.5 * Θ .^ 2

immutable LogisticRegression <: GLMModel end
predict(m::LogisticRegression, η::VecF) = 1.0 ./ (1.0 + exp(-η))
_Θ(m::LogisticRegression, η::VecF) = η
_b(m::LogisticRegression, Θ::VecF) = log(1.0 + exp(Θ))

immutable ProbitRegression <: GLMModel end
predict(m::ProbitRegression, η::VecF) = Distributions.cdf(Normal(), η)
_Θ(m::ProbitRegression, η::VecF) = Distributions.logcdf(Normal(), η)
_b(m::ProbitRegression, Θ::VecF) = log(1.0 + exp(Θ))

immutable PoissonRegression <: GLMModel end
predict(m::PoissonRegression, η::VecF) = exp(η)
_Θ(m::PoissonRegression, η::VecF) = η
_b(m::PoissonRegression, Θ::VecF) = exp(Θ)


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
        λs::Vector          = zeros(1)
    )
    n, p = size(x)
    @assert length(y) == n "size(x, 2) != length(y)"
    nλ = length(λs)
    GLMPath(
        zeros(nλ),
        zeros(p, nλ),
        intercept,
        model,
        penalty,
        x,
        y,
        weights,
        λs
    )
end
function Base.show(io::IO, o::GLMPath)
    print_header(io, "GLMPath")
    print_item(io, "Model", o.model)
    print_item(io, "Penalty", o.penalty)
    print_item(io, "Path of", "$(length(o.λs)) λs")
end



#--------------------------------------------------------------------# main algorithm
function fista!(o::GLMPath, k::UnitRange{Int} = 1:length(o.λs);
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

    # main loop
    for ki in k
        iters = 0
        newcost = Inf
        λ = o.λs[ki]
        s = stepsize        # step size for FISTA
        for rep in 1:maxit
            iters += 1
            oldcost = newcost

            # FISTA momentum
            copy!(β2, β1)
            copy!(β1, β)
            if rep > 2
                β += ((rep - 2) / (rep + 1)) * (β1 - β2)
            end

            # get gradient
            η = o.x*β + o.β0[ki]
            ŷ = predict(o.model, η)
            resids = o.y - ŷ
            gradient!(o.model, Δ, o.x, resids, η)

            # take step
            if o.intercept
                o.β0[ki] += mean(resids)
            end
            β -= s * Δ / n

            # evaluate prox operator
            prox!(o.penalty, λ, β, s)

            # check for convergence
            newcost = mean(lossvector(o.model, o.y, η)) + penalty(o.penalty, λ, o.β[:, ki])
            if newcost < oldcost  # decrease step size if cost doesn't decrease
                s *= 0.9
            end
            if abs(newcost - oldcost) < eps * abs(oldcost + 1.0)
                verbose && println("λ = $(o.λs[ki]) converged in $iters iterations")
                verbose && println("tolerance = $(abs(newcost - oldcost) / (abs(1.0 + oldcost)))")
                break
            end
            iters == maxit &&
                print_with_color(:red, "λ = $(o.λs[ki]) did not converge in $iters iterations!\n")
        end
        o.β[:, ki] = β
    end
    o
end






# TEST
using GLMNet
using Distributions
n, p = 10000, 11
x = randn(n, p)
β = collect(linspace(-.5, .5, p))

print_with_color(:green, "\nL2Regression\n\n")
y = x*β + randn(n)
o = GLMPath(x, y, model = L2Regression(), penalty = L1Penalty(), λs = collect(.1:.1:.4))
fista!(o)
@time fista!(o)
@time glmnet(x, y, lambda = collect(.1:.1:.4))
@display o
@display o.β


print_with_color(:green, "\nLogisticRegression\n\n")
y = Float64[rand(Bernoulli(1 / (1 + exp(-η)))) for η in x*β]
o = GLMPath(x, y, model = LogisticRegression(), penalty = L1Penalty(), λs = collect(.01:.01:.04))
fista!(o)
@time fista!(o)
@display o
@display o.β


print_with_color(:green, "\nProbitRegression\n\n")
y = Float64[rand(Bernoulli(1 / (1 + exp(-η)))) for η in x*β]
o = GLMPath(x, y, model = ProbitRegression(), penalty = L1Penalty(), λs = collect(.01:.01:.04))
fista!(o)
@time fista!(o)
@display o
@display o.β


print_with_color(:green, "\nPoissonRegression\n\n")
y = Float64[rand(Poisson(exp(η))) for η in x*β]
o = GLMPath(x, y, model = PoissonRegression(), penalty = L1Penalty(), λs = collect(.01:.01:.04))
# fista!(o)
@time fista!(o, stepsize = .1)
@display o
@display o.β
