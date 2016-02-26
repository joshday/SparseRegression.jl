#-----------------------------------------------------------------------------# Model
immutable L1Regression          <: Model            end
immutable L2Regression          <: Model            end
immutable LogisticRegression    <: Model            end
immutable ProbitRegression      <: Model            end
immutable PoissonRegression     <: Model            end
immutable SVMLike               <: Model            end
immutable QuantileRegression    <: Model τ::Float64 end
immutable HuberRegression       <: Model δ::Float64 end

# Base.show
Base.show(io::IO, o::L1Regression)          = print(io, "L1Regression")
Base.show(io::IO, o::L2Regression)          = print(io, "L2Regression")
Base.show(io::IO, o::LogisticRegression)    = print(io, "LogisticRegression")
Base.show(io::IO, o::ProbitRegression)      = print(io, "ProbitRegression")
Base.show(io::IO, o::PoissonRegression)     = print(io, "PoissonRegression")
Base.show(io::IO, o::SVMLike)               = print(io, "SVMLike")
Base.show(io::IO, o::QuantileRegression)    = print(io, "QuantileRegression($(o.τ))")
Base.show(io::IO, o::HuberRegression)       = print(io, "HuberRegression($(o.δ))")

# predict
predict(m::L1Regression,        η::Float64) = η
predict(m::L2Regression,        η::Float64) = η
predict(m::LogisticRegression,  η::Float64) = 1.0 / (1.0 + exp(-η))
predict(m::ProbitRegression,    η::Float64) = Distributions.cdf(Distributions.Normal(), η)
predict(m::PoissonRegression,   η::Float64) = exp(η)
predict(m::SVMLike,             η::Float64) = η
predict(m::QuantileRegression,  η::Float64) = η
predict(m::HuberRegression,     η::Float64) = η
predict{T <: Real}(m::Model,    η::Vector{T}) = [predict(m, ηi) for ηi in η]

# lossvector
lossvector(m::L1Regression, y, η) = abs(y - η)
lossvector(m::L2Regression, y, η) = abs2(y - η)
lossvector(m::LogisticRegression, y, η) = -y .* η + log(1.0 + exp(η))
function lossvector(m::ProbitRegression, y, η)
    probs = predict(m, η)
    y .* log(probs) + (1.0 - y) .* log(1.0 - probs)
end
lossvector(m::SVMLike, y, η) = [max(0.0, 1.0 - y[i] * η[i]) for i in 1:length(y)]
function lossvector(m::QuantileRegression, y, η)
    [(y[i] - η[i]) * (m.τ - (y[i] < η[i])) for i in 1:length(y)]
end
function lossvector(m::HuberRegression, y, η)
    [
        abs(y[i]-η[i]) < m.δ ? 0.5 * (y[i]-η[i])^2 : m.δ * (abs(y[i]-η[i]) - 0.5 * m.δ)
        for i in 1:length(y)
    ]
end

# deriv
deriv(m::L1Regression, resids, x, y, ŷ, η) =
    x' * abs(resids)
deriv(m::L2Regression, resids, x, y, ŷ, η) =
    x' * resids
deriv(m::LogisticRegression, resids, x, y, ŷ, η) =
    x' * resids
deriv(m::ProbitRegression, resids, x, y, ŷ, η) =
    x' * (resids .* Distributions.pdf(Distributions.Normal(), η))
deriv(m::PoissonRegression, resids, x, y, ŷ, η) =
    x' * resids
deriv(m::QuantileRegression, resids, x, y, ŷ, η) =
    x' * (resids .< 0 - m.τ)
deriv(m::SVMLike, resids, x, y, ŷ, η) =
    x' * [y[i] * ŷ[i] < 1 ? -y[i] : 0.0 for i in 1:length(y)]
deriv(m::HuberRegression, resids, x, y, ŷ, η) =
    x' * [abs(resids[i]) <= m.δ ? -resids[i] : m.δ * sign(-resids[i]) for i in 1:length(y)]


#---------------------------------------------------------------------# StatLearnPath
type StatLearnPath{
        M <: Model,
        P <: Penalty,
        T <: Real
    }
    model::M
    penalty::P                  # regularization term
    β0::VecF                    # intercept terms
    β::MatF                     # coefficients
    intercept::Bool             # should intercept be estimated?
    x::Matrix{T}                # design matrix
    y::Vector{T}                # response vector
    η::VecF                     # linear predictor
    ŷ::VecF                     # predicted values
    resids::VecF                # residual vector
    weights::VecF               # weights
    lambdas::VecF               # regularization parameters
end
function StatLearnPath(x::Matrix, y::Vector;
        model::Model = L2Regression(),
        penalty::Penalty = NoPenalty(),
        intercept::Bool = true,
        weights::VecF = ones(0),
        lambdas::VecF = zeros(1)
    )
    n, p = size(x)
    @assert length(y) == n
    nλ = length(lambdas)
    o = StatLearnPath(
        model,
        penalty,
        zeros(nλ), zeros(p, nλ),        # β0 and β
        intercept,
        x, y,                           # x and y
        zeros(n), zeros(n), zeros(n),   # η, ŷ, and resids
        weights,
        lambdas
    )
    fit!(o, 1)
end
function Base.show(io::IO, o::StatLearnPath)
    print_header(io, "StatLearnPath")
    print_item(io, "Model", o.model)
    print_item(io, "Penalty", o.penalty)
end

# predict(o::StatLearnPath, d::Integer = 1) = predict(o.model, o.x * o.β[:, d] + o.β0[d])

cost(o::StatLearnPath, β, y, η) = mean(lossvector(o.model, y, η)) + penalty(o.penalty, λ, β)


#----------------------------------------------------------# residuals and derivative
function res_and_deriv(o::StatLearnPath{L2Regression})
    o.η = o.x * o.β
end


#--------------------------------------------------------------------# main algorithm
# FISTA: Fast Iterative Shrinkage and Threshold Algorithm
# http://www.seas.ucla.edu/~vandenbe/236C/lectures/fgrad.pdf


# get solution for λ = lambdas[d]
function fit!(o::StatLearnPath, d::Int;
        maxit::Int = 100,
        eps::Real = 1e-4,
        verbose::Bool = true
    )

    # setup
    n, p = size(o.x)
    λ = o.lambdas[d]
    # usewts = (n == length(o.weights))
    newcost = Inf
    s = 1.0             # step size for FISTA
    iters = 0
    β = zeros(p)
    β1 = zeros(p)       # last iteration
    β2 = zeros(p)       # two iterations ago

    # main loop
    for k in 1:maxit
        iters += 1
        oldcost = newcost
        copy!(β2, β1)
        copy!(β1, β)

        o.η = o.x * β
        o.ŷ = predict(o.model, o.η)
        o.resids = y - o.ŷ

        if k > 2
            step = (k - 2) / (k + 1)
            for j in 1:p
                β[j] = β[j] + step * (β1[j] - β2[j])
            end
        end
        ∇f = deriv(o.model, o.resids, o.x, o.y, o.ŷ, o.η)
        β += s * ∇f / n
        prox!(o.penalty, λ, β, s)
        if o.intercept
            o.β0[d] += mean(o.resids)
        end

        newcost = mean(lossvector(o, resids)) + penalty(o.penalty, λ, o.β[:, ki], s)

        # check for convergence
        if abs(newcost - oldcost) < eps * abs(oldcost + 1.0)
            verbose && println("converged in $iters iterations")
            break
        end
    end
    o.β[:, d] = β
    iters == maxit && print_with_color(:red, "Did NOT converge in $iters iterations \n")
    o
end



# Test
import GLMNet
n, p = 10000, 3000
x = randn(n, p) * 2
β = collect(linspace(-.5, .5, p))
y = x * β + randn(n)


print_with_color(:red, "Linear Regression\n\n")
@time o = StatLearnPath(x, y, model = L2Regression(), lambdas = [.1])
@display o
@time o2 = GLMNet.glmnet(x, y, lambda = [.1])


print_with_color(:red, "Logistic Regression\n\n")
y = Float64[rand(Distributions.Bernoulli(1 / (1 + exp(-η)))) for η in x*β]
@time o = StatLearnPath(x, y, model = LogisticRegression(), lambdas = [.1])
@display o

y1 = y
y2 = y .== 0
ys = hcat(y1, y2)
@time o2 = GLMNet.glmnet(x, ys, Distributions.Binomial(), lambda = [.1])
