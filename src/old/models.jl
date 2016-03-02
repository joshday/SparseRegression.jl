abstract StatLearnModel


# immutable QuantileRegression <: StatLearnModel τ::Float64 end
# predict!(m::QuantileRegression, storage::VecF, η::VecF) = copy!(storage, η)
# predict(m::QuantileRegression, η::VecF) = η

#---------------------------------------------------------# Generalized Linear Models
abstract GLMModel <: StatLearnModel
function gradient!(m::GLMModel, Δ::VecF, x::MatF, resids::VecF, η::VecF)
    BLAS.gemv!('T', -1.0, x, resids, 0.0, Δ)
end
function weightedgradient!(m::GLMModel, Δ::VecF, x::MatF, resids::VecF, η::VecF, wts::VecF)
    BLAS.gemv!('T', -1.0, x, resids, 0.0, Δ)
    for j in eachindex(Δ)
        Δ[j] *= wts[j]
    end
end
# loss vector, proportional to negative loglikelihood: log f(y | β) ∝ y * Θ + b(Θ)
function lossvector!(m::GLMModel, storage::VecF, y::VecF, η::VecF)
    @inbounds for j in eachindex(storage)
        Θ = _Θ(m, η[j])
        storage[j] = -y[j] * Θ + _b(m, Θ)
    end
end
function weightedlossvector!(m::GLMModel, storage::VecF, y::VecF, η::VecF, wts::VecF)
    @inbounds for j in eachindex(storage)
        Θ = _Θ(m, η[j])
        storage[j] = (-y[j] * Θ + _b(m, Θ)) * wts[j]
    end
end

# L2Regression
immutable L2Regression <: GLMModel end
predict!(m::L2Regression, storage::VecF, η::VecF) = copy!(storage, η)
predict(m::L2Regression, η::VecF) = η
_Θ(m::L2Regression, η::Float64) = η
_b(m::L2Regression, Θ::Float64) = 0.5 * Θ * Θ

# LogisticRegression
immutable LogisticRegression <: GLMModel end
function  predict!(m::LogisticRegression, storage::VecF, η::VecF)
    for i in eachindex(storage)
        @inbounds storage[i] = 1.0 / (1.0 + exp(-η[i]))
    end
end
predict(m::LogisticRegression, η::VecF) = 1.0 ./ (1.0 + exp(-η))
_Θ(m::LogisticRegression, η::Float64) = η
_b(m::LogisticRegression, Θ::Float64) = log(1.0 + exp(Θ))

# ProbitRegression
immutable ProbitRegression <: GLMModel end
function predict!(m::ProbitRegression, storage::VecF, η::VecF)
    for i in eachindex(storage)
        @inbounds storage[i] = Distributions.cdf(Distributions.Normal(), η[i])
    end
end
predict(m::ProbitRegression, η::VecF) = Distributions.cdf(Distributions.Normal(), η)
_Θ(m::ProbitRegression, η::Float64) = Distributions.logcdf(Distributions.Normal(), η)
_b(m::ProbitRegression, Θ::Float64) = log(1.0 + exp(Θ))

# PoissonRegression
immutable PoissonRegression <: GLMModel end
function predict!(m::PoissonRegression, storage::VecF, η::VecF)
    for i in eachindex(storage)
        @inbounds storage[i] = exp(η[i])
    end
end
predict(m::PoissonRegression, η::VecF) = exp(η)
_Θ(m::PoissonRegression, η::Float64) = η
_b(m::PoissonRegression, Θ::Float64) = exp(Θ)
