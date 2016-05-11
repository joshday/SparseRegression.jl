abstract Model
abstract BivariateModel <: Model  # LogisticRegression and SVMLike
function Base.show(io::IO, m::Model)
    s = string(typeof(m))
    s = replace(s, "SparseRegression.", "")
    print(s)
end

# =====================================================================# Model
function lossvector!(m::Model, storage::VecF, y::VecF, η::VecF)
    for i in eachindex(y)
        @inbounds storage[i] = loss(m, y[i], η[i])
    end
end
function loglikelihood!(m::Model, storage::VecF, y::VecF, η::VecF)
    for i in eachindex(y)
        @inbounds storage[i] = loglikelihood(m, y[i], η[i])
    end
end
function predict!(m::Model, storage::VecF, η::VecF)
    for i in eachindex(η)
        @inbounds storage[i] = predict(m, η[i])
    end
end
function classify!(m::BivariateModel, storage::VecF, η::VecF)
    for i in eachindex(η)
        @inbounds storage[i] = classify(m, η[i])
    end
end
maxlambda(m::Model, x::AMat, y::AVec) = maxlambda(L2Regression(), x, y)




#----------------------------------------------------------------------# L2Regression
immutable L2Regression <: Model end
loss(m::L2Regression, y::Real, η::Real) = 0.5 * (y - η) ^ 2
loglikelihood(m::L2Regression, y::Real, η::Real) = -loss(m, y, η)
lossderiv(m::L2Regression, y::Real, η::Real) = -(y - η)
predict(m::L2Regression, η::Real) = η
function maxlambda(m::L2Regression, x::AMat, y::AVec)
    n = length(y)
    bias = (n - 1) / n
    maximum(abs(StatsBase.zscore(x)' * y * bias)) / length(y)
end

#----------------------------------------------------------------------# L1Regression
immutable L1Regression <: Model end
loss(m::L1Regression, y::Real, η::Real) = abs(y - η)
lossderiv(m::L1Regression, y::Real, η::Real) = -sign(y - η)
predict(m::L1Regression, η::Real) = η

#----------------------------------------------------------------# LogisticRegression
"For data in {0, 1}"
immutable LogisticRegression <: BivariateModel end
loss(m::LogisticRegression, y::Real, η::Real) = -y * η + log(1.0 + exp(η))
loglikelihood(m::LogisticRegression, y::Real, η::Real) = -loss(m, y, η)
lossderiv(m::LogisticRegression, y::Real, η::Real) = -(y - predict(m, η))
predict(m::LogisticRegression, η::Real) = 1.0 / (1.0 + exp(-η))
classify(m::LogisticRegression, η::Real) = Float64(predict(m, η) > .5)

#----------------------------------------------------------------# PoissonRegression
immutable PoissonRegression <: Model end
loss(m::PoissonRegression, y::Real, η::Real) = -y * η + exp(η)
loglikelihood(m::LogisticRegression, y::Real, η::Real) = -loss(m, y, η)
lossderiv(m::PoissonRegression, y::Real, η::Real) = -y + exp(η)
predict(m::PoissonRegression, η::Real) = exp(η)

#---------------------------------------------------------------------------# SVMLike
"For data in {-1, 1}"
immutable SVMLike <: BivariateModel end
loss(m::SVMLike, y::Real, η::Real) = max(0.0, 1.0 - y * η)
lossderiv(m::SVMLike, y::Real, η::Real) = 1.0 < y*η ? 0.0: -y
predict(m::SVMLike, η::Real) = η
classify(m::SVMLike, η::Real) = sign(η)

#----------------------------------------------------------------# QuantileRegression
immutable QuantileRegression <: Model τ::Float64 end
function loss(m::QuantileRegression, y::Real, η::Real)
    r = y - η
    r * (m.τ - (r < 0.0))
end
lossderiv(m::QuantileRegression, y::Real, η::Real) = (y - η < 0.0) - m.τ
predict(m::QuantileRegression, η::Real) = η

#-------------------------------------------------------------------# HuberRegression
immutable HuberRegression <: Model δ::Float64 end
function loss(m::HuberRegression, y::Real, η::Real)
    r = y - η
    r < m.δ ? 0.5 * r * r : m.δ * (abs(r) - 0.5 * m.δ)
end
function lossderiv(m::HuberRegression, y::Real, η::Real)
    r = y - η
    abs(r) <= m.δ ? -r : m.δ * sign(-r)
end
predict(m::HuberRegression, η::Real) = η
