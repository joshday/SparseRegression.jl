abstract Model
function Base.show(io::IO, m::Model)
    s = string(typeof(m))
    s = replace(s, "SparseRegression.", "")
    print(s)
end

# =====================================================================# LinPredModel
# Models with prediction f(η) where η = Xβ
abstract LinPredModel <: Model
abstract BivariateLinPredModel <: LinPredModel
function lossvector!(m::LinPredModel, storage::VecF, y::VecF, η::VecF)
    for i in eachindex(y)
        @inbounds storage[i] = loss(m, y[i], η[i])
    end
end
function predict!(m::LinPredModel, storage::VecF, η::VecF)
    for i in eachindex(η)
        @inbounds storage[i] = predict(m, η[i])
    end
end
function classify!(m::BivariateLinPredModel, storage::VecF, η::VecF)
    for i in eachindex(η)
        @inbounds storage[i] = classify(m, η[i])
    end
end
maxlambda(m::Model, x::MatF, y::VecF) = maxlambda(L2Regression(), x, y)




#----------------------------------------------------------------------# L2Regression
immutable L2Regression <: LinPredModel end
loss(m::L2Regression, y::Float64, η::Float64) = 0.5 * (y - η) ^ 2
lossderiv(m::L2Regression, y::Float64, η::Float64) = -(y - η)
predict(m::L2Regression, η::Float64) = η
function maxlambda(m::L2Regression, x::MatF, y::VecF)
    n = length(y)
    bias = (n - 1) / n
    maximum(abs(StatsBase.zscore(x)'StatsBase.zscore(y) * bias)) / length(y)
end

#----------------------------------------------------------------------# L1Regression
immutable L1Regression <: LinPredModel end
loss(m::L1Regression, y::Float64, η::Float64) = abs(y - η)
lossderiv(m::L1Regression, y::Float64, η::Float64) = -sign(y - η)
predict(m::L1Regression, η::Float64) = η

#----------------------------------------------------------------# LogisticRegression
immutable LogisticRegression <: BivariateLinPredModel end
loss(m::LogisticRegression, y::Float64, η::Float64) = log(1.0 + exp(-y * η))
lossderiv(m::LogisticRegression, y::Float64, η::Float64) = -y / (1.0 + exp(y * η))
predict(m::LogisticRegression, η::Float64) = 1.0 / (1.0 + exp(η))
classify(m::LogisticRegression, η::Float64) = sign(η)

#----------------------------------------------------------------# PoissonRegression
immutable PoissonRegression <: LinPredModel end
loss(m::PoissonRegression, y::Float64, η::Float64) = -y * η + exp(η)
lossderiv(m::PoissonRegression, y::Float64, η::Float64) = -y + exp(η)
predict(m::PoissonRegression, η::Float64) = exp(η)

#---------------------------------------------------------------------------# SVMLike
immutable SVMLike <: BivariateLinPredModel end
loss(m::SVMLike, y::Float64, η::Float64) = max(0.0, 1.0 - y * η)
lossderiv(m::SVMLike, y::Float64, η::Float64) = 1.0 < y*η ? 0.0: -y
predict(m::SVMLike, η::Float64) = η
classify(m::SVMLike, η::Float64) = sign(η)

#----------------------------------------------------------------# QuantileRegression
immutable QuantileRegression <: LinPredModel τ::Float64 end
function loss(m::QuantileRegression, y::Float64, η::Float64)
    r = y - η
    r * (m.τ - (r < 0.0))
end
lossderiv(m::QuantileRegression, y::Float64, η::Float64) = (y - η < 0.0) - m.τ
predict(m::QuantileRegression, η::Float64) = η

#-------------------------------------------------------------------# HuberRegression
immutable HuberRegression <: LinPredModel δ::Float64 end
function loss(m::HuberRegression, y::Float64, η::Float64)
    r = y - η
    r < m.δ ? 0.5 * r * r : m.δ * (abs(r) - 0.5 * m.δ)
end
function lossderiv(m::HuberRegression, y::Float64, η::Float64)
    r = y - η
    abs(r) <= m.δ ? -r : m.δ * sign(-r)
end
predict(m::HuberRegression, η::Float64) = η
