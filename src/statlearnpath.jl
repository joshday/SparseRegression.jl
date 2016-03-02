abstract Link

immutable IdentityLink <: Link end
predict!{T <: Real}(l::IdentityLink, storage::Vector{T}, η::Vector{T}) = copy!(storage, η)





# ============================================================================== Loss
abstract Loss
# ------------------------------------------------------------------------- ErrorLoss
# Loss in which value and deriv depend on r = y - ŷ
abstract ErrorLoss <: Loss
function lossvector!(l::ErrorLoss, lossvec::Vector, resids::Vector)
    for i in eachindex(lossvec)
        lossvec[i] = value(l, resids[i])
    end
end

# f(residual) = .5 * residual ^ 2
immutable SquaredErrorLoss <: ErrorLoss end
value{T<:Number}(o::SquaredErrorLoss, r::T) = T(1 // 2) * r * r
deriv{T<:Number}(o::SquaredErrorLoss, r::T) = r

# f(residual) = |residual|
immutable AbsoluteErrorLoss <: ErrorLoss end
value{T<:Number}(o::AbsoluteErrorLoss, r::T) = abs(r)
deriv{T<:Number}(o::AbsoluteErrorLoss, r::T) = sign(r)

# f(residual) = (residual < 0) - τ
immutable QuantileErrorLoss{T <: Real} <: ErrorLoss τ::T end
value{T<:Number}(o::QuantileErrorLoss, r::T) = r * (T(o.τ) - T(r < zero(r)))
deriv{T<:Number}(o::QuantileErrorLoss, r::T) = T(r < zero(r)) - T(o.τ)

# --------------------------------------------------------------------- AgreementLoss
# Loss in which value and deriv depend on yt = y * ŷ
abstract AgreementLoss <: Loss

# f(yt) = max(0, 1 - yt)
immutable HingeLoss <: AgreementLoss end
value{T<:Number}(o::HingeLoss, yt::T) = max(zero(yt), one(yt) - yt)
deriv{T<:Number}(o::HingeLoss, yt::T) = one(yt) - yt < zero(yt) ? zero(yt) : -one(yt)

# f(yt) = log(1 + exp(-yt))
immutable LogisticLoss <: AgreementLoss end
value{T<:Number}(o::LogisticLoss, yt::T) = log(one(yt) + exp(-yt))
deriv{T<:Number}(o::LogisticLoss, yt::T) = - one(yt) / (one(yt) + exp(yt))


# ===================================================================== StatLearnPath
immutable StatLearnPath{Lo <: Loss, Li <: Link, P <: Penalty, T <: Real}
    β0::Vector{T}       # intercepts
    β::Matrix{T}        # coefficients
    intercept::Bool     # should intercept be estimated?
    link::Li            # Link g(y) = x*β
    loss::Lo            # Loss function
    penalty::P          # regularization
    x::Matrix{T}        # design matrix
    y::Vector{T}        # response vector
    weights::Vector{T}  # weights
    λs::Vector{T}       # regularization parameters
end
function StatLearnPath{T <: Real}(x::Matrix{T}, y::Vector{T};
        intercept::Bool = true,
        link::Link = IdentityLink(),
        loss::Loss = SquaredErrorLoss(),
        penalty::Penalty = NoPenalty(),
        weights = ones(T, 0),
        lambdas = zeros(T, 1),
        standardize::Bool = true,
        algkw...
    )
    n, p = size(x)
    d = length(lambdas)
    @assert length(y) == n "size(x, 1) != length(y)"
    o = StatLearnPath(
        zeros(T, d),
        zeros(T, p, d),
        intercept, link, loss, penalty,
        _standardize(standardize, x), y,
        weights, lambdas
    )
    fit!(o; algkw...)
    o
end
function Base.show(io::IO, o::StatLearnPath)
    print_header(io, "StatLearnPath")
    print_item(io, "Link", o.link)
    print_item(io, "Loss", o.loss)
    print_item(io, "Penalty", o.penalty)
    print_item(io, "Intercept", o.intercept)
    print_item(io, "λs", "$(length(o.λs))")
end

function _standardize{T <: Real}(stdz::Bool, x::Matrix{T})
    if !stdz
        return x
    else
        return StatsBase.zscore(x)
    end
end
