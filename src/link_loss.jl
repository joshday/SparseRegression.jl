# ============================================================================== Link
abstract Link

immutable IdentityLink <: Link end
predict!(l::IdentityLink, storage::VecF, η::VecF) = copy!(storage, η)

immutable LogLink <: Link end
function predict!(l::LogLink, storage::VecF, η::VecF)
    for i in eachindex(storage)
        @inbounds storage[i] = exp(η[i])
    end
end


# ============================================================================== Loss
abstract Loss
# ------------------------------------------------------------------------- ErrorLoss
# Loss in which value and deriv depend on r = y - ŷ
abstract ErrorLoss <: Loss
function lossvector!(l::ErrorLoss, lossvec::VecF, resids::VecF)
    for i in eachindex(lossvec)
        @inbounds lossvec[i] = value(l, resids[i])
    end
end

# f(residual) = .5 * residual ^ 2
immutable SquaredErrorLoss <: ErrorLoss end
value(o::SquaredErrorLoss, r::Float64) = 0.5 * r * r
deriv(o::SquaredErrorLoss, r::Float64) = r

# f(residual) = abs(residual)
immutable AbsoluteErrorLoss <: ErrorLoss end
value(o::AbsoluteErrorLoss, r::Float64) = 0.5 * abs(r)
deriv(o::AbsoluteErrorLoss, r::Float64) = sign(r)

# f(residual) = residual * (τ - (residual < 0))
immutable QuantileErrorLoss <: ErrorLoss τ::Float64 end
value(o::QuantileErrorLoss, r::Float64) = r * (o.τ - Float64(r < 0.0))
deriv(o::QuantileErrorLoss, r::Float64) = o.τ - Float64(r < 0.0)

# --------------------------------------------------------------------- AgreementLoss
# Loss in which value and deriv depend on yt = y * ŷ
abstract AgreementLoss <: Loss
function lossvector!(l::ErrorLoss, lossvec::VecF, yt::VecF)
    for i in eachindex(lossvec)
        lossvec[i] = value(l, yt[i])
    end
end

# f(yt) = max(0, 1 - yt)
immutable HingeLoss <: AgreementLoss end
value(o::HingeLoss, yt::Float64) = max(0.0, 1.0 - yt)
deriv(o::HingeLoss, yt::Float64) = 1.0 < yt ? 0.0 : -1.0

# f(yt) = log(1 + exp(-yt))
# LogisticLoss with IdentityLink has same gradient as logreg negative loglikelihood
immutable LogisticLoss <: AgreementLoss end
value(o::LogisticLoss, yt::Float64) = log(1.0 + exp(-yt))
deriv(o::LogisticLoss, yt::Float64) = -1.0 / (1.0 + exp(yt))
