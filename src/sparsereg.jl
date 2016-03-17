# ========================================================================= SparseReg
immutable SparseReg{M <: LinPredModel, P <: Penalty}
    β0::VecF                # intercepts
    β::MatF                 # coefficients
    intercept::Bool         # should intercept be estimated?
    model::M                # Model
    penalty::P              # regularization
    penalty_factor::VecF    # weights for coefficients
    x::MatF                 # design matrix (standardized if standardize=true)
    standardize::Bool       # Is x centered and scaled?
    μx::VecF                # column means of x
    σx::VecF                # column stds of x
    y::VecF                 # response vector
    weights::VecF           # weights
    λs::VecF                # regularization parameters
end
function SparseReg(x::MatF, y::VecF;
        intercept::Bool = true,
        model::Model = L2Regression(),
        penalty::Penalty = NoPenalty(),
        penalty_factor::AVecF = ones(size(x, 2)),
        weights::VecF = ones(0),
        verbose::Bool = true,
        lambda::AVecF = get_lambda(nlambda, model, x, y, verbose),
        nlambda::Integer = 100,
        standardize::Bool = true,
        algorithm::Algorithm = default_alg(model),
        algkw...
    )
    n, p = size(x)
    d = length(lambda)
    # set lambda = [0.0] if NoPenalty
    if typeof(penalty) == NoPenalty && d > 1
        info("NoPenalty: Setting lambda = [0.0]")
        lambda = [0.0]
    end
    @assert all(diff(lambda) .> 0)      "lambda must be an increasing vector"
    @assert length(penalty_factor) == p "`penalty_factor` must have length $p"
    @assert all(penalty_factor .>= 0)   "`penalty_factor` cannot have negative values"
    @assert length(y) == n              "x and y have incompatable dimensions"
    @assert all(weights .>= 0)          "`weights` cannot have negative values"
    μx = mean(x, 1)
    σx = std(x, 1)
    o = SparseReg(
        zeros(d),
        zeros(p, d),
        intercept, model, penalty, collect(penalty_factor),
        x, standardize, vec(μx), vec(σx), y,
        weights, collect(lambda)
    )
    o.standardize && StatsBase.zscore!(o.x, o.μx', o.σx')
    fit!(algorithm, o; verbose = verbose, algkw...)
    o.standardize && scaled_to_original!(o)
    o
end
function Base.show(io::IO, o::SparseReg)
    print_header(io, "SparseReg")
    print_item(io, "Model", o.model)
    print_item(io, "Penalty", o.penalty)
    print_item(io, "Intercept", o.intercept)
    print_item(io, "λs", "$(length(o.λs))")
end
function get_lambda(nlambda::Integer, model::LinPredModel, x::MatF, y::VecF, verbose::Bool)
    maxλ = maxlambda(model, x, y)
    verbose && info("Smallest lambda calculated as: $maxλ")
    collect(linspace(0, maxλ, nlambda))
end
# Get coefficients in terms of the original predictors
function scaled_to_original!(o::SparseReg)
    p, d = size(o.β)
    scale!(1. ./ o.σx, o.β)
    if o.intercept
        for j in 1:d
            o.β[j] = o.β[j] - dot(o.μx, o.β[:, j])
        end
    end
end
StatsBase.coef(o::SparseReg) = vcat(o.β0', o.β)
function StatsBase.coef(o::SparseReg, λ::Real)
    ff = findfirst(o.λs, λ)
    ff == 0 ?
        error("Coefficients not available for unfitted λ = $λ") :
        vcat(o.β0[ff], o.β[:, ff])
end
function StatsBase.predict(o::SparseReg, x::Matrix, λ::Real = o.λs[1])
    ff = findfirst(o.λs, λ)
    storage = zeros(size(x, 1))
    ff == 0 ?
        error("Prediction not available for unfitted λ = $λ") :
        predict!(o.model, storage, x*o.β[:, ff] + o.β0[ff])
end
Base.copy(o::SparseReg) = deepcopy(o)
function null_loss(o::SparseReg)
    lossvec = zeros(length(o.y))
    lossvector!(o.model, lossvec, o.y, zeros(length(o.y)))
    mean(lossvec)
end
