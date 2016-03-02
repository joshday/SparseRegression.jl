# ===================================================================== StatLearnPath
immutable StatLearnPath{M <: Model, P <: Penalty}
    β0::VecF            # intercepts
    β::MatF             # coefficients
    intercept::Bool     # should intercept be estimated?
    model::M            # Model
    penalty::P          # regularization
    x::MatF             # design matrix
    μx::VecF            # column means of x
    σx::VecF            # column stds of x
    y::VecF             # response vector
    weights::VecF       # weights
    λs::VecF            # regularization parameters
end
function StatLearnPath(x::MatF, y::VecF;
        intercept::Bool = true,
        model::Model = L2Regression(),
        penalty::Penalty = NoPenalty(),
        weights::VecF = ones(0),
        lambdas::AVecF = zeros(1),
        standardize::Bool = true,
        algkw...
    )
    n, p = size(x)
    if typeof(penalty) == NoPenalty && length(lambdas) > 1
        info("NoPenalty: Setting lambdas = zeros(1)")
    else
        lambdas = lambda_check(lambdas)
    end
    lambdas = lambda_check(lambdas)
    d = length(lambdas)
    @assert length(y) == n "size(x, 1) != length(y)"
    μx = mean(x, 1)
    σx = std(x, 1)
    o = StatLearnPath(
        zeros(d),
        zeros(p, d),
        intercept, model, penalty,
        _standardize(standardize, x, μx, σx), vec(μx), vec(σx), y,
        weights, lambdas
    )
    fit!(o; algkw...)
    if standardize
        scaled_to_original!(o)
    end
    o
end
function Base.show(io::IO, o::StatLearnPath)
    print_header(io, "StatLearnPath")
    print_item(io, "Model", o.model)
    print_item(io, "Penalty", o.penalty)
    print_item(io, "Intercept", o.intercept)
    print_item(io, "λs", "$(length(o.λs))")
end
function _standardize(stdz::Bool, x::MatF, μx::MatF, σx::MatF)
    if !stdz
        return x
    else
        return StatsBase.zscore(x, μx, σx)
    end
end
# ensure lambdas are increasing
function lambda_check(lambdas::AVecF)
    for j in 1:length(lambdas) - 1
        @assert lambdas[j] < lambdas[j + 1]
    end
    collect(lambdas)
end
# Get coefficients in terms of the original predictors
function scaled_to_original!(o::StatLearnPath)
    p, d = size(o.β)
    scale!(1. ./ o.σx, o.β)
    if o.intercept
        for j in 1:d
            o.β[j] = o.β[j] - dot(o.μx, o.β[:, j])
        end
    end
end
StatsBase.coef(o::StatLearnPath) = vcat(o.β0', o.β)
