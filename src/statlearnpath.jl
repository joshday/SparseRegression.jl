# TODO: put βs back into original scale
# ===================================================================== StatLearnPath
immutable StatLearnPath{Lo <: Loss, Li <: Link, P <: Penalty}
    β0::VecF            # intercepts
    β::MatF             # coefficients
    intercept::Bool     # should intercept be estimated?
    link::Li            # Link g(y) = x*β
    loss::Lo            # Loss function
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
        link::Link = IdentityLink(),
        loss::Loss = SquaredErrorLoss(),
        penalty::Penalty = NoPenalty(),
        weights::VecF = ones(0),
        lambdas::AVecF = zeros(1),
        standardize::Bool = true,
        algkw...
    )
    n, p = size(x)
    lambdas = lambda_check(lambdas)
    d = length(lambdas)
    @assert length(y) == n "size(x, 1) != length(y)"
    μx = mean(x, 1)
    σx = std(x, 1)
    o = StatLearnPath(
        zeros(d),
        zeros(p, d),
        intercept, link, loss, penalty,
        _standardize(standardize, x, μx, σx), vec(μx), vec(σx), y,
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
StatsBase.coef(o::StatLearnPath) = vcat(o.β0', o.β)
