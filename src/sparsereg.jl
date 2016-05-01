immutable SparseReg{M <: Model, P <: Penalty}
    β0::VecF                # intercepts
    β::MatF                 # coefficients
    intercept::Bool         # should intercept be estimated?
    model::M                # Model
    penalty::P              # regularization
    penalty_factor::VecF    # weights for coefficients
    λ::VecF                 # regularization parameters
    crit::Symbol            # criteria for convergence (:obj or :coef)
end
function SparseReg(
        p::Integer;
        intercept::Bool         = true,
        model::Model            = L2Regression(),
        penalty::Penalty        = NoPenalty(),
        penalty_factor::AVecF   = ones(p),
        lambda::AVecF           = .1:.1:1.0,
        crit::Symbol            = :coef
    )
    if typeof(penalty) == NoPenalty
        info("NoPenalty: Setting lambda = [0.0]")
        lambda = zeros(1)
    end
    d = length(lambda)
    @assert all(lambda .>= 0)           "`lambda` values must be nonnegative"
    @assert all(diff(lambda) .> 0)      "`lambda` must be an increasing vector"
    @assert length(penalty_factor) == p "`penalty_factor` must have length $p"
    @assert all(penalty_factor .>= 0)   "`penalty_factor` values must be nonnegative"

    SparseReg(
        zeros(d), zeros(p, d), intercept, model, penalty, collect(penalty_factor),
        collect(lambda), crit
    )
end
function SparseReg(x::AMatF, y::AVecF;
        intercept::Bool         = true,
        model::Model            = L2Regression(),
        penalty::Penalty        = NoPenalty(),
        penalty_factor::AVecF   = ones(size(x, 2)),
        lambda::AVecF           = .1:.1:1.0,
        crit::Symbol            = :coef,
        alg::Algorithm          = default_algorithm(model, penalty),
        algkw...
    )
    o = SparseReg(size(x, 2), intercept = intercept, model = model, penalty = penalty,
        penalty_factor = penalty_factor, lambda = lambda, crit = crit)
    fit!(o, alg, x, y; algkw...)
end

default_algorithm(::Model, ::Penalty) = FISTA()
# default_algorithm(::L2Regression, ::NoPenalty) = Sweep()



function Base.show(io::IO, o::SparseReg)
    print_header(io, "SparseReg")
    print_item(io, "Model", o.model)
    print_item(io, "Penalty", o.penalty)
    print_item(io, "Intercept", o.intercept)
    print_item(io, "nλ", "$(length(o.λ))")
end

coef(o::SparseReg) = o.intercept? vcat(o.β0', o.β) : o.β

function coef(o::SparseReg, λ::Real)
    ff = findfirst(o.λ, λ)
    ff == 0 ?
        error("Coefficients not available for unfitted λ = $λ") :
        vcat(o.β0[ff], o.β[:, ff])
end

function predict(o::SparseReg, x::Matrix, λ::Real = o.λ[1])
    ff = findfirst(o.λ, λ)
    storage = zeros(size(x, 1))
    ff == 0 ?
        error("Prediction not available for unfitted λ = $λ") :
        predict!(o.model, storage, x * o.β[:, ff] + o.β0[ff])
    storage
end

Base.copy(o::SparseReg) = deepcopy(o)
