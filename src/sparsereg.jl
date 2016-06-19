# TODO: Add N parameter for dimension of response (allow classification)
"""
# Notes:
### Two constructors:
- `SparseReg(p; kw...)`
- `SparseReg(x, y, wts; kw...)`
"""
immutable SparseReg{M <: Model, P <: Penalty, A <: Algorithm}
    β0::VecF                # intercepts
    β::MatF                 # coefficients
    intercept::Bool         # should intercept be estimated?
    model::M                # Model
    penalty::P              # regularization
    penalty_factor::VecF    # weights for coefficients
    λ::VecF                 # regularization parameters
    algorithm::A            # Algorithm stores convergence criteria, tolerance, etc.
end
function _SparseReg(p::Integer;
        intercept::Bool         = true,
        model::Model            = LinearRegression(),
        penalty::Penalty        = NoPenalty(),
        penalty_factor::AVecF   = ones(p),
        lambda::AVecF           = 0.0 : 0.1 : 1.0,
        algorithm::Algorithm    = default_algorithm(model, penalty)
    )
    if typeof(penalty) == NoPenalty
        lambda = zeros(1)
    end
    d = length(lambda)
    @assert all(lambda .>= 0)           "`lambda` values must be nonnegative"
    @assert all(diff(lambda) .> 0)      "`lambda` must be an increasing vector"
    @assert length(penalty_factor) == p "`penalty_factor` must have length $p"
    @assert all(penalty_factor .>= 0)   "`penalty_factor` values must be nonnegative"

    SparseReg(
        zeros(d), zeros(p, d), intercept, model, penalty, collect(penalty_factor),
        VecF(collect(lambda)), algorithm
    )
end
function SparseReg(p::Integer, args...; kw...)
    mod = LinearRegression()
    pen = NoPenalty()
    alg = Fista()
    for arg in args
        T = typeof(arg)
        if T <: Model
            mod = arg
        elseif T <: Algorithm
            alg = arg
        elseif T <: Penalty
            pen = arg
        end
    end
    _SparseReg(p; model = mod, algorithm = alg, penalty = pen, kw...)
end
function SparseReg(x::AMat, y::AVec, args...; kw...)
    o = SparseReg(size(x, 2), args...; kw...)
    fit!(o, x, y, ones(0))
end
function SparseReg(x::AMat, y::AVec, wts::AVec, args...; kw...)
    o = SparseReg(size(x, 2), args...; kw...)
    fit!(o, x, y, wts)
end

function Base.show(io::IO, o::SparseReg)
    print_header(io, "SparseReg")
    print_item(io, "Model", o.model)
    print_item(io, "Penalty", o.penalty)
    print_item(io, "Intercept", o.intercept)
    print_item(io, "nλ", "$(length(o.λ))")
    print_item(io, "Algorithm", o.algorithm)
end
default_algorithm(::Model, ::Penalty) = Fista()
default_algorithm(::LinearRegression, ::NoPenalty) = Sweep()

#---------------------------------------------------------------------------# Methods
coef(o::SparseReg) = o.intercept? vcat(o.β0', o.β) : o.β

function λindex(o::SparseReg, λ::Real)
    # find i such that o.λ[i] == λ
    i = findfirst(o.λ, λ)
    i == 0 && error("Provided λ = $λ has not been fit")
    i
end

function coef(o::SparseReg, λ::Real)
    i = λindex(o, λ)
    vcat(o.β0[i], o.β[:, i])
end

function predict(o::SparseReg, x::AMat, λ::Real = o.λ[1])
    i = λindex(o, λ)
    storage = zeros(size(x, 1))
    predict!(o.model, storage, x * o.β[:, i] + o.β0[i])
    storage
end

function classify{M <: BivariateModel}(o::SparseReg{M}, x::AMat, λ::Real = o.λ[1])
    i = λindex(o, λ)
    storage = zeros(size(x, 1))
    classify!(o.model, storage, x * o.β[:, i] + o.β0[i])
    storage
end

function loss(o::SparseReg, x::AMat, y::AVec, λ::Real = o.λ[1])
    i = λindex(o, λ)
    storage = zeros(length(y))
    lossvector!(o.model, storage, y, o.β0[i] + x * o.β[:, i])
    mean(storage)
end

function cost(o::SparseReg, x::AbstractArray, y::AbstractArray, λ::Real = o.λ[1])
    i = λindex(o, λ)
    loss(o, x, y, λ) + penalty(o.penalty, o.β[:, i], λ)
end
