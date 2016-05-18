#---------------------------------------------------------------------------# Penalty
abstract Penalty
immutable NoPenalty             <: Penalty              end
immutable RidgePenalty          <: Penalty              end
immutable LassoPenalty          <: Penalty              end
immutable ElasticNetPenalty     <: Penalty α::Float64   end
immutable SCADPenalty           <: Penalty a::Float64   end


ElasticNetPenalty(α::Real = .5) = ElasticNetPenalty(α)
SCADPenalty(a::Real = 3.7) = SCADPenalty(a)

Base.show(io::IO, p::NoPenalty)         = print(io, "NoPenalty")
Base.show(io::IO, p::RidgePenalty)      = print(io, "RidgePenalty")
Base.show(io::IO, p::LassoPenalty)      = print(io, "LassoPenalty")
Base.show(io::IO, p::ElasticNetPenalty) = print(io, "ElasticNetPenalty (α = $(p.α))")
Base.show(io::IO, p::SCADPenalty)       = print(io, "SCADPenalty (a = $(p.a))")

penalty(p::NoPenalty, β, λ) = 0.0
penalty(p::RidgePenalty, β, λ) = 0.5 * λ * sumabs2(β)
penalty(p::LassoPenalty, β, λ) = λ * sumabs(β)
penalty(p::ElasticNetPenalty, β, λ) = λ * (p.α * sumabs(β) + (1. - p.α) * 0.5 * sumabs2(β))
function penalty(p::SCADPenalty, β, λ)
    val = 0.0
    for j in eachindex(β)
        βj = abs(β[j])
        if βj < λ
            val += λ * βj
        elseif βj < λ * p.a
            val -= 0.5 * (βj ^ 2 - 2.0 * p.a * λ * βj + λ ^ 2) / (p.a - 1.0)
        else
            val += 0.5 * (p.a + 1.0) * λ ^ 2
        end
    end
    return val
end


# Setup for proximal gradient algorithm (FISTA):
# For ℓ(β) = f(β) + g(β),
# βnew = prox_{s * g}(βold - s * ∇f(βold))
prox(p::NoPenalty, βj, λ, s) = βj
prox(p::RidgePenalty, βj, λ, s) = βj / (1.0 + s * λ)
prox(p::LassoPenalty, βj, λ, s) = sign(βj) * max(abs(βj) - s * λ, 0.0)
prox(p::ElasticNetPenalty, βj, λ, s) =
    sign(βj) * max(abs(βj) - s * λ * p.α, 0.0) / (1.0 + s * λ * (1.0 - p.α))
function prox(p::SCADPenalty, βj, λ, s)
    if abs(βj) > p.a * λ
    elseif abs(βj) < 2.0 * λ
        βj = sign(βj) * max(abs(βj) - s * λ, 0.0)
    else
        βj = (βj - s * sign(βj) * p.a * λ / (p.a - 1.0)) / (1.0 - (1.0 / p.a - 1.0))
    end
    βj
end

function prox!(p::Penalty, β::VecF, λ::Float64, s::Float64)
    for j in eachindex(β)
        @inbounds β[j] = prox(p, β[j], λ, s)
    end
end
# if λs are weighted by penalty_factor:
function prox!(p::Penalty, β::VecF, λ::Float64, penalty_factor::VecF, s::Float64)
    for j in eachindex(β)
        @inbounds β[j] = prox(p, β[j], λ * penalty_factor[j], s)
    end
end
# For prox.jl
function prox!(p::Penalty, β::VecF, λ::Float64, penalty_factor::VecF, s::VecF)
    for j in eachindex(β)
        @inbounds β[j] = prox(p, β[j], λ * penalty_factor[j], s[j])
    end
end
