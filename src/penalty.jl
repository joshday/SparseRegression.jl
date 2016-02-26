#---------------------------------------------------------------------------# Penalty
immutable NoPenalty             <: Penalty              end
immutable L2Penalty             <: Penalty              end
immutable L1Penalty             <: Penalty              end
immutable ElasticNetPenalty     <: Penalty α::Float64   end
immutable SCADPenalty           <: Penalty a::Float64   end


ElasticNetPenalty(α::Real = .5) = ElasticNetPenalty(α)
SCADPenalty(a::Real = 3.7) = SCADPenalty(a)

Base.show(io::IO, p::NoPenalty)         = print(io, "NoPenalty")
Base.show(io::IO, p::L2Penalty)         = print(io, "L2Penalty")
Base.show(io::IO, p::L1Penalty)         = print(io, "L1Penalty")
Base.show(io::IO, p::ElasticNetPenalty) = print(io, "ElasticNetPenalty (α = $(p.α))")
Base.show(io::IO, p::SCADPenalty)       = print(io, "SCADPenalty (a = $(p.a))")

penalty(p::NoPenalty, λ, β) = 0.0
penalty(p::L2Penalty, λ, β) = 0.5 * λ * sumabs2(β)
penalty(p::L1Penalty, λ, β) = λ * sumabs(β)
penalty(p::ElasticNetPenalty, λ, β) = λ * (p.α * sumabs(β) + (1. - p.α) * 0.5 * sumabs2(β))
function penalty(p::SCADPenalty, λ, β)
    val = 0.0
    for j in 1:length(β)
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

prox(p::NoPenalty, λ, βj, s) = βj
prox(p::L2Penalty, λ, βj, s) = βj / (1.0 + s * λ)
prox(p::L1Penalty, λ, βj, s) = sign(βj) * max(abs(βj) - s * λ, 0.0)
prox(p::ElasticNetPenalty, λ, βj, s) =
    sign(βj) * max(abs(βj) - s * λ * p.α, 0.0) / (1.0 + s * λ * (1.0 - p.α))
function prox(p::SCADPenalty, λ, βj, s)
    if abs(βj) > p.a * λ
    elseif abs(βj) < 2.0 * λ
        βj = sign(βj) * max(abs(βj) - s * λ, 0.0)
    else
        βj = (βj - s * sign(βj) * p.a * λ / (p.a - 1.0)) / (1.0 - (1.0 / p.a - 1.0))
    end
    βj
end

function prox!(p::Penalty, λ::Float64, β::VecF, s::Float64)
    for j in 1:length(β)
        β[j] = prox(p, λ, β[j], s)
    end
end
