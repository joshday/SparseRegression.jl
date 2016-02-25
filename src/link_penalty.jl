################################################################################ Link
predict{T <: Real}(l::Link, η::Vector{T}) = [predict(l, ηi) for ηi in η]

immutable IdentityLink  <: Link end
immutable LogitLink     <: Link end
immutable LogLink       <: Link end
immutable ProbitLink    <: Link end

canonical(::Normal)    = IdentityLink()
canonical(::Bernoulli) = LogitLink()
canonical(::Poisson)   = Logink()

predict(l::IdentityLink, η::Real)   = η
predict(l::LogitLink, η::Real)      = 1.0 / (1.0 + exp(-η))
predict(l::LogLink, η::Real)        = exp(η)
predict(l::ProbitLink, η::Real)     = Distributions.cdf(Normal(), η)


############################################################################# Penalty
immutable NoPenalty             <: Penalty                          end
immutable L2Penalty             <: Penalty λ::Float64               end
immutable L1Penalty             <: Penalty λ::Float64               end
immutable ElasticNetPenalty     <: Penalty λ::Float64; α::Float64   end
immutable SCADPenalty           <: Penalty λ::Float64; a::Float64   end

ElasticNetPenalty(λ::Real, α::Real = .5) = ElasticNetPenalty(λ, α)
SCADPenalty(λ::Real, a::Real = 3.7) = SCADPenalty(λ, a)

Base.show(io::IO, p::NoPenalty)         = print(io, "NoPenalty")
Base.show(io::IO, p::L2Penalty)         = print(io, "L2Penalty (λ = $(p.λ))")
Base.show(io::IO, p::L1Penalty)         = print(io, "L1Penalty (λ = $(p.λ))")
Base.show(io::IO, p::ElasticNetPenalty) = print(io, "ElasticNetPenalty (λ = $(p.λ), α = $(p.α))")
Base.show(io::IO, p::SCADPenalty)       = print(io, "SCADPenalty (λ = $(p.λ), a = $(p.a))")

penalty(p::NoPenalty, β) = 0.0
penalty(p::L2Penalty, β) = 0.5 * sumabs2(β)
penalty(p::L1Penalty, β) = sumabs(β)
penalty(p::ElasticNetPenalty, β) = p.λ * (p.α * sumabs(β) + (1. - p.α) * 0.5 * sumabs2(β))
function penalty(p::SCADPenalty, β)
    val = 0.0
    for j in 1:length(β)
        βj = abs(β[j])
        if βj < p.λ
            val += p.λ * βj
        elseif βj < p.λ * p.a
            val -= 0.5 * (βj ^ 2 - 2.0 * p.a * p.λ * βj + p.λ ^ 2) / (p.a - 1.0)
        else
            val += 0.5 * (p.a + 1.0) * p.λ ^ 2
        end
    end
    return val
end


#---------------------------------------------------------------------# prox operator
prox(p::NoPenalty, βj::Float64, s::Float64) = βj
prox(p::L2Penalty, βj::Float64, s::Float64) = βj / (1.0 + s * p.λ)
prox(p::L1Penalty, βj::Float64, s::Float64) = sign(βj) * max(abs(βj) - s * p.λ, 0.0)
prox(p::ElasticNetPenalty, βj::Float64, s::Float64) =
    sign(βj) * max(abs(βj) - s * p.λ * p.α, 0.0) / (1.0 + s * p.λ * (1.0 - p.α))
function prox(p::SCADPenalty, βj::Float64, s::Float64)
    if abs(βj) > p.a * p.λ
    elseif abs(βj) < 2.0 * p.λ
        βj = sign(βj) * max(abs(βj) - s * p.λ, 0.0)
    else
        βj = (βj - s * sign(βj) * p.a * p.λ / (p.a - 1.0)) / (1.0 - (1.0 / p.a - 1.0))
    end
    βj
end


function prox!(p::Penalty, β::Vector{Float64}, s::Float64)
    for j in 1:length(β)
        β[j] = prox(p, β[j], s)
    end
end
