abstract Penalty
# New Penalties need a:
#   - Constructor
#   - one line show method
#   - penalty method
#   - prox method prox_{s * λ * penalty}(βj)
#------------------------------------------------------------------# abstract methods
# c = step * λ (step size * regularization parameter)
function prox!(p::Penalty, β::VecF, c::Float64)
    for j in eachindex(β)
        @inbounds β[j] = prox(p, β[j], c)
    end
end
# if λs are not constant
function prox!(p::Penalty, β::VecF, c::VecF)
    for j in eachindex(β)
        @inbounds β[j] = prox(p, β[j], c[j])
    end
end

#-------------------------------------------------------------------------# NoPenalty
immutable NoPenalty <: Penalty end
Base.show(io::IO, p::NoPenalty) = print(io, "NoPenalty")
penalty(p::NoPenalty, β::VecF, λ::Float64) = 0.0
prox(p::NoPenalty, βj::Float64, c::Float64) = βj

#----------------------------------------------------------------------# RidgePenalty
immutable RidgePenalty <: Penalty end
Base.show(io::IO, p::RidgePenalty) = print(io, "RidgePenalty")
penalty(p::RidgePenalty, β::VecF, λ::Float64) = 0.5 * λ * sumabs2(β)
prox(p::RidgePenalty, βj::Float64, c::Float64) = βj / (1.0 + c)

#----------------------------------------------------------------------# LassoPenalty
immutable LassoPenalty <: Penalty end
Base.show(io::IO, p::LassoPenalty) = print(io, "LassoPenalty")
penalty(p::LassoPenalty, β::VecF, λ::Float64) = λ * sumabs(β)
prox(p::LassoPenalty, βj::Float64, c::Float64) = sign(βj) * max(abs(βj) - c, 0.0)

#-----------------------------------------------------------------# ElasticNetPenalty
immutable ElasticNetPenalty <: Penalty
    a::Float64
    function ElasticNetPenalty(a::Real = .5)
        @assert a != 0 "Use RidgePenalty instead"
        @assert a != 1 "Use LassoPenalty instead"
        @assert 0 < a < 1
        new(a)
    end
end
Base.show(io::IO, p::ElasticNetPenalty) = print(io, "ElasticNetPenalty (α = $(p.a))")
function penalty(p::ElasticNetPenalty, β::VecF, λ::Float64)
    λ * (p.a * sumabs(β) + (1. - p.a) * 0.5 * sumabs2(β))
end
function prox(p::ElasticNetPenalty, βj::Float64, c::Float64)
    sign(βj) * max(abs(βj) - c * p.a, 0.0) / (1.0 + c * (1.0 - p.a))
end

#----------------------------------------------------------------------# SCADPenalty
# TODO
# immutable SCADPenalty <: Penalty
#     a::Float64
#     function SCADPenalty(a::Real = 3.7)
#         @assert a > 2
#         new(a)
#     end
# end
# Base.show(io::IO, p::SCADPenalty) = print(io, "SCADPenalty (a = $(p.a))")
# function penalty(p::SCADPenalty, β::VecF, λ::Float64)
#     val = 0.0
#     for j in eachindex(β)
#         βj = abs(β[j])
#         if βj < λ
#             val += λ * βj
#         elseif βj < λ * p.a
#             val -= 0.5 * (βj ^ 2 - 2.0 * p.a * λ * βj + λ ^ 2) / (p.a - 1.0)
#         else
#             val += 0.5 * (p.a + 1.0) * λ ^ 2
#         end
#     end
#     return val
# end
# function prox(p::SCADPenalty, βj::Float64, c::Float64)
#     if abs(βj) > p.a * c
#     elseif abs(βj) < 2.0 * c
#         βj = sign(βj) * max(abs(βj) - c, 0.0)
#     else
#         βj = (βj - c * sign(βj) * p.a / (p.a - 1.0)) / (1.0 - (1.0 / p.a - 1.0))
#     end
#     βj
# end
