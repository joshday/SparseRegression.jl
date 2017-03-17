immutable StreamReg{
            A <: OnlineAlgorithm,
            L <: Loss,
            P <: Penalty,
            W <: Weight
        } <: AbstractSparseReg
    β::VecF
    loss::L
    penalty::P
    algorithm::A
    λ::Float64
    factor::VecF
    weight::W
    η::Float64
end


_defaults2(p::Integer) = LinearRegression(), NoPenalty(), SGD(), 0.1, ones(p)

# Type-stable constructor with arbitrary order of arguments.
# There must be a better way to do this
# generated functions?
function StreamReg{W <: Weight}(p::Integer; weight::W = LearningRate(), η::Float64 = 1.0)
    StreamReg(zeros(p), _defaults2(p)..., weight, η)
end
function StreamReg(p::Integer, a; kw...)
    args = _defaults2(p)
    args2 = _b(args..., a)
    StreamReg(zeros(p), args2...)
end
function StreamReg(p::Integer, a1, a2; kw...)
    args = _defaults2(p)
    args2 = _b(args..., a1)
    args3 = _b(args2..., a2)
    StreamReg(zeros(p), args3...)
end
function StreamReg(p::Integer, a1, a2, a3; kw...)
    args = _defaults2(p)
    args2 = _b(args..., a1)
    args3 = _b(args2..., a2)
    args4 = _b(args3..., a3)
    StreamReg(zeros(p), args4...; kw...)
end
function StreamReg(p::Integer, a1, a2, a3, a4; kw...)
    args = _defaults2(p)
    args2 = _b(args..., a1)
    args3 = _b(args2..., a2)
    args4 = _b(args3..., a3)
    args5 = _b(args4..., a4)
    StreamReg(zeros(p), args5...; kw...)
end
function StreamReg(p::Integer, a1, a2, a3, a4, a5; kw...)
    args = _defaults2(p)
    args2 = _b(args..., a1)
    args3 = _b(args2..., a2)
    args4 = _b(args3..., a3)
    args5 = _b(args4..., a4)
    args6 = _b(args5..., a5)
    StreamReg(zeros(p), args6...; kw...)
end

# "overwrite" one argument in a tuple based on type
_b(l::Loss,r::Penalty,a::Algorithm,λ::Float64,f::VecF,t::Loss)      = t,r,a,λ,f
_b(l::Loss,r::Penalty,a::Algorithm,λ::Float64,f::VecF,t::Penalty)   = l,t,a,λ,f
_b(l::Loss,r::Penalty,a::Algorithm,λ::Float64,f::VecF,t::Algorithm) = l,r,t,λ,f
_b(l::Loss,r::Penalty,a::Algorithm,λ::Float64,f::VecF,t::Float64)   = l,r,a,t,f
_b(l::Loss,r::Penalty,a::Algorithm,λ::Float64,f::VecF,t::VecF)      = l,r,a,λ,t

init(p, l, r, a, λ, f) = zeros(p), l, r, init(a, p), λ, f

function StreamReg(x::AMatF, y::AVecF, args...; kw...)
    o = StreamReg(size(x, 2), args...; kw...)
    fit!(o, Obs(x, y))
    o
end
function StreamReg(x::AMatF, y::AVecF, w::AVecF, args...; kw...)
    o = StreamReg(size(x, 2), args...; kw...)
    fit!(o, Obs(x, y, w))
    o
end
