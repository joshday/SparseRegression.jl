#----------------------------------------------------------------------# SparseReg
immutable SparseReg{A <: OfflineAlgorithm, L <: Loss, P <: Penalty} <: AbstractSparseReg
    β::VecF
    loss::L
    penalty::P
    algorithm::A
    λ::Float64
    factor::VecF
end

_defaults(p::Integer) = LinearRegression(), NoPenalty(), ProxGrad(), 0.1, ones(p)

# Type-stable constructor with arbitrary order of arguments.
# There must be a better way to do this
# generated functions?
SparseReg(p::Integer) = SparseReg(zeros(p), _defaults(p)...)
function SparseReg(p::Integer, a)
    args = _defaults(p)
    args2 = _a(args..., a)
    SparseReg(zeros(p), args2...)
end
function SparseReg(p::Integer, a1, a2)
    args = _defaults(p)
    args2 = _a(args..., a1)
    args3 = _a(args2..., a2)
    SparseReg(zeros(p), args3...)
end
function SparseReg(p::Integer, a1, a2, a3)
    args = _defaults(p)
    args2 = _a(args..., a1)
    args3 = _a(args2..., a2)
    args4 = _a(args3..., a3)
    SparseReg(zeros(p), args4...)
end
function SparseReg(p::Integer, a1, a2, a3, a4)
    args = _defaults(p)
    args2 = _a(args..., a1)
    args3 = _a(args2..., a2)
    args4 = _a(args3..., a3)
    args5 = _a(args4..., a4)
    SparseReg(zeros(p), args5...)
end
function SparseReg(p::Integer, a1, a2, a3, a4, a5)
    args = _defaults(p)
    args2 = _a(args..., a1)
    args3 = _a(args2..., a2)
    args4 = _a(args3..., a3)
    args5 = _a(args4..., a4)
    args6 = _a(args5..., a5)
    SparseReg(zeros(p), args6...)
end

# "overwrite" one argument in a tuple based on type
_a(l::Loss,r::Penalty,a::Algorithm,λ::Float64,f::VecF,t::Loss)      = t,r,a,λ,f
_a(l::Loss,r::Penalty,a::Algorithm,λ::Float64,f::VecF,t::Penalty)   = l,t,a,λ,f
_a(l::Loss,r::Penalty,a::Algorithm,λ::Float64,f::VecF,t::Algorithm) = l,r,t,λ,f
_a(l::Loss,r::Penalty,a::Algorithm,λ::Float64,f::VecF,t::Float64)   = l,r,a,t,f
_a(l::Loss,r::Penalty,a::Algorithm,λ::Float64,f::VecF,t::VecF)      = l,r,a,λ,t


function SparseReg(x::AMatF, y::AVecF, args...)
    o = SparseReg(size(x, 2), args...)
    fit!(o, Obs(x, y))
    o
end
function SparseReg(x::AMatF, y::AVecF, w::AVecF, args...)
    o = SparseReg(size(x, 2), args...)
    fit!(o, Obs(x, y, w))
    o
end
