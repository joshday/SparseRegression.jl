#----------------------------------------------------------------------# SparseReg
immutable SparseReg{A <: OfflineAlgorithm, L <: Loss, P <: Penalty} <: AbstractSparseReg
    β::VecF
    loss::L
    penalty::P
    algorithm::A
    λ::Float64
    factor::VecF
end

_defaults(p) = LinearRegression(), NoPenalty(), ProxGrad(), 0.1, zeros(p)


# Type-stable constructor with arbitrary order of arguments.
# There must be a better way to do this
# generated functions?

# t is a tuple of types: (Loss, Penalty, Algorithm, Float64, VecF)
init(n, p, t::Tuple) = zeros(p), t[1], t[2], init(n, p, t[3]), t[4], t[5]

SparseReg(n::Integer, p::Integer, t::Tuple) = SparseReg(init(n, p, t)...)

SparseReg(n::Integer, p::Integer) = SparseReg(n, p, _defaults(p))

function SparseReg(n::Integer, p::Integer, a)
    args = _defaults(p)
    SparseReg(n, p, _a(args, a))
end
function SparseReg(n::Integer, p::Integer, a1, a2)
    args = _defaults(p)
    args2 = _a(args, a1)
    SparseReg(n, p, _a(args2, a2))
end
function SparseReg(n::Integer, p::Integer, a1, a2, a3)
    args = _defaults(p)
    args2 = _a(args, a1)
    args3 = _a(args2, a2)
    SparseReg(n, p, _a(args3, a3))
end
function SparseReg(n::Integer, p::Integer, a1, a2, a3, a4)
    args = _defaults(p)
    args2 = _a(args, a1)
    args3 = _a(args2, a2)
    args4 = _a(args3, a3)
    SparseReg(n, p, _a(args4, a4))
end
function SparseReg(n::Integer, p::Integer, a1, a2, a3, a4, a5)
    args = _defaults(p)
    args2 = _a(args, a1)
    args3 = _a(args2, a2)
    args4 = _a(args3, a3)
    args5 = _a(args4, a4)
    SparseReg(n, p, _a(args5, a5))
end

# "overwrite" one argument in a tuple based on last argument
# Loss, Penalty, Algorithm, Float64, VecF
_a(t::Tuple, arg::Loss)        = arg, t[2], t[3], t[4], t[5]
_a(t::Tuple, arg::Penalty)     = t[1], arg, t[3], t[4], t[5]
_a(t::Tuple, arg::Algorithm)   = t[1], t[2], arg, t[4], t[5]
_a(t::Tuple, arg::Float64)     = t[1], t[2], t[3], arg, t[5]
_a(t::Tuple, arg::VecF)        = t[1], t[2], t[3], t[4], arg


function SparseReg(x::AMatF, y::AVecF, args...)
    o = SparseReg(size(x)..., args...)
    fit!(o, Obs(x, y))
    o
end

function SparseReg(x::AMatF, y::AVecF, w::AVecF, args...)
    o = SparseReg(size(x)..., args...)
    fit!(o, Obs(x, y, w))
    o
end
