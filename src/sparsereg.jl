#----------------------------------------------------------------------# SparseReg
immutable SparseReg{L <: Loss, P <: Penalty} <: AbstractSparseReg
    β::VecF
    loss::L
    penalty::P
    λ::Float64
    factor::VecF
end

# For solution path
SparseReg(o::SparseReg, λ::Float64) = SparseReg(copy(o.β), o.loss, o.penalty, λ, o.factor)

# Constructors
# Messy because: type-stable with arbitrary number/order of arguments

# defaults
_d(p::Integer) = LinearRegression(), NoPenalty(), 0.1, ones(p)

# "overwrite" one argument of tuple (Loss, Penalty, Float64, VecF)
a(t::Tuple, arg::Loss)        = arg, t[2], t[3], t[4]
a(t::Tuple, arg::Penalty)     = t[1], arg, t[3], t[4]
a(t::Tuple, arg::Float64)     = t[1], t[2], arg, t[4]
a(t::Tuple, arg::VecF)        = t[1], t[2], t[3], arg

# Everything below calls this constructor
SparseReg(p::Integer, t::Tuple) = SparseReg(zeros(p), t...)

SparseReg(p::Integer)                = SparseReg(p, _d(p))
SparseReg(p::Integer,a1)             = SparseReg(p, a(_d(p),a1))
SparseReg(p::Integer,a1,a2)          = SparseReg(p, a(a(_d(p),a1),a2))
SparseReg(p::Integer,a1,a2,a3)       = SparseReg(p, a(a(a(_d(p),a1),a2),a3))
SparseReg(p::Integer,a1,a2,a3,a4)    = SparseReg(p, a(a(a(a(_d(p),a1),a2),a3),a4))
SparseReg(p::Integer,a1,a2,a3,a4,a5) = SparseReg(p, a(a(a(a(a(_d(p),a1),a2),a3),a4),a5))

function Base.show(io::IO, o::SparseReg)
    header(io, "SparseReg")
    print_item(io, "β", o.β)
    print_item(io, "Loss", o.loss)
    print_item(io, "Penalty", o.penalty)
    print_item(io, "λ", o.λ, false)
    if any(x -> x != 1.0, o.factor)
        println(io)
        print_item(io, "λ scaling", o.factor)
    end
end
