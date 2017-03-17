#----------------------------------------------------------------------# SparseReg
immutable SparseReg{L <: Loss, P <: Penalty} <: AbstractSparseReg
    β::VecF
    loss::L
    penalty::P
    λ::Float64
    factor::VecF
end

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
    print_with_color(default_color, io, "■ Sparse Regression Model\n")
    print_item(io, "β", o.β)
    print_item(io, "Loss", o.loss)
    print_item(io, "Penalty", o.penalty)

    showpen = !isa(o.penalty, NoPenalty)
    showpen && print_item(io, "λ", o.λ)
    showpen && any(x -> x != 1.0, o.factor) && print_item(io, "λ scaling", o.factor)
end

#----------------------------------------------------------------------# FittedModel
immutable FittedModel{S <: SparseReg, A <: Algorithm}
    model::S
    algorithm::A
end
function Base.show(io::IO, o::FittedModel)
    print_with_color(default_color, io, "■■■■■■ Fitted Model ■■■■■■\n")
    show(io, o.model)
    println(io)
    show(io, o.algorithm)
end
