#----------------------------------------------------------------------# SparseReg
immutable SparseReg{
            A <: OfflineAlgorithm,
            L <: Loss,
            P <: Penalty,
            O <: Obs
        } <: AbstractSparseReg
    β::VecF
    loss::L
    penalty::P
    λ::Float64
    factor::VecF
    algorithm::A
    obs::O
end


# forget about type stability for now
function SparseReg(obs::Obs, args...)
    n, p = size(obs.x)
    l = LinearRegression()
    r = NoPenalty()
    λ = .01
    f = ones(p)
    a = ProxGrad(obs)
    for arg in args
        if isa(arg, Loss)
            l = arg
        elseif isa(arg, Penalty)
            r = arg
        elseif isa(arg, Float64)
            λ = arg
        elseif isa(arg, VecF)
            f = arg
        elseif isa(arg, Algorithm)
            a = init(arg, obs)
        else
            warn("Unused argument")
        end
    end
    SparseReg(zeros(p), l, r, λ, f, a, obs)
end

function SparseReg(x::AMatF, y::AVecF, args...)
    o = SparseReg(Obs(x, y), args...)
    fit!(o)
    o
end

function SparseReg(x::AMatF, y::AVecF, w::AVecF, args...)
    o = SparseReg(Obs(x, y, w), args...)
    fit!(o)
    o
end




# The mess below is due to:
# Type-stable constructors with arbitrary order of arguments!
# There must be a better way to do this (generated functions?)

# `t` is a tuple containing: (Loss, Penalty, Float64, VecF, Algorithm)

# _defaults(p::Integer) =
#     LinearRegression(), # Loss
#     NoPenalty(),        # Penalty
#     0.1,                # λ (Float64)
#     zeros(p),           # factor (VecF)
#     ProxGrad()          # Algorithm
#
#
# # Create an empty coefficient vector and also initialize the offline algorithm
# # based on n, p
# init(n, p, t::Tuple) = zeros(p), t[1], t[2], t[3], t[4], init(n, p, t[5])
#
# # Everything below calls this constructor
# SparseReg(n::Integer, p::Integer, t::Tuple) = SparseReg(init(n, p, t)...)
#
# # Constructors for providing 0 to 5 additional arguments
# function SparseReg(n::Integer, p::Integer)
#     SparseReg(n, p, _defaults(p))
# end
# function SparseReg(n::Integer, p::Integer, a)
#     args = _defaults(p)
#     SparseReg(n, p, _a(args, a))
# end
# function SparseReg(n::Integer, p::Integer, a1, a2)
#     args = _defaults(p)
#     args2 = _a(args, a1)
#     SparseReg(n, p, _a(args2, a2))
# end
# function SparseReg(n::Integer, p::Integer, a1, a2, a3)
#     args = _defaults(p)
#     args2 = _a(args, a1)
#     args3 = _a(args2, a2)
#     SparseReg(n, p, _a(args3, a3))
# end
# function SparseReg(n::Integer, p::Integer, a1, a2, a3, a4)
#     args = _defaults(p)
#     args2 = _a(args, a1)
#     args3 = _a(args2, a2)
#     args4 = _a(args3, a3)
#     SparseReg(n, p, _a(args4, a4))
# end
# function SparseReg(n::Integer, p::Integer, a1, a2, a3, a4, a5)
#     args = _defaults(p)
#     args2 = _a(args, a1)
#     args3 = _a(args2, a2)
#     args4 = _a(args3, a3)
#     args5 = _a(args4, a4)
#     SparseReg(n, p, _a(args5, a5))
# end
#
# # "overwrite" one argument in a tuple based on last argument
# # Loss, Penalty, Float64, VecF, Algorithm
# _a(t::Tuple, arg::Loss)        = arg, t[2], t[3], t[4], t[5]
# _a(t::Tuple, arg::Penalty)     = t[1], arg, t[3], t[4], t[5]
# _a(t::Tuple, arg::Float64)     = t[1], t[2], arg, t[4], t[5]
# _a(t::Tuple, arg::VecF)        = t[1], t[2], t[3], arg, t[5]
# _a(t::Tuple, arg::Algorithm)   = t[1], t[2], t[3], t[4], arg
