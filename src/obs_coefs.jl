#----------------------------------------------------------------------#  Ones
# constant vector of ones
immutable Ones <: AVecF n::Int end
Ones(y::AVec) = Ones(length(y))
Base.size(o::Ones) = (o.n, )
Base.getindex(o::Ones, i::Integer) = 1.
Base.getindex{I <: Integer}(o::Ones, rng::AVec{I}) = Ones(length(rng))

#----------------------------------------------------------------------#  Obs
immutable Obs{W <: AVec, X <: AMat, Y <: AVec}
    w::W
    x::X
    y::Y
end
function Obs(x::AMat, y::AVec, w::AVec = Ones(y))
    n1 = size(x, 1)
    n2 = length(y)
    n3 = length(w)
    n1 == n2 == n3 || throw(DimensionMismatch("number of rows should match: $n1, $n2, $n3"))
    Obs(w, x, y)
end
function Base.show(io::IO, o::Obs)
    header(io, name(o))
    print_item(io, "x", summary(o.x))
    print_item(io, "y", summary(o.y))
    print_item(io, "weights", summary(o.w), false)
end

nobs(o::Obs) = length(o.y)
Base.size(o::Obs) = size(o.x)

#-------------------------------------------------------------------------------# Coefficients
immutable Coefficients
    β::MatF
    λ::VecF
    function Coefficients(β::MatF, λ)
        size(β, 2) == length(λ) || throw(DimensionMismatch())
        new(β, λ)
    end
end
Coefficients(o::Obs, λ::VecF) = Coefficients(zeros(size(o.x, 2), length(λ)), λ)
function Base.show(io::IO, o::Coefficients)
    header(io, name(o))
    for (i, λ) in enumerate(o.λ)
        print(io, @sprintf("%4s", "$i:"))
        print(io, @sprintf("%12s", "β($(round(λ,3))) = "))
        show(io, o.β[:, i]')
        println(io)
    end
end
