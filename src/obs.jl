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
    print_with_color(default_color, io, "  □ Observations\n")
    # x
    print(io, "  ")
    print_item(io, "x", typeof(o.x), false)
    println(io, " $(size(o.x))")
    # y
    print(io, "  ")
    print_item(io, "y", typeof(o.y), false)
    println(io, " $(size(o.y))")
    # w
    print(io, "  ")
    print_item(io, "weights", typeof(o.w), false)
    print(io, " $(size(o.w))")
end
