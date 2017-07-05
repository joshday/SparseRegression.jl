struct Model{L <: Loss, P <: Penalty}
    β::Vector{Float64}
    λfactor::Vector{Float64}
    loss::L
    penalty::P
end

# Some hacks for type-stable arbitrary argument order
Model(p::Integer, t::Tuple) = Model(zeros(p), t...)
d(p::Integer) = (fill(.1, p), LinearRegression(), L2Penalty())
a(argu::Vector{Float64}, t::Tuple)  = (argu, t[2], t[3])
a(argu::Loss, t::Tuple)             = (t[1], argu, t[3])
a(argu::Penalty, t::Tuple)          = (t[1], t[2], argu)
Model(p::Integer)               = Model(p, d(p))
Model(p::Integer, a1)           = Model(p, a(a1, d(p)))
Model(p::Integer, a1, a2)       = Model(p, a(a2, a(a1, d(p))))
Model(p::Integer, a1, a2, a3)   = Model(p, a(a3, a(a2, a(a1, d(p)))))
Model(obs::Obs, args...) = Model(size(obs, 2), args...)

function Base.show(io::IO, o::Model)
    println(io, typeof(o))
    println(io, "  > β        :", o.β)
    println(io, "  > λ factor :", o.λfactor)
    println(io, "  > Loss     :", o.loss)
    print(io,   "  > Penalty  :", o.penalty)
end

coef(o::Model) = o.β
factor(o::Model) = o.λfactor
loss(o::Model) = o.loss
penalty(o::Model) = o.penalty
value(o::Model) = o.β
predict(o::Model, x::AbstractVector) = At_mul_B(x, o.β)
predict(o::Model, x::AbstractMatrix) = x * o.β




#-----------------------------------------------------------------------# ModelPath
struct Path{E <: Model}
    path::Vector{E}
    λfactor::Vector{Float64}
    αs::Vector{Float64}
end
function Path(o::Model, αs::AbstractVector{Float64})
    λf = o.λfactor
    path = [Model(length(o.β), o.loss, o.penalty, α * λf) for α in αs]
    Path(path, λf, collect(αs))
end
function Base.show(io::IO, P::Path)
    println(io, typeof(P))
    println(io, "  > λ factor :", P.λfactor)
    println(io, "  > αs       :", P.αs)
    println(io, "  > Loss     :", P.path[1].loss)
    println(io, "  > Penalty  :", P.path[1].penalty)
    println(io, "  > Path     :")
    for j in eachindex(P.path)
        o = P.path[j]
        println(io,  "    > $(@sprintf("%8s", "β($(P.αs[j]))")) :", coef(o))
    end
end


#-----------------------------------------------------------------------# learn!
function learn!(o::Model, a::Algorithm, m::MaxIter = MaxIter(1), args...)
    ml = make_learner(a, m, args...)
    learn!(o, ml)
    o
end

function learn!(P::Path, a::Algorithm, m::MaxIter = MaxIter(1), args...)
    ml = make_learner(a, m, args...)
    for o in P.path
        learn!(o, ml)
    end
    P
end
