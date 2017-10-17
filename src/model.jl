"""
    SModel(p::Int, args...)

Create a SparseRegression model of `p` coefficients.  Additional arguments can be given in any
order (and is still type stable):

| argument  | type              | default             |
|-----------|-------------------|---------------------|
| `λfactor` | `Vector{Float64}` | `fill(.1, p)`       |
| `loss`    | `Loss`            | `.5 * L2DistLoss()` |
| `penalty` | `Penalty`         | `L2Penalty()`       |

# Example

    SModel(10, L1Penalty(), vcat(0.0, ones(9)), LogitMarginLoss())
"""
struct SModel{L <: Loss, P <: Penalty}
    β::Vector{Float64}
    λfactor::Vector{Float64}
    loss::L
    penalty::P
end

# hacks for type-stable arbitrary argument order
d(p::Integer) = (fill(.1, p), .5 * L2DistLoss(), L2Penalty())
a(argu::Vector{Float64}, t::Tuple)  = (argu, t[2], t[3])
a(argu::Loss, t::Tuple)             = (t[1], argu, t[3])
a(argu::Penalty, t::Tuple)          = (t[1], t[2], argu)

SModel(p::Integer, t::Tuple)     = SModel(zeros(p), t...)
SModel(p::Integer)               = SModel(p, d(p))
SModel(p::Integer, a1)           = SModel(p, a(a1, d(p)))
SModel(p::Integer, a1, a2)       = SModel(p, a(a2, a(a1, d(p))))
SModel(p::Integer, a1, a2, a3)   = SModel(p, a(a3, a(a2, a(a1, d(p)))))
SModel(obs::Obs, args...)        = SModel(size(obs, 2), args...)

function Base.show(io::IO, o::SModel)
    println(io, typeof(o))
    println(io, "  > β        : ", o.β')
    println(io, "  > λ factor : ", o.λfactor')
    println(io, "  > Loss     : ", o.loss)
    print(io,   "  > Penalty  : ", o.penalty)
end

coef(o::SModel) = o.β
factor(o::SModel) = o.λfactor
loss(o::SModel) = o.loss
penalty(o::SModel) = o.penalty
value(o::SModel) = o.β
predict(o::SModel, x::AbstractVector) = At_mul_B(x, o.β)
predict(o::SModel, x::AbstractMatrix) = x * o.β




#-----------------------------------------------------------------------# SModelPath
struct Path{E <: SModel}
    path::Vector{E}
    λfactor::Vector{Float64}
    αs::Vector{Float64}
end
function Path(o::SModel, αs::AbstractVector{Float64})
    λf = o.λfactor
    path = [SModel(length(o.β), o.loss, o.penalty, α * λf) for α in αs]
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
function learn!(o::SModel, a::Algorithm, m::MaxIter = MaxIter(1), args...)
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
