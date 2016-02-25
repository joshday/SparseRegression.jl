#-------------------------------------------------------------------------------# GLM
type GLM{
        D<:Ds.UnivariateDistribution,
        L<:Link,
        P<:Penalty
    }
    β0::Float64
    β::Vector{Float64}
    intercept::Bool
    x::Matrix{Float64}
    y::Vector{Float64}
    family::D
    link::L
    penalty::P
end
function Base.show(io::IO, o::GLM)
    print_header(io, "GLM")
    o.intercept && print_item(io, "β0", o.β0)
    print_item(io, "β", o.β)
    print_item(io, "Family", typeof(o.family))
    print_item(io, "Link", typeof(o.link))
    print_item(io, "Penalty", o.penalty)
end
function GLM(x::Matrix, y::Vector;
        intercept::Bool = true,
        family::Ds.UnivariateDistribution = Ds.Normal(),
        link::Link = canonical(family),
        penalty::Penalty = NoPenalty()
    )
    GLM(0.0, zeros(size(x, 2)), intercept, x, y, family, link, penalty)
end
has_canonical_link(o::GLM) = o.link == canonical(o.family)
StatsBase.predict{T <: Real}(o::GLM, x::Matrix{T}) = predict(o.link, x * o.β + o.β0)
StatsBase.predict(o::GLM) = predict(o, o.x)


#------------------------------------------------------------------------------# loss
cost(o::GLM{Ds.Normal}) = 0.5 * mean(abs2(o.y - predict(o))) + penalty(o.penalty, o.β)
function cost(o::GLM{Ds.Bernoulli})
    probs = predict(o)
    mean(o.y .* log(π) + (1.0 - o.y) .* log(1.0 - probs)) + penalty(o.penalty, o.β)
end


function fista!(o::GLM;
        maxit::Int = 100,
        eps::Real = 1e-4,
        verbose::Bool = true
    )
    n, p = size(o.x)
    newcost = Inf
    s = 1.0
    iters = 0

    for i in 1:maxit
        iters += 1
        oldcost = newcost
        resid = o.y - predict(o)
        ∇f = o.x' * resid
        o.β += s * ∇f / n
        prox!(o.penalty, o.β, s)
        if o.intercept
            o.β0 += mean(resid)
        end

        newcost = cost(o)

        if abs(newcost - oldcost) < eps * abs(oldcost + 1.0)
            verbose && println("converged in $iters iterations")
            break
        end
    end
    o
end


n, p = 100, 10
x = randn(n, p)
y = x * collect(1.:p) + randn(n)

o = GLM(x, y; family = Ds.Normal(), penalty = L1Penalty(3))
@time fista!(o)

@display o
@display has_canonical_link(o)
@display cost(o)
