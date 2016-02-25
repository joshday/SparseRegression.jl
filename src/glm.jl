#-------------------------------------------------------------------------------# GLM
type GLM{
        D <: UnivariateDistribution,
        L <: Link,
        P <: Penalty,
        Tx <: Real,
        Ty <: Real
    }
    β0::Float64             # intercept term
    β::Vector{Float64}      # coefficients
    intercept::Bool         # should intercept be estimated?
    x::Matrix{Tx}           # design matrix
    y::Vector{Ty}           # response vector
    family::D               # Bernoulli, Normal, or Poisson
    link::L                 # link function: canonical_link(family) by default
    penalty::P              # regularization term
end
function Base.show(io::IO, o::GLM)
    print_header(io, "GLM")
    o.intercept && print_item(io, "β0", o.β0)
    nz = sum(o.β .!= 0)
    nz_percent = nz / length(o.β)
    print_item(io, "β", "nz = $nz / $(length(o.β)) ($nz_percent)")
    println(io, UnicodePlots.scatterplot(o.β))
    print_item(io, "Family", typeof(o.family))
    print_item(io, "Link", typeof(o.link))
    print_item(io, "Penalty", o.penalty)
end
function GLM(x::Matrix, y::Vector;
        intercept::Bool = true,
        family::UnivariateDistribution = Normal(),
        link::Link = canonical(family),
        penalty::Penalty = NoPenalty()
    )
    GLM(0.0, zeros(size(x, 2)), intercept, x, y, family, link, penalty)
end
has_canonical_link(o::GLM) = o.link == canonical(o.family)
StatsBase.predict{T <: Real}(o::GLM, x::Matrix{T}) = predict(o.link, x * o.β + o.β0)
StatsBase.predict(o::GLM) = predict(o, o.x)


#------------------------------------------------------------------------------# cost
# cost = objective function we are trying to minimize
# cost = f(β) + g(β) where f() is proportional to negative loglikelihood and g() is regularization

# canonical link
cost(o::GLM{Normal}) = 0.5 * mean(abs2(o.y - predict(o))) + penalty(o.penalty, o.β)
function cost(o::GLM{Bernoulli, LogitLink})
    probs = predict(o)
    mean(o.y .* log(probs) + (1.0 - o.y) .* log(1.0 - probs)) + penalty(o.penalty, o.β)
end
function cost(o::GLM{Poisson, LogLink})
    mean(o.y .* o.x * o.β - predict(o))
end

# noncanonical link
function cost(o::GLM{Bernoulli, ProbitLink})
    probs = predict(o)
    mean(o.y .* log(probs) + (1.0 - o.y) .* log(1.0 - probs)) + penalty(o.penalty, o.β)
end

#--------------------------------------------------------------------------# gradient
# canonical link
deriv(o::GLM{Normal, IdentityLink}, resid::Vector) = o.x' * resid
deriv(o::GLM{Bernoulli, LogitLink}, resid::Vector) = o.x' * resid
deriv(o::GLM{Poisson, LogLink}, resid::Vector) = o.x' * resid

# noncanonical link
function deriv(o::GLM{Bernoulli, ProbitLink}, resid::Vector)
    probs = predict(o)
    v = Distributions.pdf(Normal(), o.x *o.β) ./ (probs .* (1.0 - probs))
    o.x' * (resid .* v)
end


#--------------------------------------------------------------------# main algorithm
# FISTA: Fast Iterative Shrinkage and Threshold Algorithm
# http://www.seas.ucla.edu/~vandenbe/236C/lectures/fgrad.pdf


# todo: actually use FISTA (currently just proximal gradient)
# todo: weights
# todo: solution path
function fista!(o::GLM;
        maxit::Int = 100,
        eps::Real = 1e-4,
        verbose::Bool = true
    )

    # setup
    n, p = size(o.x)
    newcost = Inf
    s = 1.0
    iters = 0
    β1 = zeros(p)   # last iteration
    β2 = zeros(p)   # two iterations ago

    # main loop
    for k in 1:maxit
        iters += 1
        oldcost = newcost
        copy!(β2, β1)
        copy!(β1, o.β)
        if k > 2
            o.β += ((k - 2) / (k + 1)) * (β1 - β2)
        end
        resid = o.y - predict(o)
        ∇f = deriv(o, resid)
        o.β += s * ∇f / n
        prox!(o.penalty, o.β, s)
        if o.intercept
            o.β0 += mean(resid)
        end

        newcost = cost(o)

        # check for convergence
        if abs(newcost - oldcost) < eps * abs(oldcost + 1.0)
            verbose && println("converged in $iters iterations")
            break
        end
    end
    o
end


n, p = 10000, 20
x = randn(n, p)
β = collect(linspace(-.5, .5, p))
# β = 3 * randn(p)

y = x * β + randn(n)
o = GLM(x, y; family = Normal(), penalty = L1Penalty(.1))
@time fista!(o)
@display o

y = Float64[rand(Bernoulli(1 / (1 + exp(-η)))) for η in x*β]
o = GLM(x, y; family = Bernoulli(), link = ProbitLink(), penalty = L1Penalty(.01))
@time fista!(o)
@display o

@display has_canonical_link(o)
@display cost(o)
