#-------------------------------------------------------------------------------# GLM
type GLM{
        D <: UnivariateDistribution,
        L <: Link,
        P <: Penalty,
        Tx <: Real,
        Ty <: Real
    }
    β0::Float64                 # intercept term
    β::Vector{Float64}          # coefficients
    intercept::Bool             # should intercept be estimated?
    x::Matrix{Tx}               # design matrix
    y::Vector{Ty}               # response vector
    wts::Vector{Float64}        # weights
    family::D                   # Bernoulli, Normal, or Poisson
    link::L                     # link function: canonical_link(family) by default
    penalty::P                  # regularization term
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
        penalty::Penalty = NoPenalty(),
        wts::VecF = ones(0)
    )
    n, p = size(x)
    @assert length(y) == n "length(y) != size(x, 1)"
    if length(wts) == n
        wts = wgts / sum(wgts) * n
    end
    o = GLM(0.0, zeros(p), intercept, x, y, wts, family, link, penalty)
    fit!(o)
end
has_canonical_link(o::GLM) = o.link == canonical(o.family)
StatsBase.predict{T <: Real}(o::GLM, x::Matrix{T}) = predict(o.link, x * o.β + o.β0)
StatsBase.predict(o::GLM) = predict(o, o.x)
StatsBase.coef(o::GLM) = o.β
penalty(o::GLM) = penalty(o.penalty, o.β)


#------------------------------------------------------------------------------# cost
# cost = objective function we are trying to minimize
# cost = f(β) + g(β) where f() is proportional to negative loglikelihood and g() is regularization

# canonical links
lossvector(o::GLM{Normal, IdentityLink}) = 0.5 * abs2(o.y - predict(o))
function lossvector(o::GLM{Bernoulli})  # for both Logit and Probit
    probs = predict(o)
    o.y .* log(probs ./ (1.0 - probs)) + log(1.0 - probs)
end
lossvector(o::GLM{Poisson, LogLink}) = o.y .* o.x * o.β - predict(o)


function cost(o::GLM)
    if length(o.wts) == length(o.y)
        return mean(o.wts .* lossvector(o)) + penalty(o)
    else
        return mean(lossvector(o)) + penalty(o)
    end
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


# todo: solution path
# todo: coeftable
function fit!(o::GLM;
        maxit::Int = 100,
        eps::Real = 1e-4,
        verbose::Bool = true
    )

    # setup
    n, p = size(o.x)
    usewts = (n == length(o.wts))
    newcost = Inf
    s = 1.0             # step size for FISTA
    iters = 0
    β1 = zeros(p)       # last iteration
    β2 = zeros(p)       # two iterations ago

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
        if usewts
            resid .* o.wts
        end
        ∇f = deriv(o, resid)
        o.β += s * ∇f / n
        prox!(o.penalty, o.β, s)
        if o.intercept
            o.β0 += mean(resid)
        end

        if usewts
            newcost = weightedcost(o)
        else
            newcost = cost(o)
        end

        # check for convergence
        if abs(newcost - oldcost) < eps * abs(oldcost + 1.0)
            verbose && println("converged in $iters iterations")
            break
        end
    end
    iters == maxit && print_with_color(:red, "Did NOT converge in $iters iterations \n")
    o
end




n, p = 10000, 30
x = randn(n, p)
β = collect(linspace(-.5, .5, p))

y = x * β + randn(n)
@time o = GLM(x, y; family = Normal(), penalty = SCADPenalty(.1))
@display o

y = Float64[rand(Bernoulli(1 / (1 + exp(-η)))) for η in x*β]
@time o = GLM(x, y; family = Bernoulli(), link = ProbitLink(), penalty = L2Penalty(.01))
@display o
