struct ProximalGradientModel{
        L <: Loss,
        P <: ConvexElementPenalty,
        O <: Obs
    } <: AbstractSparseReg
    # AbstractSparseReg
    θ::Coefficients
    loss::L
    penalty::P
    factor::VecF
    # observations
    obs::O
    # config
    maxit::Int
    tol::Float64
    verbose::Bool
    step::Float64
    # buffers
    ∇::VecF
    xβ::VecF
    deriv_vec::VecF
end
function ProximalGradientModel(obs::Obs;
        λ::VecF             = defaultλ(),
        loss::Loss          = defaultloss(),
        penalty::Penalty    = defaultpenalty(),
        factor::VecF        = ones(size(obs.x, 2)),
        maxit::Int          = 100,
        tol::Float64        = 1e-6,
        verbose::Bool       = false,
        step::Float64       = 1.0,
    )
    n, p = size(obs)
    c = Coefficients(obs, λ)
    o = ProximalGradientModel(c, loss, penalty, factor, obs, maxit, tol, verbose, step,
                              zeros(p), zeros(n), zeros(n))
    fit!(o)
    o
end

# TODOs:
# - Estimate Lipschitz constant for step size?
# - FISTA acceleration?
function fit!(o::ProximalGradientModel)
    for (k, λ) in enumerate(o.θ.λ)
        β = @view(o.θ.β[:, k])
        newL = Inf
        niters = 0
        for _ in 1:o.maxit
            update_g!(o)
            update_β!(o, β, λ)
            update_xβ!(o, β)
            newL, niters, isconverged = converged(o, newL, niters, β)
            isconverged && break
        end
    end
end

#--------------------------------------------------------------# objective value
_L(o, β) = value(o.loss, o.obs.y, o.xβ, AvgMode.Mean()) + value(o.penalty, β)

#--------------------------------------------------------------# update_xβ!
update_xβ!(o, β) = A_mul_B!(o.xβ, o.obs.x, β)

#--------------------------------------------------------------# update_g!
function update_g!(o)
    for i in eachindex(o.obs.y)
        @inbounds o.deriv_vec[i] = deriv(o.loss, o.obs.y[i], o.xβ[i])
    end
    add_weight!(o.deriv_vec, o.obs.w)
    At_mul_B!(o.∇, o.obs.x, o.deriv_vec)
    scale!(o.∇, 1 / length(o.obs.y))
end
add_weight!(v::VecF, w::Ones) = return
function add_weight!(v::VecF, w::AVecF)
    for i in eachindex(v)
        @inbounds v[i] *= w[i]
    end
end

#--------------------------------------------------------------# update_β!
function update_β!(o, β, λ)
    s = o.step
    for j in eachindex(β)
        @inbounds λj = λ * o.factor[j]
        @inbounds β[j] = prox(o.penalty, β[j] - s * o.∇[j], s * λj)
    end
end

#--------------------------------------------------------------# converged
function converged(o, newL, niters, β)
    oldL = newL
    newL = _L(o, β)
    niters += 1
    tolerance = abs(newL - oldL) #/ min(abs(newL), abs(oldL))
    isconverged = tolerance < o.tol
    isconverged || niters == o.maxit &&
        warn("DID NOT CONVERGE, RelTol = $tolerance")
    isconverged ?
        o.verbose && info("CONVERGED: $niters, RelTol = $tolerance") :
        o.verbose && info("Iteration: $niters, RelTol = $tolerance")
    newL, niters, isconverged
end
