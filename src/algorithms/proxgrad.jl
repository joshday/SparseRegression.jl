immutable ProxGrad{O <: Obs} <: OfflineAlgorithm
    # config
    maxit::Int
    tol::Float64
    verbose::Bool
    step::Float64
    # buffers
    ∇::VecF
    ŷ::VecF
    deriv_vec::VecF
    # observations
    obs::O
end
function Base.show(io::IO, alg::ProxGrad)
    header(io, name(alg))
    print_items(io, alg, [:maxit, :tol, :verbose, :step])
    println(io)
    show(io, alg.obs)
end
function ProxGrad(o::Obs ;maxit = 100, tol = 1e-6, verbose = false, step = 1.0)
    n, p = size(o.x)
    ProxGrad(maxit, tol, verbose, step, zeros(p), zeros(n), zeros(n), o)
end



# TODOs:
# - Estimate Lipschitz constant for step size?
# - FISTA acceleration?
function fit!(o::SparseReg, A::ProxGrad)
    n, p = size(A.obs.x)
    p == length(o.β) || throw(ArgumentError("x dimension does not match β"))

    oldcost = -Inf
    newcost = objective_value(o, A)
    niters = 0
    for k in 1:A.maxit
        oldcost = newcost
        niters += 1

        get_gradient!(o, A)
        update_β!(o, A)
        update_ŷ!(o, A)

        newcost = objective_value(o, A)
        converged(oldcost, newcost, niters, A) && break
    end
    o, A
end

#--------------------------------------------------------------# objective_value
function objective_value(o::SparseReg, A::Algorithm)
    value(o.loss, A.obs.y, A.ŷ, AvgMode.Mean()) + value(o.penalty, o.β)
end

#--------------------------------------------------------------# get_gradient!
function get_gradient!(o::SparseReg, A::ProxGrad)
    for i in eachindex(A.obs.y)
        @inbounds A.deriv_vec[i] = deriv(o.loss, A.obs.y[i], A.ŷ[i])
    end
    add_weight!(A.deriv_vec, A.obs.w)
    At_mul_B!(A.∇, A.obs.x, A.deriv_vec)
    scale!(A.∇, 1 / length(A.obs.y))
end
add_weight!(v::VecF, w::Ones) = return
function add_weight!(v::VecF, w::AVecF)
    for i in eachindex(v)
        @inbounds v[i] *= w[i]
    end
end

#--------------------------------------------------------------# update_β!
function update_β!(o, A)
    s = A.step
    @simd for j in eachindex(o.β)
        @inbounds λj = o.λ * o.factor[j]
        @inbounds o.β[j] = prox(o.penalty, o.β[j] - s * A.∇[j], s * λj)
    end
end

#--------------------------------------------------------------# objective_ŷ!
function update_ŷ!(o, A)
    A_mul_B!(A.ŷ, A.obs.x, o.β)
    xβ_to_ŷ!(o.loss, A.ŷ)
end

#--------------------------------------------------------------# converged
function converged(oldcost, newcost, niters, A)
    tolerance = abs(newcost - oldcost) / min(abs(newcost), abs(oldcost))
    isconverged = tolerance < A.tol
    isconverged ?
        A.verbose && info("CONVERGED: $niters, Relative Tolerance = $tolerance") :
        A.verbose && info("Iteration: $niters, Relative Tolerance = $tolerance")
        niters == A.maxit && warn("DID NOT CONVERGE in $niters iterations, Tol = $tolerance")
    isconverged
end
