"""
Proximal Gradient Method
"""
immutable ProxGrad <: OfflineAlgorithm
    maxit::Int
    tol::Float64
    verbose::Bool
    step::Float64
end
function ProxGrad(;maxit::Int=100, tol::Float64=1e-6, verbose::Bool=false, step::Float64=1.0)
    ProxGrad(maxit, tol, verbose, step)
end
function Base.show(io::IO, o::ProxGrad)
    print(io,"ProxGrad(maxit=$(o.maxit), tol=$(o.tol), verbose=$(o.verbose), step=$(o.step))")
end

default(::Type{Algorithm}) = ProxGrad()

immutable ProxGradBuffer
    ∇::VecF
    ŷ::VecF
    deriv_vec::VecF
end
function makebuffer(a::ProxGrad, x, y)
    n = size(x, 1)
    p = size(x, 2)
    ProxGradBuffer(zeros(p), zeros(n), zeros(n))
end


# TODOs:
# - line search?
# - Estimate Lipschitz constant for step size?
# - Use FISTA acceleration?
# - weighted version
function fit!(o::SparseReg{ProxGrad}, x::AMat, y::AVec, buffer = makebuffer(o.algorithm, x, y))
    n, p = size(x)
    p == length(o.β) || throw(ArgumentError("x dimension does not match β"))

    oldcost = -Inf
    newcost = value(o.loss, y, buffer.ŷ, AvgMode.Mean()) + value(o.penalty, o.β)
    niters = 0
    for k in 1:o.algorithm.maxit
        oldcost = newcost
        niters += 1

        get_gradient!(o.loss, x, y, buffer)
        update_β!(o, buffer)
        update_ŷ!(o, x, buffer)

        newcost = value(o.loss, y, buffer.ŷ, AvgMode.Mean()) + value(o.penalty, o.β)
        converged(oldcost, newcost, niters, o.algorithm) && break
    end

    if niters == o.algorithm.maxit
        tolerance = abs(newcost - oldcost) / min(abs(newcost), abs(oldcost))
        warn("DID NOT CONVERGE in $niters iterations, Relative Tolerance = $tolerance")
    end
    o
end

#--------------------------------------------------------------# components of loop
function get_gradient!(L, x, y, buffer)
    for i in eachindex(y)
        @inbounds buffer.deriv_vec[i] = deriv(L, y[i], buffer.ŷ[i])
    end
    At_mul_B!(buffer.∇, x, buffer.deriv_vec)
    scale!(buffer.∇, 1 / length(y))
end

function update_β!(o, buffer)
    s = o.algorithm.step
    @simd for j in eachindex(o.β)
        @inbounds λj = o.λ * o.factor[j]
        @inbounds o.β[j] = prox(o.penalty, o.β[j] - s * buffer.∇[j], s * λj)
    end
end

function update_ŷ!(o, x, buffer)
    A_mul_B!(buffer.ŷ, x, o.β)
    xβ_to_ŷ!(o.loss, buffer.ŷ)
end

@inline function converged(oldcost, newcost, niters, alg)
    tolerance = abs(newcost - oldcost) / min(abs(newcost), abs(oldcost))
    isconverged = tolerance < alg.tol
    isconverged ?
        alg.verbose && info("CONVERGED: $niters, Relative Tolerance = $tolerance") :
        alg.verbose && info("Iteration: $niters, Relative Tolerance = $tolerance")
    isconverged
end
