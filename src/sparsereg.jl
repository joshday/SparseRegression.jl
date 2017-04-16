struct SparseReg{L <: Loss, P <: Penalty, O <: Obs}
    β::VecF
    λfactor::VecF
    loss::L
    penalty::P
    obs::O
end
function SparseReg(o::Obs, l::Loss, p::Penalty, λ::Float64 = .1)
    SparseReg(o, l, p, .1 * ones(size(o)[2]))
end
function SparseReg(o::Obs, l::Loss, pen::Penalty, λfactor::VecF)
    SparseReg(zeros(size(o)[2]), λfactor, l, pen, o)
end
function Base.show(io::IO, o::SparseReg)
    header(io, name(o))
    print_item(io, "β", o.β)
    print_item(io, "λ factor", o.λfactor')
    print_item(io, "Loss", o.loss)
    print_item(io, "Penalty", o.penalty)
end
coef(o::SparseReg) = o.β


# To calculate a gradient, we need two storage buffers
#  - length n: stores derivatives with respect to x*β
#  - length p: stores x' * derivatives
function gradient!(nvec::VecF, pvec::VecF, o::SparseReg)
    n, p = size(o.obs)
    @boundscheck length(nvec) == n
    @boundscheck length(pvec) == p
    A_mul_B!(nvec, o.obs.x, o.β)                      # x * β
    derivatives!(nvec, o)                             # nvec = deriv(loss, y, x*β)
    BLAS.gemv!('T', 1 / n, o.obs.x, nvec, 0.0, pvec)  # ∇ = mean(x' * nvec)
end

function derivatives!{L,P}(nvec, o::SparseReg{L, P, Obs{Ones}})
    deriv!(nvec, o.loss, o.obs.y, nvec)
end
function derivatives!(nvec, o::SparseReg)
    deriv!(nvec, o.loss, o.obs.y, nvec)
    for i in eachindex(nvec)
        nvec[i] *= o.obs.w[i]
    end
end





# function ProxGrad(obs::Obs; maxit::Int = 100, tol::Float64 = 1e-6,
#                   verbose::Bool = false, step::Float64 = 1.0)
#     n, p = size(obs)
#     ProxGrad(maxit, tol, verbose, step, zeros(p), zeros(n))
# end
# nparams(o::ProxGrad) = length(o.∇)
# function Base.show(io::IO, p::ProxGrad)
#     print(io, "ProxGrad")
#     printfields(io, p, [:maxit, :tol, :verbose, :step])
# end
#
# function fit!(o::SparseReg{ProxGrad})
#     for (k, λ) in enumerate(o.θ.λ)
#         avgloss = Inf
#         iters = 0
#         isconverged = false
#         for _ in 1:o.algorithm.maxit
#             iters = update!(o, k, iters)
#             iters >= o.algorithm.maxit && break
#         end
#     end
# end
#
# function update!(o::SparseReg, k, iters)
#     A = o.algorithm
#     O = o.obs
#     β = @view(o.θ.β[:, k])
#     # buffer = xβ, then get objective value
#     A_mul_B!(A.buffer, O.x, β)
#     objval = value(o.loss, O.y, A.buffer, AvgMode.Mean()) + value(o.penalty, β)
#     # buffer = vector of deriv(loss, y, xβ)
#     A.buffer .= deriv.(o.loss, O.y, A.buffer)
#     for i in eachindex(O.y)
#         @inbounds A.buffer[i] = deriv(o.loss, O.y[i], A.buffer[i])
#     end
#     # multiply_weights!(o.deriv_vec, O.w)
#     At_mul_B!(A.∇, O.x, A.buffer)
#     scale!(A.∇, 1 / length(O.y))
#     iters + 1
# end
#
#
# function converged(o::SparseReg{ProxGrad}, newL, niters, β)
#     oldL = newL
#     newL = _L(o, β)
#     niters += 1
#     tolerance = abs(newL - oldL) #/ min(abs(newL), abs(oldL))
#     isconverged = tolerance < o.algorithm.tol
#     isconverged || niters == o.algorithm.maxit &&
#         warn("DID NOT CONVERGE, RelTol = $tolerance")
#     isconverged ?
#         o.algorithm.verbose && info("CONVERGED: $niters, RelTol = $tolerance") :
#         o.algorithm.verbose && info("Iteration: $niters, RelTol = $tolerance")
#     newL, niters, isconverged
# end
