"""
Constructors take form: `MyAlgorithm(obs; kw...)`
"""
abstract type Algorithm end

struct SparseReg{A <: Algorithm, L <: Loss, P <: Penalty, O <: Obs}
    θ::Coefficients
    loss::L
    penalty::P
    factor::VecF
    obs::O
    algorithm::A
end
function SparseReg(o::Obs, l::Loss, pen::Penalty, a::Type = ProxGrad;
                   λ::VecF = collect(linspace(0, 1, 10)),
                   factor::VecF = ones(nparams(o)),
                   kw...)
    s = SparseReg(Coefficients(o, λ), l, pen, factor, o, a(o; kw...))
    fit!(s)
    s
end
# function SparseReg(x::AMat, y::AVec, alg::Type, args...; kw...)
#     SparseReg(alg(Obs(x, y); kw...), args...)
# end
function Base.show(io::IO, o::SparseReg)
    header(io, name(o))
    show(io, o.θ)
    header(io, "Model Specification")
    print_item(io, "Loss", o.loss)
    print_item(io, "Penalty", o.penalty)
    print_item(io, "λ factor", o.factor')
    print_item(io, "Algorithm", o.algorithm)
end

factor!(o::SparseReg, f::VecF) = (o.factor[:] = f)

#-------------------------------------------------------------------------# ProxGrad
struct ProxGrad <: Algorithm
    # config
    maxit::Int
    tol::Float64
    verbose::Bool
    step::Float64
    # buffers
    ∇::VecF
    buffer::VecF
end
function ProxGrad(obs::Obs; maxit::Int = 100, tol::Float64 = 1e-6,
                  verbose::Bool = false, step::Float64 = 1.0)
    n, p = size(obs)
    ProxGrad(maxit, tol, verbose, step, zeros(p), zeros(n))
end
nparams(o::ProxGrad) = length(o.∇)
function Base.show(io::IO, p::ProxGrad)
    print(io, "ProxGrad")
    printfields(io, p, [:maxit, :tol, :verbose, :step])
end

function fit!(o::SparseReg{ProxGrad})
    for (k, λ) in enumerate(o.θ.λ)
        avgloss = Inf
        iters = 0
        isconverged = false
        A = o.algorithm
        O = o.obs
        for _ in 1:o.algorithm.maxit
            update!(o, A, O, k)
            # iters, avgloss, isconverged = converged(iters, avgloss, isconverged)
        end
    end
end

function update!(o::SparseReg, A::ProxGrad, O::Obs, k)
    β = @view(o.θ.β[:, k])
    # buffer = xβ, then get objective value
    A_mul_B!(A.buffer, O.x, β)
    objval = value(o.loss, O.y, A.buffer, AvgMode.Mean()) + value(o.penalty, β)
    # buffer = vector of deriv(loss, y, xβ)
    for i in eachindex(O.y)
        @inbounds A.buffer[i] = deriv(o.loss, O.y[i], A.buffer[i])
    end
    # multiply_weights!(o.deriv_vec, O.w)
    At_mul_B!(A.∇, O.x, A.buffer)
    scale!(A.∇, 1 / length(O.y))
end


function converged(o::SparseReg{ProxGrad}, newL, niters, β)
    oldL = newL
    newL = _L(o, β)
    niters += 1
    tolerance = abs(newL - oldL) #/ min(abs(newL), abs(oldL))
    isconverged = tolerance < o.algorithm.tol
    isconverged || niters == o.algorithm.maxit &&
        warn("DID NOT CONVERGE, RelTol = $tolerance")
    isconverged ?
        o.algorithm.verbose && info("CONVERGED: $niters, RelTol = $tolerance") :
        o.algorithm.verbose && info("Iteration: $niters, RelTol = $tolerance")
    newL, niters, isconverged
end
