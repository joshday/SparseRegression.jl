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
