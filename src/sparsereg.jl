struct SparseReg{L <: Loss, P <: Penalty, O <: Obs}
    β::VecF
    λfactor::VecF
    loss::L
    penalty::P
    obs::O
end

function SparseReg(o::Obs, t::Tuple)
    l = t[1]
    p = t[2]
    λf = t[3]
    n, d = size(o)
    SparseReg(zeros(d), λf, l, p, o)
end

d(o::Obs, λ::Float64 = .1) = (LinearRegression(), L2Penalty(), fill(λ, size(o.x, 2)))

a(argu::Loss, t::Tuple)     = (argu, t[2], t[3])
a(argu::Penalty, t::Tuple)  = (t[1], argu, t[3])
a(argu::VecF, t::Tuple)     = (t[1], t[2], argu)

SparseReg(o::Obs)               = SparseReg(o, d(o))
SparseReg(o::Obs, a1)           = SparseReg(o, a(a1, d(o)))
SparseReg(o::Obs, a1, a2)       = SparseReg(o, a(a2, a(a1, d(o))))
SparseReg(o::Obs, a1, a2, a3)   = SparseReg(o, a(a3, a(a2, a(a1, d(o)))))

function Base.show(io::IO, o::SparseReg)
    header(io, name(o))
    print_item(io, "β", o.β')
    print_item(io, "λ factor", o.λfactor')
    print_item(io, "Loss", o.loss)
    print_item(io, "Penalty", o.penalty)
end

coef(o::SparseReg) = o.β

xβ(o::SparseReg, x::AMat = o.obs.x) = x * o.β
xβ(o::SparseReg, xi::AVec) = dot(x, o.β)

predict(o::SparseReg, x::AMat = o.obs.x) = xβ(o, x)
predict(o::SparseReg{MarginLoss}, x::AMat = o.obs.x) = map(x -> 1 / (1 + exp(-x)), xβ(o, x))

factor!(o::SparseReg, f::VecF) = (o.λfactor[:] = f; o)


#------------------------------------------------------------------------# SparseRegPath
struct SparseRegPath{S <: SparseReg}
    path::Vector{S}
    λs::VecF
end
function SparseRegPath(o::SparseReg, λs::AVecF)
    λf = o.λfactor
    SparseRegPath([SparseReg(o.obs, o.loss, o.penalty, λ * λf) for λ in λs], collect(λs))
end
function Base.show(io::IO, o::SparseRegPath)
    println(io, name(o, true))
    for i in 1:length(o.path)
        β = coef(o.path[i])
        println(io, "  > " * @sprintf("β(%.2f) : ", o.λs[i]) * "$β")
    end
end

#------------------------------------------------------------------------# gradient!
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

#------------------------------------------------------------------------# learn!
function learn!(o::SparseReg, a::AlgorithmStrategy, m::MaxIter = MaxIter(1), args...)
    a2 = typeof(a)(a, o.obs)
    ml = MetaLearner(a2, m, args...)
    learn!(o, ml)
    o
end
function learn!(path::SparseRegPath, a::AlgorithmStrategy, m::MaxIter = MaxIter(1), args...)
    for o in path.path
        learn!(o, a, m, args...)
    end
    path
end
