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

d(o::Obs) = (LinearRegression(), L2Penalty(), fill(.1, size(o.x, 2)))

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
xβ(o::SparseReg, xi::AVec) = dot(xi, o.β)

predict(o::SparseReg, x::AVec) = predict(o, x')
predict(o::SparseReg, x::AMat = o.obs.x) = xβ(o, x)
predict(o::SparseReg{MarginLoss}, x::AMat = o.obs.x) = map(x -> 1 / (1 + exp(-x)), xβ(o, x))

factor!(o::SparseReg, f::VecF) = (o.λfactor[:] = f; o)

fitted(o::SparseReg) = predict(o)
residuals(o::SparseReg) = o.obs.y - fitted(o)


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
    header(io, name(o))
    print_item(io, "λ factor", o.path[end].λfactor)
    print_item(io, "Loss", o.path[1].loss)
    print_item(io, "Penalty", o.path[1].penalty)
    for i in 1:length(o.path)
        β = coef(o.path[i])
        println(io, "  > " * @sprintf("β(%.2f) : ", o.λs[i]) * "$β")
    end
end
coef(o::SparseRegPath) = coef.(o.path)

#------------------------------------------------------------------------# gradient!
# To calculate a gradient, we need two storage buffers
#  - length n: stores derivatives with respect to x*β
#  - length p: stores x' * derivatives
function gradient!(nvec::VecF, pvec::VecF, o::SparseReg)
    n, p = size(o.obs)
    A_mul_B!(nvec, o.obs.x, o.β)                      # x * β
    derivatives!(nvec, o)                             # nvec = deriv(loss, y, x*β)
    BLAS.gemv!('T', 1 / n, o.obs.x, nvec, 0.0, pvec)  # ∇ = mean(x' * nvec)
end

function derivatives!(nvec::VecF, o::SparseReg)
    deriv!(nvec, o.loss, o.obs.y, nvec)
    multiply_by_weights!(nvec, o.obs)
end

multiply_by_weights!(nvec::VecF, obs::Obs{Ones}) = nothing
multiply_by_weights!(nvec::VecF, obs::Obs) = (nvec .*= obs.w)

#------------------------------------------------------------------------# learn!
function learn!(o::SparseReg, a::AlgorithmStrategy, m::MaxIter = MaxIter(1), args...)
    ml = MetaLearner(a, m, args...)
    learn!(o, ml)
    o
end
function learn!(path::SparseRegPath, a::AlgorithmStrategy, m::MaxIter = MaxIter(1), args...)
    for o in path.path
        learn!(o, a, m, args...)
    end
    path
end



#------------------------------------------------------------------------# recipes
@recipe function f(o::SparseReg)
    β = coef(o)
    seriestype --> :scatter
    xlab --> "Coefficients"
    ylab --> "Value"
    label --> "Estimate"
    group --> β .!= 0
    1:length(β), β
end

@recipe function f(o::SparseRegPath)
    λfactor = o.path[end].λfactor
    xlab --> "lambda"
    label --> ["factor[$j] = $(λfactor[j])" for j in 1:length(λfactor)]
    β = coef(o)
    nλ, p = length(o.λs), length(β[1])
    mat = zeros(p, nλ)
    for i in 1:nλ
        mat[:, i] = β[i]
    end
    o.λs, mat'
end
