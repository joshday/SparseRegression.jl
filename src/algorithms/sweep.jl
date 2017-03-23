SweepLoss = Union{LinearRegression, L2DistLoss}
SweepPenalty = Union{NoPenalty, L2Penalty}

immutable SweepModel{L <: SweepLoss, P <: SweepPenalty, O <: Obs} <: AbstractSparseReg
    # AbstractSparseReg
    θ::Coefficients
    loss::L
    penalty::P
    factor::VecF
    # observations
    obs::O
    # buffers
    a::MatF   # [x y]'[x y]
    s::MatF   # sweep!(a, 1:p)
end
function SweepModel(obs::Obs; λ::VecF = defaultλ(), loss::SweepLoss = LinearRegression(),
        penalty::SweepPenalty = NoPenalty(), factor::VecF = ones(size(obs.x, 2)))
    c = Coefficients(obs, λ)
    a = make_a(obs)
    o = SweepModel(c, loss, penalty, factor, obs, a, copy(a))
    fit!(o)
    o
end

function make_a(obs::Obs)
    n, p = size(obs.x)
    a = zeros(p + 1, p + 1)
    b = zeros(p + 1, p + 1)
    BLAS.syrk!('U', 'T', 1 / n, Diagonal(sqrt.(obs.w)) * obs.x, 0.0, view(a, 1:p, 1:p))  # x'wx
    BLAS.gemv!('T', 1 / n, Diagonal(obs.w) * obs.x, obs.y, 0.0, @view(a[1:p, end]))     # x'wy
    a[end, end] = dot(obs.y, Diagonal(obs.w) * obs.y) / n                               # y'wy
    a
end
function make_a(obs::Obs{Ones})
    n, p = size(obs.x)
    a = zeros(p + 1, p + 1)
    BLAS.syrk!('U', 'T', 1 / n, obs.x, 0.0, view(a, 1:p, 1:p))      # x'x
    BLAS.gemv!('T', 1 / n, obs.x, obs.y, 0.0, @view(a[1:p, end]))   # x'y
    a[end, end] = dot(obs.y, obs.y) / n                             # y'y
    a
end

function fit!(o::SweepModel)
    p = nparams(o)
    for (k, λ) in enumerate(o.θ.λ)
        copy!(o.s, o.a)
        add_ridge!(o, λ)
        SweepOperator.sweep!(o.s, 1:p)
        o.θ.β[:, k] = o.s[1:p, end]
    end
end
#
#
function add_ridge!{L}(o::SweepModel{L, NoPenalty}, λ) end
function add_ridge!{L}(o::SweepModel{L, L2Penalty}, λ)
    for i in 1:nparams(o)
        o.s[i, i] += λ * o.factor[i]  # TODO: check math
    end
end
