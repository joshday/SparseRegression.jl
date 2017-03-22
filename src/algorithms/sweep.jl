immutable Sweep{O <: Obs} <: OfflineAlgorithm
    obs::O
    a::MatF
    s::MatF
end
function Sweep(obs::Obs)
    n, p = size(obs.x)
    a = zeros(p + 1, p + 1)
    b = zeros(p + 1, p + 1)
    BLAS.syrk!('U', 'T', 1 / n, Diagonal(sqrt.(obs.w)) * obs.x, 0.0, view(a, 1:p, 1:p))  # x'wx
    BLAS.gemv!('T', 1 / n, Diagonal(obs.w) * obs.x, obs.y, 0.0, @view(a[1:p, end]))     # x'wy
    a[end, end] = dot(obs.y, Diagonal(obs.w) * obs.y) / n                               # y'wy
    Sweep(obs, a, b)
end
function Sweep(obs::Obs{Ones})
    n, p = size(obs.x)
    a = zeros(p + 1, p + 1)
    b = zeros(p + 1, p + 1)
    BLAS.syrk!('U', 'T', 1 / n, obs.x, 0.0, view(a, 1:p, 1:p))      # x'x
    BLAS.gemv!('T', 1 / n, obs.x, obs.y, 0.0, @view(a[1:p, end]))   # x'y
    a[end, end] = dot(obs.y, obs.y) / n                             # y'y
    Sweep(obs, a, b)
end

Sweepable = Union{NoPenalty, L2Penalty}

function fit!{P<:Sweepable}(o::SparseReg{LinearRegression, P}, A::Sweep, obs::Obs)
    p = size(obs.x, 2)
    copy!(A.s, A.a)
    add_ridge!(o, A, obs)
    SweepOperator.sweep!(A.s, 1:p)
    o.β[:] = A.s[1:p, end]
    o
end


function add_ridge!{L}(o::SparseReg{L, NoPenalty}, A, obs) end
function add_ridge!{L}(o::SparseReg{L, L2Penalty}, A, obs)
    for i in 1:size(obs.x, 2)
        A.s[i, i] += o.λ * o.factor[i]
    end
end
