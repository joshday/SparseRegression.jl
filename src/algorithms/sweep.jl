immutable Sweep <: OfflineAlgorithm end
immutable SweepBuffer
    a::MatF
    s::MatF
end


function makebuffer(o::SparseReg{Sweep}, obs::Obs{Ones})
    n, p = size(obs.x)
    a = zeros(p + 1, p + 1)
    # xtx
    BLAS.syrk!('U', 'T', 1 / n, obs.x, 0.0, view(a, 1:p, 1:p))
    # xty
    a[1:p, end] = obs.x'obs.y
    scale!(@view(a[1:p, end]), 1 / n)
    # yty
    a[end, end] = dot(obs.y, obs.y)
    a[end, end] /= n
    SweepBuffer(a, copy(a))
end
# weighted version
function makebuffer(o::SparseReg{Sweep}, obs::Obs)
    n, p = size(obs.x)
    a = zeros(p + 1, p + 1)
    wmat = Diagonal(sqrt.(obs.w))
    # x'wx
    BLAS.syrk!('U', 'T', 1 / n, wmat * obs.x, 0.0, view(a, 1:p, 1:p))
    # x'wy
    a[1:p, end] = (wmat * obs.x)'obs.y
    scale!(@view(a[1:p, end]), 1 / n)
    # y'wy
    a[end, end] = dot(obs.y, obs.y .* obs.w)
    a[end, end] /= n
    SweepBuffer(a, copy(a))
end


Sweepable = Union{NoPenalty, L2Penalty}

function fit!(o::SparseReg{Sweep, LinearRegression}, obs::Obs, buffer = makebuffer(o, obs))
    isa(o.penalty, Sweepable) || throw(ArgumentError("Sweep requires NoPenalty or L2Penalty"))
    p = size(obs.x, 2)
    copy!(buffer.s, buffer.a)
    add_ridge!(o, obs, buffer)
    SweepOperator.sweep!(buffer.s, 1:p)
    o.β[:] = buffer.s[1:p, end]
    o
end


function add_ridge!{A, L}(o::SparseReg{A, L, NoPenalty}, obs, buffer) end
function add_ridge!{A, L}(o::SparseReg{A, L, L2Penalty}, obs, buffer)
    for i in 1:size(obs.x, 2)
        buffer.s[i, i] += o.λ * o.factor[i]
    end
end
function add_ridge!{A, L}(o::SparseReg{A, L, L2Penalty}, obs::Obs{Ones}, buffer)
    for i in 1:size(obs.x, 2)
        buffer.s[i, i] += o.λ
    end
end
