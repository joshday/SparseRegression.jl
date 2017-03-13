immutable Sweep <: OfflineAlgorithm end
immutable SweepBuffer
    a::MatF
    s::MatF
end

# TODO: optimize for GC
# TODO: doesn't work with weights yet
function makebuffer(o::SparseReg{Sweep}, obs::Obs)
    xy = [obs.x obs.y]
    a = xy'xy
    scale!(a, 1 / length(obs.y))
    SweepBuffer(a, a)
end

Sweepable = Union{NoPenalty, L2Penalty}

function fit!(o::SparseReg{Sweep, LinearRegression}, obs::Obs, buffer = makebuffer(o, obs))
    isa(o.penalty, Sweepable) || throw(ArgumentError("Sweep requires NoPenalty or L2Penalty"))
    p = size(obs.x, 2)
    add_ridge!(o, obs, buffer)
    copy!(buffer.s, buffer.a)
    SweepOperator.sweep!(buffer.s, 1:p)
    o.β[:] = buffer.s[1:p, end]
    o
end


function add_ridge!{A, L}(o::SparseReg{A, L, NoPenalty}, obs, buffer) end
function add_ridge!{A, L}(o::SparseReg{A, L, L2Penalty}, obs, buffer)
    for i in 1:size(obs.x, 2)
        buffer.a[i, i] += o.λ * o.factor[i]
    end
end
function add_ridge!{A, L}(o::SparseReg{A, L, L2Penalty}, obs::Obs{Ones}, buffer)
    for i in 1:size(obs.x, 2)
        buffer.a[i, i] += o.λ
    end
end
