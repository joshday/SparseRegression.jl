immutable Sweep <: OfflineAlgorithm end
immutable SweepBuffer
    a::MatF
    s::MatF
end
function makebuffer(o::SparseReg{Sweep}, obs::Obs)
    xy = [obs.x obs.y]
    a = xy'xy / length(obs.y)
    SweepBuffer(a, a)
end

function fit!(o::SparseReg{Sweep, LinearRegression}, obs::Obs, buffer = makebuffer(o, obs))
    isa(o.penalty, NoPenalty) ||
        isa(o.penalty, L2Penalty) ||
            throw(ArgumentError("Sweep algorithm only works with NoPenalty or L2Penalty"))
    n, p = size(obs.x)
    copy!(buffer.s, buffer.a)
    SweepOperator.sweep!(buffer.s, 1:p)
    o.β[:] = buffer.s[1:p, end]
    o
end
