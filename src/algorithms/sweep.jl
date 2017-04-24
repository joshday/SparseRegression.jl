mutable struct Sweep <: AlgorithmStrategy
    A::MatF  # [x y]'[x y]
    S::MatF  # sweep!(a, 1:p)
    Sweep() = new(zeros(0,0), zeros(0,0))
end

function pre_hook(a::Sweep, s::SparseReg)
    if length(a.A) == 0
        a.A = make_A(s.obs)
        a.S = zeros(a.A)
    end
end

function make_A(obs::Obs)
    n, p = size(obs.x)
    a = zeros(p + 1, p + 1)
    b = zeros(p + 1, p + 1)
    BLAS.syrk!('U', 'T', 1 / n, Diagonal(sqrt.(obs.w)) * obs.x, 0.0, view(a, 1:p, 1:p)) # x'wx
    BLAS.gemv!('T', 1 / n, Diagonal(obs.w) * obs.x, obs.y, 0.0, @view(a[1:p, end]))     # x'wy
    a[end, end] = dot(obs.y, Diagonal(obs.w) * obs.y) / n                               # y'wy
    a
end
function make_A(obs::Obs{Ones})
    n, p = size(obs.x)
    a = zeros(p + 1, p + 1)
    BLAS.syrk!('U', 'T', 1 / n, obs.x, 0.0, view(a, 1:p, 1:p))      # x'x
    BLAS.gemv!('T', 1 / n, obs.x, obs.y, 0.0, @view(a[1:p, end]))   # x'y
    a[end, end] = dot(obs.y, obs.y) / n                             # y'y
    a
end

function learn!(o::SparseReg, a::Sweep, item::Void)
    n, p = size(o.obs)
    copy!(a.S, a.A)
    add_ridge!(o, a, o.λfactor)
    SweepOperator.sweep!(a.S, 1:p)
    o.β[:] = a.S[1:p, end]
end


function add_ridge!{L}(o::SparseReg{L, NoPenalty}, a::Sweep, λf::VecF) end
function add_ridge!{L}(o::SparseReg{L, L2Penalty}, a::Sweep, λf::VecF)
    for i in eachindex(o.β)
        a.S[i, i] += λf[i]
    end
end
