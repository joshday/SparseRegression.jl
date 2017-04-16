# -------------------------------------------------------------------------# ProxGrad
struct ProxGrad <: LearningStrategy
    step::Float64
    derivs::VecF
    ∇::VecF
end
ProxGrad(step::Float64 = 1.0, n = 0, p = 0) = ProxGrad(step, zeros(n), zeros(p))
ProxGrad(a::ProxGrad, o::Obs) = ProxGrad(a.step, size(o)...)

function learn!(o::SparseReg, a::ProxGrad, i)
    gradient!(a.derivs, a.∇, o)
    s = a.step
    for j in eachindex(o.β)
        @inbounds o.β[j] = prox(o.penalty, o.β[j] - s * a.∇[j], s * o.λfactor[j])
    end
end

function fit!(o::SparseReg, a::ProxGrad, m::MaxIter = MaxIter(), args::LearningStrategy...)
    a2 = ProxGrad(a, o.obs)
    ml = MetaLearner(a2, args...)
    learn!(o, ml)
    o
end
