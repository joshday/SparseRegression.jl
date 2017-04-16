# -------------------------------------------------------------------------# ProxGrad
"""
    ProxGrad(stepsize)
Proximal Gradient Descent.  `stepsize` is the step size for the gradient descent part of the algorithm.
### Example
    using DataGenerator, SparseRegression
    x, y, b = linregdata(1000, 10)
    s = SparseReg(Obs(x,y), LinearRegression(), L2Penalty(), .1)
    fit!(s, ProxGrad(), MaxIter(50), Converged(coef))
"""
struct ProxGrad <: AlgorithmStrategy
    step::Float64
    derivs::VecF
    ∇::VecF
end
ProxGrad(step::Float64 = 1.0, n = 0, p = 0) = ProxGrad(step, zeros(n), zeros(p))
ProxGrad(a::ProxGrad, o::Obs) = ProxGrad(a.step, size(o)...)

function learn!(o::SparseReg, a::ProxGrad, item)
    @inbounds gradient!(a.derivs, a.∇, o)
    s = a.step
    for j in eachindex(o.β)
        @inbounds o.β[j] = prox(o.penalty, o.β[j] - s * a.∇[j], s * o.λfactor[j])
    end
end
