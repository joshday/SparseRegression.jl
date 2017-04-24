# -------------------------------------------------------------------------# ProxGrad
"""
    ProxGrad(s)
Proximal Gradient Descent.  `s` is the step size for the gradient descent part of the algorithm.
### Example
    using DataGenerator, SparseRegression
    x, y, b = linregdata(1000, 10)
    s = SparseReg(Obs(x,y), LinearRegression(), L2Penalty(), .1)
    fit!(s, ProxGrad(), MaxIter(50), Converged(coef))
"""
mutable struct ProxGrad <: AlgorithmStrategy
    step::Float64
    derivs::VecF
    ∇::VecF
    ProxGrad(step::Float64 = 1.0) = new(step, zeros(0), zeros(0))
end

function pre_hook(a::ProxGrad, s::SparseReg)
    n, p = size(s.obs)
    a.derivs = zeros(n)
    a.∇ = zeros(p)
end

function learn!(o::SparseReg, a::ProxGrad, item::Void)
    gradient!(a.derivs, a.∇, o)
    s = a.step
    for j in eachindex(o.β)
        @inbounds o.β[j] = prox(o.penalty, o.β[j] - s * a.∇[j], s * o.λfactor[j])
    end
    o
end
