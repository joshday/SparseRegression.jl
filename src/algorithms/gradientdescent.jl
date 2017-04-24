# -------------------------------------------------------------------------# GradientDescent
"""
    GradientDescent(s)
Gradient Descent with step size `s`.
"""
mutable struct GradientDescent <: AlgorithmStrategy
    step::Float64
    derivs::VecF
    ∇::VecF
    GradientDescent(step::Float64 = 1.0) = new(step, zeros(0), zeros(0))
end

function pre_hook(a::GradientDescent, s::SparseReg)
    if length(a.derivs) == 0
        n, p = size(s.obs)
        a.derivs = zeros(n)
        a.∇ = zeros(p)
    end
end

function learn!(o::SparseReg, a::GradientDescent, item::Void)
    gradient!(a.derivs, a.∇, o)
    s = a.step
    for j in eachindex(o.β)
        @inbounds o.β[j] -= s * (a.∇[j] + deriv(o.penalty, o.β[j]))
    end
    o
end
