# -------------------------------------------------------------------------# GradientDescent
"""
    GradientDescent(s)
Gradient Descent with step size `s`.
"""
struct GradientDescent <: AlgorithmStrategy
    step::Float64
    derivs::VecF
    ∇::VecF
end
GradientDescent(step::Float64 = 1.0, n = 0, p = 0) = GradientDescent(step, zeros(n), zeros(p))
GradientDescent(a::GradientDescent, o::Obs) = GradientDescent(a.step, size(o)...)

function learn!(o::SparseReg, a::GradientDescent, item::Void)
    gradient!(a.derivs, a.∇, o)
    s = a.step
    for j in eachindex(o.β)
        @inbounds o.β[j] -= s * (a.∇[j] + deriv(o.penalty, o.β[j]))
    end
end
