"""
```julia
Fista(s)
```
Fast Iterative Shrinkage and Thresholding Algorithm with step size `s`.
"""
mutable struct Fista <: AlgorithmStrategy
    step::Float64
    derivs::VecF
    ∇::VecF
    β1::VecF
    β2::VecF
    t::Int
    Fista(step::Float64 = 1.0) = new(step, zeros(0), zeros(0), zeros(0), zeros(0), 0)
end

function pre_hook(a::Fista, s::SparseReg)
    n, p = size(s.obs)
    a.derivs = zeros(n)
    a.∇ = zeros(p)
    a.β1 = zeros(p)
    a.β2 = zeros(p)
end

function learn!(o::SparseReg, a::Fista, item::Void)
    copy!(a.β2, a.β1)
    copy!(a.β1, o.β)
    a.t += 1
    if a.t > 2
        γ = (a.t - 2) / (a.t + 1)
        for j in eachindex(o.β)
            @inbounds o.β[j] = a.β1[j] + γ * (a.β1[j] - a.β2[j])
        end
    end
    gradient!(a.derivs, a.∇, o)
    s = a.step
    for j in eachindex(o.β)
        @inbounds o.β[j] = prox(o.penalty, o.β[j] - s * a.∇[j], s * o.λfactor[j])
    end
end

post_hook(a::Fista, o::SparseReg) = (a.t = 0)
