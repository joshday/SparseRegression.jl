"""
```julia
Fista(stepsize)
```
Fast Iterative Shrinkage and Thresholding Algorithm.
"""
mutable struct Fista <: AlgorithmStrategy
    step::Float64
    derivs::VecF
    ∇::VecF
    β1::VecF
    β2::VecF
    t::Int
end
function Fista(step::Float64 = 1.0, n = 0, p = 0)
    Fista(step, zeros(n), zeros(p), zeros(p), zeros(p), 0)
end
Fista(a::Fista, o::Obs) = Fista(a.step, size(o)...)

function learn!(o::SparseReg, a::Fista, item)
    copy!(a.β2, a.β1)
    copy!(a.β1, o.β)
    a.t += 1
    if a.t > 2
        γ = (a.t - 2) / (a.t + 1)
        for j in eachindex(o.β)
            @inbounds o.β[j] = a.β1[j] + γ * (a.β1[j] - a.β2[j])
        end
    end
    @inbounds gradient!(a.derivs, a.∇, o)
    s = a.step
    for j in eachindex(o.β)
        @inbounds o.β[j] = prox(o.penalty, o.β[j] - s * a.∇[j], s * o.λfactor[j])
    end
end

post_hook(a::Fista, o::SparseReg) = (a.t = 0)
