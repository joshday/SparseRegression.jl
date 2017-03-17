immutable SolutionPath{S <: SparseReg, A <: Algorithm, T <: AVecF}
    models::Vector{S}
    λs::T
    algorithm::A
end

function SolutionPath{L, P}(m::SparseReg{L, P}, λs::AVecF, A::Algorithm)
    models = SparseReg{L, P}[]
    for λ in reverse(sort(λs))
        o = SparseReg(m, λ)
        fit!(o, A)
        push!(models, o)
    end
    @show typeof(models)
    SolutionPath(models, λs, A)
end

function Base.show(io::IO, path::SolutionPath)
    print_with_color(default_color, io, "■ $(name(path))\n")
    for o in path.models
        print_item(io, "β(λ=$(round(o.λ, 4)))", o.β)
    end
end

function fitpath(x::AMat, y::AVec, args...; λs::AVecF = linspace(0,1,20), kw...)
    n, p = size(x)
    o = SparseReg(p, args...)
    alg = default_algorithm(o, Obs(x, y); kw...)
    SolutionPath(o, λs, alg)
end
function fitpath(x::AMat, y::AVec, w::AVec, args...; λs::VecF = linspace(0,1,20), kw...)
    n, p = size(x)
    o = SparseReg(p, args...)
    alg = default_algorithm(o, Obs(x, y, w); kw...)
    SolutionPath(o, λs, alg)
end
