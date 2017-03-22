immutable SolutionPath{T <: SparseReg}
    models::Vector{T}
    λs::VecF
end
function Base.show(io::IO, path::SolutionPath)
    header(io, name(path, true))
    for i in eachindex(path.models)
        print_item(io, "β(λ=$(path.λs[i]))", coef(path.models[i]))
    end
end



function SolutionPath(o::SparseReg, λs::VecF = collect(0:.01:.1))
    p = length(o.β)
    models = [SparseReg(p, o.loss, o.penalty, o.factor, λs[1])]
    for λ in λs[2:end]
        push!(models, SparseReg(p, o.loss, o.penalty, o.factor, λ))
    end
    SolutionPath(models, λs)
end
