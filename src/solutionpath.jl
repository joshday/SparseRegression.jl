type SolutionPath{T <: SparseReg}
    o::Vector{T}
end
function SolutionPath(o::SparseReg, λ::AVec)
    all(x -> x>=0, λ) || throw(ArgumentError("λs must all be nonnegative"))
    path = typeof(o)[deepcopy(o) for i in λ]
    for i in eachindex(λ)
        path[i].penalty.λ = λ[i]
    end
    SolutionPath(path)
end
function show(io::IO, o::SolutionPath)
    println(io, "SolutionPath of $(length(o.λ)) λs")
end
