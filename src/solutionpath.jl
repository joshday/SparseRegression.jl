type SolutionPath{T <: SparseReg}
    o::Vector{T}
end
function SolutionPath(o::SparseReg, λ::AVec, x::AMat, y::AVec; verbose = true)
    all(x -> x>=0, λ) || throw(ArgumentError("λs must all be nonnegative"))
    path = typeof(o)[deepcopy(o) for i in λ]
    for i in eachindex(λ)
        path[i].penalty.λ = λ[i]
    end
    nλs = length(λ)
    for i in reverse(eachindex(path))
        verbose && info("$i models left")
        if i > 1
            path[i].β = path[i - 1].β  # warm start
        end
        fit!(path[i], x, y)
    end
    SolutionPath(path)
end
function show(io::IO, o::SolutionPath)
    println(io, "SolutionPath of $(length(o.λ)) λs")
end
