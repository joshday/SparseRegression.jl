type SolutionPath{T <: SparseReg}
    o::Vector{T}
end

# Init should be SparseReg with largest penalty
function SolutionPath(init::SparseReg, x::AMat, y::AVec; verbose = true)

end
function show(io::IO, o::SolutionPath)
    println(io, "SolutionPath of $(length(o.λ)) λs")
end
