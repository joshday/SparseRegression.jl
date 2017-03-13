immutable SolutionPath{T <: SparseReg, S}
    o::Vector{T}
    buffer::S
end

# initialize
function SolutionPath(n::Integer, o::SparseReg, x::AMat, y::AVec)
    buffer = makebuffer(o.algorithm, x, y)
    fit!(o, x, y, buffer)
    SolutionPath([deepcopy(o) for i in 1:n], buffer)
end


function Base.show(io::IO, o::SolutionPath)
    println(io, "Solution Path of $(length(o.o)) Models")
    for obj in o.o
        print_item(io, "")
    end
end
