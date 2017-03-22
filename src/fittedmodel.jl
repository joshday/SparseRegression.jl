#----------------------------------------------------------------------# FullModelSpec
immutable FittedModel{S <: SparseReg, A <: Algorithm}
    model::S
    algorithm::A
end
function Base.show(io::IO, o::FittedModel)
    print_with_color(:light_cyan, io, "■■■■ FittedModel\n")
    show(io, o.model); println(io)
    show(io, o.algorithm)
end
