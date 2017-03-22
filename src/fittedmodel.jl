#----------------------------------------------------------------------# FullModelSpec
immutable FittedModel{S <: SparseReg, A <: Algorithm, O <: Obs}
    model::S
    algorithm::A
    obs::O
end
function Base.show(io::IO, o::FittedModel)
    print_with_color(:light_cyan, io, "■■■■ FittedModel\n")
    show(io, o.model); println(io)
    show(io, o.algorithm); println(io)
    show(io, o.obs)
end
