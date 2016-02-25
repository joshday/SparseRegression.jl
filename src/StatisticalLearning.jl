module StatisticalLearning

import StatsBase, Distributions
import StatsBase: predict
Ds = Distributions


#-----------------------------------------------------------------------------# types
abstract Penalty
abstract Link


#--------------------------------------------------------------------------# printing
print_header(io::IO, s::AbstractString) = print_with_color(:blue, io, "▌ $s \n")

function print_item(io::IO, name::AbstractString, value)
    println(io, "  >" * @sprintf("%12s", name * ": "), value)
end





macro display(expr) :(display($expr)) end

#----------------------------------------------------------------------# source files
include("link_penalty.jl")
include("glm.jl")


end # module


S = StatisticalLearning
