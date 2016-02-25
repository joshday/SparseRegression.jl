module StatisticalLearning

import StatsBase
import StatsBase: predict
import UnicodePlots

import Distributions
using Distributions: UnivariateDistribution, Bernoulli, Normal, Poisson


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
