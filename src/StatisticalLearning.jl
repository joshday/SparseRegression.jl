module StatisticalLearning

import StatsBase: StatisticalModel, predict, coef
import Distributions
import Base.LinAlg.BLAS: BlasFloat


export predict, coef, GLM

#-----------------------------------------------------------------------------# types
typealias VecF Vector{Float64}
typealias MatF Matrix{Float64}
typealias AVec{T} AbstractVector{T}
typealias AMat{T} AbstractMatrix{T}
typealias AVecF AVec{Float64}
typealias AMatF AMat{Float64}

abstract Penalty
abstract ModelPath




#--------------------------------------------------------------------------# printing
print_header(io::IO, s::AbstractString) = print_with_color(:blue, io, "■ $s \n")
function print_item(io::IO, name::AbstractString, value)
    println(io, "  >" * @sprintf("%12s", name * ": "), value)
end

macro display(expr) :(display($expr)) end

#----------------------------------------------------------------------# source files
include("penalty.jl")
include("glmpath.jl")

end # module


S = StatisticalLearning
