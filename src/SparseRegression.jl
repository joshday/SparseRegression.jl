module SparseRegression

import StatsBase: coef, predict, zscore, fit!, loglikelihood
import Requires
import StandardizedMatrices; SM = StandardizedMatrices

export
    SparseReg,
    NoPenalty, RidgePenalty, LassoPenalty, ElasticNetPenalty, SCADPenalty,
    L2Regression, L1Regression, LogisticRegression, SVMLike, QuantileRegression,
    HuberRegression, PoissonRegression,
    FISTA,
    coef, predict, fit!, loglikelihood

#-----------------------------------------------------------------------------# types
typealias VecF Vector{Float64}
typealias MatF Matrix{Float64}
typealias AVec{T} AbstractVector{T}
typealias AMat{T} AbstractMatrix{T}
typealias AVecF AVec{Float64}
typealias AMatF AMat{Float64}

abstract Algorithm

#--------------------------------------------------------------------------# printing
print_header(io::IO, s::AbstractString) = print_with_color(:blue, io, "■ $s \n")
function print_item(io::IO, name::AbstractString, value)
    println(io, "  >" * @sprintf("%14s", name * ":  "), value)
end


#----------------------------------------------------------------------# source files
include("penalty.jl")
include("model.jl")
include("sparsereg.jl")
include("algorithms/fista.jl")
include("algorithms/sweep.jl")
include("algorithms/coordinate_descent.jl")


Requires.@require Plots include("plots.jl")

end # module
sp = SparseRegression
