#-----------------------------------------------------------------------------# FISTA
immutable CD <: Algorithm
    maxit::Int
    tol::Float64
    verbose::Bool
    step::Float64
    criteria::Symbol
    standardize::Bool
end

#------------------------------------------------------------------------------# fit!
"""
Coordinate Descent
"""
function fit!{M <: Model, P <: Penalty, T <: Real}(
        o::SparseReg{M, P, CD}, x::AMat{T}, y::AVec{T}, wts::AVecF = ones(0)
    )



end
