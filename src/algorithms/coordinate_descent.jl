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
function fit!{M <: Model, P <: Penalty}(
        o::SparseReg{M, P, CD}, x::AMat, y::AVec, wts::AVecF = ones(0)
    )



end
