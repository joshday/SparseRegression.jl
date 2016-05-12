#-----------------------------------------------------------------------------# FISTA
immutable CD <: Algorithm
    maxit::Int
    tol::Float64
    verbose::Bool
    step::Float64
    criteria::Symbol
    standardize::Bool
end
