#----------------------------------------------------------------# coordinate descent
function cd!{M <: Model}(o::SparseReg{M}, x::AMatF, y::AVecF;
        maxit::Integer      = 100,
        tol::Float64        = 1e-7,
        verbose::Bool       = true,
        step::Float64       = 1.0,
        weights::AVecF      = ones(0),
        standardize::Bool   = false
    )
    error("Not implemented yet.")
end
