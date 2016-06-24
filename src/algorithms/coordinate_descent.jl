# #-----------------------------------------------------------------------------# Fista
# immutable CD <: Algorithm
#     maxit::Int
#     tol::Float64
#     verbose::Bool
#     step::Float64
#     criteria::Symbol
#     standardize::Bool
#
#     function CD(;
#             maxit::Integer      = 100,
#             tol::Real           = 1e-7,
#             verbose::Bool       = true,
#             step::Real          = 0.5,
#             crit::Symbol    = :obj,
#             standardize::Bool   = true
#         )
#         new(maxit, tol, verbose, step, crit, standardize)
#     end
# end
#
# #------------------------------------------------------------------------------# fit!
# """
# Coordinate Descent
# """
# function fit!{M <: Model, P <: Penalty}(
#         o::SparseReg{M, P, CD}, x::AMat, y::AVec, wts::AVecF = ones(0)
#     )
#     #----------------------------------------------------------------# error checking
#     @assert typeof(o.penalty) == ElasticNetPenalty "Only ElasticNetPenalty is supported"
#     n, p = size(x)
#     alg = o.algorithm
#     @assert size(o.β, 1) == p "Columns of `x` don't match columns in `β`"
#     use_weights = length(wts) > 0  # use weights if they are provided
#     @assert !use_weights || length(wts) == n "`weights` must have length $n"
#     @assert !(o.intercept == false && alg.standardize == true) "standardizing implies an intercept"
#     #-------------------------------------------------------------------------# setup
#     β0 = 0.0
#     β = zeros(p)
#     Δ = zeros(p)            # Δ = x' * deriv_vec
#     deriv_vec = zeros(n)    # derivative of loss with respect to η
#     η = zeros(n)            # linear predictor
#     x_std = SM.StandardizedMatrix(x)
#
#     #/////////////////////////////////////////////////////////////////////# main loop
#     for k in reverse(eachindex(o.λ))
#         iters = 0
#         newcost = Inf
#         oldcost = Inf
#         @inbounds λ = o.λ[k]
#         s = alg.step
#
#         for rep in 1:alg.maxit
#             #--------------------------------------------# linear predictor η = x * β
#             alg.standardize ? A_mul_B!(η, x_std, β) : A_mul_B!(η, x, β)
#             o.intercept && add_constant!(η, β0)
#             #-----------------------------------------------------# derivative vector
#             for i in eachindex(deriv_vec)
#                 @inbounds deriv_vec[i] = lossderiv(o.model, y[i], η[i])
#             end
#             use_weights && mult_vector!(deriv_vec, wts)
#             #-------------------------------------# calculate gradient from deriv_vec
#             alg.standardize ? At_mul_B!(Δ, x_std, deriv_vec) : At_mul_B!(Δ, x, deriv_vec)
#             scale!(Δ, 1 / n)
#             for j in eachindex(β)
#                 #------------------------------------------------# coordinate-wise update
#                 β[k, j]
#             end
#         end
#     end
# end
