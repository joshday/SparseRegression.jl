# #------------------------------------------------------------------------------# Prox
# # TODO: second deriv version
# # TODO: MM second deriv version
# # TODO: other?
# """
# Experimental Algorithm similar to Fista.
# """
# immutable Prox <: Algorithm
#     maxit::Int
#     tol::Float64
#     verbose::Bool
#     step::Float64
#     crit::Symbol
#     standardize::Bool
# end
# function Prox(;
#         maxit::Integer      = 100,
#         tol::Real           = 1e-7,
#         verbose::Bool       = true,
#         step::Real          = 0.5,
#         crit::Symbol        = :obj,
#         standardize::Bool   = true
#     )
#     Prox(maxit, tol, verbose, step, crit, standardize)
# end
#
# #------------------------------------------------------------------------------# fit!
# function fit!{M <: Model, P <: Penalty}(o::SparseReg{M, P, Prox}, x::AMat, y::AVec, wts::AVec)
#     #------------------------------------------------------------------------# checks
#     n, p = size(x)
#     alg = o.algorithm
#     @assert size(o.β, 1) == p "Columns of `x` don't match columns in `β`"
#     use_weights = length(wts) > 0  # use weights if they are provided
#     @assert !use_weights || length(wts) == n "`weights` must have length $n"
#     @assert !(o.intercept == false && alg.standardize == true) "standardizing implies an intercept"
#
#     #-------------------------------------------------------------------------# setup
#     use_step_halving = (alg.crit == :obj)
#     β0 = 0.0
#     β = zeros(p)
#     Θ1 = zeros(p)           # last iteration
#     Θ2 = zeros(p)           # two iterations ago
#     Θ0_1 = 0.0              # last iteration
#     Θ0_2 = 0.0              # two iterations ago
#     Δ = zeros(p)            # Δ = x' * deriv_vec
#     deriv_vec = zeros(n)    # derivative of loss with respect to η
#     η = zeros(n)            # linear predictor
#     if alg.crit == :obj       # need lossvec if using objective as convergence criteria
#         lossvec = zeros(n)
#     elseif alg.crit == :coef
#         lossvec = zeros(0)
#     end
#     x_std = SM.StandardizedMatrix(x)
#     H0 = 0.0 + .001
#     H = zeros(p) + .01
#
#     # main loop
#     for k in reverse(eachindex(o.λ))
#         #----------------------------------------------------------# setup for next λ
#         iters = 0
#         newcost = Inf
#         oldcost = Inf
#         @inbounds λ = o.λ[k]
#         s = alg.step
#         for rep in 1:alg.maxit
#             iters += 1
#             oldcost = newcost
#             #--------------------------------------------# linear predictor η = x * β
#
#             alg.standardize ? A_mul_B!(η, x_std, β) : A_mul_B!(η, x, β)
#             o.intercept && add_constant!(η, β0)
#             #-----------------------------------------------------# derivative vector
#             for i in eachindex(deriv_vec)
#                 @inbounds deriv_vec[i] = lossderiv(o.model, y[i], η[i])
#             end
#             use_weights && add_vector!(deriv_vec, wts)
#             #-------------------------------------# calculate gradient from deriv_vec
#             alg.standardize ? At_mul_B!(Δ, x_std, deriv_vec) : At_mul_B!(Δ, x, deriv_vec)
#             scale!(Δ, 1 / n)
#             #---------------------------------------------# gradient descent and prox
# 			γ = 1 / (rep + 1)
#             if o.intercept
# 				m = mean(deriv_vec)
# 				H0 = mean(deriv_vec .^ 2)
#                 β0 -= s * m / sqrt(H0)
#             end
#             for j in eachindex(β)
# 				@inbounds H[j] = mean((deriv_vec .* x[:, j]) .^ 2)
#                 @inbounds β[j] -= s * Δ[j] / sqrt(H[j])
#             end
#             prox!(o.penalty, β, λ, o.penalty_factor, s * H)
#             #-------------------------------------------------# check for convergence
#             if alg.crit == :obj
#                 lossvector!(o.model, lossvec, y, η)
#                 use_weights && add_vector!(lossvec, wts)
#                 newcost = mean(lossvec) + penalty(o.penalty, β, λ)
#             elseif alg.crit == :coef
#                 newcost = maxabs(β - Θ1)
#             end
#             if abs(newcost - oldcost) < alg.tol * (min(abs(oldcost), abs(newcost)) + 1.0)
#                 break
#             end
#         end
#         #--------------------------------------# Did the algorithm reach convergence?
#         reltol = abs(newcost - oldcost) / min(abs(oldcost), abs(newcost))
#         if alg.maxit == iters
#             warn("Not converged for λ = $(o.λ[k]).  Tolerance = $(round(reltol, 12))")
#         end
#         #---------------------------------------------------------# update parameters
#         if o.intercept
#             o.β0[k] = β0
#         end
#         o.β[:, k] = β
#     end  # end main loop
#     if alg.standardize
#         scaled_to_original!(o, x_std)
#     end
#     o
# end
#
#
# #---------------------------------------------------------------------------# helpers
# # put scaled coefficients back in original scale
# function scaled_to_original!(o::SparseReg, x_std::SM.StandardizedMatrix)
#     p, d = size(o.β)
#     σx = x_std.σinv
#     μx = x_std.μ
#     scale!(σx, o.β)
#     for j in eachindex(o.β0)
#         o.β0[j] = o.β0[j] - dot(μx, o.β[:, j])
#     end
# end
#
# # Add constant to an array
# function add_constant!(arr, c)
#     for i in eachindex(arr)
#         @inbounds arr[i] += c
#     end
# end
#
# # Add arrays, overwrite v1
# function add_vector!(v1, v2)
#     for i in eachindex(v1)
#         @inbounds v1[i] += v2[i]
#     end
# end
