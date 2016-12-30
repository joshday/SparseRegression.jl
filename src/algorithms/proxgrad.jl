immutable PROXGRAD <: OfflineAlgorithm
    maxit::Int
    tol::Float64
end
PROXGRAD(p::Int = 0; maxit::Int = 100, tol::Float64 = 1e-6) = PROXGRAD(maxit, tol)
is_supported(loss::Loss, pen::Penalty, alg::PROXGRAD) = true

function fit!(o::SparseReg{PROXGRAD}, x::AMat, y::AVec, wts::AVec = zeros(0), penalty_factor::AVec = ones(0))
    n, p = size(x)
    @assert p == length(o.β)
    use_weights = length(wts) > 0
    @assert !use_weights || length(wts) == n "`weights` must have length $n"
    A, L, P = o.algorithm, o.loss, o.penalty

    predict_vec = zeros(n)
    deriv_vec = zeros(n)

    avgloss = meanvalue(L, y, predict(o, x))
    for k in 1:A.maxit
        deriv!(deriv_vec, L)
    end
end

#
#
# #------------------------------------------------------------------------------# fit!
# function fit!{M, P}(o::SparseReg{M, P, Fista}, x::AMat, y::AVec, wts::AVec)
#     #------------------------------------------------------------------------# checks
#     n, p = size(x)
#     alg = o.algorithm
#     @assert size(o.β, 1) == p "Columns of `x` don't match columns in `β`"
#     use_weights = length(wts) > 0  # use weights if they are provided
#     @assert !use_weights || length(wts) == n "`weights` must have length $n"
#     @assert !(o.intercept == false && alg.standardize == true) "standardizing implies an intercept"
#
#     #-------------------------------------------------------------------------# setup
#     β0 = 0.0
#     β = zeros(p)
#     Θ1 = zeros(p)           # last iteration
#     Θ2 = zeros(p)           # two iterations ago
#     Θ0_1 = 0.0              # last iteration
#     Θ0_2 = 0.0              # two iterations ago
#     Δ = zeros(p)            # Δ = x' * deriv_vec
#     deriv_vec = zeros(n)    # derivative of loss with respect to η
#     η = zeros(n)            # linear predictor
#     lossvec = zeros(n)
#     x_std = SM.StandardizedMatrix(x)
#
#     # main loop
#     for k in reverse(eachindex(o.λ))
#         #----------------------------------------------------------# setup for next λ
#         iters = 0
#         newcost = Inf
#         oldcost = Inf
#         @inbounds λ = o.λ[k]
#         s = alg.step
#         #/////////////////////////////////////////////////////////////////# main loop
#         for rep in 1:alg.maxit
#             iters += 1
#             oldcost = newcost
#             #--------------------------------------------------------# Fista momentum
#             copy!(Θ2, Θ1)
#             copy!(Θ1, β)
#             if rep > 2
#                 ratio = (rep - 2) / (rep + 1)
#                 for j in eachindex(β)
#                     @inbounds β[j] = Θ1[j] + ratio * (Θ1[j] - Θ2[j])
#                 end
#                 if o.intercept
#                     Θ0_2 = Θ0_1
#                     Θ0_1 = β0
#                     β0 = Θ0_1 + ratio * (Θ0_1 - Θ0_2)
#                 end
#             end
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
#
#             line_search = true
#             while line_search
#                 #-----------------------------------------# gradient descent and prox
#                 if o.intercept
#                     β0 -= s * mean(deriv_vec)
#                 end
#                 for j in eachindex(β)
#                     @inbounds β[j] -= s * Δ[j]
#                 end
#                 prox!(o.penalty, β, (λ * s) * o.penalty_factor)
#                 #---------------------------------------------# check for convergence
#                 alg.standardize ? A_mul_B!(η, x_std, β) : A_mul_B!(η, x, β)
#                 o.intercept && add_constant!(η, β0)
#                 lossvector!(o.model, lossvec, y, η)
#                 use_weights && mult_vector!(lossvec, wts)
#                 newcost = mean(lossvec) + penalty(o.penalty, β, λ)
#                 if (newcost > oldcost)
#                     # if objective didn't decrease:
#                         # reset coefficients, use step-halving, increase iters
#                     s *= .5
#                     β0 = Θ0_1
#                     copy!(β, Θ1)
#                     iters += 1
#                 else
#                     line_search = false
#                 end
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
# # Elementwise multiplication of arrays, overwrite v1
# function mult_vector!(v1, v2)
#     for i in eachindex(v1)
#         @inbounds v1[i] *= v2[i]
#     end
# end
