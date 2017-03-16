#-------------------------------------------------------------------------------# printing
name(a) = replace(string(typeof(a)), "SparseRegression.", "")
function print_item(io::IO, name::AbstractString, value)
    print(io, "  >" * @sprintf("%13s", name * ":  "))
    println(io, value)
end
function Base.show(io::IO, o::AbstractSparseReg)
    println(io, "Sparse Regression Model")
    print_item(io, "β", o.β)
    print_item(io, "Loss", o.loss)
    print_item(io, "Penalty", o.penalty)
    typeof(o.penalty) != NoPenalty && print_item(io, "λ", o.λ)
    any(x -> x != 1.0, o.factor) && print_item(io, "λ scaling", o.factor)
    print_item(io, "Algorithm", o.algorithm)
end

#-------------------------------------------------------------------------------# helpers
# scary names so that nobody uses them
predict_from_xβ(l::Loss, xβ::Real) = xβ
predict_from_xβ(l::LogitMarginLoss, xβ::Real) = logistic(xβ)
predict_from_xβ(l::PoissonLoss, xβ::Real) = exp(xβ)
function xβ_to_ŷ!(l::Union{LogitMarginLoss, PoissonLoss}, xβ::AVec)
    for i in eachindex(xβ)
        @inbounds xβ[i] = predict_from_xβ(l, xβ[i])
    end
    xβ
end
xβ_to_ŷ!(l::Loss, xβ::AVec) = xβ;  # no-op if linear predictor == ŷ

function objective_value(o::SparseReg, obs::Obs{Ones}, ŷ::AVec)
    value(o.loss, obs.y, ŷ, AvgMode.Mean()) + value(o.penalty, o.β)
end
function objective_value(o::SparseReg, obs::Obs, ŷ::AVec)
    value(o.loss, obs.y, ŷ, AvgMode.WeightedMean(obs.w)) + value(o.penalty, o.β)
end

#-------------------------------------------------------------------------------# methods
coef(o::AbstractSparseReg) = o.β
logistic(x::Float64) = 1.0 / (1.0 + exp(-x))
xβ(o::AbstractSparseReg, x::AMat) = x * o.β
xβ(o::AbstractSparseReg, x::AVec) = dot(x, o.β)
predict(o::AbstractSparseReg, x::AVec) = predict_from_xβ(o.loss, xβ(o, x))
predict(o::AbstractSparseReg, x::AMat) = predict_from_xβ.(o.loss, xβ(o, x))

classify{A, L<:MarginLoss}(o::SparseReg{A,L}, x::AVec) = sign(xβ(o, x))
classify{A, L<:MarginLoss}(o::SparseReg{A,L}, x::AMat) = sign.(xβ(o, x))
classify{A, L<:MarginLoss}(o::StreamReg{A,L}, x::AVec) = sign(xβ(o, x))
classify{A, L<:MarginLoss}(o::StreamReg{A,L}, x::AMat) = sign.(xβ(o, x))

loss(o::AbstractSparseReg, x::AMat, y::AVec, args...) =
    value(o.loss, y, predict(o, x), args...)
