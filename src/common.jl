#-------------------------------------------------------------------------------# printing
function name(a; withparams = false)
    s = replace(string(typeof(a)), "SparseRegression.", "")
    withparams || replace(s, r"\{(.*)", "")
end

function print_item(io::IO, name::AbstractString, value, newline = true)
    print(io, "  >" * @sprintf("%15s", name * ":  "))
    print(io, value)
    newline && println(io)
end

#----------# Display fields like: (a = 1, b = 5.0, ...)
function showfields(io::IO, o, nms)
    if length(nms) != 0
        s = "("
        for nm in nms
            s *= "$nm = $(getfield(o, nm))"
            if nms[end] != nm
                s *= ", "
            end
        end
        s *= ")"
        return print(io, s)
    else
        return print(io, "")
    end
end

showme(o) = []
function Base.show(io::IO, A::Algorithm)
    print_with_color(default_color, io, "■ $(name(A))")
    showfields(io, A, showme(A))
    println(io, "")
    show(io, A.obs)
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
# classify{A, L<:MarginLoss}(o::StreamReg{A,L}, x::AVec) = sign(xβ(o, x))
# classify{A, L<:MarginLoss}(o::StreamReg{A,L}, x::AMat) = sign.(xβ(o, x))

loss(o::AbstractSparseReg, x::AMat, y::AVec, args...) =
    value(o.loss, y, predict(o, x), args...)
