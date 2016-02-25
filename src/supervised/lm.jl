type LinearRegression{P <: Penalty}
    β0::Float64
    β::Vector{Float64}
    intercept::Bool
    A::Matrix{Float64}
    penalty::P
end
function Base.show(io::IO, o::LinearRegression)
    print_header(io, "LinearRegression")
    o.intercept && print_item(io, "β0", o.β0)
    print_item(io, "β", o.β)
    print_item(io, "Penalty", o.penalty)
end



function lm{T<:Real, S<:Real}(x::AMat{T}, y::AVec{S};
        intercept::Bool = true, penalty::Penalty = NoPenalty()
    )
    n, p = size(x)
    @assert n == length(y) "number of rows in x and y do not match"
    p = size(x, 2)
    o = LinearRegression(0.0, zeros(p), intercept, zeros(p + 1, p + 1), penalty)
    fit!(o, x, y)
end

function fit!(o::LinearRegression{NoPenalty}, x, y)
    p = size(x, 2)
    xy = hcat(x, y)
    if o.intercept
        o.A = cor(xy)
    else
        o.A = xy'xy
    end
    sweep!(o.A, 1:p)
    copy!(o.β, o.A[1:p, end])
    o.intercept && scaled_to_original!(o, xy)
    o
end

function scaled_to_original!(o::LinearRegression, xy::Matrix)
    μ = vec(mean(xy, 1))
    σ = vec(std(xy, 1))
    β₀ = μ[end] - σ[end] * sum(μ[1:end-1] ./ σ[1:end-1] .* o.β)
    for i in 1:length(o.β)
        o.β[i] = o.β[i] * σ[end] / σ[i]
    end
    o.β0 = β₀
end



n, p = 10_000, 50
x = randn(n, p)
y = x * collect(1.:p) + randn(n)

@display @time lm(x, y)
@display @time lm(x, y, intercept = false)
