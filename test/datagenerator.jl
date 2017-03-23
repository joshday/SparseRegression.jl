module DataGenerator
using Distributions; D = Distributions

defaultβ(p) = collect(linspace(-1, 1, p))

function linregdata(n, p; β = defaultβ(p), V = eye(p), μ = zeros(p), σ = 1.0)
	x = rand(D.MvNormal(μ, V), n)'
	y = x * β + σ * randn(n)
	return x, y, β
end

function logregdata(n, p, sgn = true; β = defaultβ(p), V = eye(p), μ = zeros(p))
	x = rand(D.MvNormal(μ, V), n)'
	y = Float64[rand(D.Bernoulli(1 / (1 + exp(-η)))) for η in x * β]
	if sgn
		y = 2y - 1
	end
	return x, y, β
end

function poissonregdata(n, p, β = defaultβ(p), V = eye(p), μ = zeros(p))
	x = rand(D.MvNormal(μ, V), n)'
	y = Float64[rand(D.Poisson(exp(η))) for η in x * β]
	return x, y, β
end
end
