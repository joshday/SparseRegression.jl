module SparseRegressionTest
using SparseRegression, Base.Test
include("datagenerator.jl")

n, p = 1000, 5
x, y, β = DataGenerator.linregdata(n ,p)

model = SModel(x, y, L1DistLoss())
s = strategy(Verbose(Fista(model)), Verbose(MaxIter(5)))
learn!(model, s)
@show model
#
#
#
# model = SModel(x, y, L1DistLoss())
# s = strategy(GradientDescent(model), MaxIter(5))
# learn!(model, Verbose(s))
# @show model

# losses = [LinearRegression(), L1Regression(), LogisticRegression(), PoissonRegression(),
#           HuberRegression(), SVMLike(), DWDLike(1.0), QuantileRegression(.7)]
# penalties = [NoPenalty(), L1Penalty(), L2Penalty(), ElasticNetPenalty(.5), LogPenalty(),
#           SCADPenalty(), MCPPenalty()]
#
# #------------------------------------------------------------# Show methods
# n, p = 100, 5
# x, y, β = DataGenerator.linregdata(n, p)
#
# info("Show Obs")
# show(Obs(x, y))
# println()
#
# info("Show SModel")
# show(SModel(5))
#
#
# #------------------------------------------------------------# Tests Here
# println("\n\n")
# info("Begin Tests")
# data(::Loss, n, p) = DataGenerator.linregdata(n, p)
# data(::MarginLoss, n, p) = DataGenerator.logregdata(n, p)
#
# function _test(l::Loss, p::Penalty, alg)
#     x, y, β = data(l, 100, 5)
#     obs = Obs(x, y)
#     o = @inferred SModel(obs, l, p)
#     a = @inferred alg(obs)
#     @inferred learn!(o, a)
#     @test coef(o) == o.β
#     @test length(predict(o, x)) == length(y)
#
#     w = rand(100)
#     obs = Obs(x, y, w)
#     o = @inferred SModel(obs, l, p)
#     a = @inferred alg(obs)
#     learn!(o, a)
#     coef(o)
#     predict(o, x)
# end
#
# @testset "Sanity Checks" begin
#     @testset "ProxGrad/Fista Sanity Check" begin
#         for l in losses, p in penalties
#             if isa(p, PenaltyFunctions.ConvexElementPenalty)
#                 _test(l, p, ProxGrad)
#                 _test(l, p, Fista)
#             end
#         end
#     end
#     @testset "Sweep/LinRegCholesky Sanity Check" begin
#         for l in [L2DistLoss(), LinearRegression()], p in [NoPenalty(), L2Penalty()]
#             _test(l, p, Sweep)
#             _test(l, p, LinRegCholesky)
#         end
#     end
#     @testset "GradientDescent Sanity Check" begin
#         for l in losses, p in penalties
#             _test(l, p, GradientDescent)
#         end
#     end
# end
#
#
# @testset "Obs" begin
#     n, p = 1000, 5
#     x, y, β = DataGenerator.linregdata(n, p)
#     o = Obs(x, y)
#     @test size(o) == (n, p)
#     @test size(o, 1) == n
#     @test size(o, 2) == p
#     @test nobs(o) == size(o, 1)
# end
# @testset "SModel" begin
#     n, p = 100, 5
#     x, y, β = DataGenerator.linregdata(n, p)
#     o = Obs(x, y)
#     @testset "Constructor type stability" begin
#         @inferred SModel(o)
#
#         @inferred SModel(o, L2DistLoss())
#         @inferred SModel(o, L2Penalty())
#         @inferred SModel(o, rand(5))
#
#         @inferred SModel(o, L2DistLoss(), L2Penalty())
#         @inferred SModel(o, L2DistLoss(), rand(5))
#         @inferred SModel(o, L2Penalty(), L2DistLoss())
#         @inferred SModel(o, L2Penalty(), rand(5))
#         @inferred SModel(o, rand(5), L2DistLoss())
#         @inferred SModel(o, rand(5), L2Penalty())
#
#         @inferred SModel(o, L2DistLoss(), L2Penalty(), rand(5))
#         @inferred SModel(o, L2DistLoss(), rand(5), L2Penalty())
#         @inferred SModel(o, L2Penalty(), L2DistLoss(), rand(5))
#         @inferred SModel(o, L2Penalty(), rand(5), L2DistLoss())
#         @inferred SModel(o, rand(5), L2DistLoss(), L2Penalty())
#         @inferred SModel(o, rand(5), L2Penalty(), L2DistLoss())
#     end
#     @testset "predict" begin
#         o = SModel(5)
#         @test predict(o, randn(5)) == 0.0
#         @test predict(o, randn(10, 5)) == zeros(10)
#     end
# end
#


end
