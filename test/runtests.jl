module SparseRegressionTest
using SparseRegression, Base.Test
include("datagenerator.jl")



#-----------------------------------------------------------------------# Sanity Check and show
n, p = 1000, 5
x, y, β = DataGenerator.linregdata(n, p)
show(SModel(x, y))
println("\n")

data(::Loss, n, p) = DataGenerator.linregdata(n, p)
data(::MarginLoss, n, p) = DataGenerator.logregdata(n, p)

name(o) = replace(string(typeof(o)), r"([a-zA-Z]*\.)", "")

function _test(l::Loss, p::Penalty, alg)
    x, y, β = data(l, 100, 5)
    o = @inferred SModel(x, y, l, p)
    a = @inferred alg(o)
    @inferred learn!(o, strategy(a, MaxIter(2)))
    @test coef(o) == o.β
    @test length(predict(o, x)) == length(y)

    w = rand(100)
    o = @inferred SModel(x, y, l, Weights(w), p)
    a = @inferred alg(o)
    learn!(o, strategy(a, MaxIter(2)))
    coef(o)
    predict(o, x)
end

for a in [ProxGrad, Fista, GradientDescent]
    info(a)
    for l in [L2DistLoss(), .5*L2DistLoss(), HuberLoss(2.0), QuantileLoss(.7), LogitMarginLoss(),
        DWDMarginLoss(2.0), L2HingeLoss(), PerceptronLoss(), ExpLoss()]
        println("  > $l")
        for p in [NoPenalty(), L2Penalty(), L1Penalty(), ElasticNetPenalty()]
            _test(l, p, a)
        end
    end
    println()
end


# ProxGrad/Fista/GradientDescent
# for l in [L2DistLoss(), L1DistLoss(), HuberLoss(2.0), QuantileLoss(.7)]
#     print("Sanity Check: $l"); print("    ")
#     for p in [NoPenalty(), L2Penalty(), L1Penalty(), ElasticNetPenalty()]
#         print(", $p")
#         s = SModel(x, y, l, p)
#         learn!(s, strategy(ProxGrad(s), MaxIter(3)))
#         learn!(s, strategy(Fista(s), MaxIter(3)))
#         learn!(s, strategy(GradientDescent(s), MaxIter(3)))
#         _test(l, p, ProxGrad)
#     end
#     println()
# end
# # Sweep/LinRegCholesky
# for l in [L2DistLoss(), .5 * L2DistLoss()], p in [NoPenalty(), L2Penalty()]
#     s = SModel(x, y, l, p)
#     learn!(s, Sweep(s))
#     learn!(s, LinRegCholesky(s))
# end



println()
println()
info("TESTS START HERE")


@testset "Linear Regression" begin
    n, p = 1000, 10
    x, y, β = DataGenerator.linregdata(n, p)
    s = SModel(x, y, .5 * L2DistLoss(), NoPenalty())

    for a in [ProxGrad(s, .1), Fista(s, .1), GradientDescent(s, .1), Sweep(s), LinRegCholesky(s)]
        learn!(s, strategy(a, MaxIter(25)))
        @test coef(s) ≈ x\y atol = .5
    end
end
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
data(::Loss, n, p) = DataGenerator.linregdata(n, p)
data(::MarginLoss, n, p) = DataGenerator.logregdata(n, p)


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
