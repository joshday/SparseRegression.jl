module SparseRegressionTest
using SparseRegression, Base.Test
include("datagenerator.jl")



#-----------------------------------------------------------------------# Sanity Check and show
n, p = 1000, 5
x, y, β = SparseRegression.fakedata(L2DistLoss(), n, p)
show(SModel(x, y))
println("\n")

name(o) = replace(string(typeof(o)), r"([a-zA-Z]*\.)", "")

function _test(l::Loss, p::Penalty, alg)
    x, y, β = SparseRegression.fakedata(l, 100, 5)
    o = @inferred SModel(x, y, l, p)
    a = @inferred alg(o)
    @inferred learn!(o, strategy(a, MaxIter(3)))
    @test coef(o) == o.β
    @test length(predict(o, x)) == length(y)

    w = rand(100)
    o = @inferred SModel(x, y, l, Weights(w), p)
    a = @inferred alg(o)
    learn!(o, strategy(a, MaxIter(3)))
    coef(o)
    predict(o, x)
end

@testset "Sanity Check" begin
    for a in [ProxGrad, Fista, AdaptiveProxGrad, GradientDescent]
        info(a)
        for l in [L2DistLoss(), .5*L2DistLoss(), HuberLoss(2.0), QuantileLoss(.7),
            LogitMarginLoss(), DWDMarginLoss(2.0), L2HingeLoss(), ExpLoss()]
            println("  > $l")
            for p in [NoPenalty(), L2Penalty(), L1Penalty(), ElasticNetPenalty()]
                _test(l, p, a)
            end
        end
        println()
    end
    for a in [Sweep, LinRegCholesky]
        for l in [L2DistLoss(), .5L2DistLoss()]
            for p in [NoPenalty(), L2Penalty()]
                _test(l ,p, a)
            end
        end
    end
end


@testset "Linear Regression" begin
    n, p = 1000, 10
    x, y, β = DataGenerator.linregdata(n, p)
    s = SModel(x, y, .5 * L2DistLoss(), NoPenalty())

    for a in [ProxGrad(s, .1), Fista(s, .1), GradientDescent(s, .1), Sweep(s), LinRegCholesky(s)]
        learn!(s, strategy(a, MaxIter(25)))
        @test coef(s) ≈ x\y atol = .5
    end
end



@testset "SModel" begin
    n, p = 100, 5
    x, y, β = DataGenerator.linregdata(n, p)
    @testset "Constructor type stability" begin
        @inferred SModel(x, y)

        @inferred SModel(x, y, L2DistLoss())
        @inferred SModel(x, y, L2Penalty())
        @inferred SModel(x, y, rand(5))

        @inferred SModel(x, y, L2DistLoss(), L2Penalty())
        @inferred SModel(x, y, L2DistLoss(), rand(5))
        @inferred SModel(x, y, L2Penalty(), L2DistLoss())
        @inferred SModel(x, y, L2Penalty(), rand(5))
        @inferred SModel(x, y, rand(5), L2DistLoss())
        @inferred SModel(x, y, rand(5), L2Penalty())

        @inferred SModel(x, y, L2DistLoss(), L2Penalty(), rand(5))
        @inferred SModel(x, y, L2DistLoss(), rand(5), L2Penalty())
        @inferred SModel(x, y, L2Penalty(), L2DistLoss(), rand(5))
        @inferred SModel(x, y, L2Penalty(), rand(5), L2DistLoss())
        @inferred SModel(x, y, rand(5), L2DistLoss(), L2Penalty())
        @inferred SModel(x, y, rand(5), L2Penalty(), L2DistLoss())
    end
    @testset "predict" begin
        o = SModel(x, y)
        @test predict(o, randn(5)) == 0.0
        @test predict(o, randn(10, 5)) == zeros(10)
    end
end



end
