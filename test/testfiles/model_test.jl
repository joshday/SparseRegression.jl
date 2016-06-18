module ModelDefinitionTest
using SparseRegression; sp = SparseRegression
if VERSION >= v"0.5.0-dev+7720"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

@testset "Model" begin
    m = LinearRegression()
    y = randn(100)
    η = randn(100)
    @test loss(m, y, η) == 0.5 * mean(abs2(y - η))

    storage = zeros(100)
    sp.predict!(m, storage, η)
    @test storage == η

    m = LogisticRegression()
    sp.predict!(m, storage, η)
    @test storage == 1.0 ./ (1.0 + exp(-η))
    sp.classify!(m, storage, η)
    @test storage == (η .> 0)

    @testset "LinearRegression" begin
        m = LinearRegression()
        @test predict(m, 1.0) == 1.0
        @test loss(m, 0.5, 1.0) == 0.5 ^ 3
        @test sp.lossderiv(m, 0.5, 1.0) == -(0.5 - 1.0)
    end
    @testset "L1Regression" begin
        m = L1Regression()
        @test predict(m, 1.0) == 1.0
        @test loss(m, 0.5, 1.0) == 0.5
        @test sp.lossderiv(m, 0.5, 1.0) == -sign(0.5 - 1.0)
    end
    @testset "LogisticRegression" begin
        m = LogisticRegression()
        @test predict(m, 1.0) == 1.0 / (1.0 + exp(-1.0))
        @test classify(m, 1.0) == 1.0
        @test classify(m, -1.0) == 0.0
        @test loss(m, 1.0, 0.9) == -0.9 + log(1.0 + exp(.9))
        @test sp.lossderiv(m, 1.0, .9) == -1.0 + 1.0 / (1.0 + exp(-.9))
    end
    @testset "PoissonRegression" begin
        m = PoissonRegression()
        @test predict(m, 1.0) == exp(1.0)
        @test loss(m, 1.0, 0.9) == -0.9 + exp(.9)
        @test sp.lossderiv(m, 1.0, .9) == -1.0 + exp(.9)
    end
    @testset "SVMLike" begin
        m = SVMLike()
        @test predict(m, 1.0) == 1.0
        @test classify(m, 1.0) == 1.0
        @test classify(m, -1.0) == -1.0
        @test loss(m, 1.0, 0.9) == 1.0 - 0.9
        @test loss(m, 1.0, 1.1) == 0.0
        @test sp.lossderiv(m, 1.0, 0.9) == -1.0
        @test sp.lossderiv(m, 1.0, 1.1) == 0.0
        @test sp.lossderiv(m, -1.0, 0.9) == 1.0
    end
    @testset "QuantileRegression" begin
        m = QuantileRegression(.7)
        @test predict(m, 1.0) == 1.0
        @test loss(m, 0.5, 1.0) ≈ .3 * .5
        @test loss(m, 1.0, 0.5) ≈ .7 * .5
        @test sp.lossderiv(m, 1.0, 0.5) == -.7
        @test sp.lossderiv(m, 0.5, 1.0) ≈ .3
    end
    @testset "HuberRegression" begin
        m = HuberRegression(2.0)
        @test predict(m, 1.0) == 1.0
        @test loss(m, 0.5, 1.0) == 0.5 ^ 3
        @test loss(m, 0.5, 10.0) == 2.0 * (9.5 - 1.0)
        @test sp.lossderiv(m, 0.5, 1.0) == -(0.5 - 1.0)
        @test sp.lossderiv(m, 0.5, 10.0) == -2.0 * sign(0.5 - 10.0)
    end
end

end
