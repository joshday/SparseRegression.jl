module ModelDefinitionTest
using SparseRegression
if VERSION >= v"0.5.0-dev+7720"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

@testset "Model" begin
    @testset "LinearRegression" begin
        m = LinearRegression()
        @test predict(m, 1.0) == 1.0
    end
    @testset "L1Regression" begin
        m = L1Regression()
        @test predict(m, 1.0) == 1.0
    end
    @testset "LogisticRegression" begin
        m = LogisticRegression()
        @test predict(m, 1.0) == 1.0 / (1.0 + exp(-1.0))
    end
    @testset "PoissonRegression" begin
        m = PoissonRegression()
        @test predict(m, 1.0) == exp(1.0)
    end
    @testset "SVMLike" begin
        m = SVMLike()
        @test predict(m, 1.0) == 1.0
        @test classify(m, 1.0) == 1.0
        @test classify(m, -1.0) == -1.0
    end
    @testset "QuantileRegression" begin
        m = QuantileRegression(.7)
        @test predict(m, 1.0) == 1.0
    end
    @testset "HuberRegression" begin
        m = HuberRegression(2.0)
        @test predict(m, 1.0) == 1.0
    end
end

end
