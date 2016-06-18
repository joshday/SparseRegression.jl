module PenaltyTest
using SparseRegression
if VERSION >= v"0.5.0-dev+7720"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

@testset "Penalty" begin
    β = randn(10)
    info("Show methods for penalties:")
    for pen in [NoPenalty(), LassoPenalty(), RidgePenalty(), ElasticNetPenalty()]
        show(pen); println()
    end
    println()

    @testset "NoPenalty" begin
        pen = NoPenalty()
        @test penalty(pen, β, 0.1) == 0.0
        @test prox(pen, 1.0, 0.1) == 1.0
    end

    @testset "RidgePenalty" begin
        pen = RidgePenalty()
        @test penalty(pen, β, 0.1) == 0.1 * 0.5 * sumabs2(β)
        @test prox(pen, 1.0, 0.1) == 1.0 / 1.1
    end

    @testset "LassoPenalty" begin
        pen = LassoPenalty()
        @test penalty(pen, β, 0.1) == 0.1 * sumabs(β)
        @test prox(pen, 1.0, 0.1) == 0.9
    end

    @testset "ElasticNetPenalty" begin
        @test_throws Exception ElasticNetPenalty(-1.0)
        @test_throws Exception ElasticNetPenalty(0)
        @test_throws Exception ElasticNetPenalty(1)
        pen = ElasticNetPenalty(.5)
        @test penalty(pen, β, 0.1) == 0.1 * 0.5 * (sumabs(β) + 0.5 * sumabs2(β))
        @test prox(pen, 1.0, 0.1) ==
            prox(RidgePenalty(), 1.0, 0.05) * prox(LassoPenalty(), 1.0, 0.05)
    end

    βold = copy(β)
    pen = RidgePenalty()
    prox!(pen, β, 0.1)
    for j in eachindex(β)
        @test β[j] == βold[j] / 1.1
    end
end

end
