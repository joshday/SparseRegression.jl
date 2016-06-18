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
    end

    @testset "LassoPenalty" begin
        pen = LassoPenalty()
        @test penalty(pen, β, 0.1) == 0.1 * sumabs(β)
    end

    @testset "RidgePenalty" begin
        pen = RidgePenalty()
        @test penalty(pen, β, 0.1) == 0.1 * 0.5 * sumabs2(β)
    end

    @testset "ElasticNetPenalty" begin
        pen = ElasticNetPenalty(.5)
        @test penalty(pen, β, 0.1) == 0.1 * 0.5 * (sumabs(β) + 0.5 * sumabs2(β))
    end
end

end
