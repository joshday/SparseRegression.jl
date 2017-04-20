# module SparseRegressionBenchmarks
# using SparseRegression, BenchmarkTools, DataGenerator, GLM, MultivariateStats
#
# # Benchmarks for unpenalized linear regression
# function print_naive_benchmarks(n, p)
#     x, y, β = linregdata(n, p)
#
#     println("Benchmarks (n = $n, p = $p)")
#
#     println("  > SparseRegression.SweepModel")
#     print("    ")
#     @time SweepModel(Obs(x, y); λ = [0.0], penalty = NoPenalty())
#
#     println("  > SparseRegression.ProximalGradientModel")
#     print("    ")
#     @time ProximalGradientModel(Obs(x, y); λ = [0.0], penalty = NoPenalty())
#
#     println("  > Base")
#     print("    ")
#     @time x \ y
#
#     println("  > GLM")
#     print("    ")
#     @time lm(x, y)
#
#     println("  > MultivariateStats")
#     print("    ")
#     @time llsq(x, y; bias=false)
# end
#
# print_naive_benchmarks(1_000_000, 10)
# print_naive_benchmarks(10_000, 100)
# print_naive_benchmarks(1_000, 200)
#
# end
