using SparseRegression, GLM, BenchmarkTools, MultivariateStats
n, p = 100_000, 200
x = randn(n, p)
β = collect(1.0:p)
y = x * β + randn(n)
#-------------------------------------------------------------------------# benchmark
info("SparseReg")
b1 = @benchmark SparseReg(x, y, algorithm = Sweep(), intercept = false)

info("Base")
b2 = @benchmark x \ y

info("GLM.lm")
b3 = @benchmark lm(x, y)

info("MultivariateStats.llsq")
b4 = @benchmark llsq(x, y, bias = false)

#---------------------------------------------------------------------# write results
function write_benchmark(f, name, b)
    info("Writing $name")
    write(f, "# $name \n```")
    write(f, string(b) * "\n```\n")
end

bench = Pkg.dir("SparseRegression", "test", "benchmarks")
touch(bench * "/linreg_benchmarks.md")

file = open(bench * "/linreg_benchmarks.md", "r+")
write_benchmark(file, "SparseReg", b1)
write_benchmark(file, "Base", b2)
write_benchmark(file, "GLM.lm", b3)
write_benchmark(file, "MultivariateStats.llsq", b4)
close(file)
