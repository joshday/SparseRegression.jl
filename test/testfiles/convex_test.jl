# Do not include in runtests.jl
#
# This file compare numbers against Convex.jl

module ConvexTest
    reload("SparseRegression")
    reload("OnlineStats")
    using SparseRegression, StatsBase, Convex, Mosek, FactCheck, DataFrames, GLMNet
    using Lasso, Distributions
    srand(1234)

    import OnlineStats

    # facts("Compare to Convex.jl") do
        n, p = 10_000, 10
        x = randn(n, p) * 4

        # context("L2Regression") do
            y = x * collect(1:p) + randn(n)
            λ = .0

            o = SparseReg(x, y,
                penalty = LassoPenalty(),
                intercept = false,
                lambda = [λ],
                model = L2Regression(),
                step = .5,
                tol = 1e-10
            )
            βhat = coef(o)[:, 1]

            o2 = glmnet(x, y, lambda = [λ], intercept = false)
            βhat2 = Matrix{Float64}(o2.betas)[:, 1]

            o3 = fit(LassoPath, x, y, λ = [λ], intercept = false)
            βlasso = Matrix(coef(o3))[:, 1]

            O = OnlineStats
            o4 = O.LinReg(x, y, O.LassoPenalty(λ))
            βonline = coef(o4, step = .01, maxit = 10000)

            β = Variable(p)
            problem = minimize(0.5 * sumsquares(y - x * β))
            solve!(problem, MosekSolver(LOG = 1))
            @show βcon = β.value[:, 1]

            @show DataFrame(
                SparseReg = βhat,
                Lasso = βlasso,
                OnlineStats = βonline[2:end],
                Convex = βcon,
                GLMNet = βhat2,
                diff_with_convex = maxabs(βhat - βcon),
                diff_with_glmnet = maxabs(βhat - βhat2)
            )

            print_with_color(:blue, "SparseRegression\n")
            @show mean(abs2(y - x*βhat))
            print_with_color(:blue, "OnlineStats\n")
            @show mean(abs2(y - O.predict(o4, x)))
            print_with_color(:blue, "GLMNet\n")
            @show mean(abs2(y - x*βhat2))
            print_with_color(:blue, "Convex\n")
            @show mean(abs2(y - x*βcon))
    #     end
    # end


end
