import Plots


function Plots.plot(o::SparseReg)
    Plots.plot(o.λ, o.β',
        labels = ["B_$j" for j in 1:size(o.β, 1)]',
        xlabel = "lambda",
        ylabel = "value",
        title = "Solution Path for $(replace(string(typeof(o)), "SparseRegression.", ""))"
    )
end


function Plots.plot(o::SparseReg, x::Matrix, y::Vector)
    d = length(o.β0)
    n, p = size(x)
    @assert p == size(o.β, 1) "x is incompatable with coefficient vector"
    @assert n == length(y) "x is incompatable with y"
    err = zeros(d)
    η = zeros(n)
    ŷ = zeros(n)

    for j in 1:d
        # η
        BLAS.gemv!('N', 1.0, x, o.β[:, j], 0.0, η)
        if o.intercept
            for i in eachindex(η)
                @inbounds η[i] += o.β0[j]
            end
        end
        # calculate err
        if typeof(o.model) <: BivariateModel
            classify!(o.model, ŷ, η)
            err[j] = mean(y .!= ŷ)
        else
            predict!(o.model, ŷ, η)
            err[j] = sumabs2(y - ŷ) / n
        end
    end
    if typeof(o.model) <: BivariateModel
        ylab = "Misclassification Rate"
    else
        ylab = "MSE"
    end
    p1 = Plots.plot(o)
    p2 = Plots.plot(o.λ, err, xlabel = "lambda", ylabel = ylab, title = "Test Set Error")
    Plots.subplot(p1, p2)
end
