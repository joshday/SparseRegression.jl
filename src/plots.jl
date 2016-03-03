import Plots


function Plots.plot(o::StatLearnPath)
    Plots.plot(o.λs, o.β',
        labels = ["B_$j" for j in 1:size(o.β, 1)]',
        xlabel = "lambda",
        ylabel = "value",
        title = "Solution Path for $(replace(string(typeof(o)), "StatisticalLearning.", ""))"
    )
end


function Plots.plot(o::StatLearnPath, xtest::Matrix, ytest::Vector)
    d = length(o.β0)
    @assert size(xtest, 2) == size(o.β, 1)
    lossvalues = zeros(d)
    lossvec = zeros(size(xtest, 1))
    for j in 1:d
        lossvector!(o.model, lossvec, ytest, xtest * o.β[:, j] + o.β0[j])
        lossvalues[j] = mean(lossvec)
    end
    p1 = Plots.plot(o)
    p2 = Plots.plot(o.λs, lossvalues, xlabel = "lambda", ylabel = "loss", title = "Test Error")
    Plots.subplot(p1, p2)
end
