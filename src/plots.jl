import Plots


function Plots.plot(o::StatLearnPath)
    Plots.plot(o.λs, o.β',
        labels = ["B_$j" for j in 1:size(o.β, 1)]',
        xlabel = "lambda",
        ylabel = "value",
        title = "Solution Path for $(replace(string(typeof(o)), "StatisticalLearning.", ""))"
    )
end
