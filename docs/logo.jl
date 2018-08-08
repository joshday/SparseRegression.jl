using Plots
gr()

Random.seed!(2)

colors = [RGB(.22,.596,.149), RGB(.8,.361,.361), RGB(.702, .322, .8)]
border = [RGB(.133, .541, .133), RGB(.8, .2, .2), RGB(.584, .345, .698)]

y = rand(50)
for i in eachindex(y)
    if rand() < .8
        y[i] = 0.0
    end
end

scatter(y, c = colors, ms = 10, markerstrokewidth = 5, markerstrokecolor = border,
    legend=false, grid=false, axis=false)


savefig(joinpath(@__DIR__(), "logo.png"))
