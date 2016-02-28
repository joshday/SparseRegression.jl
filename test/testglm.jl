facts(@title "Show Methods") do
    n, p = 1000, 11
    x = randn(n, p)
    β = collect(linspace(-5, 5, p))
    y = x*β + randn(n)

    o = GLMPath(x, y, λs = collect(.1:.1:.9))
    show(o)
    coef(o)
end
