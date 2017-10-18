using Documenter, SparseRegression

makedocs(
    format = :html,
    sitename = "SparseRegression.jl",
    authors = "Josh Day",
    clean = true,
    pages = [
        "index.md",
        "usage.md",
        "algorithms.md"
    ]
)

deploydocs(
    repo   = "github.com/joshday/SparseRegression.jl.git",
    target = "build",
    osname = "linux",
    julia  = "0.6",
    deps   = nothing,
    make   = nothing
)
