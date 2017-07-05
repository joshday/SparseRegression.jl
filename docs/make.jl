using Documenter, SparseRegression

makedocs(
    format = :html,
    sitename = "SparseRegression",
    pages = ["index.md"]
)

deploydocs(
    repo = "github.com/joshday/SparseRegression.jl.git",
    julia = "0.6",
    deps = nothing,
    make = nothing
)
