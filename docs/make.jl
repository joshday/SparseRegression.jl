using Documenter, SparseRegression

makedocs()

deploydocs(
    repo = "github.com/joshday/SparseRegression.jl.git",
    julia = "0.6",
    deps = Deps.pip("pygments", "mkdocs", "python-markdown-math")
)
