# SparseRegression

This package relies on primitives defined in the JuliaML ecosystem to implement high-performance algorithms for linear models which often produce sparsity in the coefficients.

## Usage

I think this should get you set up:
```
Pkg.clone("https://github.com/joshday/SparseRegression.jl")
```

Quick example:
```julia
Pkg.clone("https://github.com/joshday/DataGenerator.jl")  # for simulating data

using SparseRegression, DataGenerator

x, y, b = linregdata(100_000, 5)

# Absolute loss with Ridge penalty
o = SparseReg(5; loss = L1DistLoss(), penalty = L2Penalty())
@time fit!(o, x, y, PROXGRAD(verbose = true))
# INFO: Iteration: 1, Relative Tolerance = 0.3473853196294418
# INFO: Iteration: 2, Relative Tolerance = 0.2386436858877052
# INFO: Iteration: 3, Relative Tolerance = 0.08486287402355355
# INFO: Iteration: 4, Relative Tolerance = 0.017244280030338035
# INFO: Iteration: 5, Relative Tolerance = 0.003580554873613252
# INFO: Iteration: 6, Relative Tolerance = 0.0007958709334973529
# INFO: Iteration: 7, Relative Tolerance = 0.00016713085747966776
# INFO: Iteration: 8, Relative Tolerance = 4.374220880495914e-5
# INFO: Iteration: 9, Relative Tolerance = 1.6230694309988682e-5
# INFO: Iteration: 10, Relative Tolerance = 7.743133531993131e-6
# INFO: Iteration: 11, Relative Tolerance = 4.012741694875729e-6
# INFO: CONVERGED: 12, Relative Tolerance = 1.462267983540721e-7
#   0.011061 seconds (356 allocations: 1.544 MB)
```

## Models

The available models come from LossFunctions.jl and assuming a linear transformation.  SparseRegression provides aliases for these types

- Linear Regression: `LinearRegression()`
- Logistic Regression: `LogisticRegression()`
- Poisson Regression: `PoissonRegression()`
- Absolute Loss Regression: `L1Regression()`
- Quantile Regression: `QuantileRegression(q)`
- Huber Regression: `HuberRegression(c)`
- SVM: `SVMLike()`
- Distance Weighted Discrimination: `DWDLike(q)`

## Penalties/Regularizers

The available penalties come from PenaltyFunctions.jl

## Algorithms

### Offline Algorithms

Offline algorithms are the standard solvers that people think of for statistical learning.  

- `PROXGRAD()`
  - Proximal Gradient Method.  

### Online Algorithms
- WIP
