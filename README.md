# SparseRegression

This package relies on primitives defined in the JuliaML ecosystem to implement high-performance algorithms for linear models which often produce sparsity in the coefficients.

## Usage

I think this should get you set up:
```
Pkg.clone("https://github.com/joshday/SparseRegression.jl")
Pkg.checkout("LossFunctions")
```

Quick example:
```julia
Pkg.clone("https://github.com/joshday/DataGenerator.jl")  # for simulating data

using SparseRegression, DataGenerator

x, y, b = linregdata(100_000, 10)

# Absolute loss with Ridge penalty
@time o = SparseReg(x, y, L1DistLoss(), PROXGRAD(verbose=true), L2Penalty(.1))
# INFO: Iteration: 1, Relative Tolerance = 0.3278716198007006
# INFO: Iteration: 2, Relative Tolerance = 0.29606007437944964
# INFO: Iteration: 3, Relative Tolerance = 0.17725425000688583
# INFO: Iteration: 4, Relative Tolerance = 0.06164121705556583
# INFO: Iteration: 5, Relative Tolerance = 0.014080327521919796
# INFO: Iteration: 6, Relative Tolerance = 0.0032216462390018893
# INFO: Iteration: 7, Relative Tolerance = 0.0008673949445801162
# INFO: Iteration: 8, Relative Tolerance = 0.00020781680579618777
# INFO: Iteration: 9, Relative Tolerance = 2.579150923996323e-5
# INFO: Iteration: 10, Relative Tolerance = 5.845266072802313e-6
# INFO: Iteration: 11, Relative Tolerance = 1.0160477264827278e-5
# INFO: Iteration: 12, Relative Tolerance = 5.362242262064752e-6
# INFO: Iteration: 13, Relative Tolerance = 2.987010885114535e-6
# INFO: Iteration: 14, Relative Tolerance = 2.582794111780323e-6
# INFO: Iteration: 15, Relative Tolerance = 6.787445423844011e-6
# INFO: Iteration: 16, Relative Tolerance = 4.0869511665504665e-6
# INFO: Iteration: 17, Relative Tolerance = 4.350354429294964e-6
# INFO: CONVERGED: 18, Relative Tolerance = 6.580971629277822e-8
#   0.016987 seconds (532 allocations: 1.553 MB)
```

## Models

The available models come from LossFunctions.jl and assuming a linear transformation.

- Linear Regression: `ScaledLoss(L2DistLoss(), .5)`
- Logistic Regression: `LogitMarginLoss()`
- Poisson Regression: `PoissonLoss()`
- Absolute Loss Regression: `L1DistLoss()`
- Quantile Regression: `QuantileLoss(q)`
- Huber Regression: `HuberLoss(c)`
- SVM: `L1HingeLoss()`

## Penalties/Regularizers

The available penalties come from PenaltyFunctions.jl

- Ridge: `L2Penalty(λ)`
- Lasso: `L1Penalty(λ)`
- Elastic Net: `ElasticNetPenalty(λ, α)`
- SCAD: `SCADPenalty(λ, a)`


## Algorithms

### Offline Algorithms

Offline algorithms are the standard solvers that people think of for statistical learning.  

- `PROXGRAD()`
  - Proximal Gradient Method.  

### Online Algorithms

Online algorithms perform a single pass through the data with each call to `fit!`.  They are approximations of the true solution.  While they are fast (and can run on datasets larger than memory), the "closeness" of the approximation can depend heavily on the learning rate, defined by `OnlineStats.Weight` types and a constant.  For example, `SGD(LearningRate(r), c)` is stochastic gradient descent using a learning rate of `w(t) = c / t^r`.

- `SGD(wt, c)`
  - Stochastic Gradient Descent.
- `MOMENTUM(wt, c, a)`
  - Stochastic Gradient Descent with momentum (`a in (0, 1)`  is the momentum rate).  
