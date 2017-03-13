# SparseRegression

This package relies on primitives defined in the JuliaML ecosystem to implement high-performance algorithms for linear models which often produce sparsity in the coefficients.

# Disclaimer

This package is a work in progress, and the API/internals may change without warning.

# Example

```julia
# for generating data
Pkg.clone("https://github.com/joshday/DataGenerator.jl")
```

```julia
using DataGenerator, SparseRegression
x, y, b = linregdata(10_000, 50)

# Proximal Gradient algorithm with options
# - maxit:    maximum number of iterations
# - tol:      tolerance for convergence
# - step:     step size for gradient descent
# - verbose:  print out tolerance at each iteration?
alg = ProxGrad(maxit=50, tol=1e-5, step=.7, verbose=true)

# Linear regression with Lasso penalty with a tuning parameter of .1
o = SparseReg(x, y, alg, LinearRegression(), L1Penalty(), .1)
# INFO: Iteration: 1, Relative Tolerance = 0.7992926640305482
# INFO: Iteration: 2, Relative Tolerance = 0.21158589427567479
# INFO: Iteration: 3, Relative Tolerance = 0.06217273407155267
# INFO: Iteration: 4, Relative Tolerance = 0.019790192221616928
# INFO: Iteration: 5, Relative Tolerance = 0.006563202078845049
# INFO: Iteration: 6, Relative Tolerance = 0.002213809294601788
# INFO: Iteration: 7, Relative Tolerance = 0.0007549002315283089
# INFO: Iteration: 8, Relative Tolerance = 0.00026036475832886126
# INFO: Iteration: 9, Relative Tolerance = 9.071250950871088e-5
# INFO: Iteration: 10, Relative Tolerance = 3.18841103372272e-5
# INFO: Iteration: 11, Relative Tolerance = 1.1293402486153018e-5
# INFO: CONVERGED: 12, Relative Tolerance = 4.027213159695074e-6
# Sparse Regression Model
#   >         β:  [-0.894656,-0.848999,...,0.84992,0.913742]
#   >      Loss:  0.5 * (L2DistLoss)
#   >   Penalty:  L1Penalty
#   >         λ:  0.1
#   > Algorithm:  ProxGrad(maxit=50, tol=1.0e-5, verbose=true, step=0.7)
```
