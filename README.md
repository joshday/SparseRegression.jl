# SparseRegression

This package relies on primitives defined in the JuliaML ecosystem to implement high-performance algorithms for linear models which often produce sparsity in the coefficients.

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

- `SWEEP()`
- `PROXGRAD()`
