# Introduction

**SparseRegression** is a Julia package which combines [JuliaML](https://github.com/JuliaML) primitives to implement high-performance algorithms for fitting linear models.

## Objective Function
The objective function that SparseRegression can solve takes the form:

```math
\frac{1}{n}\sum_{i=1}^n w_i f(y_i, x_i^T\beta) + \sum_{j=1}^p \lambda_j J(\beta_j),
```
where $f$ is a loss function, $J$ is a penalty or regularization function, the $w_i$'s are nonnegative observation weights and the $\lambda_j$'s are nonnegative element-wise regularization parameters.

## JuliaML
The three core JuliaML packages that SparseRegression brings together are:

- [LossFunctions](https://github.com/JuliaML/LossFunctions.jl)
- [PenaltyFunctions](https://github.com/JuliaML/PenaltyFunctions.jl)
- [LearningStrategies](https://github.com/JuliaML/LearningStrategies.jl)
