# SparseRegression.jl

SparseRegression relies on primitives defined in the [JuliaML](https://github.com/JuliaML) ecosystem to implement high-performance algorithms for linear models which often produce sparsity in the coefficients.  

### Objective
The objective functions that SparseRegression can solve are of the form:

```math
\frac{1}{n}\sum_{i=1}^n w_i f(y_i, x_i^T\beta) + \sum_{j=1}^p \lambda_j J(\beta_j),
```
where $f$ is a loss function, $J$ is a penalty or regularization function, the $w_i$'s are nonnegative observation weights and the $\lambda_j$'s are nonnegative element-wise regularization parameters.

The three core JuliaML packages that SparseRegression brings together are:

- [LossFunctions](https://github.com/JuliaML/LossFunctions.jl)
  - "grammar of losses"
- [PenaltyFunctions](https://github.com/JuliaML/PenaltyFunctions.jl)
  - "grammar of regularization"
- [LearningStrategies](https://github.com/JuliaML/LearningStrategies.jl)
  - "grammar of iterative learning"

With few exceptions, SparseRegression can handle:
  - any Loss from [LossFunctions.jl](https://github.com/JuliaML/LossFunctions.jl#available-losses)
  - any ElementPenalty from [PenaltyFunctions.jl](https://github.com/JuliaML/PenaltyFunctions.jl#available-penalties)
