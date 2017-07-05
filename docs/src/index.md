# SparseRegression Documentation

SparseRegression implements high performance algorithms for regularized linear models using primitives in the JuliaML ecosystem.  The SparseRegression model takes the form

$\frac{1}{n}\sum_{i=1}^n f_i(\beta) + \sum_{j=1}^p\lambda_j \psi(\beta_j),$

where $f_i$ is the loss for observation $i$, $\lambda_j$'s are nonnegative element-wise regularization parameters, and $\psi$ is a penalty/regularization function applied to $\beta_j$.


```@docs
SModel
```

```@example
a = 1
b = 2
a + b
```

## Index

```@index
```
