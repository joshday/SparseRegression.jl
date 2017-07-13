# Algorithms

An `Algorithm` contains `Obs`, parameters for the algorithm, and storage buffers.  Some algorithms only work with specific loss/penalty combinations.

## `ProxGrad(obs, s)`
Proximal Gradient Method with step size `s`.  Handles any loss and convex penalty.

## `Fista(obs, s)`
Fast Iterative Shrinkage-Thresholding Algorithm (accelerated proximal gradient) with step size `s`.  Handles any loss and convex penalty.

## `GradientDescent(obs, s)`
Gradient Descent with step size `s`.  Handles any loss and penalty.

## `Sweep(obs)`
Linear or Ridge regression via the [sweep operator](https://github.com/joshday/SweepOperator.jl).

## `LinRegCholesky(obs)`
Linear or Ridge regression via Cholesky decomposition
