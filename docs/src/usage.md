# Usage

1. Create a model
1. Fit the model

## SModel

The model type used by SparseRegression is `SModel`.  An `SModel` holds onto the sufficient
information for generating a solution fo the SparseRegression objective.

!!! note
    Constructing an `SModel` does not create a fitted model.  It must be `learn!`-ed.

```@docs
SModel
```


## [LearningStrategies](https://github.com/JuliaML/LearningStrategies.jl)

An `SModel` can be learned with the default learning strategy with `learn!(model)`.  You
can provide more control over the learning process by providing your own LearningStrategy.

SparseRegression implements several `Algorithm <: LearningStrategy` types to do `SModel`
fitting.  An `Algorithm` must be constructed with an `SModel` to ensure storage buffers
are the correct size.

```julia
using SparseRegression

# Make some fake data
x = randn(1000, 10)
y = x * range(-1, stop=1, length=10) + randn(1000)

# Create an SModel
s = SModel(x, y)

# All of the following are valid ways to calculate a solution
learn!(s)
learn!(s, strategy(ProxGrad(s), MaxIter(25), TimeLimit(.5)))
learn!(s, Sweep(s))
learn!(s, LinRegCholesky(s))
```