# Usage

## SModel

The model type used by SparseRegression is `SModel`.  An `SModel` holds onto the sufficient
information for generating a solution fo the SparseRegression objective.

```@docs
SModel
```

!!! note
    Constructing an `SModel` does not create a solution.  It must be `learn!`-ed.

## [LearningStrategies](https://github.com/JuliaML/LearningStrategies.jl)

An `SModel` can be learned with the default learning strategy with `learn!(model)`.  You 
can provide more control over the learning process by providing your own LearningStrategy.

SparseRegression implements several `Algorithm <: LearningStrategy` types to do the heavy lifting.  An `Algorithm` must be constructed with an `SModel` to ensure storage buffers are the correct size.


```julia
using SparseRegression

# Make some fake data
x = randn(1000, 10)
y = x * linspace(-1, 1, 10) + randn(1000)

# Create an SModel
s = SModel(x, y)

# All of the following are valid ways to calculate a solution
learn!(s)
learn!(s, strategy(ProxGrad(s), MaxIter(25), TimeLimit(.5)))
learn!(s, Sweep(s))
learn!(s, LinRegCholesky(s))
```